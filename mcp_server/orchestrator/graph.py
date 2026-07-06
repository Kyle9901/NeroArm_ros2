"""
LangGraph execution engine — dynamically builds a StateGraph from a pipeline.

Each pipeline step becomes a graph node.  Conditional edges handle success / retry / fail.
MemorySaver provides checkpointing for future needs_input support.
"""

import sys
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import StateGraph, START, END

from .planner import plan_pipeline, SKILL_SCHEMA
from .planner_config import PLANNING_LLM_CONFIG
from ..components import motion
from ..skills import manipulation, perception, prepare as prepare_skill
from ..skills.base import SkillResult, GraspGeometry
from .place_resolver import resolve_place as _resolve_place

# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Skill registry — functions callable by graph nodes
# ═══════════════════════════════════════════════════════════════════════════════════════════

_MAX_RETRIES = 3


def _skill_go_home(bridge, **kwargs) -> SkillResult:
    if bridge.is_at_home():
        return SkillResult.success(already_home=True)
    result = motion.go_home(bridge)
    if result.ok:
        return SkillResult.success(already_home=False)
    return SkillResult.failure(result.error or "go_home failed", failed_step="go_home", retryable=True)


def _skill_resolve_place(bridge, place: str, **kwargs) -> SkillResult:
    try:
        coords = _resolve_place(bridge, place)
        return SkillResult.success(**coords)
    except Exception as e:
        return SkillResult.failure(str(e), failed_step="resolve_place", retryable=False)


def _skill_stack_on(bridge, x: float, y: float, z: float, height: float = 0.05, **kwargs) -> SkillResult:
    return SkillResult.success(x=x, y=y, z=z + height)


def _skill_open_gripper(bridge, **kwargs) -> SkillResult:
    result = motion.control_gripper(bridge, 0.10, duration=1.5)
    if result.ok:
        return SkillResult.success()
    return SkillResult.failure(result.error or "open gripper failed", failed_step="open_gripper", retryable=True)


def _skill_close_gripper(bridge, **kwargs) -> SkillResult:
    result = motion.control_gripper(bridge, 0.02, duration=1.5)
    if result.ok:
        return SkillResult.success()
    return SkillResult.failure(result.error or "close gripper failed", failed_step="close_gripper", retryable=True)


def _skill_wave(bridge, times: int = 5, **kwargs) -> SkillResult:
    right = [0, -20, 0, 40, 0, 20, 10]
    left = [0, -20, 0, 40, 0, -20, 10]
    for i in range(times):
        target = right if i % 2 == 0 else left
        result = motion.move_joints(bridge, target, timeout=10.0, velocity=0.3, accel=0.5)
        if not result.ok:
            return SkillResult.failure(result.error or "wave failed", failed_step="wave", retryable=False)
    return SkillResult.success(waves=times)


def _skill_nod(bridge, times: int = 4, **kwargs) -> SkillResult:
    up = [0, -20, 0, 80, 0, 0, 90]
    down = [0, -20, 0, 80, 0, 0, 70]
    for i in range(times):
        target = up if i % 2 == 0 else down
        result = motion.move_joints(bridge, target, timeout=10.0, velocity=0.4, accel=0.4)
        if not result.ok:
            return SkillResult.failure(result.error or "nod failed", failed_step="nod", retryable=False)
    return SkillResult.success(nods=times)


def _skill_handshake(bridge, times: int = 4, **kwargs) -> SkillResult:
    start = [0, 40, 0, 70, 0, 0, 25]
    down = [0, 40, 0, 65, 0, 0, 25]
    up = [0, 40, 0, 75, 0, 0, 25]
    result = motion.move_joints(bridge, start, timeout=10.0, velocity=0.4, accel=0.4)
    if not result.ok:
        return SkillResult.failure(result.error or "handshake failed", failed_step="handshake", retryable=False)
    for i in range(times):
        target = down if i % 2 == 0 else up
        result = motion.move_joints(bridge, target, timeout=10.0, velocity=0.4, accel=0.4)
        if not result.ok:
            return SkillResult.failure(result.error or "handshake failed", failed_step="handshake", retryable=False)
    return SkillResult.success(shakes=times)


def _skill_detect_by_color(bridge, vlm, target: str, **kwargs) -> SkillResult:
    return perception.locate_object(bridge, vlm, target, use_vlm=False)


# All skills the LLM can reference
_SKILL_FNS = {
    "go_home": _skill_go_home,
    "locate_object": perception.locate_object,
    "detect_by_color": _skill_detect_by_color,
    "scan_scene": perception.scan_scene,
    "grasp_object": manipulation.grasp_object,
    "place_object": manipulation.place_object,
    "resolve_place": _skill_resolve_place,
    "stack_on": _skill_stack_on,
    "prepare": prepare_skill.prepare,
    "open_gripper": _skill_open_gripper,
    "close_gripper": _skill_close_gripper,
    "wave": _skill_wave,
    "nod": _skill_nod,
    "handshake": _skill_handshake,
}

_VLM_SKILLS = {"locate_object", "detect_by_color"}


# ═══════════════════════════════════════════════════════════════════════════════════════════
#  TaskState
# ═══════════════════════════════════════════════════════════════════════════════════════════

class TaskState(TypedDict, total=False):
    pipeline: list[dict]
    step_outputs: dict[str, Any]
    messages: list[str]
    status: Literal["running", "completed", "failed"]
    error: str | None
    holding: bool | None
    user_output: dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Graph builder
# ═══════════════════════════════════════════════════════════════════════════════════════════

def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _resolve_args(args: dict, step_outputs: dict) -> dict:
    resolved = {}
    for key, val in args.items():
        if isinstance(val, str) and val.startswith("$"):
            ref = val[1:]
            parts = ref.split(".")
            cur = step_outputs
            for p in parts:
                cur = cur[p]
            resolved[key] = cur
        else:
            resolved[key] = val
    return resolved


def _make_step_node(step: dict, bridge, vlm, max_retries: int = _MAX_RETRIES):
    skill_name = step["skill"]
    skill_fn = _SKILL_FNS[skill_name]
    needs_vlm = skill_name in _VLM_SKILLS
    step_name = step["name"]
    is_grasp = skill_name == "grasp_object"

    def node(state: TaskState) -> dict:
        result = None
        args = None
        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"[graph] retry {step_name} ({attempt}/{max_retries})", file=sys.stderr, flush=True)
                motion.go_home(bridge)

            args = _resolve_args(step.get("args", {}), state.get("step_outputs", {}))
            if needs_vlm:
                result = skill_fn(bridge, vlm, **args)
            else:
                result = skill_fn(bridge, **args)

            if isinstance(result, SkillResult):
                if not result.ok:
                    print(f"[graph] {step_name} attempt {attempt}: error={result.error}, failed_step={result.failed_step}, retryable={result.retryable}", file=sys.stderr, flush=True)
                if result.ok and is_grasp and result.holding is False:
                    print(f"[graph] {step_name} attempt {attempt}: holding=false, retrying", file=sys.stderr, flush=True)
                    result = SkillResult.failure(
                        "holding=false: no object grasped",
                        failed_step=result.failed_step or step_name,
                        retryable=True,
                        holding=False,
                        **result.data,
                    )
                if result.ok or not result.retryable:
                    break
            else:
                break

        outputs = dict(state.get("step_outputs", {}))
        if isinstance(result, SkillResult):
            outputs[step_name] = _json_safe({
                "_ok": result.ok,
                "_error": result.error,
                "_retryable": result.retryable,
                "_failed_step": result.failed_step,
                "_recovered": result.recovered,
                "_holding": result.holding,
                **result.data,
            })
        elif isinstance(result, dict):
            outputs[step_name] = _json_safe({"_ok": True, "_error": None, "_retryable": False, **result})
        else:
            outputs[step_name] = {"_ok": True, "_error": None, "_retryable": False, "value": str(result)}

        if isinstance(result, SkillResult) and result.ok:
            _update_semantic_context(bridge, skill_name, step_name, args or {}, outputs[step_name], state.get("step_outputs", {}))

        okay = outputs[step_name].get("_ok", True)
        print(f"[graph] {step_name}: {'OK' if okay else 'FAIL'}", file=sys.stderr, flush=True)
        return {"step_outputs": outputs}
    return node


def _nearest_located_object(args: dict, prior_outputs: dict) -> str | None:
    x, y, z = args.get("x"), args.get("y"), args.get("z")
    best_name, best_dist = None, float("inf")
    for output in prior_outputs.values():
        if not isinstance(output, dict) or "target" not in output:
            continue
        if not all(k in output for k in ("x", "y", "z")):
            continue
        try:
            dist = abs(float(output["x"]) - float(x)) + abs(float(output["y"]) - float(y)) + abs(float(output["z"]) - float(z))
        except Exception:
            continue
        if dist < best_dist:
            best_name, best_dist = output.get("target"), dist
    return best_name


def _update_semantic_context(bridge, skill_name: str, step_name: str,
                             args: dict, output: dict, prior_outputs: dict) -> None:
    holding = bridge.get_holding()
    if not holding:
        bridge.update_task_context(grasped_object=None)

    if skill_name == "grasp_object":
        if output.get("_holding") is True or bridge.get_holding():
            obj = _nearest_located_object(args, prior_outputs) or "unknown object"
            bridge.update_task_context(
                grasped_object=obj,
                last_action="grasp",
                pick_x=output.get("pick_x"),
                pick_y=output.get("pick_y"),
                pick_z=output.get("pick_z"),
            )
            bridge.add_recent_action(f"grasped {obj}")
        return

    if skill_name == "place_object":
        bridge.update_task_context(
            grasped_object=None,
            last_action="place",
            last_place={"x": args.get("x"), "y": args.get("y"), "z": args.get("z")},
        )
        bridge.add_recent_action("placed object")
        return

    if skill_name == "open_gripper":
        bridge.set_holding(False)
        bridge.update_task_context(grasped_object=None, last_action="open_gripper")
        bridge.add_recent_action("opened gripper")
        return

    if skill_name == "close_gripper":
        bridge.update_task_context(last_action="close_gripper")
        bridge.add_recent_action("closed gripper")


def _make_router(step_index: int, pipeline_len: int):
    def router(state: TaskState) -> Literal["next", "fail", "done"]:
        if state.get("status") == "failed":
            return "fail"

        pipeline = state["pipeline"]
        step = pipeline[step_index]
        step_name = step["name"]

        wrapper = state.get("step_outputs", {}).get(step_name, {})
        ok = wrapper.get("_ok", True)
        error = wrapper.get("_error")

        if ok or error is None:
            if step_index + 1 >= pipeline_len:
                return "done"
            return "next"
        return "fail"
    return router


def _make_done_node(user_visible: list[str] | None = None):
    def node(state: TaskState) -> dict:
        outputs = state.get("step_outputs", {})
        visible = {}
        if user_visible:
            for ref in user_visible:
                parts = ref.split(".")
                cur = outputs
                try:
                    for p in parts:
                        cur = cur[p]
                    visible[ref] = cur
                except (KeyError, TypeError):
                    continue
        return {
            "status": "completed",
            "user_output": visible,
            "messages": state.get("messages", []) + ["任务完成"],
        }
    return node


def _make_fail_node():
    def node(state: TaskState) -> dict:
        return {
            "status": "failed",
            "error": state.get("error") or "Task failed",
            "messages": state.get("messages", []) + ["任务失败"],
        }
    return node


def _make_cleanup_node(bridge):
    def node(state: TaskState) -> dict:
        result = motion.go_home(bridge)
        msg = "任务结束，已回到 home" if result.ok else f"任务结束，回 home 失败: {result.error}"
        return {"messages": state.get("messages", []) + [msg]}
    return node


def build_graph(pipeline: list[dict], bridge, vlm,
                user_visible: list[str] | None = None,
                max_retries: int = _MAX_RETRIES):
    graph = StateGraph(TaskState)
    n = len(pipeline)

    for i, step in enumerate(pipeline):
        retries = 1 if step.get("skill") == "grasp_object" else max_retries
        graph.add_node(f"step_{i}", _make_step_node(step, bridge, vlm, retries))

    graph.add_node("done", _make_done_node(user_visible))
    graph.add_node("cleanup_home", _make_cleanup_node(bridge))
    graph.add_node("fail", _make_fail_node())

    graph.add_edge(START, "step_0")

    for i in range(n - 1):
        graph.add_conditional_edges(
            f"step_{i}",
            _make_router(i, n),
            {"next": f"step_{i+1}", "fail": "fail"},
        )

    graph.add_conditional_edges(
        f"step_{n-1}",
        _make_router(n - 1, n),
        {"next": "done", "done": "done", "fail": "fail"},
    )

    graph.add_edge("done", "cleanup_home")
    graph.add_edge("cleanup_home", END)
    graph.add_edge("fail", END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Fast rule matching — skips LLM for simple, deterministic tasks
# ═══════════════════════════════════════════════════════════════════════════════════════════

def _fast_route(task: str, bridge) -> list[dict] | None:
    """Return a pipeline for simple tasks, or None to fall through to LLM."""
    text = task.strip().lower()
    holding = bridge.get_holding()

    # ── "放下" / "放回原位" / "放置" ──
    if any(w in text for w in ("放下", "放回", "放下来", "放置", "put down", "drop")):
        if not holding:
            return None  # will be caught as error by LLM
        ctx = bridge.get_task_context()
        x = ctx.get("pick_x")
        y = ctx.get("pick_y")
        z = ctx.get("pick_z") or 0.0
        if x is None or y is None:
            # No pick position stored, fall back to default place
            place = bridge.get_place_pose()
            x, y, z = place["x"], place["y"], place["z"]
        return [
            {"name": "place", "skill": "place_object", "args": {"x": x, "y": y, "z": z}},
        ]

    # ── "回home" / "归位" / "回家" ──
    if any(w in text for w in ("回home", "归位", "回家", "home", "go home")):
        return [{"name": "home", "skill": "go_home", "args": {}}]

    # ── "打开夹爪" / "松手" / "释放" ──
    if any(w in text for w in ("打开夹爪", "松手", "释放", "open gripper", "release")):
        return [{"name": "open", "skill": "open_gripper", "args": {}}]

    # ── "闭合夹爪" / "夹紧" ──
    if any(w in text for w in ("闭合夹爪", "夹紧", "close gripper")):
        return [{"name": "close", "skill": "close_gripper", "args": {}}]

    # ── "挥手" / "wave" / "摆动" ──
    if any(w in text for w in ("挥手", "wave", "摆动", "打招呼", "摇手")):
        return [{"name": "wave", "skill": "wave", "args": {}}]

    # ── "点头" / "nod" ──
    if any(w in text for w in ("点头", "nod")):
        return [{"name": "nod", "skill": "nod", "args": {}}]

    # ── "握手" / "handshake" ──
    if any(w in text for w in ("握手", "handshake")):
        return [{"name": "handshake", "skill": "handshake", "args": {}}]

    return None


# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Public executor
# ═══════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class GraphResult:
    status: Literal["completed", "failed", "needs_input"]
    messages: list[str] = field(default_factory=list)
    user_output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    template: str | None = None
    session_id: str | None = None
    question: str | None = None
    options: list[dict] = field(default_factory=list)


class GraphExecutor:
    def __init__(self, bridge, vlm):
        self.bridge = bridge
        self.vlm = vlm

    def execute_task(self, task: str, params: dict[str, Any] | None = None) -> GraphResult:
        params = params or {}

        # ── 1. Fast rule matching (skip LLM) ──
        pipeline = _fast_route(task, self.bridge)
        if pipeline is not None:
            print(f"[graph] fast route: {task}", file=sys.stderr, flush=True)
        else:
            # ── 2. Planning LLM ──
            context = self.bridge.build_planning_context()
            print(f"[graph] planning: {task}\n{context}", file=sys.stderr, flush=True)
            pipeline = plan_pipeline(task, extra_context=context)
            if pipeline is None:
                return GraphResult(status="failed", error="Planning LLM returned no pipeline")
        steps_str = " → ".join(s["name"] for s in pipeline)
        print(f"[graph] pipeline: {steps_str}", file=sys.stderr, flush=True)

        # ── 3. Auto-prepend prepare if missing ──
        if not pipeline or pipeline[0].get("skill") != "prepare":
            pipeline.insert(0, {"name": "prepare", "skill": "prepare", "args": {}})

        # ── 4. Inject user params into pipeline args ──
        for step in pipeline:
            for key, val in params.items():
                if key in step.get("args", {}):
                    step["args"][key] = val

        # ── 5. Validate pipeline ──
        for step in pipeline:
            skill = step.get("skill")
            if skill not in _SKILL_FNS:
                return GraphResult(status="failed", error=f"Unknown skill: {skill}")
            if skill in ("locate_object", "grasp_object", "place_object") and "args" not in step:
                return GraphResult(status="failed", error=f"Step {step['name']} missing args")

        # ── 6. Build + run graph ──
        graph = build_graph(pipeline, self.bridge, self.vlm)

        initial: TaskState = {
            "pipeline": pipeline,
            "step_outputs": {},
            "messages": [],
            "status": "running",
            "error": None,
            "holding": None,
            "user_output": {},
        }

        final = graph.invoke(initial)

        return GraphResult(
            status=final.get("status", "failed"),
            messages=final.get("messages", []),
            user_output=final.get("user_output", {}),
            error=final.get("error"),
        )