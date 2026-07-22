"""
LangGraph execution engine — dynamically builds a StateGraph from a pipeline.

Each pipeline step becomes a graph node.  Conditional edges handle success / retry / fail.
MemorySaver provides checkpointing for future needs_input support.
"""

import sys
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import StateGraph, START, END

from .planner import plan_task_spec
from .task_spec import (
    TaskSpecError,
    compile_task_spec,
    parse_fast_task,
    validate_pipeline,
)
from ..components import motion
from ..skills import basic, gestures, manipulation, perception, placement, prepare
from ..skills.base import SkillResult

# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Skill registry — functions callable by graph nodes
# ═══════════════════════════════════════════════════════════════════════════════════════════

_MAX_RETRIES = 3


# All skills the LLM can reference
_SKILL_FNS = {
    "go_home": basic.go_home,
    "locate_object": perception.locate_object,
    "detect_by_color": perception.detect_by_color,
    "scan_scene": perception.scan_scene,
    "grasp_object": manipulation.grasp_object,
    "place_object": manipulation.place_object,
    "resolve_place": placement.resolve_place,
    "stack_on": placement.stack_on,
    "offset_from": placement.offset_from,
    "verify_placement": placement.verify_placement,
    "prepare": prepare.prepare,
    "open_gripper": basic.open_gripper,
    "close_gripper": basic.close_gripper,
    "wave": gestures.wave,
    "nod": gestures.nod,
    "handshake": gestures.handshake,
}

_VLM_SKILLS = {"detect_by_color"}
_YOLO_SKILLS = {"locate_object", "scan_scene"}


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


def _task_stop_requested(bridge) -> bool:
    checker = getattr(bridge, "is_task_stop_requested", None)
    return bool(checker()) if callable(checker) else False


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


def _make_step_node(step: dict, bridge, vlm, yolo, max_retries: int = _MAX_RETRIES):
    skill_name = step["skill"]
    skill_fn = _SKILL_FNS[skill_name]
    needs_vlm = skill_name in _VLM_SKILLS
    needs_yolo = skill_name in _YOLO_SKILLS
    step_name = step["name"]
    is_grasp = skill_name == "grasp_object"

    def node(state: TaskState) -> dict:
        result = None
        args = None
        for attempt in range(max_retries + 1):
            if _task_stop_requested(bridge):
                result = SkillResult.failure(
                    "Task stopped by arm_stop before the next skill/retry",
                    failed_step=step_name,
                    retryable=False,
                    holding=bridge.get_holding(),
                    stop_requested=True,
                )
                break
            if attempt > 0:
                print(f"[graph] retry {step_name} ({attempt}/{max_retries})", file=sys.stderr, flush=True)
                # The grasp skill owns its retreat/recovery semantics.  Do not
                # request a software stop or force observation pose here:
                # doing so destroys the known holding state and invalidates
                # candidate-specific recovery.

            args = _resolve_args(step.get("args", {}), state.get("step_outputs", {}))
            if needs_yolo:
                # Perception skills do not share the same positional signature:
                # locate_object(..., target, yolo=None) would otherwise receive
                # yolo as ``target`` and then receive ``target`` again via args.
                result = skill_fn(bridge, vlm, yolo=yolo, **args)
            elif needs_vlm:
                result = skill_fn(bridge, vlm, **args)
            else:
                result = skill_fn(bridge, **args)

            if isinstance(result, SkillResult):
                if _task_stop_requested(bridge):
                    result = SkillResult.failure(
                        "Task stopped by arm_stop; no further motion will be submitted",
                        failed_step=step_name,
                        retryable=False,
                        holding=bridge.get_holding(),
                        stop_requested=True,
                    )
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
                selected_candidate=output.get("selected_candidate"),
                pick_geometry=output.get("geometry"),
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
        if _task_stop_requested(bridge):
            return {
                "status": "failed",
                "error": "Task stopped by arm_stop",
                "messages": state.get("messages", []) + [
                    "已停止后续软件动作；不会自动返回观察位"
                ],
            }
        holding = bridge.get_holding()
        if holding is True:
            return {
                "messages": state.get("messages", []) + [
                    "任务结束，机械臂保持 carry 姿态并继续持物"
                ]
            }
        if holding is None:
            return {
                "messages": state.get("messages", []) + [
                    "任务结束，但持物状态不确定；为避免误动作未返回观察位"
                ]
            }
        pipeline = state.get("pipeline", [])
        if isinstance(pipeline, list):
            last_home = max(
                (
                    index for index, step in enumerate(pipeline)
                    if isinstance(step, dict) and step.get("skill") == "go_home"
                ),
                default=-1,
            )
            last_other_motion = max(
                (
                    index for index, step in enumerate(pipeline)
                    if isinstance(step, dict) and step.get("skill") in {
                        "grasp_object", "place_object", "open_gripper",
                        "close_gripper", "wave", "nod", "handshake",
                    }
                ),
                default=-1,
            )
            if last_home > last_other_motion:
                return {
                    "messages": state.get("messages", []) + [
                        "任务结束，机械臂已在观察位"
                    ]
                }
        result = motion.go_home(bridge)
        if result.ok:
            return {
                "messages": state.get("messages", []) + ["任务结束，已回到观察位"]
            }
        return {
            "status": "failed",
            "error": result.error or "任务结束后返回观察位失败",
            "messages": state.get("messages", []) + [
                f"任务动作已完成，但回观察位失败: {result.error}"
            ],
        }
    return node


def build_graph(pipeline: list[dict], bridge, vlm, yolo,
                user_visible: list[str] | None = None,
                max_retries: int = _MAX_RETRIES):
    graph = StateGraph(TaskState)
    n = len(pipeline)

    for i, step in enumerate(pipeline):
        skill = step.get("skill")
        retries = 1 if skill in ("grasp_object", "prepare") else max_retries
        graph.add_node(f"step_{i}", _make_step_node(step, bridge, vlm, yolo, retries))

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
    """Compatibility entrypoint backed by typed parsing and compilation."""
    spec = parse_fast_task(task)
    return compile_task_spec(spec, bridge) if spec is not None else None


def _grasp_start_error(pipeline: list[dict], holding: bool | None) -> str | None:
    """Reject a new pick before prepare/home can move or release anything."""
    if not any(step.get("skill") == "grasp_object" for step in pipeline):
        return None
    if holding is True:
        return "Cannot start a new grasp while an object is already held"
    if holding is None:
        return "Cannot start a new grasp while the holding state is unknown"
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
    def __init__(self, bridge, vlm, yolo=None):
        self.bridge = bridge
        self.vlm = vlm
        self.yolo = yolo

    def execute_task(self, task: str, params: dict[str, Any] | None = None) -> GraphResult:
        params = params or {}

        # ── 1. Both parsers produce TaskSpec; one compiler owns behavior ──
        try:
            spec = parse_fast_task(task)
            if spec is not None:
                print(f"[graph] fast route: {task}", file=sys.stderr, flush=True)
            else:
                context = self.bridge.build_planning_context()
                print(f"[graph] planning semantics: {task}\n{context}", file=sys.stderr, flush=True)
                spec = plan_task_spec(task, extra_context=context)
            pipeline = compile_task_spec(spec, self.bridge)
        except (TaskSpecError, RuntimeError) as exc:
            return GraphResult(status="failed", error=str(exc))
        holding_error = _grasp_start_error(pipeline, self.bridge.get_holding())
        if holding_error:
            return GraphResult(status="failed", error=holding_error)
        steps_str = " → ".join(s["name"] for s in pipeline)
        print(f"[graph] pipeline: {steps_str}", file=sys.stderr, flush=True)

        # ── 3. Auto-prepend prepare if missing ──
        if not pipeline or pipeline[0].get("skill") != "prepare":
            pipeline.insert(0, {"name": "prepare", "skill": "prepare", "args": {}})

        # Runtime infrastructure overrides belong specifically to prepare; they
        # are not object-detection or manipulation arguments.
        prepare_args = pipeline[0].setdefault("args", {})
        for key in ("can_port", "calib_name", "octomap_enabled"):
            if key in params:
                prepare_args[key] = params[key]

        # ── 4. Inject user params into pipeline args ──
        for step in pipeline:
            for key, val in params.items():
                if key in step.get("args", {}):
                    step["args"][key] = val

        # ── 5. Validate dataflow and safety invariants before any skill runs ──
        try:
            validate_pipeline(pipeline, set(_SKILL_FNS))
        except TaskSpecError as exc:
            return GraphResult(status="failed", error=str(exc))

        # ── 6. Build + run graph ──
        graph = build_graph(pipeline, self.bridge, self.vlm, self.yolo)

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
