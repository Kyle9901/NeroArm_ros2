"""
Planning LLM — generates a structured pipeline from a task description.
"""

import json
import re

import requests

from .planner_config import PLANNING_LLM_CONFIG
from .task_spec import TaskSpec, TaskSpecError, task_spec_from_dict

# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Skill registry — descriptions the LLM sees
# ═══════════════════════════════════════════════════════════════════════════════════════════

SKILL_SCHEMA = {
    "go_home": {
        "description": "将机械臂移动到全局视觉观察位，确保相机有清晰视野。抓取任务的第一步。",
        "args": {},
        "returns": "ok",
    },
    "locate_object": {
        "description": "检测物体并计算 3D 基坐标。方块用 HSV 实时顶面几何；水瓶/圆柱/易拉罐强制用 YOLO 加 5 帧深度圆柱拟合；其他目标按检测级联处理。",
        "args": {"target": "string — 物体描述"},
        "returns": "{x, y, z, geometry, bbox, source, debug_image}",
    },
    "detect_by_color": {
        "description": "纯 HSV 颜色检测，不调 VLM。仅用于纯色物块：blue/red/green/yellow/purple/orange/cyan。",
        "args": {"target": "string — 颜色名，如 '蓝色方块'、'red block'"},
        "returns": "{x, y, z, geometry, bbox, source, debug_image}",
    },
    "scan_scene": {
        "description": "扫描桌面所有可见色块，返回带 3D 坐标的列表。",
        "args": {},
        "returns": "{blocks, count, debug_image}",
    },
    "grasp_object": {
        "description": "抓取 (x,y,z) 处的物体。方块及直立/横放圆柱会生成并筛选多姿态候选；其他物体使用兼容抓取。",
        "args": {"x": "float — base_link X", "y": "float — base_link Y", "z": "float — 物体表面 Z", "geometry": "object — locate 返回的完整几何信息", "target": "string — 原始目标描述"},
        "returns": "{holding, pick_x, pick_y, pick_z, gripper_width, selected_candidate}",
    },
    "place_object": {
        "description": "将手持物体放置到 (x,y,z)。关系放置使用已计算的 placement_candidate 保持物体-TCP变换。",
        "args": {"x": "float — base_link X", "y": "float — base_link Y", "z": "float — 放置面 Z", "placement_candidate": "object — 可选的几何关系放置候选"},
        "returns": "{place_x, place_y, place_z}",
    },
    "open_gripper": {
        "description": "打开夹爪，释放手持物体。",
        "args": {},
        "returns": "ok",
    },
    "close_gripper": {
        "description": "闭合夹爪。",
        "args": {},
        "returns": "ok",
    },
    "wave": {
        "description": "机械臂挥手动作。通过 joint6 左右摆动实现。",
        "args": {"times": "int — 摆动次数，默认 3"},
        "returns": "{waves}",
    },
    "nod": {
        "description": "机械臂点头动作。在视觉观察位通过 joint7 上下摆动实现。",
        "args": {"times": "int — 点头次数，默认 4"},
        "returns": "{nods}",
    },
    "handshake": {
        "description": "机械臂握手动作。先移动到握手初始位置，再通过 joint4 上下摆动实现。",
        "args": {"times": "int — 摆动次数，默认 2"},
        "returns": "{shakes}",
    },
    "resolve_place": {
        "description": "将方位词（右边/左边/中间/前面/后面）转换为 base_link 坐标。",
        "args": {"place": "string — 方位词或坐标"},
        "returns": "{x, y, z}",
    },
    "stack_on": {
        "description": "根据源物体、支撑物体实时几何和抓取候选计算叠放TCP。",
        "args": {"source_geometry": "object", "support_geometry": "object", "selected_candidate": "object"},
        "returns": "{x, y, z, placement_candidate}",
    },
    "offset_from": {
        "description": "根据两个物体实时尺寸计算左/右/前/后方的无重叠放置点。",
        "args": {"source_geometry": "object", "reference_geometry": "object", "selected_candidate": "object", "relation": "right_of|left_of|in_front_of|behind"},
        "returns": "{x, y, z, placement_candidate}",
    },
    "verify_placement": {
        "description": "释放物体并回观察位后，验证目标物体的XY位置和顶面高度是否满足放置后置条件。",
        "args": {
            "observed_geometry": "object — 释放后重新检测的几何",
            "expected_x": "float",
            "expected_y": "float",
            "expected_surface_z": "float",
        },
        "returns": "{verified, xy_error, surface_z_error}",
    },
    "prepare": {
        "description": "启动/检查机械臂、相机、手眼标定 TF 节点。",
        "args": {},
        "returns": "{already_ready}",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Few-shot examples
# ═══════════════════════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = [
    {
        "task": "把蓝色方块放到右边",
        "pipeline": [
            {"name": "prepare", "skill": "prepare", "args": {}},
            {"name": "go_home", "skill": "go_home", "args": {}},
            {"name": "locate", "skill": "detect_by_color", "args": {"target": "蓝色方块"}},
            {"name": "grasp", "skill": "grasp_object", "args": {"x": "$locate.x", "y": "$locate.y", "z": "$locate.z", "geometry": "$locate.geometry", "target": "蓝色方块"}},
            {"name": "resolve", "skill": "resolve_place", "args": {"place": "右边"}},
            {"name": "place", "skill": "place_object", "args": {"x": "$resolve.x", "y": "$resolve.y", "z": "$resolve.z"}},
        ],
    },
    {
        "task": "把蓝色方块放到红色物块上方",
        "pipeline": [
            {"name": "prepare", "skill": "prepare", "args": {}},
            {"name": "go_home", "skill": "go_home", "args": {}},
            {"name": "locate_blue", "skill": "detect_by_color", "args": {"target": "蓝色方块"}},
            {"name": "locate_red", "skill": "detect_by_color", "args": {"target": "红色物块"}},
            {"name": "grasp_blue", "skill": "grasp_object", "args": {"x": "$locate_blue.x", "y": "$locate_blue.y", "z": "$locate_blue.z", "geometry": "$locate_blue.geometry", "target": "蓝色方块"}},
            {"name": "stack", "skill": "stack_on", "args": {"source_geometry": "$grasp_blue.geometry", "support_geometry": "$locate_red.geometry", "selected_candidate": "$grasp_blue.selected_candidate"}},
            {"name": "place", "skill": "place_object", "args": {"x": "$stack.x", "y": "$stack.y", "z": "$stack.z", "placement_candidate": "$stack.placement_candidate"}},
        ],
    },
    {
        "task": "把蓝色方块放到红色物块右边",
        "pipeline": [
            {"name": "prepare", "skill": "prepare", "args": {}},
            {"name": "go_home", "skill": "go_home", "args": {}},
            {"name": "locate_blue", "skill": "detect_by_color", "args": {"target": "蓝色方块"}},
            {"name": "locate_red", "skill": "detect_by_color", "args": {"target": "红色物块"}},
            {"name": "grasp_blue", "skill": "grasp_object", "args": {"x": "$locate_blue.x", "y": "$locate_blue.y", "z": "$locate_blue.z", "geometry": "$locate_blue.geometry", "target": "蓝色方块"}},
            {"name": "relative_place", "skill": "offset_from", "args": {"source_geometry": "$grasp_blue.geometry", "reference_geometry": "$locate_red.geometry", "selected_candidate": "$grasp_blue.selected_candidate", "relation": "right_of"}},
            {"name": "place_blue", "skill": "place_object", "args": {"x": "$relative_place.x", "y": "$relative_place.y", "z": "$relative_place.z", "placement_candidate": "$relative_place.placement_candidate"}},
        ],
    },
    {
        "task": "扫描桌面",
        "pipeline": [
            {"name": "prepare", "skill": "prepare", "args": {}},
            {"name": "go_home", "skill": "go_home", "args": {}},
            {"name": "scan", "skill": "scan_scene", "args": {}},
        ],
    },
    {
        "task": "打开夹爪",
        "pipeline": [
            {"name": "prepare", "skill": "prepare", "args": {}},
            {"name": "open", "skill": "open_gripper", "args": {}},
        ],
    },
    {
        "task": "闭合夹爪",
        "pipeline": [
            {"name": "prepare", "skill": "prepare", "args": {}},
            {"name": "close", "skill": "close_gripper", "args": {}},
        ],
    },
    {
        "task": "回到观察位",
        "pipeline": [
            {"name": "prepare", "skill": "prepare", "args": {}},
            {"name": "home", "skill": "go_home", "args": {}},
        ],
    },
    {
        "task": "放下",
        "pipeline": [
            {"name": "prepare", "skill": "prepare", "args": {}},
            {"name": "resolve", "skill": "resolve_place", "args": {"place": "右边"}},
            {"name": "place", "skill": "place_object", "args": {"x": "$resolve.x", "y": "$resolve.y", "z": "$resolve.z"}},
        ],
    },
]

# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Prompt
# ═══════════════════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一个机械臂任务规划器。根据用户的任务描述和可用技能列表,输出一个 JSON pipeline。

## 可用技能

{skill_descriptions}

## 参数引用

使用 $step_name.field 引用之前步骤的输出。
例如: $locate.x 表示 locate 步骤输出的 x 字段。

## 规则

- 第一步必须是 prepare，确保机械臂和相机节点已启动。prepare 会自动跳过已就绪的节点。
- 第二步必须是 go_home（兼容技能名，实际目标是视觉观察位），确保相机有清晰视野。
- locate_object 必须在 grasp_object 之前，用 $locate.x/y/z 传入坐标。
- 对纯色物块（蓝/红/绿/黄/紫/橙/青），使用 detect_by_color（纯 HSV，不调 VLM）。速度快，不消耗 API 调用。
- 对水瓶、圆柱和易拉罐使用 locate_object；其内部强制执行 YOLO + 5 帧深度圆柱拟合，不用 VLM 猜测几何。
- 对工具和复杂纹理等其他非纯色物体，使用 locate_object 的通用检测级联。
- 方位词必须先用 resolve_place 转换，再用 $resolve.x/y/z 传入 place_object。
- "放到X上方/上面"必须先用 stack_on 叠加高度偏移，再用 $stack.x/y/z 传入 place_object。
- scan_scene 只检测不抓取，用于扫描桌面。
- open_gripper 和 close_gripper 用于单纯开关夹爪，不需要 go_home 或 locate。
- go_home 可以单独使用，用于回到视觉观察位。
- 如果额外上下文显示 holding=true 且用户说"放下/放到/放置"，说明手上已有物体，应直接 resolve_place + place_object，不要 locate_object 或 grasp_object。
- 如果额外上下文显示 holding=true 且 grasped_object 有值，应把用户说的"它/手上的/当前物体"理解为 grasped_object。
- 如果额外上下文显示 holding=false，则不能直接 place_object，必须先 locate_object + grasp_object。
- 每个步骤的 name 必须唯一。

## 关键规则：抓取后禁止检测

抓取物块后，手持物块会遮挡相机视野。因此：
- 所有 locate_object 和 scan_scene 必须在第一个 grasp_object 之前完成。
- 如果任务涉及多个物体，先全部检测完，再逐个抓取放置。
- 禁止在 grasp_object 之后出现任何 locate_object 或 scan_scene。
- open_gripper、close_gripper、go_home 不受此规则限制。

## 输出格式

只输出 JSON，不要 markdown 代码块，不要额外文字:

{{"pipeline": [{{"name": "step1", "skill": "skill_name", "args": {{...}}}}, ...]}}"""

USER_TEMPLATE = """任务: {task}

请输出 pipeline JSON。"""


TASK_SPEC_EXAMPLES = [
    {
        "task": "抓取红色物块",
        "result": {"intent": "pick", "source": {"name": "红色物块"}},
    },
    {
        "task": "抓取红色物块并放回原位置",
        "result": {
            "intent": "pick_place",
            "source": {"name": "红色物块"},
            "destination": {"kind": "original"},
        },
    },
    {
        "task": "把蓝色物块放到红色物块上方",
        "result": {
            "intent": "pick_place",
            "source": {"name": "蓝色物块"},
            "destination": {
                "kind": "relative",
                "relation": "on_top_of",
                "reference": {"name": "红色物块"},
            },
        },
    },
    {
        "task": "把蓝色物块放到桌面右边",
        "result": {
            "intent": "pick_place",
            "source": {"name": "蓝色物块"},
            "destination": {"kind": "named_zone", "name": "right"},
        },
    },
]


TASK_SPEC_PROMPT = """你是机械臂任务语义解析器，只负责把用户命令转换为 TaskSpec。
不要生成技能、步骤或 pipeline；执行顺序由确定性编译器负责。

可用 intent:
- pick: 抓取物体，需要 source
- pick_place: 抓取并放置，需要 source 和 destination
- place_held: 放置手上已有物体，需要 destination
- scan, go_home, open_gripper, close_gripper, wave, nod, handshake

source 格式: {"name": "物体描述"}
destination 只能是以下一种:
- {"kind": "original"}
- {"kind": "configured"}
- {"kind": "named_zone", "name": "right|left|center|front|back"}
- {"kind": "absolute", "x": 0.0, "y": 0.0, "z": 0.0}
- {"kind": "relative", "relation": "on_top_of|right_of|left_of|in_front_of|behind", "reference": {"name": "参考物体"}}

“红色物块右边”是相对红色物块，不是全局 right 区域。未知或含糊的目的地不要猜成默认位置。
如果上下文显示 holding=true，且用户要求放下、放回或放到某处，intent 必须是 place_held，不能再次 pick。
只输出一个 JSON 对象，不要 Markdown。参考示例:
{examples}
"""


# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════════════════

def _build_skill_descriptions() -> str:
    lines = []
    for name, schema in SKILL_SCHEMA.items():
        args_str = ", ".join(f"{k}: {v}" for k, v in schema["args"].items()) or "无"
        lines.append(f"- **{name}**({args_str}): {schema['description']} → {schema['returns']}")
    return "\n".join(lines)


def _build_few_shot() -> str:
    return json.dumps(FEW_SHOT_EXAMPLES, ensure_ascii=False, indent=2)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start >= 0:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[start:])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def plan_pipeline(task: str, extra_context: str = "") -> list[dict] | None:
    cfg = PLANNING_LLM_CONFIG
    if not cfg["api_key"]:
        raise RuntimeError(
            "PLANNING_LLM_API_KEY not set. "
            "Set the environment variable or edit orchestrator/planner_config.py"
        )

    skill_desc = _build_skill_descriptions()
    few_shot = _build_few_shot()

    system = SYSTEM_PROMPT.format(skill_descriptions=skill_desc)
    system += f"\n\n## 参考示例\n{few_shot}"
    if extra_context:
        system += f"\n\n## 额外上下文\n{extra_context}"

    user = USER_TEMPLATE.format(task=task)

    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 2000,
        "temperature": 0.1,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg['api_key']}",
    }

    timeout = float(cfg.get("timeout", 90))
    retries = int(cfg.get("retries", 2))
    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(cfg["api_url"], headers=headers, json=payload, timeout=timeout)
            break
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt >= retries:
                raise RuntimeError(f"Planning LLM request failed after {retries + 1} attempts: {e}") from e
    else:
        raise RuntimeError(f"Planning LLM request failed: {last_error}")

    if resp.status_code != 200:
        raise RuntimeError(f"Planning LLM HTTP {resp.status_code}: {resp.text[:300]}")

    body = resp.json()
    msg = body["choices"][0]["message"]
    text = msg.get("content", "").strip()

    # DeepSeek reasoning models: content may be empty if reasoning consumed all tokens.
    # Fall back to reasoning_content and try to extract JSON from the reasoning traces.
    if not text and "reasoning_content" in msg:
        text = msg["reasoning_content"].strip()

    parsed = _extract_json(text)
    if parsed is None:
        raise RuntimeError(f"Planning LLM returned unparseable output:\n{text[:500]}")
    pipeline = parsed.get("pipeline", [])
    if not pipeline:
        raise RuntimeError("Planning LLM returned empty pipeline")
    # Normalize: replace hallucinated skill names with valid ones
    _SKILL_ALIASES = {"detect_by_color": "locate_object"}
    for step in pipeline:
        skill = step.get("skill", "")
        if skill in _SKILL_ALIASES:
            step["skill"] = _SKILL_ALIASES[skill]
    return pipeline


def plan_task_spec(task: str, extra_context: str = "") -> TaskSpec:
    """Use the planning LLM only as a typed semantic-parser fallback."""
    cfg = PLANNING_LLM_CONFIG
    if not cfg["api_key"]:
        raise TaskSpecError(
            "Task is not covered by the deterministic parser and "
            "PLANNING_LLM_API_KEY is not set"
        )

    system = TASK_SPEC_PROMPT.replace(
        "{examples}",
        json.dumps(TASK_SPEC_EXAMPLES, ensure_ascii=False, indent=2),
    )
    if extra_context:
        system += f"\n\n当前机器人上下文，仅用于消解指代和持物状态:\n{extra_context}"
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"任务: {task}\n只输出 TaskSpec JSON。"},
        ],
        "max_tokens": 1000,
        "temperature": 0.0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg['api_key']}",
    }
    timeout = float(cfg.get("timeout", 90))
    retries = int(cfg.get("retries", 2))
    response = None
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                cfg["api_url"],
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            break
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt >= retries:
                raise RuntimeError(
                    f"Planning LLM request failed after {retries + 1} attempts: {exc}"
                ) from exc
    if response is None:
        raise RuntimeError(f"Planning LLM request failed: {last_error}")
    if response.status_code != 200:
        raise RuntimeError(
            f"Planning LLM HTTP {response.status_code}: {response.text[:300]}"
        )

    body = response.json()
    message = body["choices"][0]["message"]
    text = message.get("content", "").strip()
    if not text and "reasoning_content" in message:
        text = message["reasoning_content"].strip()
    parsed = _extract_json(text)
    if parsed is None:
        raise TaskSpecError(f"Planning LLM returned invalid TaskSpec JSON: {text[:500]}")
    return task_spec_from_dict(parsed)
