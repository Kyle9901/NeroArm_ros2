"""
Planning LLM — generates a structured pipeline from a task description.
"""

import json
import re

import requests

from .planner_config import PLANNING_LLM_CONFIG

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
        "description": "将手持物体放置到 (x,y,z)。z 是放置面高度。自动处理下降和松开。",
        "args": {"x": "float — base_link X", "y": "float — base_link Y", "z": "float — 放置面 Z"},
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
        "description": "在参考物体坐标上叠加高度偏移。用于'放到X上方'场景。",
        "args": {"x": "float — 参考 X", "y": "float — 参考 Y", "z": "float — 参考 Z", "height": "float — 叠加高度，默认 0.05m"},
        "returns": "{x, y, z}",
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
            {"name": "stack", "skill": "stack_on", "args": {"x": "$locate_red.x", "y": "$locate_red.y", "z": "$locate_red.z", "height": 0.05}},
            {"name": "place", "skill": "place_object", "args": {"x": "$stack.x", "y": "$stack.y", "z": "$stack.z"}},
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
            {"name": "resolve", "skill": "resolve_place", "args": {"place": "右边"}},
            {"name": "place_blue", "skill": "place_object", "args": {"x": "$resolve.x", "y": "$resolve.y", "z": "$resolve.z"}},
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
    m = re.search(r'\{[\s\S]*"pipeline"[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group())
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
