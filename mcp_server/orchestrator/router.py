"""Rule-based task router."""

from .templates import TEMPLATES
from .types import TaskTemplate


def route_template(task: str) -> TaskTemplate | None:
    text = task.lower()

    # visual_grasp: explicit visual-servo keywords or "grasp only" (no place)
    if any(p in text for p in ("视觉", "visual", "直接抓")):
        for t in TEMPLATES:
            if t.name == "visual_grasp":
                return t

    # pick_and_place: requires BOTH pick AND place semantics
    has_pick = any(p in text for p in ("抓", "拿", "取", "pick", "grasp"))
    has_place = any(p in text for p in ("放", "到", "place", "put"))
    if has_pick and has_place:
        for t in TEMPLATES:
            if t.name == "pick_and_place":
                return t

    # scan_scene: detection / scanning keywords
    if any(p in text for p in ("扫描", "识别", "检测", "scan", "detect", "物块")):
        for t in TEMPLATES:
            if t.name == "scan_scene":
                return t

    # Fallback: first template whose match_patterns hit
    for template in TEMPLATES:
        if any(p in text for p in template.match_patterns):
            return template
    return None
