"""Object location and scene scanning skills."""

from typing import TYPE_CHECKING

from .base import SkillResult
from ..components import perception
from ..config import runtime_config

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge
    from ..vlm_client import VlmClient
    from ..yolo_detector import YoloDetector


_COLOR_NAME_MAP = {
    "blue": "blue", "蓝色": "blue", "蓝": "blue",
    "red": "red", "红色": "red", "红": "red",
    "green": "green", "绿色": "green", "绿": "green",
    "yellow": "yellow", "黄色": "yellow", "黄": "yellow",
    "purple": "purple", "紫色": "purple", "紫": "purple",
    "orange": "orange", "橙色": "orange", "橙": "orange",
    "cyan": "cyan", "青色": "cyan", "青": "cyan",
}

def _target_color(target: str) -> str | None:
    target_lower = target.lower()
    for cn_name, en_name in _COLOR_NAME_MAP.items():
        if cn_name in target_lower:
            return en_name
    return None


def _with_3d(bridge: "RobotBridge", frame, detection: dict) -> SkillResult:
    center = detection.get("center_2d")
    if not center:
        return SkillResult.failure("Detection has no center_2d", failed_step="detect", retryable=True)
    pos = perception.pixel_to_3d(bridge, frame, center[0], center[1])
    if not pos.ok:
        return SkillResult.failure(pos.error or "3D projection failed", failed_step="pixel_to_3d", retryable=True)
    data = {**detection, **pos.data}
    print(f"[perception] 3D: x={pos.data['x']:.4f}, y={pos.data['y']:.4f}, z={pos.data['z']:.4f} (base_link)", flush=True)
    return SkillResult.success(**data)


def _with_bbox_3d(bridge: "RobotBridge", frame, detection: dict) -> SkillResult:
    """Use bbox_to_3d for better depth sampling from YOLO bboxes."""
    bbox = detection.get("bbox")
    if not bbox:
        return _with_3d(bridge, frame, detection)
    known_height = (
        bridge.get_color_block_height()
        if detection.get("source") == "CV" and detection.get("color") else None
    )
    pos = perception.bbox_to_3d(
        bridge, frame, bbox, known_object_height=known_height,
    )
    if not pos.ok:
        # Fallback to center point
        return _with_3d(bridge, frame, detection)
    data = {**detection, **pos.data}
    print(f"[perception] 3D(bbox): x={pos.data['x']:.4f}, y={pos.data['y']:.4f}, z={pos.data['z']:.4f} "
          f"method={pos.data.get('method', '?')} (base_link)", flush=True)
    return SkillResult.success(**data)


def _compute_iou(bbox_a: list[int], bbox_b: list[int]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    return inter / (area_a + area_b - inter)


def _compute_3d_distance(pos_a: dict, pos_b: dict) -> float:
    """Compute Euclidean distance between two 3D positions."""
    try:
        dx = float(pos_a.get("x", 0)) - float(pos_b.get("x", 0))
        dy = float(pos_a.get("y", 0)) - float(pos_b.get("y", 0))
        dz = float(pos_a.get("z", 0)) - float(pos_b.get("z", 0))
        return (dx * dx + dy * dy + dz * dz) ** 0.5
    except (TypeError, ValueError):
        return float("inf")


def fuse_detections(hsv_blocks: list[dict], yolo_objects: list[dict],
                    iou_threshold: float = 0.3,
                    distance_threshold: float = 0.05) -> list[dict]:
    """Fuse HSV and YOLO detections into a unified object list.

    Strategy (see yolo-integration-plan.md #13):
      - If IoU > 0.3 or 3D distance < 0.05m → same object
      - Fused label: HSV color + YOLO class (e.g., "blue_cube")
      - HSV bbox takes priority (pixel-level precision)
      - YOLO-only and HSV-only objects are preserved independently

    Returns a unified list of dicts, each with: class_name, bbox, center_2d, sources, color.
    """
    fused = []
    used_hsv = set()
    used_yolo = set()

    for i, hsv in enumerate(hsv_blocks):
        for j, yolo in enumerate(yolo_objects):
            if j in used_yolo:
                continue
            iou = _compute_iou(hsv.get("bbox", []), yolo.get("bbox", []))
            dist = _compute_3d_distance(hsv, yolo)
            if iou > iou_threshold or dist < distance_threshold:
                # Fuse: HSV color + YOLO class
                color = hsv.get("color", "unknown")
                yolo_class = yolo.get("class", "object")
                fused_label = f"{color}_{yolo_class}"
                fused.append({
                    "class_name": fused_label,
                    "color": color,
                    "yolo_class": yolo_class,
                    "bbox": hsv.get("bbox"),  # HSV bbox is more precise
                    "center_2d": hsv.get("center_2d"),
                    "confidence": yolo.get("confidence"),
                    "sources": ["HSV", "YOLO"],
                    "area_px": hsv.get("area_px", 0),
                    "has_3d": hsv.get("has_3d", False),
                    "x": hsv.get("x"), "y": hsv.get("y"), "z": hsv.get("z"),
                })
                used_hsv.add(i)
                used_yolo.add(j)
                break

    # Add unmatched HSV blocks
    for i, hsv in enumerate(hsv_blocks):
        if i not in used_hsv:
            hsv["class_name"] = hsv.get("color", "unknown")
            hsv["sources"] = ["HSV"]
            fused.append(hsv)

    # Add unmatched YOLO objects
    for j, yolo in enumerate(yolo_objects):
        if j not in used_yolo:
            yolo["class_name"] = yolo.get("class", "unknown")
            yolo["color"] = None
            yolo["sources"] = ["YOLO"]
            yolo["area_px"] = yolo.get("area", 0)
            fused.append(yolo)

    fused.sort(key=lambda o: o.get("area_px", 0), reverse=True)
    return fused


def locate_object(bridge: "RobotBridge", vlm: "VlmClient", target: str,
                  yolo: "YoloDetector" = None,
                  location_hint: str = "", use_vlm: bool = True) -> SkillResult:
    """Locate an object using HSV → YOLO → VLM cascade.

    Detection priority:
      1. HSV color detection (pixel-level, fast)
      2. YOLO detection (general objects, fast, offline)
      3. VLM fallback (semantic, slow, API-dependent)

    Args:
        bridge: RobotBridge instance.
        vlm: VlmClient instance.
        target: Object description, e.g. "bottle", "蓝色方块".
        yolo: YoloDetector instance (optional). Injected by GraphExecutor.
        location_hint: Spatial hint (left/right/center/top/bottom).
        use_vlm: Whether to use VLM as final fallback.
    """
    frame_result = perception.capture_image(bridge)
    if not frame_result.ok:
        return SkillResult.failure(frame_result.error or "capture failed", failed_step="capture", retryable=True)
    frame = frame_result.data["frame"]

    # ── Step 1: HSV color detection ──
    color_name = _target_color(target)
    if color_name:
        detected = perception.detect_by_color(frame, color_name, location_hint)
        if detected.ok and detected.data.get("found"):
            return _with_bbox_3d(bridge, frame, {"target": target, **detected.data})

    # ── Step 2: YOLO detection ──
    if yolo is not None:
        detected = perception.detect_by_yolo(yolo, frame, target, location_hint)
        if detected.ok and detected.data.get("found"):
            return _with_bbox_3d(bridge, frame, {"target": target, **detected.data})

    # ── Step 3: VLM fallback ──
    if not use_vlm or not runtime_config.vlm_fallback:
        msg = f"Object '{target}' not found (HSV + YOLO exhausted"
        if not runtime_config.vlm_fallback:
            msg += ", VLM fallback disabled"
        msg += ")"
        return SkillResult.failure(msg, failed_step="detect", retryable=True)

    detected = perception.detect_by_vlm(vlm, frame, target)
    if not detected.ok:
        return SkillResult.failure(detected.error or "VLM detection failed", failed_step="detect", retryable=True)
    if not detected.data.get("found"):
        return SkillResult.failure(f"Object '{target}' not found", failed_step="detect", retryable=True)
    return _with_3d(bridge, frame, {"target": target, **detected.data})


def detect_by_color(bridge: "RobotBridge", vlm: "VlmClient", target: str,
                    **kwargs) -> SkillResult:
    """Locate a solid-colour target without YOLO or VLM fallback."""
    return locate_object(bridge, vlm, target, yolo=None, use_vlm=False)


def scan_scene(bridge: "RobotBridge", vlm: "VlmClient" = None,
               yolo: "YoloDetector" = None,
               location_hint: str = "") -> SkillResult:
    """Scan the desktop — HSV color blocks + YOLO objects, fused and deduplicated.

    Returns a unified object list with class_name, color, bbox, 3D position, sources.
    """
    frame_result = perception.capture_image(bridge)
    if not frame_result.ok:
        return SkillResult.failure(frame_result.error or "capture failed", failed_step="capture", retryable=True)
    frame = frame_result.data["frame"]

    # ── HSV color blocks ──
    hsv_result = perception.detect_all_blocks(frame, location_hint)
    hsv_blocks = hsv_result.data.get("blocks", []) if hsv_result.ok else []

    # ── YOLO objects ──
    yolo_objects = []
    if yolo is not None:
        yolo_result = perception.detect_all_by_yolo(yolo, frame)
        yolo_objects = yolo_result.data.get("objects", []) if yolo_result.ok else []

    # ── Project 3D for HSV blocks ──
    for block in hsv_blocks:
        center = block.get("center_2d")
        if center:
            pos = perception.pixel_to_3d(bridge, frame, center[0], center[1])
            if pos.ok:
                block.update(pos.data)
                block["has_3d"] = True
            else:
                block["has_3d"] = False
                block["error"] = pos.error

    # ── Project 3D for YOLO objects (use bbox_to_3d for better sampling) ──
    for obj in yolo_objects:
        bbox = obj.get("bbox")
        if bbox:
            pos = perception.bbox_to_3d(bridge, frame, bbox)
            if pos.ok:
                obj.update(pos.data)
                obj["has_3d"] = True
            else:
                obj["has_3d"] = False
                obj["error"] = pos.error

    # ── Fuse and deduplicate ──
    fused = fuse_detections(hsv_blocks, yolo_objects)

    if not fused:
        return SkillResult.success(
            found=False,
            count=0,
            blocks=[],
            debug_image=hsv_result.data.get("debug_image"),
        )

    return SkillResult.success(
        found=True,
        count=len(fused),
        blocks=fused,
        debug_image=hsv_result.data.get("debug_image"),
    )
