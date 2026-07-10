"""Object location and scene scanning skills."""

from typing import TYPE_CHECKING

from .base import SkillResult
from ..components import perception

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge
    from ..vlm_client import VlmClient


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


def locate_object(bridge: "RobotBridge", vlm: "VlmClient", target: str,
                  location_hint: str = "", use_vlm: bool = True) -> SkillResult:
    frame_result = perception.capture_image(bridge)
    if not frame_result.ok:
        return SkillResult.failure(frame_result.error or "capture failed", failed_step="capture", retryable=True)
    frame = frame_result.data["frame"]

    color_name = _target_color(target)
    if color_name:
        detected = perception.detect_by_color(frame, color_name, location_hint)
        if detected.ok and detected.data.get("found"):
            return _with_3d(bridge, frame, {"target": target, **detected.data})

    if not use_vlm:
        return SkillResult.failure(f"Object '{target}' not found (HSV only, no VLM)", failed_step="detect", retryable=True)

    detected = perception.detect_by_vlm(vlm, frame, target)
    if not detected.ok:
        return SkillResult.failure(detected.error or "VLM detection failed", failed_step="detect", retryable=True)
    if not detected.data.get("found"):
        return SkillResult.failure(f"Object '{target}' not found", failed_step="detect", retryable=True)
    return _with_3d(bridge, frame, {"target": target, **detected.data})


def scan_scene(bridge: "RobotBridge", location_hint: str = "") -> SkillResult:
    frame_result = perception.capture_image(bridge)
    if not frame_result.ok:
        return SkillResult.failure(frame_result.error or "capture failed", failed_step="capture", retryable=True)
    frame = frame_result.data["frame"]

    detected = perception.detect_all_blocks(frame, location_hint)
    if not detected.ok:
        return SkillResult.failure(detected.error or "scan failed", failed_step="detect_all_blocks", retryable=True)
    if not detected.data.get("found"):
        return SkillResult.success(found=False, count=0, blocks=[], debug_image=detected.data.get("debug_image"))

    blocks = []
    for block in detected.data["blocks"]:
        center = block["center_2d"]
        pos = perception.pixel_to_3d(bridge, frame, center[0], center[1])
        item = dict(block)
        if pos.ok:
            item.update(pos.data)
            item["has_3d"] = True
        else:
            item["has_3d"] = False
            item["error"] = pos.error
        blocks.append(item)

    return SkillResult.success(
        found=True,
        count=len(blocks),
        blocks=blocks,
        debug_image=detected.data.get("debug_image"),
    )
