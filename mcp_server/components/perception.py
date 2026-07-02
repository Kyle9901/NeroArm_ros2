"""Atomic perception components."""

import os
import time
from itertools import count
from typing import TYPE_CHECKING

from .base import ComponentResult, ImageFrame

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge
    from ..vlm_client import VlmClient


_DEBUG_DIR = os.environ.get("VLM_DEBUG_DIR", "/tmp/vlm_debug")
_FRAME_IDS = count(1)


def _save_debug(img, bboxes, labels, prefix="detect") -> str:
    import cv2
    from ..vlm_client import _draw_bboxes

    os.makedirs(_DEBUG_DIR, exist_ok=True)
    annotated = _draw_bboxes(img, bboxes, labels)
    now = time.time()
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    ms = int((now % 1) * 1000)
    path = os.path.join(_DEBUG_DIR, f"{prefix}_{ts}_{ms:03d}.jpg")
    cv2.imwrite(path, annotated)
    return path


def capture_image(bridge: "RobotBridge", timeout: float = 3.0) -> ComponentResult:
    pair = bridge.node.get_latest_images(timeout=timeout)
    if pair is None:
        return ComponentResult.failure("No image — camera may not be running")
    color, depth = pair
    frame = ImageFrame(
        frame_id=next(_FRAME_IDS),
        color=color,
        depth=depth,
        timestamp_s=time.time(),
    )
    return ComponentResult.success(frame=frame)


def detect_by_color(frame: ImageFrame, color_name: str, location_hint: str = "") -> ComponentResult:
    from ..vlm_client import detect_by_color as _detect_by_color

    bbox = _detect_by_color(frame.color, color_name, location_hint)
    if bbox is None:
        return ComponentResult.success(found=False, color=color_name, source="CV")
    xmin, ymin, xmax, ymax = bbox
    debug_path = _save_debug(frame.color, [[xmin, ymin, xmax, ymax]], [color_name])
    return ComponentResult.success(
        found=True,
        color=color_name,
        bbox=[xmin, ymin, xmax, ymax],
        center_2d=[int((xmin + xmax) / 2), int((ymin + ymax) / 2)],
        source="CV",
        debug_image=debug_path,
    )


def detect_all_blocks(frame: ImageFrame, location_hint: str = "") -> ComponentResult:
    from ..vlm_client import detect_all_color_blocks

    blocks = detect_all_color_blocks(frame.color, location_hint)
    if not blocks:
        return ComponentResult.success(found=False, count=0, blocks=[], source="CV")
    bboxes = [b["bbox"] for b in blocks]
    labels = [b["color"] for b in blocks]
    debug_path = _save_debug(frame.color, bboxes, labels, prefix="scan")
    return ComponentResult.success(
        found=True,
        count=len(blocks),
        blocks=blocks,
        source="CV",
        debug_image=debug_path,
    )


def detect_by_vlm(vlm: "VlmClient", frame: ImageFrame, target: str) -> ComponentResult:
    try:
        result = vlm.detect(frame.color, target)
    except Exception as e:
        return ComponentResult.failure(f"VLM detection failed: {e}")
    if result is None or not result.get("found"):
        return ComponentResult.success(found=False, target=target, source="VLM")
    return ComponentResult.success(**result)


def pixel_to_3d(bridge: "RobotBridge", frame: ImageFrame, u: int, v: int) -> ComponentResult:
    cam3d = bridge.node.compute_3d(u, v, frame.depth)
    if cam3d is None:
        return ComponentResult.failure(f"No valid depth at ({u},{v})", u=u, v=v)
    try:
        base = bridge.node.transform_to_base(cam3d["x_c"], cam3d["y_c"], cam3d["z_c"])
    except Exception as e:
        return ComponentResult.failure(f"TF transform failed: {e}")
    return ComponentResult.success(
        x=base["x"],
        y=base["y"],
        z=base["z"],
        frame_id="base_link",
        image_frame_id=frame.frame_id,
        depth_mm=cam3d["depth_mm"],
        valid_depth_points=cam3d["valid_points"],
    )
