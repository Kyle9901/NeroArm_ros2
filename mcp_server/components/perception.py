"""Atomic perception components."""

import os
import time
from itertools import count
from typing import TYPE_CHECKING

from .base import ComponentResult, ImageFrame

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge
    from ..vlm_client import VlmClient
    from ..yolo_detector import YoloDetector


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


def detect_by_yolo(yolo_detector: "YoloDetector", frame: ImageFrame,
                   target: str, location_hint: str = "") -> ComponentResult:
    """Detect a single object using YOLO.

    Args:
        yolo_detector: YoloDetector instance.
        frame: ImageFrame with color image.
        target: User-facing target description, e.g. "bottle", "矿泉水瓶".
        location_hint: Optional spatial hint.

    Returns:
        ComponentResult with found=True/False, bbox, center_2d, class, confidence, source.
    """
    try:
        result = yolo_detector.detect(frame.color, target, location_hint=location_hint)
    except Exception as e:
        return ComponentResult.failure(f"YOLO detection failed: {e}")
    if result is None or not result.get("found"):
        return ComponentResult.success(found=False, target=target, source="YOLO")

    bbox = result["bbox"]
    debug_path = _save_debug(
        frame.color, [bbox],
        [f"{result.get('class', '?')}:{result.get('confidence', 0):.2f}"],
        prefix="yolo",
    )
    return ComponentResult.success(
        found=True,
        target=target,
        bbox=bbox,
        center_2d=result["center_2d"],
        class_name=result.get("class", "unknown"),
        confidence=result.get("confidence", 0.0),
        source="YOLO",
        debug_image=debug_path,
    )


def detect_all_by_yolo(yolo_detector: "YoloDetector", frame: ImageFrame) -> ComponentResult:
    """Detect all objects in the scene using YOLO.

    Returns:
        ComponentResult with found=True/False, count, objects list.
    """
    try:
        objects = yolo_detector.detect_all(frame.color)
    except Exception as e:
        return ComponentResult.failure(f"YOLO detect_all failed: {e}")
    if not objects:
        return ComponentResult.success(found=False, count=0, objects=[], source="YOLO")

    bboxes = [o["bbox"] for o in objects]
    labels = [f"{o['class']}:{o['confidence']:.2f}" for o in objects]
    debug_path = _save_debug(frame.color, bboxes, labels, prefix="yolo_scan")

    return ComponentResult.success(
        found=True,
        count=len(objects),
        objects=objects,
        source="YOLO",
        debug_image=debug_path,
    )


def bbox_to_3d(bridge: "RobotBridge", frame: ImageFrame,
               bbox: list[int], margin_px: int = 2) -> ComponentResult:
    """Compute 3D position from a bounding box using progressive ROI expansion.

    Strategy (see yolo-integration-plan.md #8):
      1. Shrink bbox by 10% → take median depth of the inner ROI
      2. If #1 fails → shrink by 20%
      3. If #2 fails → expand bbox outward by 20px (catch desk surface near object)
      4. If #3 fails → try full image for nearest valid depth near bbox center
      5. If all fail → return failure

    Uses the median depth to find the pixel closest to the median,
    ensuring the 3D point lands on the actual object surface, not a hole.

    Args:
        bridge: RobotBridge instance.
        frame: ImageFrame with color and depth arrays.
        bbox: [xmin, ymin, xmax, ymax] in pixel coordinates.
        margin_px: Fallback margin for point-based 3D lookup.

    Returns:
        ComponentResult with x, y, z in base_link frame.
    """
    import numpy as np

    depth_img = frame.depth
    dh, dw = depth_img.shape[:2]
    xmin, ymin, xmax, ymax = [int(v) for v in bbox]

    # Clamp to image bounds
    xmin = max(0, min(xmin, dw - 1))
    xmax = max(0, min(xmax, dw - 1))
    ymin = max(0, min(ymin, dh - 1))
    ymax = max(0, min(ymax, dh - 1))

    if xmax <= xmin or ymax <= ymin:
        return ComponentResult.failure(f"bbox too small: ({xmin},{ymin})-({xmax},{ymax})")

    cx = int((xmin + xmax) / 2)
    cy = int((ymin + ymax) / 2)

    def _median_3d(roi_xmin, roi_xmax, roi_ymin, roi_ymax):
        """Extract ROI, compute median depth, find pixel closest to median, return 3D."""
        rx1 = max(0, roi_xmin)
        rx2 = min(dw, roi_xmax + 1)
        ry1 = max(0, roi_ymin)
        ry2 = min(dh, roi_ymax + 1)
        roi = depth_img[ry1:ry2, rx1:rx2]
        valid = roi[roi > 0]
        if len(valid) == 0:
            return None
        median_mm = float(np.median(valid))
        # Find the pixel closest to the median depth
        diff = np.abs(roi.astype(np.float32) - median_mm)
        diff[roi <= 0] = np.inf
        min_idx = np.unravel_index(np.argmin(diff), diff.shape)
        u_real = rx1 + min_idx[1]
        v_real = ry1 + min_idx[0]
        return pixel_to_3d(bridge, frame, u_real, v_real), median_mm

    # Step 1: Shrink by 10%
    bw = xmax - xmin
    bh = ymax - ymin
    shrink10_x = max(1, int(bw * 0.10))
    shrink10_y = max(1, int(bh * 0.10))
    result = _median_3d(xmin + shrink10_x, xmax - shrink10_x,
                        ymin + shrink10_y, ymax - shrink10_y)
    if result is not None:
        pos, depth_mm = result
        if pos.ok:
            pos.data["depth_mm"] = depth_mm
            pos.data["method"] = "bbox_shrink_10pct"
            return pos

    # Step 2: Shrink by 20%
    shrink20_x = max(1, int(bw * 0.20))
    shrink20_y = max(1, int(bh * 0.20))
    result = _median_3d(xmin + shrink20_x, xmax - shrink20_x,
                        ymin + shrink20_y, ymax - shrink20_y)
    if result is not None:
        pos, depth_mm = result
        if pos.ok:
            pos.data["depth_mm"] = depth_mm
            pos.data["method"] = "bbox_shrink_20pct"
            return pos

    # Step 3: Expand outward by 20px (catch desk surface near object)
    expand = 20
    result = _median_3d(xmin - expand, xmax + expand, ymin - expand, ymax + expand)
    if result is not None:
        pos, depth_mm = result
        if pos.ok:
            pos.data["depth_mm"] = depth_mm
            pos.data["method"] = "bbox_expand_20px"
            return pos

    # Step 4: Fallback to center point with margin
    result = pixel_to_3d(bridge, frame, cx, cy)
    if result.ok:
        result.data["method"] = "center_fallback"
        return result

    return ComponentResult.failure(
        f"No valid depth in bbox ({xmin},{ymin})-({xmax},{ymax}) or center ({cx},{cy}). "
        "Object may be too close, transparent, or black (IR-absorbing)."
    )


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
