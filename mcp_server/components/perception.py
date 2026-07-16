"""Atomic perception components."""

import os
import time
from itertools import count
from typing import TYPE_CHECKING

from .base import ComponentResult, ImageFrame
from ..config import runtime_dir
from ..models import ObjectGeometry
from ..perception.color_detector import detect_all_color_blocks, detect_by_color as detect_color_bbox
from ..perception.debug import draw_bboxes

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge
    from ..vlm_client import VlmClient
    from ..yolo_detector import YoloDetector


_DEBUG_DIR = os.environ.get("VLM_DEBUG_DIR", str(runtime_dir("debug")))
_FRAME_IDS = count(1)


def _save_debug(img, bboxes, labels, prefix="detect") -> str:
    import cv2
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    annotated = draw_bboxes(img, bboxes, labels)
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
    stamp = getattr(pair, "stamp", None)
    timestamp_s = (
        stamp.sec + stamp.nanosec / 1_000_000_000.0
        if stamp is not None else time.time()
    )
    frame = ImageFrame(
        frame_id=next(_FRAME_IDS),
        color=color,
        depth=depth,
        timestamp_s=timestamp_s,
        ros_stamp=stamp,
        source_frame=getattr(pair, "source_frame", "camera_color_optical_frame"),
    )
    return ComponentResult.success(frame=frame)


def detect_by_color(frame: ImageFrame, color_name: str, location_hint: str = "") -> ComponentResult:
    bbox = detect_color_bbox(frame.color, color_name, location_hint)
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
               bbox: list[int], margin_px: int = 2,
               known_object_height: float | None = None) -> ComponentResult:
    """Compute 3D position from a bounding box using progressive ROI expansion.

    Strategy (see yolo-integration-plan.md #8):
      1. Shrink bbox by 30% → take median depth of inner ROI (stable center)
      2. If #1 fails → shrink by 10% → take median
      3. If #2 fails → expand bbox outward by 20px
      4. If #3 fails → center point fallback

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

    def _known_height_from_local_desk():
        """Fit the surrounding desk depth and lift its base-frame point vertically."""
        if known_object_height is None or known_object_height <= 0:
            return None

        bw = xmax - xmin
        bh = ymax - ymin
        pad_x = max(24, int(bw * 0.8))
        pad_y = max(24, int(bh * 0.8))
        rx1, rx2 = max(0, xmin - pad_x), min(dw, xmax + pad_x + 1)
        ry1, ry2 = max(0, ymin - pad_y), min(dh, ymax + pad_y + 1)
        patch = depth_img[ry1:ry2, rx1:rx2]

        yy, xx = np.indices(patch.shape)
        uu, vv = xx + rx1, yy + ry1
        gap = max(4, int(min(bw, bh) * 0.1))
        outside_object = (
            (uu < xmin - gap) | (uu > xmax + gap)
            | (vv < ymin - gap) | (vv > ymax + gap)
        )
        valid = (patch > 0) & outside_object
        u = uu[valid].astype(np.float64)
        v = vv[valid].astype(np.float64)
        d = patch[valid].astype(np.float64)
        if len(d) < 100:
            return None

        # RANSAC rejects nearby objects and isolated depth noise. The table is
        # locally represented as depth_mm = a*u + b*v + c.
        points = np.column_stack((u, v, np.ones_like(u)))
        rng = np.random.default_rng(0)
        best_inliers = None
        best_count = 0
        for _ in range(80):
            indices = rng.choice(len(d), 3, replace=False)
            sample = points[indices]
            if abs(np.linalg.det(sample)) < 1e-6:
                continue
            coeff = np.linalg.solve(sample, d[indices])
            residual = np.abs(points @ coeff - d)
            inliers = residual < 3.0
            count_inliers = int(np.count_nonzero(inliers))
            if count_inliers > best_count:
                best_count = count_inliers
                best_inliers = inliers
        if best_inliers is None or best_count < max(80, int(len(d) * 0.25)):
            return None

        coeff, *_ = np.linalg.lstsq(points[best_inliers], d[best_inliers], rcond=None)
        desk_depth_mm = float(np.array([cx, cy, 1.0]) @ coeff)
        cinfo = bridge.node.get_color_info()
        if cinfo is None or desk_depth_mm <= 0:
            return None
        z_c = desk_depth_mm / 1000.0
        x_c = (cx - cinfo["cx"]) * z_c / cinfo["fx"]
        y_c = (cy - cinfo["cy"]) * z_c / cinfo["fy"]
        try:
            desk_base = bridge.node.transform_to_base(
                x_c, y_c, z_c,
                stamp=frame.ros_stamp,
                source_frame=frame.source_frame,
            )
        except Exception:
            return None
        return ComponentResult.success(
            x=desk_base["x"],
            y=desk_base["y"],
            z=desk_base["z"] + known_object_height,
            frame_id="base_link",
            depth_mm=desk_depth_mm,
            valid_depth_points=best_count,
            local_desk_z=desk_base["z"],
            object_height=known_object_height,
            method="known_height_local_desk_ransac",
            depth_is_estimated=True,
            geometry=ObjectGeometry(
                surface_xyz=(desk_base["x"], desk_base["y"],
                             desk_base["z"] + known_object_height),
                center_xyz=(desk_base["x"], desk_base["y"],
                            desk_base["z"] + known_object_height / 2.0),
                size_xyz=(known_object_height,) * 3,
                local_desk_z=desk_base["z"],
                height=known_object_height,
                height_source="configured_color_block",
            ).to_dict(),
        )

    # Solid colour blocks often have no usable IR return. Their height is a
    # calibrated object property, so prefer the local desk model instead of a
    # sparse ROI value that is likely to belong to the table behind the block.
    known_height_result = _known_height_from_local_desk()
    if known_height_result is not None:
        return known_height_result

    def _median_3d(roi_xmin, roi_xmax, roi_ymin, roi_ymax):
        """Extract ROI, compute median depth, compute 3D at bbox center."""
        rx1 = max(0, roi_xmin)
        rx2 = min(dw, roi_xmax + 1)
        ry1 = max(0, roi_ymin)
        ry2 = min(dh, roi_ymax + 1)
        roi = depth_img[ry1:ry2, rx1:rx2]
        valid = roi[roi > 0]
        if len(valid) == 0:
            return None

        # Median depth of inner ROI → object surface depth
        depth_mm = float(np.median(valid))

        # Always deproject at the geometric bbox center.  The depth value comes
        # from the whole inner ROI, so the center pixel itself does not need to
        # contain a valid depth sample.  Searching for the first valid pixel
        # made sparse-depth objects jump in X/Y (and in base-frame Z when the
        # camera is tilted), even when bbox and median depth were unchanged.
        u_center = cx
        v_center = cy

        cinfo = bridge.node.get_color_info()
        if cinfo is None:
            return None
        z_c = depth_mm / 1000.0
        x_c = (u_center - cinfo["cx"]) * z_c / cinfo["fx"]
        y_c = (v_center - cinfo["cy"]) * z_c / cinfo["fy"]
        try:
            base = bridge.node.transform_to_base(
                x_c, y_c, z_c,
                stamp=frame.ros_stamp,
                source_frame=frame.source_frame,
            )
        except Exception:
            return None
        return ComponentResult.success(
            x=base["x"], y=base["y"], z=base["z"],
            frame_id="base_link",
            depth_mm=depth_mm,
            valid_depth_points=len(valid),
            geometry=ObjectGeometry(
                surface_xyz=(base["x"], base["y"], base["z"]),
                height_source="depth_surface_only",
            ).to_dict(),
        ), depth_mm

    # Step 1: Shrink by 30% (aggressive: inner ROI is mostly object surface)
    bw = xmax - xmin
    bh = ymax - ymin
    shrink_x = max(1, int(bw * 0.30))
    shrink_y = max(1, int(bh * 0.30))
    result = _median_3d(xmin + shrink_x, xmax - shrink_x,
                        ymin + shrink_y, ymax - shrink_y)
    if result is not None:
        pos, depth_mm = result
        if pos.ok:
            pos.data["depth_mm"] = depth_mm
            pos.data["method"] = "bbox_shrink_30pct"
            return pos

    # Step 2: Shrink by 10% (relaxed)
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

    # Step 3: Fallback to center point only. Never expand into the surrounding
    # table: that silently turns a missing object depth into a false low target.
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
        base = bridge.node.transform_to_base(
            cam3d["x_c"], cam3d["y_c"], cam3d["z_c"],
            stamp=frame.ros_stamp,
            source_frame=frame.source_frame,
        )
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
        geometry=ObjectGeometry(
            surface_xyz=(base["x"], base["y"], base["z"]),
            height_source="depth_surface_only",
        ).to_dict(),
    )
