"""Object location and scene scanning skills."""

import math
from collections import Counter
from typing import TYPE_CHECKING

from .base import SkillResult
from ..components import perception
from ..config import runtime_config
from ..models.geometry import (
    ObjectGeometry,
    aggregate_cylinder_geometries,
    aggregate_object_geometries,
)
from ..object_types import is_cylinder_target

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
    pos = perception.bbox_to_3d(bridge, frame, bbox)
    if not pos.ok:
        # Fallback to center point
        return _with_3d(bridge, frame, detection)
    data = {**detection, **pos.data}
    print(f"[perception] 3D(bbox): x={pos.data['x']:.4f}, y={pos.data['y']:.4f}, z={pos.data['z']:.4f} "
          f"method={pos.data.get('method', '?')} (base_link)", flush=True)
    return SkillResult.success(**data)


def _locate_color_multiframe(
    bridge: "RobotBridge",
    target: str,
    color_name: str,
    location_hint: str,
    frame_count: int | None = None,
) -> SkillResult:
    """Detect and median-fuse fresh HSV/RGB-D geometry before motion."""
    if frame_count is None:
        frame_count = bridge.get_block_depth_frames()
    samples: list[ObjectGeometry] = []
    successful: list[tuple[object, dict, dict]] = []
    frame_diagnostics: list[dict] = []
    detected_frames = 0

    for sample_index in range(frame_count):
        captured = perception.capture_image(bridge, timeout=1.5)
        if not captured.ok:
            frame_diagnostics.append({
                "sample": sample_index,
                "ok": False,
                "error": captured.error or "capture failed",
            })
            continue
        frame = captured.data["frame"]
        detected = perception.detect_by_color(frame, color_name, location_hint)
        if not detected.ok or not detected.data.get("found"):
            frame_diagnostics.append({
                "sample": sample_index,
                "ok": False,
                "error": "HSV target not found",
            })
            continue
        detected_frames += 1
        measured = perception.color_detection_to_3d(
            bridge, frame, detected.data,
        )
        if not measured.ok:
            frame_diagnostics.append({
                "sample": sample_index,
                "ok": False,
                "error": measured.error or "geometry failed",
            })
            continue
        geometry = ObjectGeometry.from_dict(measured.data["geometry"])
        samples.append(geometry)
        successful.append((frame, detected.data, measured.data))
        frame_diagnostics.append({
            "sample": sample_index,
            "ok": True,
            "surface_xyz": list(geometry.surface_xyz),
            "height": geometry.height,
            "yaw_rad": geometry.yaw_rad,
            "surface_depth_mm": geometry.surface_depth_mm,
            "quality": geometry.quality,
        })

    if detected_frames == 0:
        return SkillResult.failure(
            f"HSV target '{target}' not found in {frame_count} frames",
            failed_step="detect",
            retryable=True,
            frame_diagnostics=frame_diagnostics,
        )

    aggregation = aggregate_object_geometries(
        samples,
        requested_frames=frame_count,
        min_valid_frames=max(3, math.ceil(frame_count * 0.6)),
        max_position_deviation_m=bridge.get_block_xy_max_spread(),
        max_height_deviation_m=bridge.get_block_depth_max_spread(),
        max_size_deviation_m=bridge.get_block_depth_max_spread(),
        max_desk_deviation_m=bridge.get_block_depth_max_spread(),
        max_yaw_deviation_rad=math.radians(
            bridge.get_block_yaw_max_spread_deg(),
        ),
        max_depth_deviation_mm=bridge.get_block_depth_max_spread() * 1000.0,
    )
    if aggregation.geometry is None:
        quality = aggregation.quality.to_dict()
        reasons = "; ".join(quality["rejection_reasons"]) or "unknown inconsistency"
        return SkillResult.failure(
            f"Real-time color geometry rejected: {reasons}",
            failed_step="geometry_consistency",
            retryable=True,
            geometry_quality=quality,
            frame_diagnostics=frame_diagnostics,
        )

    geometry = aggregation.geometry
    _, representative_detection, representative_measurement = successful[-1]
    data = {
        "target": target,
        **representative_detection,
        "x": geometry.surface_xyz[0],
        "y": geometry.surface_xyz[1],
        "z": geometry.surface_xyz[2],
        "frame_id": "base_link",
        "depth_mm": geometry.surface_depth_mm,
        "valid_depth_points": sum(
            int(item[2].get("valid_depth_points", 0)) for item in successful
        ),
        "local_desk_z": geometry.local_desk_z,
        "object_height": geometry.height,
        "yaw_rad": geometry.yaw_rad,
        "method": f"realtime_depth_{frame_count}frame_median",
        "depth_is_estimated": False,
        "geometry": geometry.to_dict(),
        "geometry_quality": aggregation.quality.to_dict(),
        "frame_diagnostics": frame_diagnostics,
    }
    print(
        f"[perception] 3D(color,{frame_count}f): x={data['x']:.4f}, "
        f"y={data['y']:.4f}, z={data['z']:.4f}, "
        f"height={geometry.height:.4f}, yaw={geometry.yaw_rad:.3f}, "
        f"inliers={aggregation.quality.inlier_frames}/{frame_count} "
        f"(base_link)",
        flush=True,
    )
    return SkillResult.success(**data)


def _locate_cylinder_multiframe(
    bridge: "RobotBridge",
    target: str,
    yolo: "YoloDetector",
    location_hint: str,
    color_name: str | None,
) -> SkillResult:
    """YOLO-track and fuse live depth fits for upright or lying cylinders."""

    # The first lazy YOLO inference can spend longer than the TF buffer cache
    # loading weights and compiling CUDA kernels. Warm it before acquiring the
    # first timestamped RGB-D frame so that its exact acquisition transform is
    # still available when metric fitting begins.
    ensure_loaded = getattr(yolo, "ensure_loaded", None)
    if callable(ensure_loaded):
        try:
            ensure_loaded()
        except Exception as error:
            return SkillResult.failure(
                f"YOLO initialization failed: {error}",
                failed_step="detect",
                retryable=False,
            )

    frame_count = bridge.get_cylinder_depth_frames()
    samples: list[ObjectGeometry] = []
    successful: list[tuple[dict, dict]] = []
    diagnostics: list[dict] = []
    detected_frames = 0
    for sample_index in range(frame_count):
        captured = perception.capture_image(bridge, timeout=1.5)
        if not captured.ok:
            diagnostics.append({
                "sample": sample_index,
                "ok": False,
                "error": captured.error or "capture failed",
            })
            continue
        frame = captured.data["frame"]
        detected = perception.detect_by_yolo(
            yolo, frame, target, location_hint,
        )
        if not detected.ok or not detected.data.get("found"):
            diagnostics.append({
                "sample": sample_index,
                "ok": False,
                "error": detected.error or "YOLO cylinder not found",
            })
            continue
        detected_frames += 1
        measured = perception.cylinder_detection_to_3d(
            bridge,
            frame,
            detected.data,
            color_name=color_name,
        )
        if not measured.ok:
            frame_error = measured.error or "cylinder geometry failed"
            print(
                f"[perception] cylinder frame[{sample_index}] rejected: "
                f"{frame_error}",
                flush=True,
            )
            diagnostics.append({
                "sample": sample_index,
                "ok": False,
                "error": frame_error,
                "bbox": detected.data.get("bbox"),
                "debug_image": measured.data.get("debug_image"),
            })
            continue
        geometry = ObjectGeometry.from_dict(measured.data["geometry"])
        samples.append(geometry)
        successful.append((detected.data, measured.data))
        diagnostics.append({
            "sample": sample_index,
            "ok": True,
            "center_xyz": list(geometry.center_xyz),
            "axis_xyz": list(geometry.axis_xyz),
            "diameter_m": geometry.diameter_m,
            "length_m": geometry.length_m,
            "orientation": geometry.orientation_class,
            "quality": geometry.quality,
        })

    if detected_frames == 0:
        return SkillResult.failure(
            f"YOLO target '{target}' not found in {frame_count} frames",
            failed_step="detect",
            retryable=True,
            frame_diagnostics=diagnostics,
        )
    aggregation = aggregate_cylinder_geometries(
        samples,
        requested_frames=frame_count,
        min_valid_frames=max(3, math.ceil(frame_count * 0.6)),
        max_position_deviation_m=(
            bridge.get_cylinder_position_max_spread()
        ),
        max_dimension_deviation_m=bridge.get_cylinder_depth_max_spread(),
        max_desk_deviation_m=bridge.get_cylinder_depth_max_spread(),
        max_axis_deviation_rad=math.radians(
            bridge.get_cylinder_axis_max_spread_deg()
        ),
        max_depth_deviation_mm=(
            bridge.get_cylinder_depth_max_spread() * 1000.0
        ),
    )
    if aggregation.geometry is None:
        quality = aggregation.quality.to_dict()
        reasons = "; ".join(quality["rejection_reasons"])
        def category(error: str) -> str:
            lowered = error.lower()
            if "extrapolation" in lowered or "transform" in lowered:
                return "TF timestamp"
            if "axis is diagonal" in lowered:
                return "axis classification"
            if "connected" in lowered:
                return "connected depth support"
            if "circle" in lowered:
                return "circle fit"
            if "diameter" in lowered or "length" in lowered:
                return "dimension range"
            if "depth" in lowered or "points above" in lowered:
                return "depth support"
            return error[:100]

        failure_counts = Counter(
            category(str(item["error"]))
            for item in diagnostics
            if not item.get("ok") and item.get("error")
        )
        failure_summary = ", ".join(
            f"{label}={count}"
            for label, count in failure_counts.most_common(3)
        )
        return SkillResult.failure(
            f"Real-time cylinder geometry rejected: "
            f"{reasons or 'unknown inconsistency'}"
            f"{'; frame failures: ' + failure_summary if failure_summary else ''}",
            failed_step="geometry_consistency",
            retryable=True,
            geometry_quality=quality,
            frame_diagnostics=diagnostics,
        )

    geometry = aggregation.geometry
    representative_detection, representative_measurement = successful[-1]
    data = {
        "target": target,
        **representative_detection,
        "x": geometry.surface_xyz[0],
        "y": geometry.surface_xyz[1],
        "z": geometry.surface_xyz[2],
        "frame_id": "base_link",
        "depth_mm": geometry.surface_depth_mm,
        "valid_depth_points": sum(
            int(measurement.get("valid_depth_points", 0))
            for _, measurement in successful
        ),
        "local_desk_z": geometry.local_desk_z,
        "object_height": geometry.height,
        "cylinder_diameter": geometry.diameter_m,
        "cylinder_length": geometry.length_m,
        "cylinder_orientation": geometry.orientation_class,
        "method": f"yolo_depth_cylinder_{frame_count}frame_median",
        "depth_is_estimated": False,
        "geometry": geometry.to_dict(),
        "geometry_quality": aggregation.quality.to_dict(),
        "frame_diagnostics": diagnostics,
        "debug_image": representative_measurement.get("debug_image"),
    }
    print(
        f"[perception] 3D(cylinder,{frame_count}f): "
        f"center=({geometry.center_xyz[0]:.4f},"
        f"{geometry.center_xyz[1]:.4f},{geometry.center_xyz[2]:.4f}), "
        f"diameter={geometry.diameter_m:.4f}, "
        f"length={geometry.length_m:.4f}, "
        f"orientation={geometry.orientation_class}, "
        f"inliers={aggregation.quality.inlier_frames}/{frame_count} "
        f"(base_link)",
        flush=True,
    )
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
    color_name = _target_color(target)
    # Cylinder geometry is shape-specific. A colored bottle must never be
    # interpreted as a rectangular HSV top plane.
    if is_cylinder_target(target):
        if yolo is None:
            return SkillResult.failure(
                "Cylinder detection requires the lazy YOLO detector",
                failed_step="detect",
                retryable=False,
            )
        return _locate_cylinder_multiframe(
            bridge, target, yolo, location_hint, color_name,
        )

    # ── Step 1: HSV color detection ──
    if color_name:
        color_result = _locate_color_multiframe(
            bridge, target, color_name, location_hint,
        )
        if color_result.ok:
            return color_result
        # Do not let a generic detector bypass an explicit geometry safety
        # rejection. It may remain a semantic fallback only when HSV never saw
        # the requested color at all.
        if color_result.failed_step != "detect":
            return color_result

    frame_result = perception.capture_image(bridge)
    if not frame_result.ok:
        return SkillResult.failure(frame_result.error or "capture failed", failed_step="capture", retryable=True)
    frame = frame_result.data["frame"]

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

    # ── Project oriented real-time geometry for HSV blocks ──
    for block in hsv_blocks:
        pos = perception.color_detection_to_3d(bridge, frame, block)
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
