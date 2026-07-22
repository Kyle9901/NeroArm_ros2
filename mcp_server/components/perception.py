"""Atomic perception components."""

import math
import os
import time
from dataclasses import replace
from itertools import count
from typing import TYPE_CHECKING

from .base import ComponentResult, ImageFrame
from ..config import runtime_dir
from ..models import ObjectGeometry
from ..perception.color_detector import (
    build_color_mask,
    detect_all_color_blocks,
    detect_color_object,
)
from ..perception.cylinder_geometry import fit_cylinder_geometry
from ..perception.debug import draw_bboxes
from ..perception.transparent_bottle import (
    analyze_transparent_bottle_points,
)

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge
    from ..vlm_client import VlmClient
    from ..yolo_detector import YoloDetector


_DEBUG_DIR = os.environ.get("VLM_DEBUG_DIR", str(runtime_dir("debug")))
_FRAME_IDS = count(1)


def _save_debug(img, bboxes, labels, prefix="detect") -> str:
    import cv2
    from pathlib import Path

    os.makedirs(_DEBUG_DIR, exist_ok=True)
    annotated = draw_bboxes(img, bboxes, labels)
    # Keep one latest artifact per prefix instead of piling up timestamped files
    debug_dir = Path(_DEBUG_DIR)
    path = debug_dir / f"{prefix}_latest.jpg"
    for legacy_path in debug_dir.glob(f"{prefix}_*.jpg"):
        if legacy_path != path:
            try:
                legacy_path.unlink()
            except OSError:
                pass
    cv2.imwrite(str(path), annotated)
    return str(path)


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
    detection = detect_color_object(frame.color, color_name, location_hint)
    if detection is None:
        return ComponentResult.success(found=False, color=color_name, source="CV")
    xmin, ymin = int(detection["x"]), int(detection["y"])
    xmax, ymax = int(detection["xmax"]), int(detection["ymax"])
    debug_path = _save_debug(frame.color, [[xmin, ymin, xmax, ymax]], [color_name])
    return ComponentResult.success(
        found=True,
        color=color_name,
        bbox=[xmin, ymin, xmax, ymax],
        center_2d=[
            int(round(detection["rotated_center_2d"][0])),
            int(round(detection["rotated_center_2d"][1])),
        ],
        rotated_center_2d=detection["rotated_center_2d"],
        rotated_box_2d=detection["rotated_box_2d"],
        contour_2d=detection["contour_2d"],
        edge_axes_2d=detection["edge_axes_2d"],
        yaw_image_rad=detection["yaw_image_rad"],
        yaw_period_rad=detection["yaw_period_rad"],
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


def _deproject_samples(u, v, depth_mm, camera_info):
    """Vectorized pinhole deprojection into the color optical frame."""
    import numpy as np

    z = np.asarray(depth_mm, dtype=np.float64) / 1000.0
    x = (
        (np.asarray(u, dtype=np.float64) - camera_info["cx"])
        * z / camera_info["fx"]
    )
    y = (
        (np.asarray(v, dtype=np.float64) - camera_info["cy"])
        * z / camera_info["fy"]
    )
    return np.column_stack((x, y, z))


def _fit_plane_ransac_3d(
    points,
    *,
    threshold_m: float = 0.003,
    iterations: int = 80,
) -> dict | None:
    """Fit ``normal dot point + offset = 0`` to metric 3D samples."""
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    if len(points) < 100:
        return None
    rng = np.random.default_rng(0)
    best_inliers = None
    best_count = 0
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        inliers = np.abs(points @ normal + offset) <= threshold_m
        count_inliers = int(np.count_nonzero(inliers))
        if count_inliers > best_count:
            best_count = count_inliers
            best_inliers = inliers
    if best_inliers is None or best_count < max(80, int(len(points) * 0.25)):
        return None

    selected = points[best_inliers]
    centroid = np.mean(selected, axis=0)
    _, _, vh = np.linalg.svd(selected - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(normal @ centroid)
    residuals = np.abs(selected @ normal + offset)
    return {
        "normal": normal,
        "offset": offset,
        "inliers": best_inliers,
        "inlier_count": best_count,
        "sample_count": len(points),
        "residual_median_mm": float(np.median(residuals) * 1000.0),
    }


def _camera_to_base_rigid_transform(
    bridge: "RobotBridge",
    frame: ImageFrame,
) -> tuple:
    """Recover one timestamped camera-to-base rigid transform.

    ``transform_to_base`` intentionally exposes point transforms rather than
    tf2 internals.  Transforming the origin and the three unit basis points
    recovers the same rigid transform without looking it up at different
    timestamps for every depth sample.
    """
    import numpy as np

    probes = []
    for x, y, z in (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ):
        probes.append(bridge.node.transform_to_base(
            x, y, z,
            stamp=frame.ros_stamp,
            source_frame=frame.source_frame,
        ))
    origin = np.asarray([
        probes[0]["x"], probes[0]["y"], probes[0]["z"],
    ])
    rotation = np.column_stack([
        np.asarray([probe["x"], probe["y"], probe["z"]]) - origin
        for probe in probes[1:]
    ])
    return origin, rotation


def _transform_camera_points_to_base(
    bridge: "RobotBridge",
    frame: ImageFrame,
    points,
    *,
    rigid_transform: tuple | None = None,
):
    """Apply one timestamped rigid TF to an array of camera-frame points."""
    import numpy as np

    origin, rotation = (
        rigid_transform
        if rigid_transform is not None
        else _camera_to_base_rigid_transform(bridge, frame)
    )
    return np.asarray(points, dtype=np.float64) @ rotation.T + origin


def _fit_local_desk_plane(
    depth_img,
    bbox: list[int],
    contour_2d: list[list[int]] | None,
    camera_info: dict,
) -> dict | None:
    """Fit the surrounding desk as a physical 3D plane using RANSAC."""
    import cv2
    import numpy as np

    height, width = depth_img.shape[:2]
    xmin, ymin, xmax, ymax = [int(value) for value in bbox]
    bw, bh = max(1, xmax - xmin), max(1, ymax - ymin)
    pad_x = max(24, int(bw * 0.8))
    pad_y = max(24, int(bh * 0.8))
    rx1, rx2 = max(0, xmin - pad_x), min(width, xmax + pad_x + 1)
    ry1, ry2 = max(0, ymin - pad_y), min(height, ymax + pad_y + 1)
    patch = depth_img[ry1:ry2, rx1:rx2]

    excluded = np.zeros((height, width), dtype=np.uint8)
    if contour_2d:
        contour = np.asarray(contour_2d, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(excluded, [contour], 255)
    else:
        cv2.rectangle(excluded, (xmin, ymin), (xmax, ymax), 255, -1)
    gap = max(5, int(min(bw, bh) * 0.12))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1),
    )
    excluded = cv2.dilate(excluded, kernel)

    yy, xx = np.indices(patch.shape)
    uu = xx + rx1
    vv = yy + ry1
    valid = (patch > 0) & (excluded[ry1:ry2, rx1:rx2] == 0)
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    d = patch[valid].astype(np.float64)
    if len(d) < 100:
        return None

    # Cap work per frame while retaining spatial coverage across the patch.
    if len(d) > 8000:
        indices = np.linspace(0, len(d) - 1, 8000, dtype=int)
        u, v, d = u[indices], v[indices], d[indices]
    fitted = _fit_plane_ransac_3d(
        _deproject_samples(u, v, d, camera_info),
    )
    if fitted is None:
        return None
    fitted["sample_u"] = u
    fitted["sample_v"] = v
    return fitted


def _connected_top_support(
    uu,
    vv,
    candidate_mask,
    *,
    image_shape: tuple[int, int],
    silhouette_area_px: float,
) -> dict:
    """Keep the dominant spatially connected top-depth support.

    A minimum-area rectangle only describes the *extent* of its points.  It
    does not prove that the rectangle contains real depth support: a handful
    of flying pixels near its corners can look like 100% coverage.  This gate
    therefore measures three independent properties on the original samples:

    * most candidate points belong to one locally connected component;
    * actual samples occupy a non-trivial fraction of that component's extent;
    * the extent covers a plausible fraction of the HSV silhouette.

    A small dilation is used only to establish neighbourhood connectivity for
    sparse structured-light returns.  It never contributes synthetic points
    to the occupancy count or to the returned geometry.
    """
    import cv2
    import numpy as np

    candidate_indices = np.flatnonzero(candidate_mask)
    if len(candidate_indices) < 12:
        return {"reliable": False, "reason": "fewer than 12 height inliers"}

    sample_u = np.asarray(uu[candidate_mask], dtype=np.int32)
    sample_v = np.asarray(vv[candidate_mask], dtype=np.int32)
    if (
        np.any(sample_u < 0)
        or np.any(sample_v < 0)
        or np.any(sample_u >= image_shape[1])
        or np.any(sample_v >= image_shape[0])
    ):
        return {"reliable": False, "reason": "top pixels leave image bounds"}
    min_u, max_u = int(np.min(sample_u)), int(np.max(sample_u))
    min_v, max_v = int(np.min(sample_v)), int(np.max(sample_v))
    local_u = sample_u - min_u
    local_v = sample_v - min_v
    support_mask = np.zeros(
        (max_v - min_v + 1, max_u - min_u + 1),
        dtype=np.uint8,
    )
    support_mask[local_v, local_u] = 255

    # Link samples separated by at most a few missing depth pixels.  This lets
    # a genuinely sparse top remain one component while isolated flying pixels
    # remain separate.
    linked = cv2.dilate(
        support_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    component_count, labels = cv2.connectedComponents(linked, connectivity=8)
    if component_count <= 1:
        return {"reliable": False, "reason": "no connected top component"}

    point_labels = labels[local_v, local_u]
    label_counts = np.bincount(point_labels, minlength=component_count)
    label_counts[0] = 0
    best_label = int(np.argmax(label_counts))
    keep_local = point_labels == best_label
    kept_count = int(np.count_nonzero(keep_local))
    minimum_points = max(
        20,
        int(math.ceil(max(float(silhouette_area_px), 1.0) * 0.005)),
    )
    if kept_count < minimum_points:
        return {
            "reliable": False,
            "reason": (
                f"connected top has {kept_count} points "
                f"(needs {minimum_points})"
            ),
        }

    component_pixels = np.column_stack((
        sample_u[keep_local], sample_v[keep_local],
    )).astype(np.float32)
    pixel_rect = cv2.minAreaRect(component_pixels)
    rect_area = float(pixel_rect[1][0] * pixel_rect[1][1])
    if rect_area <= 1.0:
        return {"reliable": False, "reason": "connected top extent is degenerate"}

    connected_ratio = kept_count / max(len(candidate_indices), 1)
    # minAreaRect measures distances between pixel centres. Add one pixel cell
    # per dimension so a fully populated N-by-M patch reports about 100%, not
    # slightly above 100%.
    occupancy_area = float(
        (pixel_rect[1][0] + 1.0) * (pixel_rect[1][1] + 1.0)
    )
    occupancy_ratio = kept_count / max(occupancy_area, 1.0)
    extent_ratio = rect_area / max(float(silhouette_area_px), 1.0)
    if connected_ratio < 0.65:
        return {
            "reliable": False,
            "reason": f"connected support ratio {connected_ratio:.1%} is too small",
        }
    if occupancy_ratio < 0.015:
        return {
            "reliable": False,
            "reason": f"actual top occupancy {occupancy_ratio:.1%} is too small",
        }
    if not 0.20 <= extent_ratio <= 1.15:
        return {
            "reliable": False,
            "reason": f"top extent ratio {extent_ratio:.1%} is implausible",
        }

    kept_mask = np.zeros(len(candidate_mask), dtype=bool)
    kept_mask[candidate_indices[keep_local]] = True
    return {
        "reliable": True,
        "mask": kept_mask,
        "pixel_rect": pixel_rect,
        "connected_ratio": connected_ratio,
        "occupancy_ratio": occupancy_ratio,
        "extent_ratio": extent_ratio,
        "point_count": kept_count,
    }


def _select_highest_credible_top(
    signed_heights,
    plausible,
    uu,
    vv,
    *,
    image_shape: tuple[int, int],
    silhouette_area_px: float,
) -> dict | None:
    """Return the highest height cluster with credible 2-D spatial support.

    Selecting the globally densest height bin is unsafe for sparse depth:
    vertical sides can contribute more pixels to one quantized slice than the
    true top.  Candidate bins are instead tried from high to low.  A bin is
    accepted only after the independent connectivity/occupancy/extent gate
    proves that it behaves like a surface rather than a side strip or flying
    pixels.
    """
    import numpy as np

    plausible_heights = np.asarray(signed_heights[plausible], dtype=np.float64)
    height_bins = np.floor(plausible_heights / 0.003).astype(np.int64)
    unique_bins, bin_counts = np.unique(height_bins, return_counts=True)
    minimum_seed_points = max(
        4,
        int(math.ceil(max(float(silhouette_area_px), 1.0) * 0.002)),
    )
    plausible_indices = np.flatnonzero(plausible)
    attempted_masks: set[bytes] = set()
    rejection_reasons: list[str] = []

    for bin_id, bin_count in sorted(
        zip(unique_bins.tolist(), bin_counts.tolist()),
        reverse=True,
    ):
        if bin_count < minimum_seed_points:
            continue
        selected_plausible = height_bins == bin_id
        seed = np.zeros(len(signed_heights), dtype=bool)
        seed[plausible_indices[selected_plausible]] = True
        seed_height = float(np.median(signed_heights[seed]))
        seed_mad = float(np.median(np.abs(
            signed_heights[seed] - seed_height,
        )))
        # The 3.5 mm floor bridges a top plane split by a fixed 3 mm histogram
        # boundary.  The 8 mm cap prevents a broad side band from becoming a
        # synthetic horizontal plane.
        tolerance = min(0.008, max(0.0035, 3.0 * seed_mad))
        candidate = plausible & (
            np.abs(signed_heights - seed_height) <= tolerance
        )
        fingerprint = np.packbits(candidate).tobytes()
        if fingerprint in attempted_masks:
            continue
        attempted_masks.add(fingerprint)

        support = _connected_top_support(
            uu,
            vv,
            candidate,
            image_shape=image_shape,
            silhouette_area_px=silhouette_area_px,
        )
        if support.get("reliable"):
            support["seed_height_m"] = seed_height
            support["height_tolerance_m"] = tolerance
            support["rejection_reasons"] = tuple(rejection_reasons)
            return support
        rejection_reasons.append(
            f"bin {bin_id}: {support.get('reason', 'unreliable support')}"
        )
    return {
        "reliable": False,
        "rejection_reasons": tuple(rejection_reasons[-6:]),
    }


def color_detection_to_3d(
    bridge: "RobotBridge",
    frame: ImageFrame,
    detection: dict,
) -> ComponentResult:
    """Estimate oriented block geometry solely from live RGB-D data.

    A metric 3D desk plane is fitted outside the HSV contour. Points inside
    the contour are clustered by physical height above that plane to recover
    the top face, center, dimensions and yaw without a configured height.
    """
    import cv2
    import numpy as np

    bbox = detection.get("bbox")
    if not bbox or len(bbox) != 4:
        return ComponentResult.failure("Color detection has no valid bbox")
    depth_img = frame.depth
    height, width = depth_img.shape[:2]
    xmin, ymin, xmax, ymax = [
        int(round(float(value))) for value in bbox
    ]
    xmin, xmax = max(0, xmin), min(width - 1, xmax)
    ymin, ymax = max(0, ymin), min(height - 1, ymax)
    if xmax <= xmin or ymax <= ymin:
        return ComponentResult.failure("Color detection bbox is empty")

    contour_points = detection.get("contour_2d")
    mask = np.zeros((height, width), dtype=np.uint8)
    if contour_points:
        contour = np.asarray(contour_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [contour], 255)
    else:
        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)
    camera_info = bridge.node.get_color_info()
    if camera_info is None:
        return ComponentResult.failure("Color camera intrinsics are unavailable")

    # Use the full HSV contour here. Boundary/table pixels are rejected later
    # by their metric height above the fitted desk rather than by a fixed
    # erosion that would systematically shorten the recovered block edges.
    object_valid = (mask > 0) & (depth_img > 0)
    vv, uu = np.nonzero(object_valid)
    depths_mm = depth_img[object_valid].astype(np.float64)
    if len(depths_mm) < 12:
        return ComponentResult.failure(
            f"Only {len(depths_mm)} valid depth samples inside HSV contour",
            valid_depth_points=int(len(depths_mm)),
        )

    desk_fit = _fit_local_desk_plane(
        depth_img,
        [xmin, ymin, xmax, ymax],
        contour_points,
        camera_info,
    )
    if desk_fit is None:
        return ComponentResult.failure(
            "Unable to fit a metric 3D local desk plane around object"
        )

    object_points = _deproject_samples(
        uu, vv, depths_mm, camera_info,
    )
    desk_normal = np.asarray(desk_fit["normal"], dtype=np.float64)
    desk_offset = float(desk_fit["offset"])
    signed_heights = object_points @ desk_normal + desk_offset
    direction = 1.0 if float(np.median(signed_heights)) >= 0.0 else -1.0
    desk_normal *= direction
    desk_offset *= direction
    signed_heights *= direction

    plausible = (signed_heights >= 0.002) & (signed_heights <= 0.50)
    if int(np.count_nonzero(plausible)) < 12:
        return ComponentResult.failure(
            "HSV contour has too few points above the fitted desk",
            valid_depth_points=int(np.count_nonzero(plausible)),
        )

    detected_axes = detection.get("edge_axes_2d") or ()
    if len(detected_axes) >= 2:
        silhouette_area = (
            float(detected_axes[0]["length_px"])
            * float(detected_axes[1]["length_px"])
        )
    else:
        silhouette_area = float((xmax - xmin) * (ymax - ymin))

    top_support = _select_highest_credible_top(
        signed_heights,
        plausible,
        uu,
        vv,
        image_shape=(height, width),
        silhouette_area_px=silhouette_area,
    )
    if not top_support or not top_support.get("reliable"):
        reasons = "; ".join(
            top_support.get("rejection_reasons", ())
            if top_support else ()
        )
        return ComponentResult.failure(
            (
                "Unable to isolate a connected, occupied top plane"
                + (f": {reasons}" if reasons else "")
            ),
            valid_depth_points=0,
        )
    top_inliers = top_support["mask"]
    top_height_normal_m = float(np.median(signed_heights[top_inliers]))
    top_offset = desk_offset - top_height_normal_m
    top_residual_mm = float(np.median(np.abs(
        signed_heights[top_inliers] - top_height_normal_m,
    )) * 1000.0)

    # Intersect every accepted top pixel with the denoised top plane first.
    # The resulting metric points are transformed to base before fitting the
    # rectangle; minAreaRect in image pixels is not perspective invariant.
    top_pixels = np.column_stack((
        uu[top_inliers], vv[top_inliers],
    )).astype(np.float64)
    top_rays = np.column_stack((
        (top_pixels[:, 0] - camera_info["cx"]) / camera_info["fx"],
        (top_pixels[:, 1] - camera_info["cy"]) / camera_info["fy"],
        np.ones(len(top_pixels), dtype=np.float64),
    ))
    ray_denominators = top_rays @ desk_normal
    if np.any(np.abs(ray_denominators) <= 1e-9):
        return ComponentResult.failure(
            "Top plane is parallel to one or more accepted camera rays"
        )
    top_depths_mm = (
        -top_offset / ray_denominators * 1000.0
    )
    if np.any(top_depths_mm <= 0.0):
        return ComponentResult.failure(
            "Accepted top plane intersects behind the camera"
        )
    top_points_camera = _deproject_samples(
        top_pixels[:, 0],
        top_pixels[:, 1],
        top_depths_mm,
        camera_info,
    )
    try:
        rigid_transform = _camera_to_base_rigid_transform(bridge, frame)
        top_points_base = _transform_camera_points_to_base(
            bridge,
            frame,
            top_points_camera,
            rigid_transform=rigid_transform,
        )
    except Exception as error:
        return ComponentResult.failure(
            f"Unable to transform fitted planes to base_link: {error}"
        )

    metric_rect = cv2.minAreaRect(
        top_points_base[:, :2].astype(np.float32)
    )
    top_box_base_xy = cv2.boxPoints(metric_rect).astype(np.float64)
    edge_0 = top_box_base_xy[1] - top_box_base_xy[0]
    edge_1 = top_box_base_xy[2] - top_box_base_xy[1]
    length_0 = float(np.linalg.norm(edge_0))
    length_1 = float(np.linalg.norm(edge_1))
    if min(length_0, length_1) <= 1e-5:
        return ComponentResult.failure("Recovered metric top edges are degenerate")

    origin, rotation = rigid_transform
    normal_base_raw = rotation @ desk_normal
    normal_norm = float(np.linalg.norm(normal_base_raw))
    if normal_norm <= 1e-9:
        return ComponentResult.failure("Transformed desk normal is degenerate")
    # Transform n_c·p_c+d=0 through p_b=R p_c+t, then normalize both the
    # transformed normal and offset by the same factor.
    desk_offset_base = (
        desk_offset - float(normal_base_raw @ origin)
    ) / normal_norm
    top_offset_base = (
        top_offset - float(normal_base_raw @ origin)
    ) / normal_norm
    normal_base = normal_base_raw / normal_norm
    if normal_base[2] <= 0.90:
        return ComponentResult.failure(
            "Fitted desk normal is not sufficiently aligned with base +Z"
        )

    center_x, center_y = (
        float(metric_rect[0][0]), float(metric_rect[0][1])
    )
    surface_z = -(
        normal_base[0] * center_x
        + normal_base[1] * center_y
        + top_offset_base
    ) / normal_base[2]
    desk_z = -(
        normal_base[0] * center_x
        + normal_base[1] * center_y
        + desk_offset_base
    ) / normal_base[2]
    get_expected_desk = getattr(bridge, "get_desk_surface_z", None)
    get_desk_tolerance = getattr(
        bridge, "get_desk_measurement_max_error", None
    )
    desk_error = None
    if callable(get_expected_desk) and callable(get_desk_tolerance):
        expected_desk_z = float(get_expected_desk())
        desk_error = abs(float(desk_z) - expected_desk_z)
        desk_tolerance = float(get_desk_tolerance())
        if desk_error > desk_tolerance:
            return ComponentResult.failure(
                (
                    f"Measured desk z differs from configured desk by "
                    f"{desk_error:.4f}m (limit {desk_tolerance:.4f}m)"
                ),
                local_desk_z=float(desk_z),
                configured_desk_z=expected_desk_z,
            )
    surface_base = np.asarray([center_x, center_y, surface_z])
    object_height = float(surface_z - desk_z)
    if not 0.002 <= object_height <= 0.50:
        return ComponentResult.failure(
            f"Measured object height {object_height:.4f}m is outside [0.002, 0.50]m",
            local_desk_z=desk_z,
        )

    # Report the camera optical depth of the recovered physical top center.
    surface_camera = (surface_base - origin) @ rotation
    surface_depth_mm = float(surface_camera[2] * 1000.0)
    if surface_depth_mm <= 0.0:
        return ComponentResult.failure(
            "Recovered top center lies behind the camera"
        )

    if length_1 > length_0:
        primary_vector, primary_length = edge_1, length_1
        secondary_length = length_0
    else:
        primary_vector, primary_length = edge_0, length_0
        secondary_length = length_1
    primary_axis = tuple((primary_vector / primary_length).tolist())
    yaw_rad = float(np.arctan2(primary_axis[1], primary_axis[0]))
    # cv2.minAreaRect in metric base XY returns an orthogonal rectangle.
    secondary_axis = (-primary_axis[1], primary_axis[0])
    yaw_period_rad = (
        float(np.pi / 2.0)
        if primary_length / secondary_length < 1.15
        else float(np.pi)
    )
    per_frame_quality = {
        "reliable": True,
        "valid_depth_points": int(np.count_nonzero(top_inliers)),
        "surface_depth_mad_mm": top_residual_mm,
        # Compatibility name now carries actual observed-point occupancy, not
        # a bounding-rectangle extent ratio.
        "top_plane_coverage": float(top_support["occupancy_ratio"]),
        "top_occupancy_ratio": float(top_support["occupancy_ratio"]),
        "top_connected_ratio": float(top_support["connected_ratio"]),
        "top_extent_ratio": float(top_support["extent_ratio"]),
        "desk_inliers": int(desk_fit["inlier_count"]),
        "desk_samples": int(desk_fit["sample_count"]),
        "desk_residual_median_mm": float(desk_fit["residual_median_mm"]),
        "desk_normal_base": [float(value) for value in normal_base],
        "configured_desk_error_m": desk_error,
    }
    geometry = ObjectGeometry(
        surface_xyz=(
            float(surface_base[0]),
            float(surface_base[1]),
            float(surface_base[2]),
        ),
        center_xyz=(
            float(surface_base[0]),
            float(surface_base[1]),
            float(desk_z + object_height / 2.0),
        ),
        size_xyz=(
            primary_length, secondary_length, object_height,
        ),
        local_desk_z=float(desk_z),
        height=object_height,
        height_source="realtime_depth_local_desk",
        yaw_rad=yaw_rad,
        yaw_period_rad=yaw_period_rad,
        primary_axis_xy=primary_axis,
        secondary_axis_xy=secondary_axis,
        surface_depth_mm=surface_depth_mm,
        quality=per_frame_quality,
    )
    return ComponentResult.success(
        x=float(surface_base[0]),
        y=float(surface_base[1]),
        z=float(surface_base[2]),
        frame_id="base_link",
        depth_mm=surface_depth_mm,
        valid_depth_points=int(np.count_nonzero(top_inliers)),
        local_desk_z=float(desk_z),
        object_height=object_height,
        yaw_rad=yaw_rad,
        method="color_top_plane_local_desk_3d_ransac",
        depth_is_estimated=False,
        geometry=geometry.to_dict(),
        geometry_quality=per_frame_quality,
    )


def _adaptive_depth_connected_support(
    sample_u,
    sample_v,
    sample_depths_mm,
    bbox: list[int],
) -> dict:
    """Join sparse image support only across locally continuous depth.

    Transparent and reflective containers often leave small holes between
    otherwise consistent depth returns.  The link radius scales with the
    detected object size, while the depth threshold scales with working
    distance.  Nearby pixels from a different depth layer therefore remain
    disconnected even when their image coordinates are close.
    """
    import numpy as np

    u = np.asarray(sample_u, dtype=np.int32)
    v = np.asarray(sample_v, dtype=np.int32)
    depths = np.asarray(sample_depths_mm, dtype=np.float64)
    if not (len(u) == len(v) == len(depths)) or len(u) == 0:
        return {
            "keep_local": np.zeros(len(u), dtype=bool),
            "connected_count": 0,
            "connected_ratio": 0.0,
            "component_count": 0,
            "gap_px": 0,
            "depth_tolerance_mm": 0.0,
        }

    xmin, ymin, xmax, ymax = [int(value) for value in bbox]
    bbox_min_dimension = max(1, min(xmax - xmin + 1, ymax - ymin + 1))
    gap_px = int(np.clip(round(bbox_min_dimension * 0.08), 3, 8))
    median_depth_mm = float(np.median(depths))
    depth_tolerance_mm = float(np.clip(
        median_depth_mm * 0.035,
        12.0,
        30.0,
    ))

    min_u, max_u = int(np.min(u)), int(np.max(u))
    min_v, max_v = int(np.min(v)), int(np.max(v))
    height = max_v - min_v + 1
    width = max_u - min_u + 1
    index_grid = np.full((height, width), -1, dtype=np.int32)
    depth_grid = np.full((height, width), np.nan, dtype=np.float64)
    local_u = u - min_u
    local_v = v - min_v
    point_indices = np.arange(len(u), dtype=np.int32)
    index_grid[local_v, local_u] = point_indices
    depth_grid[local_v, local_u] = depths

    parent = np.arange(len(u), dtype=np.int32)
    sizes = np.ones(len(u), dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    # Inspect each undirected offset once.  Unlike a plain dilation, an edge
    # exists only when the two measured endpoints are close in both pixels and
    # depth. Missing pixels between them are allowed and never enter geometry.
    for delta_v in range(0, gap_px + 1):
        for delta_u in range(-gap_px, gap_px + 1):
            if delta_v == 0 and delta_u <= 0:
                continue
            if delta_u * delta_u + delta_v * delta_v > gap_px * gap_px:
                continue
            source_y0 = max(0, -delta_v)
            source_y1 = min(height, height - delta_v)
            source_x0 = max(0, -delta_u)
            source_x1 = min(width, width - delta_u)
            if source_y1 <= source_y0 or source_x1 <= source_x0:
                continue
            target_y0, target_y1 = (
                source_y0 + delta_v,
                source_y1 + delta_v,
            )
            target_x0, target_x1 = (
                source_x0 + delta_u,
                source_x1 + delta_u,
            )
            source_indices = index_grid[
                source_y0:source_y1, source_x0:source_x1
            ]
            target_indices = index_grid[
                target_y0:target_y1, target_x0:target_x1
            ]
            source_depths = depth_grid[
                source_y0:source_y1, source_x0:source_x1
            ]
            target_depths = depth_grid[
                target_y0:target_y1, target_x0:target_x1
            ]
            linkable = (
                (source_indices >= 0)
                & (target_indices >= 0)
                & (
                    np.abs(source_depths - target_depths)
                    <= depth_tolerance_mm
                )
            )
            for left, right in zip(
                source_indices[linkable],
                target_indices[linkable],
            ):
                union(int(left), int(right))

    roots = np.asarray([find(index) for index in range(len(u))], dtype=np.int32)
    unique_roots, counts = np.unique(roots, return_counts=True)
    best_root = int(unique_roots[int(np.argmax(counts))])
    keep_local = roots == best_root
    connected_count = int(np.count_nonzero(keep_local))
    return {
        "keep_local": keep_local,
        "connected_count": connected_count,
        "connected_ratio": connected_count / len(u),
        "component_count": int(len(unique_roots)),
        "gap_px": gap_px,
        "depth_tolerance_mm": depth_tolerance_mm,
    }


def _save_cylinder_depth_debug(
    depth_img,
    bbox: list[int],
    raw_u,
    raw_v,
    desk_fit: dict,
    candidate_u,
    candidate_v,
    connected_u,
    connected_v,
    connectivity: dict,
) -> str:
    """Write one four-panel cylinder depth diagnostic image."""
    import cv2
    import numpy as np
    from pathlib import Path

    depth = np.asarray(depth_img)
    valid = depth > 0
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.quantile(depth[valid].astype(np.float64), (0.02, 0.98))
        if high <= low:
            high = low + 1.0
        normalized[valid] = np.clip(
            (depth[valid].astype(np.float64) - low)
            * 255.0 / (high - low),
            0.0,
            255.0,
        ).astype(np.uint8)
    base = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    base[~valid] = 0
    xmin, ymin, xmax, ymax = [int(value) for value in bbox]

    def panel(title: str, uu, vv, color, detail: str = ""):
        image = base.copy()
        mask = np.zeros(depth.shape, dtype=np.uint8)
        u_values = np.asarray(uu, dtype=np.int32)
        v_values = np.asarray(vv, dtype=np.int32)
        inside = (
            (u_values >= 0)
            & (u_values < depth.shape[1])
            & (v_values >= 0)
            & (v_values < depth.shape[0])
        )
        mask[v_values[inside], u_values[inside]] = 255
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        image[mask > 0] = color
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
        cv2.rectangle(image, (0, 0), (image.shape[1], 48), (0, 0, 0), -1)
        cv2.putText(
            image, title, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            cv2.LINE_AA,
        )
        if detail:
            cv2.putText(
                image, detail, (10, 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1,
                cv2.LINE_AA,
            )
        return image

    desk_indices = np.asarray(
        desk_fit.get("inliers", ()), dtype=bool,
    )
    desk_u = np.asarray(desk_fit.get("sample_u", ()))
    desk_v = np.asarray(desk_fit.get("sample_v", ()))
    if len(desk_indices) == len(desk_u):
        desk_u = desk_u[desk_indices]
        desk_v = desk_v[desk_indices]
    panels = (
        panel(
            "1 raw depth in YOLO bbox",
            raw_u, raw_v, (255, 255, 255),
            f"points={len(raw_u)}",
        ),
        panel(
            "2 local desk RANSAC inliers",
            desk_u, desk_v, (255, 128, 0),
            f"inliers={len(desk_u)}",
        ),
        panel(
            "3 above-desk cylinder candidates",
            candidate_u, candidate_v, (0, 165, 255),
            f"points={len(candidate_u)}",
        ),
        panel(
            "4 depth-continuous component",
            connected_u, connected_v, (0, 255, 0),
            (
                f"kept={len(connected_u)}/{len(candidate_u)} "
                f"gap={connectivity['gap_px']}px "
                f"depth={connectivity['depth_tolerance_mm']:.1f}mm"
            ),
        ),
    )
    combined = np.vstack((
        np.hstack(panels[:2]),
        np.hstack(panels[2:]),
    ))
    debug_dir = Path(_DEBUG_DIR)
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / "cylinder_depth_latest.png"
    for legacy_path in debug_dir.glob("cylinder_depth_*.png"):
        if legacy_path != path:
            try:
                legacy_path.unlink()
            except OSError:
                pass
    cv2.imwrite(str(path), combined)
    return str(path)


def _save_transparent_bottle_debug(
    color_img,
    depth_img,
    bbox: list[int],
    sample_u,
    sample_v,
    heights_m,
    desk_fit: dict,
    analysis,
) -> dict[str, str]:
    """Save the latest sparse bottle pose and height diagnostics."""

    import cv2
    import numpy as np
    from pathlib import Path

    color = np.asarray(color_img).copy()
    depth = np.asarray(depth_img)
    u = np.asarray(sample_u, dtype=np.int32)
    v = np.asarray(sample_v, dtype=np.int32)
    heights = np.asarray(heights_m, dtype=np.float64)
    xmin, ymin, xmax, ymax = [int(value) for value in bbox]

    def paint_points(image, mask, bgr, radius=1):
        indices = np.flatnonzero(np.asarray(mask, dtype=bool))
        for index in indices:
            cv2.circle(
                image,
                (int(u[index]), int(v[index])),
                radius,
                bgr,
                -1,
                cv2.LINE_AA,
            )

    pose = color.copy()
    pose[:] = (pose.astype(np.float32) * 0.72).astype(np.uint8)
    cv2.rectangle(pose, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

    desk_inliers = np.asarray(desk_fit.get("inliers", ()), dtype=bool)
    desk_u = np.asarray(desk_fit.get("sample_u", ()), dtype=np.int32)
    desk_v = np.asarray(desk_fit.get("sample_v", ()), dtype=np.int32)
    if len(desk_inliers) == len(desk_u):
        visible_desk_u = desk_u[desk_inliers]
        visible_desk_v = desk_v[desk_inliers]
        if len(visible_desk_u) > 800:
            visible_indices = np.linspace(
                0, len(visible_desk_u) - 1, 800, dtype=int,
            )
            visible_desk_u = visible_desk_u[visible_indices]
            visible_desk_v = visible_desk_v[visible_indices]
        for du, dv in zip(visible_desk_u, visible_desk_v):
            cv2.circle(pose, (int(du), int(dv)), 1, (255, 128, 0), -1)

    paint_points(pose, analysis.reliable_mask, (0, 165, 255), 1)
    paint_points(pose, analysis.label_mask, (0, 255, 0), 2)
    paint_points(pose, analysis.cap_mask, (255, 0, 255), 2)

    center = np.asarray(analysis.image_axis_center_uv, dtype=np.float64)
    axis = np.asarray(analysis.image_axis_uv, dtype=np.float64)
    perpendicular = np.asarray([-axis[1], axis[0]])
    minor_span = max(20.0, min(xmax - xmin, ymax - ymin) * 0.65)
    for side in (-1.0, 1.0):
        boundary_center = (
            center + axis * analysis.label_axis_half_span_px * side
        )
        start = boundary_center - perpendicular * minor_span
        end = boundary_center + perpendicular * minor_span
        cv2.line(
            pose,
            tuple(np.rint(start).astype(int)),
            tuple(np.rint(end).astype(int)),
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    label_pixels = np.column_stack((u, v))[analysis.label_mask]
    label_pixel = np.rint(np.median(label_pixels, axis=0)).astype(int)
    cv2.drawMarker(
        pose,
        tuple(label_pixel),
        (255, 255, 255),
        cv2.MARKER_CROSS,
        16,
        2,
    )
    if analysis.orientation == "lying":
        arrow_end = (
            label_pixel.astype(np.float64)
            + perpendicular * max(28.0, minor_span * 0.7)
        )
        cv2.arrowedLine(
            pose,
            tuple(label_pixel),
            tuple(np.rint(arrow_end).astype(int)),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
            tipLength=0.25,
        )
    text_lines = (
        f"pose={analysis.orientation}",
        (
            f"p90={analysis.height_p90_m * 1000:.1f}mm "
            f"p95={analysis.height_p95_m * 1000:.1f}mm"
        ),
        (
            f"support={analysis.reliable_count} "
            f"label={analysis.label_count} "
            f"connected={analysis.connected_ratio:.0%}"
        ),
        (
            f"tcp=({analysis.tcp_xyz[0]:.3f},"
            f"{analysis.tcp_xyz[1]:.3f},"
            f"{analysis.tcp_xyz[2]:.3f})"
        ),
    )
    cv2.rectangle(pose, (0, 0), (pose.shape[1], 80), (0, 0, 0), -1)
    for index, text in enumerate(text_lines):
        cv2.putText(
            pose,
            text,
            (8, 17 + index * 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    height_view = color.copy()
    height_view[:] = (height_view.astype(np.float32) * 0.30).astype(np.uint8)
    reliable_indices = np.flatnonzero(analysis.reliable_mask)
    reliable_heights = heights[analysis.reliable_mask]
    if len(reliable_heights):
        maximum = max(float(np.quantile(reliable_heights, 0.98)), 0.001)
        normalized = np.clip(
            reliable_heights / maximum * 255.0, 0.0, 255.0,
        ).astype(np.uint8)
        colors = cv2.applyColorMap(
            normalized.reshape(-1, 1), cv2.COLORMAP_TURBO,
        ).reshape(-1, 3)
        for index, bgr in zip(reliable_indices, colors):
            cv2.circle(
                height_view,
                (int(u[index]), int(v[index])),
                2,
                tuple(int(value) for value in bgr),
                -1,
            )
    cv2.rectangle(
        height_view, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2,
    )
    cv2.rectangle(
        height_view, (0, 0), (height_view.shape[1], 42), (0, 0, 0), -1,
    )
    cv2.putText(
        height_view,
        "height above local desk: blue=low red=high",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        height_view,
        (
            f"p90={analysis.height_p90_m * 1000:.1f}mm "
            f"p95={analysis.height_p95_m * 1000:.1f}mm"
        ),
        (8, 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    debug_dir = Path(_DEBUG_DIR)
    debug_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pose": debug_dir / "bottle_pose_latest.png",
        "height": debug_dir / "bottle_height_latest.png",
    }
    cv2.imwrite(str(paths["pose"]), pose)
    cv2.imwrite(str(paths["height"]), height_view)
    return {name: str(path) for name, path in paths.items()}


def transparent_bottle_detection_to_3d(
    bridge: "RobotBridge",
    frame: ImageFrame,
    detection: dict,
) -> ComponentResult:
    """Measure the current transparent bottle from sparse real depth only."""

    import numpy as np

    bbox = detection.get("bbox")
    if not bbox or len(bbox) != 4:
        return ComponentResult.failure(
            "Transparent bottle detection has no valid bbox"
        )
    depth_img = frame.depth
    image_height, image_width = depth_img.shape[:2]
    xmin, ymin, xmax, ymax = [
        int(round(float(value))) for value in bbox
    ]
    xmin, xmax = max(0, xmin), min(image_width - 1, xmax)
    ymin, ymax = max(0, ymin), min(image_height - 1, ymax)
    if xmax <= xmin or ymax <= ymin:
        return ComponentResult.failure(
            "Transparent bottle detection bbox is empty"
        )

    camera_info = bridge.node.get_color_info()
    if camera_info is None:
        return ComponentResult.failure(
            "Color camera intrinsics are unavailable"
        )
    desk_fit = _fit_local_desk_plane(
        depth_img,
        [xmin, ymin, xmax, ymax],
        None,
        camera_info,
    )
    if desk_fit is None:
        return ComponentResult.failure(
            "Unable to fit a metric local desk plane around transparent bottle"
        )

    roi_valid = np.zeros(depth_img.shape[:2], dtype=bool)
    roi_valid[ymin:ymax + 1, xmin:xmax + 1] = True
    roi_valid &= depth_img > 0
    vv, uu = np.nonzero(roi_valid)
    depths_mm = depth_img[roi_valid].astype(np.float64)
    if len(depths_mm) < 30:
        return ComponentResult.failure(
            f"Only {len(depths_mm)} valid depth samples inside bottle bbox"
        )

    points_camera = _deproject_samples(
        uu, vv, depths_mm, camera_info,
    )
    desk_normal = np.asarray(desk_fit["normal"], dtype=np.float64)
    desk_offset = float(desk_fit["offset"])
    signed_heights = points_camera @ desk_normal + desk_offset
    # Most bbox pixels may see through the bottle to the desk, so the median
    # can be close to zero. Orient the plane from camera geometry instead of
    # relying on the sign of object samples.
    if desk_normal[2] > 0.0:
        desk_normal *= -1.0
        desk_offset *= -1.0
        signed_heights *= -1.0

    try:
        rigid_transform = _camera_to_base_rigid_transform(bridge, frame)
        points_base = _transform_camera_points_to_base(
            bridge,
            frame,
            points_camera,
            rigid_transform=rigid_transform,
        )
    except Exception as error:
        return ComponentResult.failure(
            f"Unable to transform transparent bottle points to base_link: {error}"
        )

    origin, rotation = rigid_transform
    normal_base_raw = rotation @ desk_normal
    normal_norm = float(np.linalg.norm(normal_base_raw))
    if normal_norm <= 1e-9:
        return ComponentResult.failure(
            "Transformed bottle desk normal is degenerate"
        )
    normal_base = normal_base_raw / normal_norm
    desk_offset_base = (
        desk_offset - float(normal_base_raw @ origin)
    ) / normal_norm
    if normal_base[2] < 0.0:
        normal_base *= -1.0
        desk_offset_base *= -1.0
        signed_heights *= -1.0
    if normal_base[2] <= 0.90:
        return ComponentResult.failure(
            "Fitted bottle desk normal is not aligned with base +Z"
        )

    profile = bridge.get_transparent_bottle_profile()
    provisional = (
        (signed_heights >= profile["transparent_bottle_min_height_m"])
        & (
            signed_heights
            <= profile["transparent_bottle_height_m"] + 0.03
        )
    )
    if int(np.count_nonzero(provisional)) < int(
        profile["transparent_bottle_min_label_points"]
    ):
        return ComponentResult.failure(
            "Bottle bbox has too few reliable points above the local desk",
            valid_depth_points=int(np.count_nonzero(provisional)),
        )
    provisional_xy = np.median(points_base[provisional, :2], axis=0)
    local_desk_z = -(
        normal_base[0] * provisional_xy[0]
        + normal_base[1] * provisional_xy[1]
        + desk_offset_base
    ) / normal_base[2]
    desk_error = abs(
        float(local_desk_z) - float(bridge.get_desk_surface_z())
    )
    if desk_error > bridge.get_desk_measurement_max_error():
        return ComponentResult.failure(
            f"Measured desk z differs from configured desk by "
            f"{desk_error:.4f}m "
            f"(limit {bridge.get_desk_measurement_max_error():.4f}m)"
        )

    try:
        analysis = analyze_transparent_bottle_points(
            np.column_stack((uu, vv)),
            depths_mm,
            points_base,
            signed_heights,
            local_desk_z=float(local_desk_z),
            minimum_height_m=float(
                profile["transparent_bottle_min_height_m"]
            ),
            maximum_height_m=float(
                profile["transparent_bottle_height_m"] + 0.03
            ),
            label_axis_fraction=float(
                profile["transparent_bottle_label_height_m"]
                / profile["transparent_bottle_height_m"]
            ),
            minimum_label_points=int(
                profile["transparent_bottle_min_label_points"]
            ),
            upright_min_p90_m=float(
                profile["transparent_bottle_upright_min_p90_m"]
            ),
            upright_min_p95_m=float(
                profile["transparent_bottle_upright_min_p95_m"]
            ),
            lying_min_p90_m=float(
                profile["transparent_bottle_lying_min_p90_m"]
            ),
            lying_min_p95_m=float(
                profile["transparent_bottle_lying_min_p95_m"]
            ),
            lying_max_p90_m=float(
                profile["transparent_bottle_lying_max_p90_m"]
            ),
            lying_max_p95_m=float(
                profile["transparent_bottle_lying_max_p95_m"]
            ),
        )
    except ValueError as error:
        return ComponentResult.failure(
            f"Transparent bottle sparse depth rejected: {error}"
        )

    if analysis.orientation in {"upright", "lying"}:
        center_2d = detection.get("center_2d")
        if not center_2d or len(center_2d) != 2:
            center_2d = [
                (xmin + xmax) * 0.5,
                (ymin + ymax) * 0.5,
            ]
        center_u, center_v = [float(value) for value in center_2d]
        ray_camera = np.asarray([
            (center_u - float(camera_info["cx"]))
            / float(camera_info["fx"]),
            (center_v - float(camera_info["cy"]))
            / float(camera_info["fy"]),
            1.0,
        ])
        ray_base = rotation @ ray_camera
        if abs(float(ray_base[2])) <= 1e-9:
            return ComponentResult.failure(
                "YOLO bbox center ray is parallel to the label-height plane"
            )
        target_plane_z = (
            float(analysis.label_surface_xyz[2])
            if analysis.orientation == "upright"
            else float(analysis.tcp_xyz[2])
        )
        scale = (target_plane_z - float(origin[2])) / float(ray_base[2])
        if scale <= 0.0:
            return ComponentResult.failure(
                "YOLO bbox center ray intersects behind the camera"
            )
        axis_point = origin + ray_base * scale
        analysis = replace(
            analysis,
            cap_or_axis_xy=(
                float(axis_point[0]),
                float(axis_point[1]),
            ),
            tcp_xyz=(
                float(axis_point[0]),
                float(axis_point[1]),
                target_plane_z,
            ),
        )

    measured_surface_height = (
        float(analysis.height_p90_m)
        if analysis.orientation == "lying"
        else float(analysis.label_surface_xyz[2]) - float(local_desk_z)
    )
    configured_desk_z = float(bridge.get_desk_surface_z())
    if analysis.orientation == "upright":
        anchored_tcp_z = configured_desk_z + measured_surface_height
    else:
        anchored_tcp_z = (
            configured_desk_z + measured_surface_height * 0.5
        )
    analysis = replace(
        analysis,
        tcp_xyz=(
            float(analysis.tcp_xyz[0]),
            float(analysis.tcp_xyz[1]),
            float(anchored_tcp_z),
        ),
    )

    try:
        debug_paths = _save_transparent_bottle_debug(
            frame.color,
            depth_img,
            [xmin, ymin, xmax, ymax],
            uu,
            vv,
            signed_heights,
            desk_fit,
            analysis,
        )
    except Exception:
        debug_paths = {}

    return ComponentResult.success(
        x=analysis.tcp_xyz[0],
        y=analysis.tcp_xyz[1],
        z=analysis.tcp_xyz[2],
        frame_id="base_link",
        orientation=analysis.orientation,
        height_p90_m=analysis.height_p90_m,
        height_p95_m=analysis.height_p95_m,
        local_desk_z=float(local_desk_z),
        configured_desk_z=configured_desk_z,
        measured_surface_height_m=measured_surface_height,
        label_surface_xyz=list(analysis.label_surface_xyz),
        cap_or_axis_xy=list(analysis.cap_or_axis_xy),
        horizontal_axis_xy=list(analysis.horizontal_axis_xy),
        horizontal_span_m=analysis.horizontal_span_m,
        tcp_xyz=list(analysis.tcp_xyz),
        valid_depth_points=analysis.reliable_count,
        label_depth_points=analysis.label_count,
        label_connected_ratio=analysis.connected_ratio,
        reliable_heights_m=[
            float(value)
            for value in signed_heights[analysis.reliable_mask]
        ],
        method="transparent_bottle_sparse_label_depth",
        depth_is_estimated=False,
        geometry_quality={
            "reliable": analysis.orientation in {"upright", "lying"},
            "orientation": analysis.orientation,
            "horizontal_span_m": analysis.horizontal_span_m,
            "height_p90_m": analysis.height_p90_m,
            "height_p95_m": analysis.height_p95_m,
            "reliable_depth_points": analysis.reliable_count,
            "label_depth_points": analysis.label_count,
            "label_connected_ratio": analysis.connected_ratio,
            "desk_inliers": int(desk_fit["inlier_count"]),
            "desk_samples": int(desk_fit["sample_count"]),
            "desk_residual_median_mm": float(
                desk_fit["residual_median_mm"]
            ),
            "configured_desk_error_m": float(desk_error),
        },
        debug_image=debug_paths.get("pose"),
        height_debug_image=debug_paths.get("height"),
    )


def cylinder_detection_to_3d(
    bridge: "RobotBridge",
    frame: ImageFrame,
    detection: dict,
    *,
    color_name: str | None = None,
) -> ComponentResult:
    """Fit a desk-supported upright/lying cylinder inside a YOLO bbox."""

    import cv2
    import numpy as np

    bbox = detection.get("bbox")
    if not bbox or len(bbox) != 4:
        return ComponentResult.failure("Cylinder detection has no valid bbox")
    depth_img = frame.depth
    image_height, image_width = depth_img.shape[:2]
    xmin, ymin, xmax, ymax = [
        int(round(float(value))) for value in bbox
    ]
    xmin, xmax = max(0, xmin), min(image_width - 1, xmax)
    ymin, ymax = max(0, ymin), min(image_height - 1, ymax)
    if xmax <= xmin or ymax <= ymin:
        return ComponentResult.failure("Cylinder detection bbox is empty")
    camera_info = bridge.node.get_color_info()
    if camera_info is None:
        return ComponentResult.failure("Color camera intrinsics are unavailable")

    desk_fit = _fit_local_desk_plane(
        depth_img,
        [xmin, ymin, xmax, ymax],
        None,
        camera_info,
    )
    if desk_fit is None:
        return ComponentResult.failure(
            "Unable to fit a metric local desk plane around cylinder"
        )

    roi_valid = np.zeros(depth_img.shape[:2], dtype=bool)
    roi_valid[ymin:ymax + 1, xmin:xmax + 1] = True
    roi_valid &= depth_img > 0
    vv, uu = np.nonzero(roi_valid)
    depths_mm = depth_img[roi_valid].astype(np.float64)
    if len(depths_mm) < 80:
        return ComponentResult.failure(
            f"Only {len(depths_mm)} valid depth samples inside cylinder bbox"
        )
    points_camera = _deproject_samples(uu, vv, depths_mm, camera_info)
    desk_normal = np.asarray(desk_fit["normal"], dtype=np.float64)
    desk_offset = float(desk_fit["offset"])
    signed_heights = points_camera @ desk_normal + desk_offset
    if float(np.median(signed_heights)) < 0.0:
        desk_normal *= -1.0
        desk_offset *= -1.0
        signed_heights *= -1.0
    plausible = (
        (signed_heights >= 0.002)
        & (signed_heights <= bridge.get_cylinder_max_length_m() + 0.03)
    )
    if int(np.count_nonzero(plausible)) < 80:
        return ComponentResult.failure(
            "Cylinder bbox has too few measured points above the fitted desk",
            valid_depth_points=int(np.count_nonzero(plausible)),
        )

    # A requested color may narrow the YOLO bbox, but never supplies shape or
    # metric geometry. If the mask is weak (transparent/printed bottle), retain
    # the full depth support and let the cylinder fit decide.
    color_assisted = False
    if color_name:
        color_mask = build_color_mask(frame.color, color_name)
        if color_mask is not None:
            color_hits = color_mask[vv, uu] > 0
            assisted = plausible & color_hits
            if (
                int(np.count_nonzero(assisted)) >= 80
                and np.count_nonzero(assisted)
                >= 0.35 * np.count_nonzero(plausible)
            ):
                plausible = assisted
                color_assisted = True

    # Preserve one measured object while allowing small transparent-surface
    # holes. Links adapt to bbox size and require locally continuous depth, so
    # nearby cap/background/desk layers cannot join solely by pixel proximity.
    selected_indices = np.flatnonzero(plausible)
    sample_u = uu[plausible].astype(np.int32)
    sample_v = vv[plausible].astype(np.int32)
    sample_depths = depths_mm[plausible]
    connectivity = _adaptive_depth_connected_support(
        sample_u,
        sample_v,
        sample_depths,
        [xmin, ymin, xmax, ymax],
    )
    keep_local = connectivity["keep_local"]
    connected_count = int(connectivity["connected_count"])
    connected_ratio = float(connectivity["connected_ratio"])
    connected_u = sample_u[keep_local]
    connected_v = sample_v[keep_local]
    try:
        depth_debug_path = _save_cylinder_depth_debug(
            depth_img,
            [xmin, ymin, xmax, ymax],
            uu,
            vv,
            desk_fit,
            sample_u,
            sample_v,
            connected_u,
            connected_v,
            connectivity,
        )
    except Exception:
        # Diagnostics must never decide whether a physical grasp is safe.
        depth_debug_path = ""
    if connected_count < 80 or connected_ratio < 0.55:
        return ComponentResult.failure(
            f"Cylinder connected depth support is weak "
            f"({connected_count} points, {connected_ratio:.1%}; "
            f"adaptive gap={connectivity['gap_px']}px, "
            f"depth tolerance={connectivity['depth_tolerance_mm']:.1f}mm)",
            valid_depth_points=connected_count,
            connected_ratio=connected_ratio,
            debug_image=depth_debug_path,
        )
    keep = np.zeros(len(plausible), dtype=bool)
    keep[selected_indices[keep_local]] = True
    selected_camera = points_camera[keep]
    selected_depths = depths_mm[keep]

    try:
        rigid_transform = _camera_to_base_rigid_transform(bridge, frame)
        selected_base = _transform_camera_points_to_base(
            bridge,
            frame,
            selected_camera,
            rigid_transform=rigid_transform,
        )
    except Exception as error:
        return ComponentResult.failure(
            f"Unable to transform cylinder points to base_link: {error}",
            valid_depth_points=connected_count,
            connected_ratio=connected_ratio,
            debug_image=depth_debug_path,
        )

    origin, rotation = rigid_transform
    normal_base_raw = rotation @ desk_normal
    normal_norm = float(np.linalg.norm(normal_base_raw))
    if normal_norm <= 1e-9:
        return ComponentResult.failure("Transformed desk normal is degenerate")
    normal_base = normal_base_raw / normal_norm
    desk_offset_base = (
        desk_offset - float(normal_base_raw @ origin)
    ) / normal_norm
    if normal_base[2] <= 0.90:
        return ComponentResult.failure(
            "Fitted desk normal is not sufficiently aligned with base +Z"
        )
    median_xy = np.median(selected_base[:, :2], axis=0)
    local_desk_z = -(
        normal_base[0] * median_xy[0]
        + normal_base[1] * median_xy[1]
        + desk_offset_base
    ) / normal_base[2]
    desk_error = abs(local_desk_z - bridge.get_desk_surface_z())
    if desk_error > bridge.get_desk_measurement_max_error():
        return ComponentResult.failure(
            f"Measured desk z differs from configured desk by "
            f"{desk_error:.4f}m "
            f"(limit {bridge.get_desk_measurement_max_error():.4f}m)"
        )

    try:
        geometry = fit_cylinder_geometry(
            selected_base,
            local_desk_z=float(local_desk_z),
            min_diameter_m=bridge.get_cylinder_min_diameter_m(),
            max_diameter_m=bridge.get_cylinder_max_diameter_m(),
            min_length_m=bridge.get_cylinder_min_length_m(),
            max_length_m=bridge.get_cylinder_max_length_m(),
            lying_axis_max_deviation_deg=(
                bridge.get_cylinder_lying_axis_max_deviation_deg()
            ),
            surface_depth_mm=float(np.median(selected_depths)),
        )
    except ValueError as error:
        return ComponentResult.failure(
            f"Cylinder geometry rejected: {error}",
            valid_depth_points=connected_count,
            connected_ratio=connected_ratio,
            debug_image=depth_debug_path,
        )
    quality = {
        **geometry.quality,
        "connected_ratio": connected_ratio,
        "connected_components": int(connectivity["component_count"]),
        "adaptive_gap_px": int(connectivity["gap_px"]),
        "depth_link_tolerance_mm": float(
            connectivity["depth_tolerance_mm"]
        ),
        "desk_inliers": int(desk_fit["inlier_count"]),
        "desk_samples": int(desk_fit["sample_count"]),
        "desk_residual_median_mm": float(
            desk_fit["residual_median_mm"]
        ),
        "configured_desk_error_m": float(desk_error),
        "color_mask_assisted": color_assisted,
    }
    geometry = replace(geometry, quality=quality)
    return ComponentResult.success(
        x=geometry.surface_xyz[0],
        y=geometry.surface_xyz[1],
        z=geometry.surface_xyz[2],
        frame_id="base_link",
        depth_mm=geometry.surface_depth_mm,
        valid_depth_points=connected_count,
        local_desk_z=geometry.local_desk_z,
        object_height=geometry.height,
        cylinder_diameter=geometry.diameter_m,
        cylinder_length=geometry.length_m,
        cylinder_orientation=geometry.orientation_class,
        method="yolo_depth_cylinder_fit",
        depth_is_estimated=False,
        geometry=geometry.to_dict(),
        geometry_quality=quality,
        debug_image=depth_debug_path,
    )


def bbox_to_3d(bridge: "RobotBridge", frame: ImageFrame,
               bbox: list[int], margin_px: int = 2) -> ComponentResult:
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
