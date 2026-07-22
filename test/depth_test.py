#!/usr/bin/env python3
"""Measure an HSV block or the calibrated transparent water bottle.

Usage:
    source scripts/run_mcp.sh
    python -m test.depth_test 红色物块
    python -m test.depth_test 水瓶
"""

from collections import Counter
import os
import sys
import time
import math

import cv2
import numpy as np


_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from mcp_server.ros_bridge import RobotBridge
from mcp_server.components import perception
from mcp_server.config import runtime_dir
from mcp_server.models.geometry import ObjectGeometry, aggregate_object_geometries
from mcp_server.object_types import is_transparent_bottle_target
from mcp_server.perception.transparent_bottle import (
    aggregate_transparent_bottle_measurements,
)
from mcp_server.skills.perception import _target_color
from mcp_server.yolo_detector import LazyYoloDetector


_OUTPUT_DIR = str(runtime_dir("depth_test"))


def _sample_roi(bridge, frame, roi: tuple[int, int, int, int]):
    """Return depth and 3D position using a real median-depth pixel in an ROI."""
    x1, y1, x2, y2 = roi
    depth_img = frame.depth
    patch = depth_img[y1:y2, x1:x2]
    ys, xs = np.nonzero(patch > 0)
    if len(xs) == 0:
        return None

    values = patch[ys, xs].astype(np.float64)
    depth_mm = float(np.median(values))
    index = int(np.argmin(np.abs(values - depth_mm)))
    u, v = x1 + int(xs[index]), y1 + int(ys[index])

    cam = bridge.node.compute_3d(u, v, depth_img, margin_px=0)
    if cam is None:
        return None
    base = bridge.node.transform_to_base(
        cam["x_c"], cam["y_c"], cam["z_c"],
        stamp=frame.ros_stamp,
        source_frame=frame.source_frame,
    )
    return {
        "roi": roi,
        "pixel": (u, v),
        "depth_mm": depth_mm,
        "valid_points": len(values),
        "x": base["x"],
        "y": base["y"],
        "z": base["z"],
    }


def _desk_rois(bbox, width: int, height: int):
    """Build four non-overlapping desk sampling bands around an object bbox."""
    xmin, ymin, xmax, ymax = bbox
    bw, bh = xmax - xmin, ymax - ymin
    gap = max(6, int(min(bw, bh) * 0.12))
    band = max(20, int(min(bw, bh) * 0.55))

    candidates = {
        "left": (xmin - gap - band, ymin, xmin - gap, ymax),
        "right": (xmax + gap, ymin, xmax + gap + band, ymax),
        "top": (xmin, ymin - gap - band, xmax, ymin - gap),
        "bottom": (xmin, ymax + gap, xmax, ymax + gap + band),
    }
    result = {}
    for name, (x1, y1, x2, y2) in candidates.items():
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 - x1 >= 5 and y2 - y1 >= 5:
            result[name] = (x1, y1, x2, y2)
    return result


def _run_transparent_bottle_test(bridge, target: str) -> int:
    """Measure five sparse-depth frames without commanding robot motion."""

    yolo = LazyYoloDetector()
    try:
        yolo.ensure_loaded()
    except Exception as error:
        print(f"YOLO initialization failed: {error}")
        return 1

    frame_count = bridge.get_transparent_bottle_depth_frames()
    maximum_capture_frames = (
        bridge.get_transparent_bottle_max_capture_frames()
    )
    minimum_consensus = max(3, math.ceil(frame_count * 0.6))
    profile = bridge.get_transparent_bottle_profile()

    def build_consensus(values):
        return aggregate_transparent_bottle_measurements(
            values,
            minimum_frames=minimum_consensus,
            minimum_label_points=int(
                profile["transparent_bottle_min_label_points"]
            ),
            maximum_tcp_spread_m=float(
                profile["transparent_bottle_tcp_max_spread_m"]
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

    measurements = []
    last_debug = {}
    debug_snapshots = {}
    consensus = None
    for index in range(maximum_capture_frames):
        captured = perception.capture_image(bridge, timeout=1.5)
        if not captured.ok:
            print(f"frame[{index}]: capture failed: {captured.error}")
            continue
        frame = captured.data["frame"]
        detected = perception.detect_by_yolo(yolo, frame, target)
        if not detected.ok or not detected.data.get("found"):
            print(
                f"frame[{index}]: YOLO bottle not found"
                + (f": {detected.error}" if detected.error else "")
            )
            continue
        measured = perception.transparent_bottle_detection_to_3d(
            bridge,
            frame,
            detected.data,
        )
        if not measured.ok:
            print(
                f"frame[{index}]: bottle depth failed: {measured.error}"
            )
            continue
        data = measured.data
        measurements.append(data)
        last_debug = {
            "pose": data.get("debug_image"),
            "height": data.get("height_debug_image"),
        }
        debug_snapshots[id(data)] = {
            name: (
                cv2.imread(path)
                if path and os.path.isfile(path)
                else None
            )
            for name, path in last_debug.items()
        }
        label = data["label_surface_xyz"]
        tcp = data["tcp_xyz"]
        print(
            f"frame[{index}]: pose={data['orientation']} "
            f"p90={data['height_p90_m'] * 1000.0:.1f}mm "
            f"p95={data['height_p95_m'] * 1000.0:.1f}mm "
            f"support={data['valid_depth_points']} "
            f"label={data['label_depth_points']} "
            f"label_surface=({label[0]:.4f},{label[1]:.4f},"
            f"{label[2]:.4f}) "
            f"tcp=({tcp[0]:.4f},{tcp[1]:.4f},{tcp[2]:.4f})"
        )
        consensus = build_consensus(measurements)
        if consensus.ready:
            print(
                f"adaptive capture reached stable consensus after "
                f"{index + 1} frame(s)"
            )
            break

    if consensus is None:
        consensus = build_consensus(measurements)
    if not consensus.ready:
        print(
            f"bottle measurement rejected after at most "
            f"{maximum_capture_frames} captures: {consensus.reason}"
        )
        return 1

    pose_counts = Counter(
        item["orientation"]
        for item in measurements
        if item["orientation"] in {"upright", "lying"}
    )
    orientation = consensus.orientation
    p90 = float(consensus.height_p90_m)
    p95 = float(consensus.height_p95_m)
    minimum_quality_points = consensus.label_gate
    inliers = [measurements[index] for index in consensus.inlier_indices]
    inlier_count = len(inliers)
    representative = max(
        inliers,
        key=lambda item: int(item["label_depth_points"]),
    )
    representative_snapshot = debug_snapshots.get(id(representative), {})
    for name, image in representative_snapshot.items():
        path = last_debug.get(name)
        if path and image is not None:
            cv2.imwrite(path, image)
    tcp_samples = np.asarray(
        [item["tcp_xyz"] for item in inliers], dtype=np.float64,
    )
    label_samples = np.asarray(
        [item["label_surface_xyz"] for item in inliers], dtype=np.float64,
    )
    tcp = np.asarray(consensus.tcp_xyz, dtype=np.float64)
    label = np.median(label_samples, axis=0)
    desk_z = float(np.median([
        item["local_desk_z"] for item in inliers
    ]))
    tcp_spread = float(consensus.tcp_spread_m)
    maximum_tcp_spread = float(
        profile["transparent_bottle_tcp_max_spread_m"]
    )
    tcp_stable = consensus.ready

    print(
        f"target={target} source=YOLO quality_measurements="
        f"{len(measurements)} captures<= {maximum_capture_frames}"
    )
    print(
        f"pose={orientation} quality_frames={inlier_count}/{frame_count} "
        f"per_frame_counts={dict(pose_counts)} "
        f"label_gate>={minimum_quality_points}"
    )
    print(
        f"height_above_desk: p90={p90 * 1000.0:.1f}mm "
        f"p95={p95 * 1000.0:.1f}mm local_desk_z={desk_z:.4f}"
    )
    print(
        f"label_surface_median=({label[0]:.4f}, {label[1]:.4f}, "
        f"{label[2]:.4f})"
    )
    print(
        f"recommended_tcp=({tcp[0]:.4f}, {tcp[1]:.4f}, "
        f"{tcp[2]:.4f}) spread={tcp_spread * 1000.0:.1f}mm"
    )
    print(
        "tcp_quality="
        + (
            "stable"
            if tcp_stable
            else (
                f"provisional: quality_frames={inlier_count}/"
                f"{minimum_consensus}, spread={tcp_spread * 1000.0:.1f}/"
                f"{maximum_tcp_spread * 1000.0:.1f}mm"
            )
        )
    )
    print(
        "grasp_strategy="
        + (
            "horizontal_side_grasp_at_measured_label_z"
            if orientation == "upright"
            else "vertical_top_grasp_at_measured_bottle_axis"
        )
    )
    if last_debug.get("pose"):
        print(f"pose_debug_image={last_debug['pose']}")
    if last_debug.get("height"):
        print(f"height_debug_image={last_debug['height']}")
    return 0 if tcp_stable else 1


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "红色物块"

    bridge = RobotBridge()
    bridge.start()
    try:
        if is_transparent_bottle_target(target):
            return _run_transparent_bottle_test(bridge, target)

        color_name = _target_color(target)
        if color_name is None:
            print(
                f"target must be the configured transparent bottle or "
                f"contain a supported color: {target}"
            )
            return 1
        frame_count = bridge.get_block_depth_frames()
        observations = []
        successful_frames = []
        for index in range(frame_count):
            captured = perception.capture_image(bridge, timeout=1.5)
            if not captured.ok:
                print(f"frame[{index}]: capture failed: {captured.error}")
                continue
            candidate_frame = captured.data["frame"]
            detected = perception.detect_by_color(candidate_frame, color_name)
            if not detected.ok or not detected.data.get("found"):
                print(f"frame[{index}]: HSV target not found")
                continue
            measured = perception.color_detection_to_3d(
                bridge, candidate_frame, detected.data,
            )
            if not measured.ok:
                print(f"frame[{index}]: geometry failed: {measured.error}")
                continue
            geometry = ObjectGeometry.from_dict(measured.data["geometry"])
            observations.append(geometry)
            successful_frames.append((candidate_frame, detected.data, measured.data))
            print(
                f"frame[{index}]: depth_mm={geometry.surface_depth_mm:.1f} "
                f"height={geometry.height:.4f} yaw={geometry.yaw_rad:.3f} "
                f"xyz=({geometry.surface_xyz[0]:.4f}, "
                f"{geometry.surface_xyz[1]:.4f}, "
                f"{geometry.surface_xyz[2]:.4f})"
            )

        aggregation = aggregate_object_geometries(
            observations,
            requested_frames=frame_count,
            min_valid_frames=max(3, math.ceil(frame_count * 0.6)),
            max_position_deviation_m=bridge.get_block_xy_max_spread(),
            max_height_deviation_m=bridge.get_block_depth_max_spread(),
            max_yaw_deviation_rad=math.radians(
                bridge.get_block_yaw_max_spread_deg(),
            ),
            max_depth_deviation_mm=bridge.get_block_depth_max_spread() * 1000.0,
        )
        if aggregation.geometry is None:
            quality = aggregation.quality.to_dict()
            print(
                f"{frame_count}-frame geometry rejected: "
                f"{quality['rejection_reasons']}"
            )
            return 1

        geometry = aggregation.geometry
        frame, detection, representative = successful_frames[-1]
        bbox = [int(value) for value in detection["bbox"]]
        obj = {
            **representative,
            "x": geometry.surface_xyz[0],
            "y": geometry.surface_xyz[1],
            "z": geometry.surface_xyz[2],
            "depth_mm": geometry.surface_depth_mm,
            "local_desk_z": geometry.local_desk_z,
            "object_height": geometry.height,
            "yaw_rad": geometry.yaw_rad,
            "method": f"realtime_depth_{frame_count}frame_median",
            "valid_depth_points": sum(
                int(item[2].get("valid_depth_points", 0))
                for item in successful_frames
            ),
        }

        depth_h, depth_w = frame.depth.shape[:2]
        desk_samples = []
        for name, roi in _desk_rois(bbox, depth_w, depth_h).items():
            try:
                sample = _sample_roi(bridge, frame, roi)
            except Exception as exc:
                print(f"desk_{name}: transform failed: {exc}")
                continue
            if sample is not None:
                sample["name"] = name
                desk_samples.append(sample)

        image = frame.color.copy()
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        rotated_box = np.asarray(detection["rotated_box_2d"], dtype=np.int32)
        cv2.polylines(image, [rotated_box], True, (0, 255, 255), 2)
        cv2.putText(image, f"object {obj['depth_mm']:.0f}mm",
                    (xmin, max(18, ymin - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2, cv2.LINE_AA)
        for sample in desk_samples:
            x1, y1, x2, y2 = sample["roi"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 128, 0), 2)
            cv2.circle(image, sample["pixel"], 3, (0, 0, 255), -1)
            cv2.putText(image, f"{sample['name']} {sample['depth_mm']:.0f}mm",
                        (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 128, 0), 1, cv2.LINE_AA)

        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(_OUTPUT_DIR, f"depth_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(output_path, image)

        print(f"target={target}  color={color_name}  bbox={bbox}")
        print(f"object: depth_mm={obj['depth_mm']:.1f}  valid_points={obj['valid_depth_points']} "
              f"base_xyz=({obj['x']:.4f}, {obj['y']:.4f}, {obj['z']:.4f}) "
              f"method={obj.get('method', '?')} yaw_rad={obj['yaw_rad']:.3f}")
        print(
            f"object_depth_source=realtime_depth_{frame_count}frame_median "
            f"local_desk_z={obj['local_desk_z']:.4f} "
            f"measured_height={obj['object_height']:.4f}"
        )
        print(f"geometry_quality={aggregation.quality.to_dict()}")
        for sample in desk_samples:
            print(f"desk_{sample['name']}: depth_mm={sample['depth_mm']:.1f} "
                  f"valid_points={sample['valid_points']} pixel={sample['pixel']} "
                  f"base_z={sample['z']:.4f}")

        if desk_samples:
            desk_z = float(np.median([sample["z"] for sample in desk_samples]))
            print(f"local_desk_z_median={desk_z:.4f}")
            reference_desk_z = obj.get("local_desk_z", desk_z)
            print(f"object_height_above_local_desk={(obj['z'] - reference_desk_z):.4f} m")
        else:
            print("no valid surrounding desk samples")
        print(f"debug_image={output_path}")
        return 0
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    sys.exit(main())
