#!/usr/bin/env python3
"""Measure an HSV block and the surrounding local desk in one frame.

Usage:
    source scripts/run_mcp.sh
    python -m test.depth_test 红色物块
"""

import os
import sys
import time

import cv2
import numpy as np


_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from mcp_server.ros_bridge import RobotBridge
from mcp_server.components import perception
from mcp_server.config import runtime_dir
from mcp_server.skills.perception import _target_color


_OUTPUT_DIR = str(runtime_dir("depth_test"))


def _sample_roi(bridge, depth_img: np.ndarray, roi: tuple[int, int, int, int]):
    """Return depth and 3D position using a real median-depth pixel in an ROI."""
    x1, y1, x2, y2 = roi
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
    base = bridge.node.transform_to_base(cam["x_c"], cam["y_c"], cam["z_c"])
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


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "红色物块"
    color_name = _target_color(target)
    if color_name is None:
        print(f"target must contain a supported color: {target}")
        return 1

    bridge = RobotBridge()
    bridge.start()
    try:
        captured = perception.capture_image(bridge)
        if not captured.ok:
            print(f"capture failed: {captured.error}")
            return 1
        frame = captured.data["frame"]

        detected = perception.detect_by_color(frame, color_name)
        if not detected.ok or not detected.data.get("found"):
            print(f"HSV target not found: {target}")
            return 1
        bbox = [int(value) for value in detected.data["bbox"]]

        object_pos = perception.bbox_to_3d(bridge, frame, bbox)
        if not object_pos.ok:
            print(f"object depth failed: {object_pos.error}")
            return 1

        depth_h, depth_w = frame.depth.shape[:2]
        desk_samples = []
        for name, roi in _desk_rois(bbox, depth_w, depth_h).items():
            try:
                sample = _sample_roi(bridge, frame.depth, roi)
            except Exception as exc:
                print(f"desk_{name}: transform failed: {exc}")
                continue
            if sample is not None:
                sample["name"] = name
                desk_samples.append(sample)

        image = frame.color.copy()
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f"object {object_pos.data['depth_mm']:.0f}mm",
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

        obj = object_pos.data
        print(f"target={target}  color={color_name}  bbox={bbox}")
        print(f"object: depth_mm={obj['depth_mm']:.1f}  valid_points={obj['valid_depth_points']} "
              f"base_xyz=({obj['x']:.4f}, {obj['y']:.4f}, {obj['z']:.4f})")
        for sample in desk_samples:
            print(f"desk_{sample['name']}: depth_mm={sample['depth_mm']:.1f} "
                  f"valid_points={sample['valid_points']} pixel={sample['pixel']} "
                  f"base_z={sample['z']:.4f}")

        if desk_samples:
            desk_z = float(np.median([sample["z"] for sample in desk_samples]))
            print(f"local_desk_z_median={desk_z:.4f}")
            print(f"object_height_above_local_desk={(obj['z'] - desk_z):.4f} m")
        else:
            print("no valid surrounding desk samples")
        print(f"debug_image={output_path}")
        return 0
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    sys.exit(main())
