#!/usr/bin/env python3
"""
Collect raw color + depth image pairs from the Orbbec camera.
Saves 16-bit depth as PNG to preserve millimeter precision.

Usage:
    source scripts/run_mcp.sh && python -m test.collect_data --count 20 --output test_data/
"""

import argparse
import os
import sys
import time

_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

import cv2
import numpy as np

from mcp_server.ros_bridge import RobotBridge


def main():
    parser = argparse.ArgumentParser(description="Collect color+depth image pairs from camera")
    parser.add_argument("--count", type=int, default=20, help="Number of frames to collect")
    parser.add_argument("--output", type=str, default="test_data", help="Output directory")
    parser.add_argument("--interval", type=float, default=1.5, help="Seconds between captures")
    args = parser.parse_args()

    output_dir = os.path.join(_pkg_dir, args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"[collect] Output directory: {output_dir}")
    print(f"[collect] Target frames: {args.count}")

    bridge = RobotBridge()
    bridge.start()
    print("[collect] ROS bridge ready. Waiting for camera...")

    # Wait for first valid image pair
    for _ in range(30):
        images = bridge.node.get_latest_images(timeout=2.0)
        if images is not None:
            break
        print("[collect] Waiting for camera images...")
        time.sleep(1.0)
    else:
        print("[collect] ERROR: No camera images received after 30s. Is the camera running?")
        bridge.shutdown()
        return 1

    print(f"[collect] Camera ready. Starting capture in 3 seconds...")
    print(f"[collect] Move objects between captures to vary the dataset.")
    time.sleep(3.0)

    saved = 0
    for i in range(args.count):
        images = bridge.node.get_latest_images(timeout=2.0)
        if images is None:
            print(f"[collect] Frame {i:04d}: no image, skipping")
            continue

        color_bgr, depth_mm = images

        # Save color as JPEG
        color_path = os.path.join(output_dir, f"{i:04d}_color.jpg")
        cv2.imwrite(color_path, color_bgr)

        # Save depth as 16-bit PNG (preserve mm precision)
        depth_path = os.path.join(output_dir, f"{i:04d}_depth.png")
        cv2.imwrite(depth_path, depth_mm.astype(np.uint16))

        saved += 1
        print(f"[collect] Saved {i:04d} | color={color_bgr.shape} | depth={depth_mm.shape} "
              f"| depth_range=[{depth_mm[depth_mm>0].min() if (depth_mm>0).any() else 0}, "
              f"{depth_mm.max()}]mm")

        if i < args.count - 1:
            print(f"[collect] Next capture in {args.interval}s... (move objects now)")
            time.sleep(args.interval)

    bridge.shutdown()
    print(f"[collect] Done. Saved {saved}/{args.count} frames to {output_dir}/")
    print(f"[collect] Ready for offline testing: python -m test.perception_test --data {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())