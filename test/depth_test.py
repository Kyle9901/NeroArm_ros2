#!/usr/bin/env python3
"""Depth diagnostic — detect object and print 3D coordinates without moving."""

import os
import sys

_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from mcp_server.ros_bridge import RobotBridge
from mcp_server.vlm_client import VlmClient
from mcp_server.components import perception
from mcp_server.skills.perception import _target_color


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "红色物块"
    vlm = VlmClient()
    bridge = RobotBridge()
    bridge.start()

    frame = perception.capture_image(bridge)
    if not frame.ok:
        print(f"capture failed: {frame.error}")
        bridge.shutdown()
        return
    img = frame.data["frame"]

    color = _target_color(target)
    detected = None
    if color:
        r = perception.detect_by_color(img, color)
        if r.ok and r.data.get("found"):
            detected = r.data

    if not detected:
        r = perception.detect_by_vlm(vlm, img, target)
        if r.ok and r.data.get("found"):
            detected = r.data

    if not detected:
        print(f"not found: {target}")
        bridge.shutdown()
        return

    cx, cy = detected["center_2d"]
    pos = perception.pixel_to_3d(bridge, img, cx, cy)
    if not pos.ok:
        print(f"3D failed: {pos.error}")
        bridge.shutdown()
        return

    d = pos.data
    print(f"x={d['x']:.4f} y={d['y']:.4f} z={d['z']:.4f} depth_mm={d.get('depth_mm')} valid={d.get('valid_depth_points')}")

    bridge.shutdown()


if __name__ == "__main__":
    main()