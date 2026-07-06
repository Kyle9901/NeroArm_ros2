"""Prepare/bringup skill."""

import sys
import time
from typing import TYPE_CHECKING

from .base import SkillResult
from ..components import infra, perception

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge


def _check_tf_frame(bridge: "RobotBridge", frame_id: str, timeout: float = 5.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            bridge.node.tf_buffer.can_transform(
                "base_link", frame_id, bridge.node.get_clock().now())
            return True
        except Exception:
            time.sleep(0.5)
    return False


def _check_duplicate_nodes(bridge: "RobotBridge") -> int:
    """Return count of each critical node. Returns 0 if none, 1 if clean, >1 if duplicates."""
    try:
        nodes = bridge.node.get_node_names_and_namespaces()
        counts = {}
        for full_name, _ in nodes:
            base = full_name.split("/")[-1]
            if base in ("move_group", "agx_arm_ctrl_single_node"):
                counts[base] = counts.get(base, 0) + 1
        return max(counts.values()) if counts else 0
    except Exception:
        return 0


def _kill_extra_nodes(bridge: "RobotBridge") -> None:
    """Kill duplicate processes, keeping only the newest (highest PID)."""
    import subprocess
    for node_name in ("move_group", "agx_arm_ctrl_single_node"):
        try:
            result = subprocess.run(
                ["pgrep", "-f", node_name], capture_output=True, text=True, timeout=5)
            pids = [int(p) for p in result.stdout.strip().split("\n") if p]
            if len(pids) <= 1:
                continue
            pids.sort()
            keep = pids[-1]
            kill = pids[:-1]
            print(f"[prepare] {node_name}: keeping pid {keep}, killing {kill}",
                  file=sys.stderr, flush=True)
            for pid in kill:
                subprocess.run(["kill", "-9", str(pid)], timeout=5)
        except Exception:
            pass
    time.sleep(1)
    with bridge._proc_lock:
        bridge._managed_procs = {}


def prepare(bridge: "RobotBridge", can_port: str = "can0",
            calib_name: str = "my_eih_calib_v6") -> SkillResult:
    count = _check_duplicate_nodes(bridge)
    if count > 1:
        print(f"[prepare] {count} duplicate control nodes detected — keeping newest, killing extras",
              file=sys.stderr, flush=True)
        _kill_extra_nodes(bridge)
    elif count == 0:
        print("[prepare] no control nodes — starting", file=sys.stderr, flush=True)
        result = infra.bringup_nodes(bridge, can_port=can_port, calib_name=calib_name)
        if not result.ok:
            return SkillResult.failure(
                result.error or "bringup failed",
                failed_step="bringup_nodes",
                retryable=True,
                **result.data,
            )

    status = infra.bringup_status(bridge)
    endpoints = status.data.get("endpoints", {}) if status.ok else {}

    if endpoints.get("move_action") and endpoints.get("camera_color") and endpoints.get("tf"):
        frame = perception.capture_image(bridge, timeout=3.0)
        if frame.ok:
            if _check_tf_frame(bridge, "camera_color_optical_frame", timeout=3.0):
                return SkillResult.success(already_ready=True, **status.data)
            print("[prepare] TF frame missing — restarting calib", file=sys.stderr, flush=True)
        else:
            print("[prepare] camera topic exists but no image — restarting camera", file=sys.stderr, flush=True)
        result = infra.bringup_nodes(bridge, can_port=can_port, calib_name=calib_name)
    else:
        result = infra.bringup_nodes(bridge, can_port=can_port, calib_name=calib_name)

    if not result.ok:
        return SkillResult.failure(
            result.error or "bringup failed",
            failed_step="bringup_nodes",
            retryable=True,
            **result.data,
        )

    frame = perception.capture_image(bridge, timeout=5.0)
    if not frame.ok:
        return SkillResult.failure(
            "Camera started but no image received. Check camera hardware.",
            failed_step="camera_check",
            retryable=True,
            **result.data,
        )

    if not _check_tf_frame(bridge, "camera_color_optical_frame", timeout=8.0):
        return SkillResult.failure(
            "TF frame camera_color_optical_frame not found. Check handeye calibration.",
            failed_step="tf_check",
            retryable=True,
            **result.data,
        )

    return SkillResult.success(already_ready=False, **result.data)