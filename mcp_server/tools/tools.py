"""
MCP tool implementations — pure functions that call RobotBridge or VlmClient.
"""

import os
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge
    from ..vlm_client import VlmClient


_DEBUG_DIR = os.environ.get("VLM_DEBUG_DIR", "/tmp/vlm_debug")


def _log(msg: str) -> None:
    print(f"[robot-arm] {msg}", file=sys.stderr, flush=True)


def _get_image(bridge: "RobotBridge", timeout=3.0):
    """Capture latest color+depth pair, or return error dict."""
    pair = bridge.node.get_latest_images(timeout=timeout)
    if pair is None:
        return None, {"success": False, "error": "No image — camera may not be running"}
    return pair, None


def _save_debug(img, bboxes, labels, prefix="detect") -> str:
    """Draw bboxes on image, save to disk, return path."""
    import cv2
    from ..vlm_client import _draw_bboxes
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    annotated = _draw_bboxes(img, bboxes, labels)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_DEBUG_DIR, f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, annotated)
    return path


# ══════════════════════════════════════════════════
#  Vision tools
# ══════════════════════════════════════════════════

def arm_capture_image(bridge: "RobotBridge") -> dict:
    pair, err = _get_image(bridge)
    if err:
        return err
    color, depth = pair
    import base64, cv2
    ok, buf = cv2.imencode(".jpg", color)
    if not ok:
        return {"success": False, "error": "JPEG encode failed"}
    b64 = base64.b64encode(buf).decode()
    return {
        "success": True, "image_base64": b64,
        "width": color.shape[1], "height": color.shape[0], "channels": 3,
        "depth_mean_mm": float(depth[depth > 0].mean()) if (depth > 0).any() else None,
    }


def arm_detect_vlm(bridge: "RobotBridge", vlm: "VlmClient", target: str) -> dict:
    pair, err = _get_image(bridge)
    if err:
        return err
    img = pair[0]
    try:
        result = vlm.detect(img, target)
    except Exception as e:
        return {"success": False, "error": f"VLM detection failed: {e}"}
    if result is None or not result.get("found"):
        return {
            "success": True, "found": False, "retryable": False,
            "message": f"Object '{target}' not found. "
            "Try arm_go_home first, use a simpler description, "
            "or arm_detect_color for solid-colour blocks.",
        }
    # result already includes debug_image from VlmClient.detect()
    return {"success": True, **result}


def arm_detect_color(bridge: "RobotBridge", color_name: str, location_hint: str = "") -> dict:
    from ..vlm_client import detect_by_color
    pair, err = _get_image(bridge)
    if err:
        return err
    bbox = detect_by_color(pair[0], color_name, location_hint)
    if bbox is None:
        colors = "blue, red, green, yellow, purple, orange, cyan"
        return {
            "success": True, "found": False, "retryable": False,
            "message": f"No {color_name} block found. Supported: {colors}. "
            "Use arm_detect_blocks to scan all blocks at once.",
        }
    xmin, ymin, xmax, ymax = bbox
    debug_path = _save_debug(pair[0], [[xmin, ymin, xmax, ymax]], [color_name])
    return {
        "success": True, "found": True, "color": color_name,
        "bbox": [xmin, ymin, xmax, ymax],
        "center_2d": [int((xmin + xmax) / 2), int((ymin + ymax) / 2)],
        "debug_image": debug_path,
    }


def arm_detect_blocks(bridge: "RobotBridge", location_hint: str = "") -> dict:
    from ..vlm_client import detect_all_color_blocks
    pair, err = _get_image(bridge)
    if err:
        return err
    blocks = detect_all_color_blocks(pair[0], location_hint)
    if not blocks:
        return {
            "success": True, "found": False, "count": 0, "blocks": [], "retryable": False,
            "message": "No blocks found. Try arm_go_home for a clearer view.",
        }
    bboxes = [b["bbox"] for b in blocks]
    labels = [b["color"] for b in blocks]
    debug_path = _save_debug(pair[0], bboxes, labels)
    return {
        "success": True, "found": True, "count": len(blocks), "blocks": blocks,
        "debug_image": debug_path,
    }


def arm_get_3d_position(bridge: "RobotBridge", u: int, v: int) -> dict:
    pair, err = _get_image(bridge)
    if err:
        return err
    cam3d = bridge.node.compute_3d(u, v, pair[1])
    if cam3d is None:
        return {"success": False, "error": f"No valid depth at ({u},{v}). Try adjusting pixel coords."}
    try:
        base = bridge.node.transform_to_base(cam3d["x_c"], cam3d["y_c"], cam3d["z_c"])
    except Exception as e:
        return {"success": False, "error": f"TF transform failed: {e}. Check TF2 is running."}
    return {
        "success": True, "x": base["x"], "y": base["y"], "z": base["z"],
        "frame_id": "base_link", "depth_mm": cam3d["depth_mm"],
        "valid_depth_points": cam3d["valid_points"],
    }


def arm_configure_vlm(bridge: "RobotBridge", vlm: "VlmClient",
                      api_key: str | None = None, api_url: str | None = None,
                      model: str | None = None) -> dict:
    changes = []
    if api_key is not None:
        vlm.api_key = api_key; changes.append("api_key")
    if api_url is not None:
        vlm.api_url = api_url; changes.append("api_url")
    if model is not None:
        vlm.model_name = model; changes.append("model")
    cur = {"api_url": vlm.api_url, "model": vlm.model_name, "api_key_set": bool(vlm.api_key)}
    return {"success": True, "message": f"Updated: {', '.join(changes)}" if changes else "No changes", "current": cur}


# ══════════════════════════════════════════════════
#  Motion tools
# ══════════════════════════════════════════════════

def arm_get_status(bridge: "RobotBridge") -> dict:
    js = bridge.node.get_joint_state()
    return {
        "success": True, **js,
        "holding": bridge.get_holding(),
        "workspace": {
            "x_min": bridge.node._workspace_x_min, "x_max": bridge.node._workspace_x_max,
            "y_min": bridge.node._workspace_y_min, "y_max": bridge.node._workspace_y_max,
        },
        "safe_height": bridge.get_safe_height(),
        "desk_surface_z": bridge.node._get_param("desk_z_surface"),
        "grasp_geometry": {
            "flange_to_tip": bridge.get_flange_to_tip(),
            "fingertip_overlap": bridge.get_fingertip_overlap(),
            "grasp_depth": bridge.get_grasp_depth(),
        },
    }


def arm_move_joints(bridge: "RobotBridge", joint_angles_deg: list[float], timeout: float = 20.0) -> dict:
    ok, msg = bridge.node.move_joints(joint_angles_deg, timeout)
    return {"success": ok, "message": msg}


def arm_move_to_pose(bridge: "RobotBridge", x: float, y: float, z: float,
                     quat: list[float] | None = None, timeout: float = 60.0) -> dict:
    ok, msg = bridge.node.move_to_pose(x, y, z, quat, timeout)
    return {"success": ok, "message": msg}


def arm_move_cartesian(bridge: "RobotBridge", x: float, y: float, z: float,
                       quat: list[float] | None = None, timeout: float = 30.0) -> dict:
    ok, msg = bridge.node.move_cartesian(x, y, z, quat, timeout)
    return {"success": ok, "message": msg}


def arm_control_gripper(bridge: "RobotBridge", width: float, duration: float = 1.5, timeout: float = 5.0) -> dict:
    ok, msg = bridge.node.control_gripper(width, duration, timeout)
    return {"success": ok, "message": msg}


def arm_go_home(bridge: "RobotBridge", timeout: float = 20.0) -> dict:
    ok, msg = bridge.node.go_home(timeout)
    return {"success": ok, "message": msg}


def arm_stop(bridge: "RobotBridge") -> dict:
    ok, msg = bridge.emergency_stop()
    return {"success": ok, "message": msg}


# ══════════════════════════════════════════════════
#  Sequence tools
# ══════════════════════════════════════════════════

def _recover_to_safe(bridge: "RobotBridge", x: float, y: float, safe_h: float, quat: list[float]) -> bool:
    """Best-effort recovery: cartesian lift → planned lift → go_home → emergency_stop."""
    node = bridge.node
    _log("RECOVERY: attempting to lift to safe height")
    for label, fn in [("cartesian lift", lambda: node.move_cartesian(x, y, safe_h, quat)),
                       ("planned lift", lambda: node.move_to_pose(x, y, safe_h, quat)),
                       ("go_home", lambda: node.go_home(timeout=20.0))]:
        ok, _ = fn()
        if ok:
            _log(f"RECOVERY: {label} successful")
            return True
        _log(f"RECOVERY: {label} failed, trying next")
    _log("RECOVERY: all attempts failed — EMERGENCY STOP")
    node.emergency_stop()
    return False


def arm_execute_grasp(bridge: "RobotBridge", x: float, y: float, z: float,
                      quat: list[float] | None = None) -> dict:
    """
    Pick up an object at (x, y, z) in base_link frame.
    z = object SURFACE height (from arm_get_3d_position). All z-offsets handled internally.
    Steps: 1) open gripper + approach (z+0.26m)  2) SLOW descent to grasp (z+0.135m → fingertip at z-0.04m)
           3) close gripper  4) measure gripper width  5) lift to safe height (0.40m).
    Returns holding=True if object physically blocked gripper closure, False if fully closed.
    On failure, auto-recovers to safe height.
    """
    t0 = time.monotonic()
    if quat is None:
        quat = bridge.get_grasp_quat()
    approach_h, grasp_d, safe_h = bridge.get_approach_height(), bridge.get_grasp_depth(), bridge.get_safe_height()
    open_w, close_w = bridge.get_gripper_open_width(), bridge.get_gripper_close_width()
    descent_vel, descent_accel = bridge.get_descent_velocity_scaling(), bridge.get_descent_accel_scaling()
    node = bridge.node
    steps, failed_step, at_grasp_pose, actually_holding = [], None, False, False

    if not node.workspace_check(x, y):
        return {"success": False, "step": "workspace_check", "error": "Target outside workspace"}

    def _elapsed():
        return time.monotonic() - t0

    def _gripper_is_closed():
        w = node.get_joint_state().get("gripper_width")
        return None if w is None else w < 0.03

    try:
        _log(f"GRASP 1/4: open gripper + approach (z={z + approach_h:.3f})")
        ok, msg = node.control_gripper(open_w, duration=2.0)
        if not ok:
            failed_step = "1/4 open_gripper"; return {"success": False, "step": failed_step, "error": msg}
        ok, msg = node.move_to_pose(x, y, z + approach_h, quat)
        if not ok:
            failed_step = "1/4 approach"; return {"success": False, "step": failed_step, "error": msg}

        _log(f"GRASP 2/4: SLOW descent to grasp (z={z + grasp_d:.3f}, vel={descent_vel:.0%})")
        ok, msg = node.move_cartesian(x, y, z + grasp_d, quat,
                                       velocity_override=descent_vel, accel_override=descent_accel)
        if not ok:
            failed_step = "2/4 descent"; return {"success": False, "step": failed_step, "error": msg}
        at_grasp_pose = True

        _log("GRASP 3/4: closing gripper")
        ok, msg = node.control_gripper(close_w, duration=2.0)
        if not ok:
            failed_step = "3/4 close"; return {"success": False, "step": failed_step, "error": msg}

        is_closed = _gripper_is_closed()
        actually_holding = (is_closed is False)
        if is_closed is True:
            _log("GRASP: gripper nearly closed — NO object")
        elif is_closed is False:
            _log("GRASP: gripper stays open — object held")
        else:
            _log("GRASP: cannot read gripper width, assuming not holding")

        _log(f"GRASP 4/4: lifting to safe height (z={safe_h:.3f})")
        ok, msg = node.move_cartesian(x, y, safe_h, quat)
        if not ok:
            ok, msg = node.move_to_pose(x, y, safe_h, quat)
        if not ok:
            failed_step = "4/4 lift"; return {"success": False, "step": failed_step, "error": msg}

        _log(f"GRASP complete in {_elapsed():.1f}s, holding={actually_holding}")
        instruction = (
            "Object is held at safe height. Use arm_execute_place to put it down." if actually_holding
            else "Gripper fully closed — NO object inside. TRUST this hardware measurement; "
                 "do NOT call arm_capture_image/arm_detect_vlm to verify. "
                 "Retry grasp or report miss to user.")
        return {
            "success": True, "steps": steps, "holding": actually_holding,
            "state": "holding" if actually_holding else "empty",
            "gripper_closed": is_closed, "pick_x": x, "pick_y": y,
            "instruction": instruction, "elapsed_s": round(_elapsed(), 1),
        }
    finally:
        if failed_step is None:
            bridge.set_holding(actually_holding)
        if failed_step and at_grasp_pose:
            _recover_to_safe(bridge, x, y, safe_h, quat)


def arm_execute_place(bridge: "RobotBridge", x: float, y: float, z: float,
                      quat: list[float] | None = None) -> dict:
    """
    Place the currently held object at (x, y, z). z = SURFACE height (e.g. desk top 0.0).
    Flange descends to z + grasp_depth so gripper tip stays safely above surface.
    Steps: 1) move to safe height  2) Cartesian descent  3) open gripper.
    On failure, auto-recovers to safe height.
    """
    if quat is None:
        quat = bridge.get_grasp_quat()
    node = bridge.node
    open_w, safe_h = bridge.get_gripper_open_width(), bridge.get_safe_height()
    place_offset = bridge.get_grasp_depth()
    steps, failed_step, at_descent = [], None, False

    if z < 0.0:
        return {"success": False, "step": "safety_check", "error": f"place z={z:.3f} below desk surface"}
    target_z = z + place_offset

    try:
        _log(f"PLACE 1/3: moving above (z={safe_h:.3f})")
        ok, msg = node.move_to_pose(x, y, safe_h, quat)
        if not ok:
            failed_step = "1/3 move_above"; return {"success": False, "step": failed_step, "error": msg}

        _log(f"PLACE 2/3: descending (flange z={target_z:.3f}, surface z={z:.3f})")
        ok, msg = node.move_cartesian(x, y, target_z, quat)
        if not ok:
            failed_step = "2/3 descent"; return {"success": False, "step": failed_step, "error": msg}
        at_descent = True

        _log("PLACE 3/3: opening gripper")
        ok, msg = node.control_gripper(open_w)
        if not ok:
            failed_step = "3/3 open"; return {"success": False, "step": failed_step, "error": msg}

        _log("PLACE complete")
        return {"success": True, "steps": steps, "state": "empty", "place_x": x, "place_y": y}
    finally:
        if failed_step is None:
            bridge.set_holding(False)
        if failed_step and at_descent:
            _recover_to_safe(bridge, x, y, safe_h, quat)