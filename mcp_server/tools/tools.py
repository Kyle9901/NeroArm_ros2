"""
MCP tool implementations — pure functions that call RobotBridge or VlmClient.

Each tool function returns a JSON-serialisable dict.
"""

import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge
    from ..vlm_client import VlmClient


# ═══════════════════════════════════════════════════════════════════════════════════════════
#   Vision tools
# ═══════════════════════════════════════════════════════════════════════════════════════════

def arm_capture_image(bridge: "RobotBridge") -> dict:
    """Capture latest colour + depth image from eye-in-hand camera."""
    pair = bridge.node.get_latest_images(timeout=3.0)
    if pair is None:
        return {"success": False, "error": "No image received within 3s. "
                "Check: (1) camera is powered and publishing, "
                "(2) /camera/color/image_raw and /camera/depth/image_raw topics are active"}
    color, depth = pair
    import base64
    import cv2
    ok, buf = cv2.imencode(".jpg", color)
    if not ok:
        return {"success": False, "error": "JPEG encode failed — image may be corrupt"}
    b64 = base64.b64encode(buf).decode()
    return {
        "success": True,
        "image_base64": b64,
        "width": color.shape[1],
        "height": color.shape[0],
        "channels": 3,
        "depth_mean_mm": float(depth[depth > 0].mean()) if (depth > 0).any() else None,
    }


def arm_detect_vlm(bridge: "RobotBridge", vlm: "VlmClient", target: str) -> dict:
    """Detect an object in the current camera frame using VLM + OpenCV fallback."""
    pair = bridge.node.get_latest_images(timeout=3.0)
    if pair is None:
        return {"success": False, "error": "No image received within 3s — camera may not be running"}
    color, _depth = pair

    try:
        result = vlm.detect(color, target)
    except Exception as e:
        return {"success": False, "error": f"VLM detection failed: {e}"}

    if result is None or not result.get("found"):
        return {"success": True, "found": False,
                "message": f"Object '{target}' not found in image. "
                "This is a normal detection result, not a tool failure. "
                "Suggestions: (1) call arm_go_home first for an unobstructed view, "
                "(2) use a simpler description like 'blue block', "
                "(3) try arm_detect_color if it's a solid-colour block, "
                "(4) check if the object is in the camera's field of view",
                "retryable": False}

    return {"success": True, **result}


def arm_detect_color(bridge: "RobotBridge", color_name: str, location_hint: str = "") -> dict:
    """Detect a colour block using OpenCV HSV colour detection."""
    from ..vlm_client import detect_by_color

    pair = bridge.node.get_latest_images(timeout=3.0)
    if pair is None:
        return {"success": False, "error": "No image received — camera may not be running"}
    color, _depth = pair

    bbox = detect_by_color(color, color_name, location_hint)
    if bbox is None:
        supported = ", ".join(["blue", "red", "green", "yellow", "purple", "orange", "cyan"])
        return {"success": True, "found": False,
                "message": f"No {color_name} block found in current image. "
                "This is a normal detection result, not a tool failure. "
                f"Supported colours: {supported}. "
                "Do not retry other colours just to search all blocks — use arm_detect_blocks instead.",
                "retryable": False}

    xmin, ymin, xmax, ymax = bbox
    cx = int((xmin + xmax) / 2)
    cy = int((ymin + ymax) / 2)
    return {
        "success": True,
        "found": True,
        "color": color_name,
        "bbox": [xmin, ymin, xmax, ymax],
        "center_2d": [cx, cy],
    }


def arm_detect_blocks(bridge: "RobotBridge", location_hint: str = "") -> dict:
    """Detect all visible solid-colour blocks in one camera frame."""
    from ..vlm_client import detect_all_color_blocks

    pair = bridge.node.get_latest_images(timeout=3.0)
    if pair is None:
        return {"success": False, "error": "No image received — camera may not be running"}
    color, _depth = pair

    blocks = detect_all_color_blocks(color, location_hint)
    if not blocks:
        return {
            "success": True,
            "found": False,
            "count": 0,
            "blocks": [],
            "message": "No solid-colour blocks found in the current image. This is not a tool failure. Try arm_go_home for a clearer top-down view.",
            "retryable": False,
        }

    return {
        "success": True,
        "found": True,
        "count": len(blocks),
        "blocks": blocks,
        "message": f"Detected {len(blocks)} visible colour block(s)",
    }


def arm_get_3d_position(bridge: "RobotBridge", u: int, v: int) -> dict:
    """Convert 2D pixel (u,v) to 3D position in base_link frame."""
    pair = bridge.node.get_latest_images(timeout=3.0)
    if pair is None:
        return {"success": False, "error": "No image received — camera may not be running"}
    _color, depth = pair

    cam3d = bridge.node.compute_3d(u, v, depth)
    if cam3d is None:
        return {"success": False, "error": f"No valid depth data at pixel ({u},{v}). "
                "The point may be outside the image bounds or in a depth hole. "
                "Try adjusting the pixel coordinates slightly"}

    try:
        base = bridge.node.transform_to_base(cam3d["x_c"], cam3d["y_c"], cam3d["z_c"])
    except Exception as e:
        return {"success": False,
                "error": f"TF transform from camera to base_link failed: {e}. "
                "Check that TF2 is running and the camera_frame transform is published"}

    return {
        "success": True,
        "x": base["x"],
        "y": base["y"],
        "z": base["z"],
        "frame_id": "base_link",
        "depth_mm": cam3d["depth_mm"],
        "valid_depth_points": cam3d["valid_points"],
    }


# ═══════════════════════════════════════════════════════════════════════════════════════════
#   Motion tools
# ═══════════════════════════════════════════════════════════════════════════════════════════

def arm_get_status(bridge: "RobotBridge") -> dict:
    """Get current state: joint angles, gripper position, holding state, workspace bounds."""
    js = bridge.node.get_joint_state()
    return {
        "success": True,
        **js,
        "holding": bridge.get_holding(),
        "workspace": {
            "x_min": bridge.node._workspace_x_min,
            "x_max": bridge.node._workspace_x_max,
            "y_min": bridge.node._workspace_y_min,
            "y_max": bridge.node._workspace_y_max,
        },
        "safe_height": bridge.get_safe_height(),
        "desk_surface_z": bridge.node._get_param("desk_z_surface"),
    }


def arm_move_joints(bridge: "RobotBridge", joint_angles_deg: list[float], timeout: float = 20.0) -> dict:
    """Move arm to joint-space target.  joint_angles_deg: [j1, j2, j3, j4, j5, j6, j7] in degrees."""
    ok, msg = bridge.node.move_joints(joint_angles_deg, timeout)
    return {"success": ok, "message": msg}


def arm_move_to_pose(
    bridge: "RobotBridge",
    x: float, y: float, z: float,
    quat: list[float] | None = None,
    timeout: float = 20.0,
) -> dict:
    """Move TCP to Cartesian pose via MoveGroup."""
    ok, msg = bridge.node.move_to_pose(x, y, z, quat, timeout)
    return {"success": ok, "message": msg}


def arm_move_cartesian(
    bridge: "RobotBridge",
    x: float, y: float, z: float,
    quat: list[float] | None = None,
    timeout: float = 20.0,
) -> dict:
    """Straight-line Cartesian motion from current pose to target."""
    ok, msg = bridge.node.move_cartesian(x, y, z, quat, timeout)
    return {"success": ok, "message": msg}


def arm_control_gripper(bridge: "RobotBridge", width: float, duration: float = 1.5, timeout: float = 5.0) -> dict:
    """Open/close gripper.  width in metres: 0.10 = open, 0.02 = close."""
    ok, msg = bridge.node.control_gripper(width, duration, timeout)
    return {"success": ok, "message": msg}


def arm_go_home(bridge: "RobotBridge", timeout: float = 20.0) -> dict:
    """Move arm to home joint configuration."""
    ok, msg = bridge.node.go_home(timeout)
    return {"success": ok, "message": msg}


def arm_stop(bridge: "RobotBridge") -> dict:
    """Emergency stop — cancel all active motion goals immediately."""
    ok, msg = bridge.emergency_stop()
    return {"success": ok, "message": msg}


# ═══════════════════════════════════════════════════════════════════════════════════════════
#   Sequence tools
# ═══════════════════════════════════════════════════════════════════════════════════════════

def _recover_to_safe(bridge: "RobotBridge", x: float, y: float, safe_h: float, quat: list[float]) -> bool:
    """Best-effort recovery: lift to safe height. Returns True if successful."""
    node = bridge.node
    _log("RECOVERY: attempting to lift to safe height")
    # Try cartesian lift first (precise), fall back to joint-space planning
    ok, _ = node.move_cartesian(x, y, safe_h, quat)
    if ok:
        _log("RECOVERY: cartesian lift successful")
        return True
    ok, _ = node.move_to_pose(x, y, safe_h, quat)
    if ok:
        _log("RECOVERY: planned lift successful")
        return True
    # Last resort: go home
    _log("RECOVERY: trying go_home as last resort")
    ok, _ = node.go_home(timeout=20.0)
    if ok:
        return True
    # Nothing worked — emergency stop
    _log("RECOVERY: all attempts failed — EMERGENCY STOP")
    node.emergency_stop()
    return False


def _log(msg: str) -> None:
    """Write progress message to stderr so OpenClaw/LLM can see intermediate progress."""
    print(f"[robot-arm] {msg}", file=sys.stderr, flush=True)


def arm_execute_grasp(
    bridge: "RobotBridge",
    x: float, y: float, z: float,
    quat: list[float] | None = None,
) -> dict:
    """
    Pick up an object at (x, y, z) in base_link frame.

    Steps:
      1. Open gripper + move to approach pose (above target)
      2. Cartesian descent to grasp pose (SLOW, precise speed)
      3. Close gripper
      4. Check actual gripper width — if closed → no object picked up
      5. Lift to safe height

    On failure, attempts recovery to safe height then returns error.
    After success, the object is held at safe height.
    Use arm_execute_place() to put it down.
    """
    t0 = time.monotonic()

    if quat is None:
        quat = bridge.get_grasp_quat()

    approach_h = bridge.get_approach_height()
    grasp_d = bridge.get_grasp_depth()
    safe_h = bridge.get_safe_height()
    open_w = bridge.get_gripper_open_width()
    close_w = bridge.get_gripper_close_width()
    descent_vel = bridge.get_descent_velocity_scaling()
    descent_accel = bridge.get_descent_accel_scaling()

    steps = []
    node = bridge.node
    failed_step = None
    at_grasp_pose = False  # track whether we descended to the object
    actually_holding = False  # determined by gripper feedback after close

    # workspace check
    if not node.workspace_check(x, y):
        return {"success": False, "step": "workspace_check", "error": "Target outside workspace"}

    def _elapsed():
        return time.monotonic() - t0

    def _gripper_is_closed():
        """Check if the gripper is nearly fully closed (no object inside)."""
        js = node.get_joint_state()
        w = js.get("gripper_width")
        if w is None:
            return None  # can't determine
        # If width < 0.03m (3cm), the gripper nearly fully closed → no object
        return w < 0.03

    try:
        # [1] Open gripper first, then move to approach pose
        _log(f"GRASP 1/4: opening gripper + moving to approach pose (z={z + approach_h:.3f})")
        steps.append("1/4 open_gripper + approach")
        ok, msg = node.control_gripper(open_w, duration=2.0)
        if not ok:
            failed_step = "1/4 open_gripper"
            return {"success": False, "step": failed_step, "error": msg}
        ok, msg = node.move_to_pose(x, y, z + approach_h, quat)
        if not ok:
            failed_step = "1/4 approach"
            return {"success": False, "step": failed_step, "error": msg}

        # [2] SLOW Cartesian descent to grasp
        _log(f"GRASP 2/4: SLOW descent to grasp (z={z + grasp_d:.3f}, vel={descent_vel:.0%})")
        steps.append("2/4 cartesian descent")
        ok, msg = node.move_cartesian(x, y, z + grasp_d, quat,
                                       velocity_override=descent_vel,
                                       accel_override=descent_accel)
        if not ok:
            failed_step = "2/4 descent"
            return {"success": False, "step": failed_step, "error": msg}
        at_grasp_pose = True

        # [3] Close gripper
        _log("GRASP 3/4: closing gripper")
        steps.append("3/4 close gripper")
        ok, msg = node.control_gripper(close_w, duration=2.0)
        if not ok:
            failed_step = "3/4 close"
            return {"success": False, "step": failed_step, "error": msg}

        # [3.5] Check actual gripper width to determine if holding
        is_closed = _gripper_is_closed()
        if is_closed is True:
            _log("GRASP: gripper nearly closed — NO object picked up")
            actually_holding = False
        elif is_closed is False:
            _log("GRASP: gripper stays open — object is held")
            actually_holding = True
        else:
            # Can't determine — assume failure (conservative)
            _log("GRASP: cannot read gripper width, assuming not holding")
            actually_holding = False

        # [4] Lift to safe height (FAST)
        _log(f"GRASP 4/4: lifting to safe height (z={safe_h:.3f})")
        steps.append("4/4 lift")
        ok, msg = node.move_cartesian(x, y, safe_h, quat)
        if not ok:
            ok, msg = node.move_to_pose(x, y, safe_h, quat)
        if not ok:
            failed_step = "4/4 lift"
            return {"success": False, "step": failed_step, "error": msg}

        _log(f"GRASP complete in {_elapsed():.1f}s, holding={actually_holding}")
        if actually_holding:
            instruction = "Object is held at safe height. Use arm_execute_place to put it down."
        else:
            instruction = ("Gripper fully closed — NO object inside. "
                           "The gripper hardware measurement is authoritative: "
                           "do NOT call arm_capture_image or arm_detect_vlm to verify. "
                           "The camera cannot reliably detect whether the gripper is holding "
                           "an object because the gripper is now at safe height out of view. "
                           "Either retry the grasp (arm_execute_grasp) or report the miss to the user.")
        return {
            "success": True,
            "steps": steps,
            "state": "holding" if actually_holding else "empty",
            "holding": actually_holding,
            "gripper_closed": is_closed,
            "pick_x": x,
            "pick_y": y,
            "instruction": instruction,
            "elapsed_s": round(_elapsed(), 1),
        }

    finally:
        # Track holding state based on actual gripper feedback
        if failed_step is None:
            bridge.set_holding(actually_holding)
        # Recovery: if we failed at or after the descent (step 2+), we're near
        # the object — try to lift to safety regardless of gripper state.
        if failed_step is not None:
            if at_grasp_pose:
                # We're down near the object — must recover
                recovered = _recover_to_safe(bridge, x, y, safe_h, quat)
                if not recovered:
                    # Recovery failed — emergency stop already triggered
                    pass
            elif failed_step.startswith("1/4"):
                # Failed at approach — we're at home or wherever, gripper open.
                # Already safe, no recovery needed.
                pass


def arm_execute_place(
    bridge: "RobotBridge",
    x: float, y: float, z: float,
    quat: list[float] | None = None,
) -> dict:
    """
    Place the currently held object at (x, y, z) in base_link frame.

    Steps:
      1. Move to place pose above (safe height)
      2. Cartesian descent to place Z
      3. Open gripper

    On failure, attempts recovery to safe height.
    After success, the hand is empty at the place position.
    Does NOT go home — caller can chain another grasp or call arm_go_home().

    z is the surface height (e.g. desk top, z=0.0).
    The flange descends to z + grasp_depth (same as grasp) to keep gripper tip
    safely above the surface — internally adds the same 0.155m flange-offset
    used during grasping, so the same z works for both pick and place.
    """
    if quat is None:
        quat = bridge.get_grasp_quat()

    node = bridge.node
    open_w = bridge.get_gripper_open_width()
    safe_h = bridge.get_safe_height()
    place_offset = bridge.get_grasp_depth()   # same 0.155m flange→tip offset as grasp

    steps = []
    failed_step = None
    at_descent = False

    # Safety: place Z must leave room for flange (gripper tip = flange_z - 0.175)
    if z < 0.0:
        return {"success": False, "step": "safety_check", "error": f"place z={z:.3f} below desk surface"}

    target_z = z + place_offset   # flange target, gripper tip stays above surface

    try:
        # [1] Move to above place
        _log(f"PLACE 1/3: moving to above (z={safe_h:.3f})")
        steps.append("1/3 move above")
        ok, msg = node.move_to_pose(x, y, safe_h, quat)
        if not ok:
            failed_step = "1/3 move_above"
            return {"success": False, "step": failed_step, "error": msg}

        # [2] Cartesian descent to place flange height
        _log(f"PLACE 2/3: descending to place (flange z={target_z:.3f}, surface z={z:.3f})")
        steps.append("2/3 descent")
        ok, msg = node.move_cartesian(x, y, target_z, quat)
        if not ok:
            failed_step = "2/3 descent"
            return {"success": False, "step": failed_step, "error": msg}
        at_descent = True

        # [3] Open gripper
        _log("PLACE 3/3: opening gripper")
        steps.append("3/3 open")
        ok, msg = node.control_gripper(open_w)
        if not ok:
            failed_step = "3/3 open"
            return {"success": False, "step": failed_step, "error": msg}

        _log("PLACE complete")
        return {
            "success": True,
            "steps": steps,
            "state": "empty",
            "place_x": x,
            "place_y": y,
        }

    finally:
        # Track holding state
        if failed_step is None:
            bridge.set_holding(False)
        if failed_step is not None and at_descent:
            # We lowered to place position — lift to safe height
            _recover_to_safe(bridge, x, y, safe_h, quat)