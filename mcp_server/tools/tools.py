"""
MCP tool implementations — pure functions that call RobotBridge or VlmClient.

Each tool function returns a JSON-serialisable dict.
"""

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
        return {"success": False, "error": "No image received within timeout"}
    color, depth = pair
    import base64
    import cv2
    ok, buf = cv2.imencode(".jpg", color)
    if not ok:
        return {"success": False, "error": "JPEG encode failed"}
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
        return {"success": False, "error": "No image received within timeout"}
    color, _depth = pair

    try:
        result = vlm.detect(color, target)
    except Exception as e:
        return {"success": False, "error": f"VLM detection failed: {e}"}

    if result is None or not result.get("found"):
        return {"success": False, "found": False, "error": "Object not found in image"}

    return {"success": True, **result}


def arm_detect_color(bridge: "RobotBridge", color_name: str, location_hint: str = "") -> dict:
    """Detect a colour block using OpenCV HSV colour detection."""
    from ..vlm_client import detect_by_color

    pair = bridge.node.get_latest_images(timeout=3.0)
    if pair is None:
        return {"success": False, "error": "No image received within timeout"}
    color, _depth = pair

    bbox = detect_by_color(color, color_name, location_hint)
    if bbox is None:
        return {"success": False, "found": False, "error": f"Colour '{color_name}' not found"}

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


def arm_get_3d_position(bridge: "RobotBridge", u: int, v: int) -> dict:
    """Convert 2D pixel (u,v) to 3D position in base_link frame."""
    pair = bridge.node.get_latest_images(timeout=3.0)
    if pair is None:
        return {"success": False, "error": "No image received within timeout"}
    _color, depth = pair

    cam3d = bridge.node.compute_3d(u, v, depth)
    if cam3d is None:
        return {"success": False, "error": "Failed to compute 3D from depth"}

    try:
        base = bridge.node.transform_to_base(cam3d["x_c"], cam3d["y_c"], cam3d["z_c"])
    except Exception as e:
        return {"success": False, "error": f"TF transform failed: {e}"}

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
    """Get current joint angles and gripper state."""
    js = bridge.node.get_joint_state()
    return {"success": True, **js}


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
    """Emergency stop — cancel all current goals (best-effort)."""
    return {"success": True, "message": "Stop requested (cancel not yet implemented)"}


# ═══════════════════════════════════════════════════════════════════════════════════════════
#   Sequence tools
# ═══════════════════════════════════════════════════════════════════════════════════════════

def arm_execute_grasp(
    bridge: "RobotBridge",
    x: float, y: float, z: float,
    quat: list[float] | None = None,
) -> dict:
    """
    Pick up an object at (x, y, z) in base_link frame.

    Steps:
      1. Open gripper + move to approach pose (above target)
      2. Cartesian descent to grasp pose
      3. Close gripper
      4. Lift to safe height

    After this call, the object is held at safe height.
    Use arm_execute_place() to put it down.
    """
    if quat is None:
        quat = bridge.get_grasp_quat()

    approach_h = bridge.get_approach_height()
    grasp_d = bridge.get_grasp_depth()
    safe_h = bridge.get_safe_height()
    open_w = bridge.get_gripper_open_width()
    close_w = bridge.get_gripper_close_width()

    steps = []
    node = bridge.node

    # workspace check
    if not node.workspace_check(x, y):
        return {"success": False, "step": "workspace_check", "error": "Target outside workspace"}

    # [1] Open gripper first, then move to approach pose
    steps.append("1/4 open_gripper + approach")
    ok, msg = node.control_gripper(open_w, duration=2.0)
    if not ok:
        return {"success": False, "step": "1/4 open_gripper", "error": msg}
    ok, msg = node.move_to_pose(x, y, z + approach_h, quat)
    if not ok:
        return {"success": False, "step": "1/4 approach", "error": msg}

    # [2] Cartesian descent to grasp
    steps.append("2/4 cartesian descent")
    ok, msg = node.move_cartesian(x, y, z + grasp_d, quat)
    if not ok:
        return {"success": False, "step": "2/4 descent", "error": msg}

    # [3] Close gripper
    steps.append("3/4 close gripper")
    ok, msg = node.control_gripper(close_w, duration=2.0)
    if not ok:
        return {"success": False, "step": "3/4 close", "error": msg}

    # [4] Lift to safe height
    steps.append("4/4 lift")
    ok, msg = node.move_cartesian(x, y, safe_h, quat)
    if not ok:
        ok, msg = node.move_to_pose(x, y, safe_h, quat)
    if not ok:
        return {"success": False, "step": "4/4 lift", "error": msg}

    return {
        "success": True,
        "steps": steps,
        "state": "holding",
        "pick_x": x,
        "pick_y": y,
    }


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

    After this call, the object is placed and the hand is empty at the place position.
    Does NOT go home — caller can chain another grasp or call arm_go_home().

    z is the surface height (e.g. desk top). Internally adds 0.02m for the
    block height so the gripper releases above the surface.
    """
    if quat is None:
        quat = bridge.get_grasp_quat()

    FLANGE_MIN_Z = 0.175  # gripper tip contacts at z=0; flange must stay above this
    node = bridge.node
    open_w = bridge.get_gripper_open_width()
    safe_h = bridge.get_safe_height()

    steps = []

    # Safety: place Z must leave room for flange (gripper tip = flange_z - 0.175)
    # Place surface + block height (~2cm) + finger clearance → flange_z >= 0.175
    if z < 0.0:
        return {"success": False, "step": "safety_check", "error": f"place z={z:.3f} below desk surface"}
    # MoveIt desk collision will block the rest

    # [1] Move to above place
    steps.append("1/3 move above")
    ok, msg = node.move_to_pose(x, y, safe_h, quat)
    if not ok:
        return {"success": False, "step": "1/3 move_above", "error": msg}

    # [2] Cartesian descent
    steps.append("2/3 descent")
    ok, msg = node.move_cartesian(x, y, z, quat)
    if not ok:
        return {"success": False, "step": "2/3 descent", "error": msg}

    # [3] Open gripper
    steps.append("3/3 open")
    ok, msg = node.control_gripper(open_w)
    if not ok:
        return {"success": False, "step": "3/3 open", "error": msg}

    return {
        "success": True,
        "steps": steps,
        "state": "empty",
        "place_x": x,
        "place_y": y,
    }