"""
Visual servoing with CSRT tracker + VLM initialization.
VLM detection (slow, ~1-2s) → CSRT tracking (fast, 30Hz) → iterative look-and-move grasp.
"""

import time
import cv2
import numpy as np


class VisualServo:
    """CSRT tracker + VLM init for iterative visual grasp."""

    def __init__(self):
        self.tracker = None
        self.last_bbox = None
        self.fail_count = 0
        self.max_fail = 5

    def init_from_vlm(self, color_img: np.ndarray, detected: dict) -> bool:
        """Initialize CSRT tracker from VLM detection result. Returns True if ok."""
        bbox = detected.get("bbox")
        if not bbox or len(bbox) != 4:
            return False
        xmin, ymin, xmax, ymax = bbox
        w, h = xmax - xmin, ymax - ymin
        if w < 10 or h < 10:
            return False
        self.last_bbox = (xmin, ymin, w, h)
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(color_img, self.last_bbox)
        self.fail_count = 0
        return True

    def update(self, color_img: np.ndarray) -> tuple[bool, tuple | None]:
        """Track with CSRT. Returns (success, bbox_xyxy)."""
        if self.tracker is None:
            return False, None
        ok, bbox_xywh = self.tracker.update(color_img)
        if not ok:
            self.fail_count += 1
            if self.fail_count >= self.max_fail:
                self.tracker = None
                return False, None
            return True, self._to_xyxy(self.last_bbox)
        self.fail_count = 0
        self.last_bbox = bbox_xywh
        return True, self._to_xyxy(bbox_xywh)

    def is_active(self) -> bool:
        return self.tracker is not None

    def _to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        return (int(x), int(y), int(x + w), int(y + h))

    def reset(self):
        self.tracker = None
        self.last_bbox = None
        self.fail_count = 0


def visual_grasp(bridge, vlm, target: str, max_steps: int = 12, step_size: float = 0.04) -> dict:
    """
    Iterative look-and-move grasp with CSRT tracking.
    1. VLM detects → init CSRT tracker
    2. Loop: CSRT track → 3D → move small step → repeat
    3. Close enough → grasp
    On tracker loss: VLM re-detection or blind descent fallback.
    """
    from .tools.tools import _log, _get_image

    node = bridge.node
    servo = VisualServo()
    grasp_d = bridge.get_grasp_depth()
    approach_h = bridge.get_approach_height()
    safe_h = bridge.get_safe_height()
    open_w = bridge.get_gripper_open_width()
    close_w = bridge.get_gripper_close_width()
    quat = bridge.get_grasp_quat()
    steps = []

    # ── Step 0: open + home ──
    _log("VISUAL_GRASP: open gripper + go home")
    ok, msg = node.control_gripper(open_w, duration=2.0)
    if not ok:
        return {"success": False, "step": "open_gripper", "error": msg}
    ok, msg = node.go_home(timeout=20.0)
    if not ok:
        return {"success": False, "step": "go_home", "error": msg}

    # ── Step 1: init detection (HSV first → VLM fallback) ──
    _log(f"VISUAL_GRASP: detecting '{target}'")
    pair, err = _get_image(bridge, timeout=3.0)
    if err:
        return {"success": False, "step": "capture", "error": err["error"]}
    color_img = pair[0]

    detected = None
    source = None

    # 1a: Try HSV color detection first (fast, ~ms)
    from .vlm_client import detect_by_color, _COLOR_HSV_RANGES
    target_lower = target.lower()
    for color_name in _COLOR_HSV_RANGES:
        if color_name in target_lower or target_lower in color_name:
            bbox = detect_by_color(color_img, color_name)
            if bbox is not None:
                detected = {"found": True, "bbox": list(bbox), "color": color_name, "source": "HSV"}
                source = "HSV"
                _log(f"VISUAL_GRASP: HSV found '{color_name}' at {bbox}")
                break

    # 1b: Fallback to VLM (slow, ~1-2s)
    if detected is None:
        _log("VISUAL_GRASP: HSV not found, falling back to VLM")
        try:
            detected = vlm.detect(color_img, target)
            source = "VLM"
        except Exception as e:
            return {"success": False, "step": "vlm_detect", "error": str(e)}

    if detected is None or not detected.get("found"):
        return {"success": False, "step": "detect",
                "error": f"Object '{target}' not found. Try arm_go_home for a clearer view."}

    if not servo.init_from_vlm(color_img, detected):
        return {"success": False, "step": "init_tracker",
                "error": "CSRT init failed — bbox too small or invalid"}

    _log(f"VISUAL_GRASP: CSRT tracker initialized ({source}), bbox={servo.last_bbox}")

    # ── Step 2: iterative track + move ──
    current_x = current_y = current_z = None
    for step in range(max_steps):
        # After any arm movement, re-capture and re-track before computing 3D.
        # This is critical: the camera has moved, so the old bbox is stale.
        pair, err = _get_image(bridge, timeout=3.0)
        if err:
            _log("VISUAL_GRASP: camera error")
            break
        color_img, depth_img = pair

        # CSRT tracking on the NEW frame (post-movement camera view)
        ok, bbox_xyxy = servo.update(color_img)
        if not ok:
            _log("VISUAL_GRASP: tracker lost, re-detecting")
            # Try HSV first (fast), then VLM (slow)
            detected = None
            for color_name in _COLOR_HSV_RANGES:
                if color_name in target_lower or target_lower in color_name:
                    bbox = detect_by_color(color_img, color_name)
                    if bbox is not None:
                        detected = {"found": True, "bbox": list(bbox), "color": color_name}
                        break
            if detected is None:
                try:
                    detected = vlm.detect(color_img, target)
                except Exception:
                    detected = None
            if detected and detected.get("found"):
                if servo.init_from_vlm(color_img, detected):
                    _log("VISUAL_GRASP: CSRT re-initialized")
                    continue
            if current_x is not None:
                _log("VISUAL_GRASP: blind descent to last known position")
                target_z = max(current_z + grasp_d, 0.18)
                ok, msg = node.move_cartesian(current_x, current_y, target_z, quat)
                if not ok:
                    return {"success": False, "step": "blind_descent", "error": msg}
                break
            return {"success": False, "step": "tracker_lost", "error": "Tracker lost, no fallback"}

        xmin, ymin, xmax, ymax = bbox_xyxy
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        cam3d = node.compute_3d(cx, cy, depth_img)
        if cam3d is None:
            _log(f"VISUAL_GRASP: step {step+1}: no depth at ({cx},{cy})")
            continue

        try:
            base = node.transform_to_base(cam3d["x_c"], cam3d["y_c"], cam3d["z_c"])
        except Exception:
            _log(f"VISUAL_GRASP: step {step+1}: TF error")
            continue

        current_x, current_y, current_z = base["x"], base["y"], base["z"]
        _log(f"VISUAL_GRASP: step {step+1}: tracked 3D=({current_x:.3f}, {current_y:.3f}, {current_z:.3f})")

        # Check if close enough to grasp (flange already near object surface + grasp_depth)
        flange_z = current_z + approach_h - step * step_size
        if flange_z <= current_z + grasp_d + 0.02:
            target_z = current_z + grasp_d
            _log(f"VISUAL_GRASP: close enough, final descent to grasp z={target_z:.3f}")
            ok, msg = node.move_cartesian(current_x, current_y, target_z, quat)
            if not ok:
                ok, msg = node.move_to_pose(current_x, current_y, target_z, quat)
            if not ok:
                return {"success": False, "step": "final_descent", "error": msg}
            steps.append("final descent")
            break

        if step == 0:
            # First iteration: move to approach height above the object
            flange_z = current_z + approach_h
            _log(f"VISUAL_GRASP: step 1: approach to flange_z={flange_z:.3f} (surface_z={current_z:.3f} + {approach_h:.3f})")
            ok, msg = node.move_to_pose(current_x, current_y, flange_z, quat)
            if not ok:
                return {"success": False, "step": "approach", "error": msg}
            steps.append("approach")
        else:
            # Subsequent iterations: descend by step_size toward the object
            target_z = max(flange_z, current_z + grasp_d)
            _log(f"VISUAL_GRASP: step {step+1}: descend to flange_z={target_z:.3f}")
            ok, msg = node.move_cartesian(current_x, current_y, target_z, quat)
            if not ok:
                ok, msg = node.move_to_pose(current_x, current_y, target_z, quat)
            if not ok:
                continue
            steps.append(f"descent to z={target_z:.3f}")

    # ── Step 3: close + lift ──
    _log("VISUAL_GRASP: closing gripper")
    ok, msg = node.control_gripper(close_w, duration=2.0)
    if not ok:
        return {"success": False, "step": "close_gripper", "error": msg}

    js = node.get_joint_state()
    w = js.get("gripper_width")
    # holding=true if gripper did NOT fully close (object inside prevents full closure)
    # Compare against close_w + margin (0.01m = 1cm) to account for thin objects
    actually_holding = w is not None and w > close_w + 0.01

    _log("VISUAL_GRASP: lifting to safe height")
    ok, msg = node.move_cartesian(current_x or 0, current_y or 0, safe_h, quat)
    if not ok:
        ok, msg = node.move_to_pose(current_x or 0, current_y or 0, safe_h, quat)

    return {
        "success": True, "steps": steps, "holding": actually_holding,
        "state": "holding" if actually_holding else "empty",
        "final_position": {"x": current_x, "y": current_y, "z": current_z},
    }