"""Visual servoing grasp skill."""

from typing import TYPE_CHECKING

from .base import GraspGeometry, SkillResult
from .perception import _target_color
from ..components import motion, perception, tracking

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge
    from ..vlm_client import VlmClient


def _detect_for_tracking(vlm: "VlmClient", frame, target: str) -> dict | None:
    color_name = _target_color(target)
    if color_name:
        detected = perception.detect_by_color(frame, color_name)
        if detected.ok and detected.data.get("found"):
            return detected.data
    detected = perception.detect_by_vlm(vlm, frame, target)
    if detected.ok and detected.data.get("found"):
        return detected.data
    return None


def visual_grasp(bridge: "RobotBridge", vlm: "VlmClient", target: str,
                 max_steps: int = 12, step_size: float = 0.04) -> SkillResult:
    geo = GraspGeometry.from_bridge(bridge)
    quat = bridge.get_grasp_quat()
    servo = tracking.create_tracker()
    steps = []

    result = motion.control_gripper(bridge, geo.gripper_open, duration=2.0)
    if not result.ok:
        return SkillResult.failure(result.error or "open gripper failed", failed_step="open_gripper", retryable=True)
    result = motion.go_home(bridge)
    if not result.ok:
        return SkillResult.failure(result.error or "go home failed", failed_step="go_home", retryable=True)

    frame_result = perception.capture_image(bridge)
    if not frame_result.ok:
        return SkillResult.failure(frame_result.error or "capture failed", failed_step="capture", retryable=True)
    frame = frame_result.data["frame"]
    detected = _detect_for_tracking(vlm, frame, target)
    if not detected:
        return SkillResult.failure(f"Object '{target}' not found", failed_step="detect", retryable=True)

    result = tracking.tracker_init(servo, frame, detected["bbox"])
    if not result.ok:
        return SkillResult.failure(result.error or "tracker init failed", failed_step="tracker_init", retryable=True)

    current_x = current_y = current_z = None
    for step in range(max_steps):
        frame_result = perception.capture_image(bridge)
        if not frame_result.ok:
            return SkillResult.failure(frame_result.error or "capture failed", failed_step="capture", retryable=True)
        frame = frame_result.data["frame"]

        tracked = tracking.tracker_update(servo, frame)
        if not tracked.data.get("active"):
            detected = _detect_for_tracking(vlm, frame, target)
            if detected and tracking.tracker_init(servo, frame, detected["bbox"]).ok:
                continue
            if current_x is None:
                return SkillResult.failure("Tracker lost, no fallback", failed_step="tracker_lost", retryable=True)
            target_z = max(current_z + geo.grasp_depth, 0.18)
            result = motion.move_cartesian(bridge, current_x, current_y, target_z, quat)
            if not result.ok:
                return SkillResult.failure(result.error or "blind descent failed", failed_step="blind_descent", retryable=True)
            steps.append("blind_descent")
            break

        bbox = tracked.data.get("bbox")
        if not bbox:
            continue
        xmin, ymin, xmax, ymax = bbox
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        pos = perception.pixel_to_3d(bridge, frame, cx, cy)
        if not pos.ok:
            continue
        current_x, current_y, current_z = pos.data["x"], pos.data["y"], pos.data["z"]

        flange_z = current_z + geo.approach_height - step * step_size
        if flange_z <= current_z + geo.grasp_depth + 0.02:
            target_z = current_z + geo.grasp_depth
            result = motion.move_cartesian(bridge, current_x, current_y, target_z, quat)
            if not result.ok:
                result = motion.move_to_pose(bridge, current_x, current_y, target_z, quat)
            if not result.ok:
                return SkillResult.failure(result.error or "final descent failed", failed_step="final_descent", retryable=True)
            steps.append("final_descent")
            break

        if step == 0:
            result = motion.move_to_pose(bridge, current_x, current_y, current_z + geo.approach_height, quat)
            if not result.ok:
                return SkillResult.failure(result.error or "approach failed", failed_step="approach", retryable=True)
            steps.append("approach")
        else:
            target_z = max(flange_z, current_z + geo.grasp_depth)
            result = motion.move_cartesian(bridge, current_x, current_y, target_z, quat)
            if not result.ok:
                result = motion.move_to_pose(bridge, current_x, current_y, target_z, quat)
            if result.ok:
                steps.append(f"descent to z={target_z:.3f}")

    if current_x is None or current_y is None:
        return SkillResult.failure("No valid 3D target found", failed_step="pixel_to_3d", retryable=True)

    result = motion.control_gripper(bridge, geo.gripper_close, duration=2.0)
    if not result.ok:
        return SkillResult.failure(result.error or "close gripper failed", failed_step="close_gripper", retryable=True)

    state = motion.read_joint_state(bridge)
    width = state.data.get("gripper_width") if state.ok else None
    holding = geo.is_holding(width)

    result = motion.move_cartesian(bridge, current_x, current_y, geo.safe_height, quat)
    if not result.ok:
        result = motion.move_to_pose(bridge, current_x, current_y, geo.safe_height, quat)
    if not result.ok:
        return SkillResult.failure(
            result.error or "lift failed",
            failed_step="lift",
            retryable=True,
            holding=holding,
            gripper_width=width,
        )

    bridge.set_holding(holding)
    return SkillResult.success(
        holding=holding,
        state="holding" if holding else "empty",
        gripper_width=width,
        steps=steps,
        final_position={"x": current_x, "y": current_y, "z": current_z},
    )
