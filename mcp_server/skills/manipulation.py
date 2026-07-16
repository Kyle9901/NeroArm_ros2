"""Pick and place skills."""

import time
from typing import TYPE_CHECKING

from .base import GraspGeometry, SkillResult
from .recovery import recover_to_safe
from ..components import motion

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge


def _fail(error: str, step: str, *, recovered: bool = False,
          retryable: bool = True) -> SkillResult:
    return SkillResult.failure(
        error,
        failed_step=step,
        recovered=recovered,
        retryable=retryable,
    )


def grasp_object(bridge: "RobotBridge", x: float, y: float, z: float,
                 quat: list[float] | None = None,
                 geometry: dict | None = None) -> SkillResult:
    geo = GraspGeometry.from_bridge(bridge)
    quat = quat or bridge.get_grasp_quat()

    check = motion.workspace_check(bridge, x, y)
    if not check.ok:
        return _fail(check.error or "Target outside workspace", "workspace_check", retryable=False)

    print(f"[grasp] target: x={x:.4f}, y={y:.4f}, z={z:.4f}, "
          f"approach_z={z + geo.approach_height:.4f}, grasp_z={geo.grasp_z(z):.4f} "
          f"(fingertip_z={geo.fingertip_z(z):.4f}, "
          f"fingertip_depth={geo.fingertip_depth:.3f})", flush=True)

    # Safety: clamp fingertip to desk surface (don't go below the desk)
    desk_z = (
        float(geometry["local_desk_z"])
        if geometry and geometry.get("local_desk_z") is not None
        else bridge.get_desk_surface_z()
    )
    fingertip_z = geo.fingertip_z(z)
    if fingertip_z < desk_z:
        clamped_depth = z - desk_z
        print(f"[grasp] WARNING: fingertip {fingertip_z:.4f} below desk {desk_z:.4f}, "
              f"clamping fingertip_depth {geo.fingertip_depth:.3f}→{clamped_depth:.3f}",
              flush=True)
        geo = geo.with_fingertip_depth(clamped_depth)

    result = motion.control_gripper(bridge, geo.gripper_open, duration=2.0)
    if not result.ok:
        return _fail(result.error or "open gripper failed", "open_gripper")

    result = motion.move_to_pose(bridge, x, y, z + geo.approach_height, quat)
    if not result.ok:
        return _fail(result.error or "approach failed", "approach")

    result = motion.move_cartesian(
        bridge, x, y, geo.grasp_z(z), quat,
        velocity=geo.descent_vel, accel=geo.descent_accel)
    if not result.ok:
        recovered = recover_to_safe(bridge, x, y, geo, quat)
        return _fail(result.error or "descent failed", "descent", recovered=recovered)

    result = motion.control_gripper(bridge, geo.gripper_close, duration=2.0)
    if not result.ok:
        recovered = recover_to_safe(bridge, x, y, geo, quat)
        return _fail(result.error or "close gripper failed", "close_gripper", recovered=recovered)

    time.sleep(0.5)
    state = motion.read_joint_state(bridge)
    width = state.data.get("gripper_width") if state.ok else None
    holding = geo.is_holding(width)

    result = motion.move_cartesian(bridge, x, y, geo.safe_height, quat)
    if not result.ok:
        result = motion.move_to_pose(bridge, x, y, geo.safe_height, quat)
    if not result.ok:
        recovered = recover_to_safe(bridge, x, y, geo, quat)
        return SkillResult.failure(
            result.error or "lift failed",
            failed_step="lift",
            recovered=recovered,
            retryable=True,
            holding=holding,
            gripper_width=width,
        )

    bridge.set_holding(holding)
    return SkillResult.success(
        holding=holding,
        state="holding" if holding else "empty",
        gripper_width=width,
        pick_x=x,
        pick_y=y,
        pick_z=z,
        geometry=geometry,
    )


def place_object(bridge: "RobotBridge", x: float, y: float, z: float,
                 quat: list[float] | None = None) -> SkillResult:
    geo = GraspGeometry.from_bridge(bridge)
    quat = quat or bridge.get_grasp_quat()

    if z < 0.0:
        return _fail(f"place z={z:.3f} below desk surface", "safety_check", retryable=False)

    result = motion.workspace_check(bridge, x, y)
    if not result.ok:
        return _fail(result.error or "Target outside workspace", "workspace_check", retryable=False)

    result = motion.move_to_pose(bridge, x, y, geo.safe_height, quat)
    if not result.ok:
        return _fail(result.error or "move above failed", "move_above")

    result = motion.move_cartesian(bridge, x, y, geo.grasp_z(z), quat)
    if not result.ok:
        recovered = recover_to_safe(bridge, x, y, geo, quat)
        return _fail(result.error or "descent failed", "descent", recovered=recovered)

    result = motion.control_gripper(bridge, geo.gripper_open)
    if not result.ok:
        recovered = recover_to_safe(bridge, x, y, geo, quat)
        return _fail(result.error or "open gripper failed", "open_gripper", recovered=recovered)

    bridge.set_holding(False)
    return SkillResult.success(
        holding=False,
        state="empty",
        place_x=x,
        place_y=y,
        place_z=z,
    )
