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
                 object_id: str = "target_object",
                 object_size: tuple[float, float, float] = (0.06, 0.06, 0.08)) -> SkillResult:
    geo = GraspGeometry.from_bridge(bridge)
    quat = quat or bridge.get_grasp_quat()

    check = motion.workspace_check(bridge, x, y)
    if not check.ok:
        return _fail(check.error or "Target outside workspace", "workspace_check", retryable=False)

    print(f"[grasp] target: x={x:.4f}, y={y:.4f}, z={z:.4f}, "
          f"approach_z={z + geo.approach_height:.4f}, grasp_z={z + geo.grasp_depth:.4f}", flush=True)

    # 清理上一次残留的 CollisionObject 和 ACM
    bridge.node.remove_target_collision(object_id)

    # 注册目标物体 → CollisionObject 会"吃掉" Octomap 体素，ACM 幽灵化
    # 宽度: 检测宽度每边 +1cm, 高度: 表面向上 0.5cm + 向下 10cm = 10.5cm
    box_w = object_size[0] + 0.03  # 每边 +1.5cm
    box_h = 0.13                    # 0.5cm 上 + 10cm 下 + 2.5cm 额外
    bridge.node.add_target_collision(
        x, y, z - object_size[2] / 2.0,  # 中心在物体几何中心
        object_id=object_id, size=(box_w, box_w, box_h))

    result = motion.control_gripper(bridge, geo.gripper_open, duration=2.0)
    if not result.ok:
        return _fail(result.error or "open gripper failed", "open_gripper")

    result = motion.move_to_pose(bridge, x, y, z + geo.approach_height, quat)
    if not result.ok:
        return _fail(result.error or "approach failed", "approach")

    result = motion.move_cartesian(
        bridge, x, y, z + geo.grasp_depth, quat,
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
    bridge.node.remove_target_collision(object_id)
    return SkillResult.success(
        holding=holding,
        state="holding" if holding else "empty",
        gripper_width=width,
        pick_x=x,
        pick_y=y,
        pick_z=z,
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

    result = motion.move_cartesian(bridge, x, y, z + geo.grasp_depth, quat)
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
