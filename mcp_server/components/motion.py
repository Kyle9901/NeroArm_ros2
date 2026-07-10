"""Atomic motion components."""

from typing import TYPE_CHECKING

from .base import ComponentResult

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge


_UNKNOWN_HOLDING = None


def move_to_pose(bridge: "RobotBridge", x: float, y: float, z: float,
                 quat: list[float] | None = None, timeout: float = 60.0,
                 velocity: float | None = None, accel: float | None = None) -> ComponentResult:
    ok, msg = bridge.node.move_to_pose(
        x, y, z, quat, timeout,
        velocity_override=velocity, accel_override=accel)
    return ComponentResult.success(message=msg) if ok else ComponentResult.failure(msg)


def move_cartesian(bridge: "RobotBridge", x: float, y: float, z: float,
                   quat: list[float] | None = None, timeout: float = 30.0,
                   velocity: float | None = None, accel: float | None = None) -> ComponentResult:
    ok, msg = bridge.node.move_cartesian(
        x, y, z, quat, timeout,
        velocity_override=velocity, accel_override=accel)
    return ComponentResult.success(message=msg) if ok else ComponentResult.failure(msg)


def move_joints(bridge: "RobotBridge", joint_angles_deg: list[float], timeout: float = 20.0,
                velocity: float | None = None, accel: float | None = None) -> ComponentResult:
    ok, msg = bridge.node.move_joints(
        joint_angles_deg, timeout,
        velocity_override=velocity, accel_override=accel)
    return ComponentResult.success(message=msg) if ok else ComponentResult.failure(msg)


def control_gripper(bridge: "RobotBridge", width: float, duration: float = 1.5,
                    timeout: float = 5.0) -> ComponentResult:
    ok, msg = bridge.node.control_gripper(width, duration, timeout)
    return ComponentResult.success(message=msg) if ok else ComponentResult.failure(msg)


def go_home(bridge: "RobotBridge", timeout: float = 60.0) -> ComponentResult:
    ok, msg = bridge.node.go_home(timeout)
    return ComponentResult.success(message=msg) if ok else ComponentResult.failure(msg)


def emergency_stop(bridge: "RobotBridge") -> ComponentResult:
    ok, msg = bridge.emergency_stop()
    # After an emergency stop, the software holding flag should not be trusted.
    bridge.set_holding(_UNKNOWN_HOLDING)
    return ComponentResult.success(message=msg) if ok else ComponentResult.failure(msg, fatal=True)


def read_joint_state(bridge: "RobotBridge") -> ComponentResult:
    state = bridge.node.get_joint_state()
    return ComponentResult.success(**state)


def workspace_check(bridge: "RobotBridge", x: float, y: float) -> ComponentResult:
    ok = bridge.node.workspace_check(x, y)
    if ok:
        return ComponentResult.success(in_workspace=True)
    return ComponentResult.failure("Target outside workspace", fatal=True, in_workspace=False)
