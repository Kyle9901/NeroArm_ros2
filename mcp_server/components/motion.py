"""Atomic motion components."""

from typing import TYPE_CHECKING

from .base import ComponentResult

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge


_UNKNOWN_HOLDING = None


def _stop_guard(bridge: "RobotBridge") -> ComponentResult | None:
    checker = getattr(bridge, "is_task_stop_requested", None)
    if callable(checker) and checker():
        return ComponentResult.failure(
            "Task stop requested; no new robot command was submitted",
            stop_requested=True,
            motion_state_unknown=False,
        )
    return None


def _execution_state_unknown(message: str | None) -> bool:
    """Whether an execution failure leaves the physical arm state unknown."""
    text = (message or "").lower()
    return any(fragment in text for fragment in (
        "may still be moving",
        "may still be executing",
        "result did not arrive in time",
        "action server did not respond",
    ))


def _execution_result(ok: bool, message: str) -> ComponentResult:
    if ok:
        return ComponentResult.success(message=message)
    return ComponentResult.failure(
        message,
        motion_state_unknown=_execution_state_unknown(message),
    )


def move_to_pose(bridge: "RobotBridge", x: float, y: float, z: float,
                 quat: list[float] | None = None, timeout: float = 60.0,
                 velocity: float | None = None, accel: float | None = None) -> ComponentResult:
    if stopped := _stop_guard(bridge):
        return stopped
    ok, msg = bridge.node.move_to_pose(
        x, y, z, quat, timeout,
        velocity_override=velocity, accel_override=accel)
    return _execution_result(ok, msg)


def plan_to_pose(bridge: "RobotBridge", x: float, y: float, z: float,
                 quat: list[float] | None = None, timeout: float = 3.0,
                 start_state=None, velocity: float | None = None,
                 accel: float | None = None) -> ComponentResult:
    """Plan a pose goal without sending it to the trajectory executor."""
    if stopped := _stop_guard(bridge):
        return stopped
    ok, msg, plan = bridge.node.plan_to_pose(
        x, y, z, quat, timeout,
        start_state=start_state,
        velocity_override=velocity,
        accel_override=accel,
    )
    if not ok or plan is None:
        return ComponentResult.failure(msg)
    return ComponentResult.success(
        message=msg,
        plan=plan,
        terminal_joints=plan.terminal_joints,
        planning_time=plan.planning_time,
    )


def move_cartesian(bridge: "RobotBridge", x: float, y: float, z: float,
                   quat: list[float] | None = None, timeout: float = 30.0,
                   velocity: float | None = None, accel: float | None = None) -> ComponentResult:
    if stopped := _stop_guard(bridge):
        return stopped
    ok, msg = bridge.node.move_cartesian(
        x, y, z, quat, timeout,
        velocity_override=velocity, accel_override=accel)
    return _execution_result(ok, msg)


def plan_cartesian(bridge: "RobotBridge", x: float, y: float, z: float,
                   quat: list[float] | None = None, timeout: float = 3.0,
                   start_state=None, velocity: float | None = None,
                   accel: float | None = None,
                   minimum_fraction: float | None = None) -> ComponentResult:
    """Plan a collision-aware Cartesian segment without executing it."""
    if stopped := _stop_guard(bridge):
        return stopped
    ok, msg, plan = bridge.node.plan_cartesian(
        x, y, z, quat, timeout,
        start_state=start_state,
        velocity_override=velocity,
        accel_override=accel,
        minimum_fraction=minimum_fraction,
    )
    if not ok or plan is None:
        return ComponentResult.failure(msg)
    return ComponentResult.success(
        message=msg,
        plan=plan,
        terminal_joints=plan.terminal_joints,
        fraction=plan.fraction,
    )


def solve_pose_ik(bridge: "RobotBridge", x: float, y: float, z: float,
                  quat: list[float] | None = None, timeout: float = 0.25,
                  seed_state=None, avoid_collisions: bool = True) -> ComponentResult:
    """Run MoveIt's IK and planning-scene collision check without motion."""
    if stopped := _stop_guard(bridge):
        return stopped
    ok, msg, joints = bridge.node.solve_pose_ik(
        x, y, z, quat, timeout,
        seed_state=seed_state,
        avoid_collisions=avoid_collisions,
    )
    if not ok or joints is None:
        return ComponentResult.failure(msg, checked=True)
    return ComponentResult.success(
        message=msg,
        checked=True,
        joints=joints,
    )


def plan_joints(bridge: "RobotBridge", joint_angles_deg: list[float],
                timeout: float = 3.0, start_state=None,
                velocity: float | None = None,
                accel: float | None = None) -> ComponentResult:
    """Plan a joint target without executing it."""
    if stopped := _stop_guard(bridge):
        return stopped
    ok, msg, plan = bridge.node.plan_joints(
        joint_angles_deg,
        timeout,
        start_state=start_state,
        velocity_override=velocity,
        accel_override=accel,
    )
    if not ok or plan is None:
        return ComponentResult.failure(msg)
    return ComponentResult.success(
        message=msg,
        plan=plan,
        terminal_joints=plan.terminal_joints,
        planning_time=plan.planning_time,
    )


def execute_planned(bridge: "RobotBridge", plan,
                    timeout: float = 20.0) -> ComponentResult:
    """Explicitly execute one previously selected plan."""
    if stopped := _stop_guard(bridge):
        return stopped
    ok, msg = bridge.node.execute_planned(plan, timeout)
    return _execution_result(ok, msg)


def move_joints(bridge: "RobotBridge", joint_angles_deg: list[float], timeout: float = 20.0,
                velocity: float | None = None, accel: float | None = None) -> ComponentResult:
    if stopped := _stop_guard(bridge):
        return stopped
    ok, msg = bridge.node.move_joints(
        joint_angles_deg, timeout,
        velocity_override=velocity, accel_override=accel)
    return _execution_result(ok, msg)


def control_gripper(bridge: "RobotBridge", width: float, duration: float = 1.5,
                    timeout: float = 5.0) -> ComponentResult:
    if stopped := _stop_guard(bridge):
        return stopped
    ok, msg = bridge.node.control_gripper(width, duration, timeout)
    return _execution_result(ok, msg)


def go_home(bridge: "RobotBridge", timeout: float = 60.0) -> ComponentResult:
    if stopped := _stop_guard(bridge):
        return stopped
    ok, msg = bridge.node.go_home(timeout)
    return _execution_result(ok, msg)


def go_carry(bridge: "RobotBridge", timeout: float = 60.0) -> ComponentResult:
    if stopped := _stop_guard(bridge):
        return stopped
    ok, msg = bridge.node.go_carry(timeout)
    return _execution_result(ok, msg)


def emergency_stop(bridge: "RobotBridge") -> ComponentResult:
    ok, msg = bridge.emergency_stop()
    # Recovery reached an indeterminate stop condition.  This helper does not
    # cancel an accepted controller trajectory, so the holding flag is unsafe.
    bridge.set_holding(_UNKNOWN_HOLDING)
    return ComponentResult.success(message=msg) if ok else ComponentResult.failure(msg, fatal=True)


def read_joint_state(bridge: "RobotBridge", *, newer_than: int | None = None,
                     timeout: float = 1.0) -> ComponentResult:
    if newer_than is None:
        state = bridge.node.get_joint_state()
    else:
        state = bridge.node.wait_for_joint_state_after(newer_than, timeout)
        if state is None:
            return ComponentResult.failure(
                f"No fresh joint feedback within {timeout:.2f}s",
                feedback_fresh=False,
            )
    return ComponentResult.success(**state)


def workspace_check(bridge: "RobotBridge", x: float, y: float) -> ComponentResult:
    ok = bridge.node.workspace_check(x, y)
    if ok:
        return ComponentResult.success(in_workspace=True)
    return ComponentResult.failure("Target outside workspace", fatal=True, in_workspace=False)
