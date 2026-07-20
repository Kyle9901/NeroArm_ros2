"""Common skill recovery helpers."""

from typing import TYPE_CHECKING

from .base import GraspGeometry
from ..components import motion

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge


def recover_to_safe(bridge: "RobotBridge", x: float, y: float,
                    geo: GraspGeometry, quat: list[float]) -> bool:
    stop_requested = getattr(bridge, "is_task_stop_requested", None)
    if callable(stop_requested) and stop_requested():
        return False
    final_recovery = (
        (lambda: motion.go_home(bridge))
        if bridge.get_holding() is False
        else (lambda: motion.go_carry(bridge))
    )
    for fn in (
        lambda: motion.move_cartesian(bridge, x, y, geo.safe_height, quat),
        lambda: motion.move_to_pose(bridge, x, y, geo.safe_height, quat),
        final_recovery,
    ):
        result = fn()
        if result.ok:
            return True
        if (
            result.data.get("motion_state_unknown")
            or result.data.get("stop_requested")
        ):
            # A previous goal may still be active. Never stack another motion
            # command on top of an execution whose physical state is unknown.
            return False
    motion.emergency_stop(bridge)
    return False
