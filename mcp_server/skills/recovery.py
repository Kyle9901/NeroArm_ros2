"""Common skill recovery helpers."""

from typing import TYPE_CHECKING

from .base import GraspGeometry
from ..components import motion

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge


def recover_to_safe(bridge: "RobotBridge", x: float, y: float,
                    geo: GraspGeometry, quat: list[float]) -> bool:
    for fn in (
        lambda: motion.move_cartesian(bridge, x, y, geo.safe_height, quat),
        lambda: motion.move_to_pose(bridge, x, y, geo.safe_height, quat),
        lambda: motion.go_home(bridge),
    ):
        result = fn()
        if result.ok:
            return True
    motion.emergency_stop(bridge)
    return False
