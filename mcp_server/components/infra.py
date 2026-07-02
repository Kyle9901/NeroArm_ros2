"""Atomic bringup/status components."""

from typing import TYPE_CHECKING

from .base import ComponentResult

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge


def bringup_nodes(bridge: "RobotBridge", can_port: str = "can0",
                  calib_name: str = "my_eih_calib_v6") -> ComponentResult:
    result = bridge.bringup_nodes(can_port=can_port, calib_name=calib_name)
    if result.get("success") is False:
        return ComponentResult.failure(result.get("hint") or "Bringup failed", **result)
    return ComponentResult.success(**result)


def bringup_status(bridge: "RobotBridge") -> ComponentResult:
    return ComponentResult.success(**bridge.bringup_status())
