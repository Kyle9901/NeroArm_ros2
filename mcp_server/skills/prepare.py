"""Prepare/bringup skill."""

from typing import TYPE_CHECKING

from .base import SkillResult
from ..components import infra

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge


def prepare(bridge: "RobotBridge", can_port: str = "can0",
            calib_name: str = "my_eih_calib_v6") -> SkillResult:
    status = infra.bringup_status(bridge)
    endpoints = status.data.get("endpoints", {}) if status.ok else {}
    if endpoints.get("move_action") and endpoints.get("camera_color") and endpoints.get("tf"):
        return SkillResult.success(already_ready=True, **status.data)

    result = infra.bringup_nodes(bridge, can_port=can_port, calib_name=calib_name)
    if not result.ok:
        return SkillResult.failure(
            result.error or "bringup failed",
            failed_step="bringup_nodes",
            retryable=True,
            **result.data,
        )
    return SkillResult.success(already_ready=False, **result.data)
