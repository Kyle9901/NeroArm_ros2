"""Shared types and geometry helpers for skills."""

from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

from ..models import OperationResult

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge


@dataclass(frozen=True)
class GraspGeometry:
    fingertip_depth: float
    approach_height: float
    safe_height: float
    gripper_open: float
    gripper_close: float
    hold_margin: float
    descent_vel: float
    descent_accel: float

    @classmethod
    def from_bridge(cls, bridge: "RobotBridge", hold_margin: float = 0.005) -> "GraspGeometry":
        return cls(
            fingertip_depth=bridge.get_fingertip_depth(),
            approach_height=bridge.get_approach_height(),
            safe_height=bridge.get_safe_height(),
            gripper_open=bridge.get_gripper_open_width(),
            gripper_close=bridge.get_gripper_close_width(),
            hold_margin=hold_margin,
            descent_vel=bridge.get_descent_velocity_scaling(),
            descent_accel=bridge.get_descent_accel_scaling(),
        )

    def is_holding(self, width: float | None) -> bool:
        return width is not None and width > self.gripper_close + self.hold_margin

    def grasp_tcp_z(self, surface_z: float) -> float:
        """Top-down TCP Z after entering below the detected object surface."""
        return surface_z - self.fingertip_depth

    def with_fingertip_depth(self, depth: float) -> "GraspGeometry":
        return replace(self, fingertip_depth=float(depth))


@dataclass
class SkillResult(OperationResult):
    """Operation result extended with task recovery and holding state."""

    holding: bool | None = None
    recovered: bool = False
    retryable: bool = False
    failed_step: str | None = None

    @classmethod
    def success(cls, *, holding: bool | None = None, **data: Any) -> "SkillResult":
        return cls(ok=True, holding=holding, retryable=False, data=data)

    @classmethod
    def failure(cls, error: str, *, failed_step: str | None = None,
                recovered: bool = False, retryable: bool = True,
                holding: bool | None = None, **data: Any) -> "SkillResult":
        return cls(
            ok=False,
            holding=holding,
            recovered=recovered,
            retryable=retryable,
            failed_step=failed_step,
            data=data,
            error=error,
        )
