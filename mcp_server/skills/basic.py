"""Basic single-purpose robot skills."""

from ..components import motion
from .base import SkillResult


def go_home(bridge, **kwargs) -> SkillResult:
    if bridge.is_at_home():
        return SkillResult.success(already_home=True)
    result = motion.go_home(bridge)
    if result.ok:
        return SkillResult.success(already_home=False)
    return SkillResult.failure(
        result.error or "go_home failed",
        failed_step="go_home",
        retryable=True,
    )


def open_gripper(bridge, **kwargs) -> SkillResult:
    result = motion.control_gripper(bridge, 0.10, duration=1.5)
    if result.ok:
        return SkillResult.success()
    return SkillResult.failure(
        result.error or "open gripper failed",
        failed_step="open_gripper",
        retryable=True,
    )


def close_gripper(bridge, **kwargs) -> SkillResult:
    result = motion.control_gripper(bridge, 0.02, duration=1.5)
    if result.ok:
        return SkillResult.success()
    return SkillResult.failure(
        result.error or "close gripper failed",
        failed_step="close_gripper",
        retryable=True,
    )
