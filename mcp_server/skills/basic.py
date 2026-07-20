"""Basic single-purpose robot skills."""

from ..components import motion
from .base import SkillResult


def go_home(bridge, **kwargs) -> SkillResult:
    holding = bridge.get_holding()
    if holding is True:
        return SkillResult.failure(
            "Cannot move to the observation pose while holding an object; "
            "use the carry pose or place the object first",
            failed_step="holding_guard",
            retryable=False,
            holding=True,
        )
    if holding is None:
        return SkillResult.failure(
            "Cannot move to the observation pose while holding state is unknown",
            failed_step="holding_guard",
            retryable=False,
            holding=None,
        )
    if bridge.is_at_home():
        return SkillResult.success(already_home=True)
    result = motion.go_home(bridge)
    if result.ok:
        return SkillResult.success(already_home=False)
    return SkillResult.failure(
        result.error or "go_home failed",
        failed_step="go_home",
        retryable=not bool(result.data.get("motion_state_unknown")),
        motion_state_unknown=bool(result.data.get("motion_state_unknown")),
    )


def open_gripper(bridge, **kwargs) -> SkillResult:
    result = motion.control_gripper(bridge, 0.10, duration=1.5)
    if result.ok:
        bridge.set_holding(False)
        return SkillResult.success(holding=False)
    bridge.set_holding(None)
    return SkillResult.failure(
        result.error or "open gripper failed",
        failed_step="open_gripper",
        retryable=False,
        holding=None,
    )


def close_gripper(bridge, **kwargs) -> SkillResult:
    result = motion.control_gripper(bridge, 0.02, duration=1.5)
    # A standalone close command has no object-aware width verification.
    # Conservatively block future grasp/open assumptions until the user
    # explicitly releases or another skill establishes holding state.
    bridge.set_holding(None)
    if result.ok:
        return SkillResult.success(holding=None)
    return SkillResult.failure(
        result.error or "close gripper failed",
        failed_step="close_gripper",
        retryable=False,
        holding=None,
    )
