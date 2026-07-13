"""Joint-space demonstration gestures."""

from ..components import motion
from .base import SkillResult


def _run_alternating(bridge, poses, times, *, velocity, accel, skill_name):
    for index in range(times):
        result = motion.move_joints(
            bridge,
            poses[index % len(poses)],
            timeout=10.0,
            velocity=velocity,
            accel=accel,
        )
        if not result.ok:
            return SkillResult.failure(
                result.error or f"{skill_name} failed",
                failed_step=skill_name,
                retryable=False,
            )
    return None


def wave(bridge, times: int = 5, **kwargs) -> SkillResult:
    result = _run_alternating(
        bridge,
        ([0, -20, 0, 40, 0, 20, 10], [0, -20, 0, 40, 0, -20, 10]),
        times,
        velocity=0.3,
        accel=0.5,
        skill_name="wave",
    )
    return result or SkillResult.success(waves=times)


def nod(bridge, times: int = 4, **kwargs) -> SkillResult:
    result = _run_alternating(
        bridge,
        ([0, -20, 0, 80, 0, 0, 90], [0, -20, 0, 80, 0, 0, 70]),
        times,
        velocity=0.4,
        accel=0.4,
        skill_name="nod",
    )
    return result or SkillResult.success(nods=times)


def handshake(bridge, times: int = 4, **kwargs) -> SkillResult:
    start = motion.move_joints(
        bridge,
        [0, 40, 0, 70, 0, 0, 25],
        timeout=10.0,
        velocity=0.4,
        accel=0.4,
    )
    if not start.ok:
        return SkillResult.failure(
            start.error or "handshake failed",
            failed_step="handshake",
            retryable=False,
        )
    result = _run_alternating(
        bridge,
        ([0, 40, 0, 65, 0, 0, 25], [0, 40, 0, 75, 0, 0, 25]),
        times,
        velocity=0.4,
        accel=0.4,
        skill_name="handshake",
    )
    return result or SkillResult.success(shakes=times)
