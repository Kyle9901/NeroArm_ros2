"""Atomic tracking components."""

from .base import ComponentResult, ImageFrame
from ..visual_servo import VisualServo


def create_tracker() -> VisualServo:
    return VisualServo()


def tracker_init(tracker: VisualServo, frame: ImageFrame, bbox: list[int]) -> ComponentResult:
    ok = tracker.init_from_vlm(frame.color, {"bbox": bbox})
    if ok:
        return ComponentResult.success(active=True)
    return ComponentResult.failure("CSRT init failed — bbox too small or invalid")


def tracker_update(tracker: VisualServo, frame: ImageFrame) -> ComponentResult:
    ok, bbox = tracker.update(frame.color)
    if not ok:
        return ComponentResult.success(active=False, lost=True)
    return ComponentResult.success(active=True, lost=False, bbox=list(bbox) if bbox else None)
