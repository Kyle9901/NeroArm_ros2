"""Detector-independent perception algorithms and visualization helpers."""

from .color_detector import detect_all_color_blocks, detect_by_color
from .debug import draw_bboxes

__all__ = ["detect_all_color_blocks", "detect_by_color", "draw_bboxes"]
