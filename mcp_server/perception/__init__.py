"""Detector-independent perception algorithms and visualization helpers."""

from .color_detector import detect_all_color_blocks, detect_by_color
from .debug import draw_bboxes
from .transparent_bottle import (
    TransparentBottleAnalysis,
    TransparentBottleConsensus,
    aggregate_transparent_bottle_measurements,
    analyze_transparent_bottle_points,
    classify_pose_from_height_percentiles,
)

__all__ = [
    "detect_all_color_blocks",
    "detect_by_color",
    "draw_bboxes",
    "TransparentBottleAnalysis",
    "TransparentBottleConsensus",
    "aggregate_transparent_bottle_measurements",
    "analyze_transparent_bottle_points",
    "classify_pose_from_height_percentiles",
]
