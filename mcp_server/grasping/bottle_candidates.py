"""Deterministic grasps for the configured transparent water bottle."""

from __future__ import annotations

import math
from typing import Sequence

from .block_candidates import (
    _as_finite_tuple,
    _candidate,
    _normalize_vector,
    normalize_quaternion_xyzw,
)
from ..models import GraspCandidate


def generate_transparent_bottle_grasp_candidates(
    grasp_center_xyz: Sequence[float],
    bottle_axis_xyz: Sequence[float],
    diameter_m: float,
    orientation_class: str,
    current_tcp_quat_xyzw: Sequence[float],
    *,
    pregrasp_distance: float = 0.08,
    retreat_distance: float = 0.08,
    upright_axis_overtravel_m: float = 0.0,
) -> tuple[GraspCandidate, ...]:
    """Generate only the two bottle strategies confirmed by the operator.

    Upright bottles expose eight horizontal approaches.  Their TCP endpoint is
    advanced past the measured bottle axis along the selected approach by the
    configured overtravel so the finger bodies, rather than only their tips,
    engage the bottle. Lying bottles expose one vertical approach, with the
    gripper closing across the bottle diameter. No tilted or fallback grasp is
    emitted.
    """

    center = _as_finite_tuple(
        grasp_center_xyz, length=3, name="grasp_center_xyz",
    )
    axis = _normalize_vector(
        bottle_axis_xyz, name="bottle_axis_xyz",
    )
    current = normalize_quaternion_xyzw(current_tcp_quat_xyzw)
    diameter = float(diameter_m)
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("diameter_m must be finite and positive")
    overtravel = float(upright_axis_overtravel_m)
    if not math.isfinite(overtravel) or not 0.0 <= overtravel < diameter * 0.5:
        raise ValueError(
            "upright_axis_overtravel_m must be finite, non-negative, and "
            "smaller than the bottle radius"
        )
    if orientation_class not in {"upright", "lying"}:
        raise ValueError("orientation_class must be upright or lying")

    def candidate(approach, closing, tilt, source):
        return _candidate(
            center=center,
            approach=approach,
            edge_axis=closing,
            width=diameter,
            tilt_deg=tilt,
            source=source,
            current_quaternion=current,
            pregrasp_distance=pregrasp_distance,
            retreat_distance=retreat_distance,
            preference_rank=0,
            ranking_mode="rotation_first",
            # Reverse-place handling already treats this as a cylindrical
            # object; the candidate source preserves the specific strategy.
            object_kind="cylinder",
        )

    if orientation_class == "upright":
        candidates = []
        for label, outward in (
            ("pos_x", (1.0, 0.0, 0.0)),
            ("pos_x_pos_y", (math.sqrt(0.5), math.sqrt(0.5), 0.0)),
            ("pos_y", (0.0, 1.0, 0.0)),
            ("neg_x_pos_y", (-math.sqrt(0.5), math.sqrt(0.5), 0.0)),
            ("neg_x", (-1.0, 0.0, 0.0)),
            ("neg_x_neg_y", (-math.sqrt(0.5), -math.sqrt(0.5), 0.0)),
            ("neg_y", (0.0, -1.0, 0.0)),
            ("pos_x_neg_y", (math.sqrt(0.5), -math.sqrt(0.5), 0.0)),
        ):
            approach = (-outward[0], -outward[1], 0.0)
            closing = (-outward[1], outward[0], 0.0)
            endpoint = tuple(
                center[index] + approach[index] * overtravel
                for index in range(3)
            )
            candidates.append(_candidate(
                center=endpoint,
                approach=approach,
                edge_axis=closing,
                width=diameter,
                tilt_deg=90.0,
                source=f"transparent_bottle_upright_side_from_{label}",
                current_quaternion=current,
                pregrasp_distance=pregrasp_distance,
                retreat_distance=retreat_distance,
                preference_rank=0,
                ranking_mode="rotation_first",
                object_kind="cylinder",
            ))
        return tuple(candidates)

    horizontal_axis = _normalize_vector(
        (axis[0], axis[1], 0.0),
        name="lying_bottle_horizontal_axis",
    )
    cross_axis = (
        -horizontal_axis[1],
        horizontal_axis[0],
        0.0,
    )
    return (candidate(
        (0.0, 0.0, -1.0),
        cross_axis,
        0.0,
        "transparent_bottle_lying_vertical_top",
    ),)
