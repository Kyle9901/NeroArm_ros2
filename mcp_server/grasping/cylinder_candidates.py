"""Deterministic grasp candidates for upright and lying cylinders."""

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


def generate_cylinder_grasp_candidates(
    cylinder_center_xyz: Sequence[float],
    cylinder_axis_xyz: Sequence[float],
    diameter_m: float,
    length_m: float,
    orientation_class: str,
    current_tcp_quat_xyzw: Sequence[float],
    *,
    local_desk_z: float,
    side_grasp_height_ratio: float = 0.45,
    pregrasp_distance: float = 0.08,
    retreat_distance: float = 0.08,
    tilt_angles_deg: Sequence[float] = (0.0, 60.0, 90.0),
) -> tuple[GraspCandidate, ...]:
    """Generate shape-prioritized cylinder grasps.

    Standing cylinders prefer four exact horizontal side grasps at 45% of
    their measured height, followed by 60-degree approaches and two top-grasp
    fallbacks. Lying cylinders prefer a top grasp across the diameter, then
    two 60-degree and two horizontal side approaches in the cross-section.
    """

    center = _as_finite_tuple(
        cylinder_center_xyz, length=3, name="cylinder_center_xyz"
    )
    axis = _normalize_vector(
        cylinder_axis_xyz, name="cylinder_axis_xyz"
    )
    current = normalize_quaternion_xyzw(current_tcp_quat_xyzw)
    diameter = float(diameter_m)
    length = float(length_m)
    if not math.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("diameter_m must be finite and positive")
    if not math.isfinite(length) or length <= diameter:
        raise ValueError("length_m must be finite and larger than diameter")
    if orientation_class not in {"upright", "lying"}:
        raise ValueError("orientation_class must be upright or lying")
    if not 0.25 <= float(side_grasp_height_ratio) <= 0.75:
        raise ValueError("side_grasp_height_ratio must be in [0.25, 0.75]")
    tilts = tuple(float(value) for value in tilt_angles_deg)
    if tilts != (0.0, 60.0, 90.0):
        raise ValueError("tilt_angles_deg must be exactly (0, 60, 90)")

    down = (0.0, 0.0, -1.0)
    candidates: list[GraspCandidate] = []

    def add(
        *,
        pose,
        approach,
        closing,
        tilt,
        source,
        preference,
    ):
        candidates.append(_candidate(
            center=pose,
            approach=approach,
            edge_axis=closing,
            width=diameter,
            tilt_deg=tilt,
            source=source,
            current_quaternion=current,
            pregrasp_distance=pregrasp_distance,
            retreat_distance=retreat_distance,
            preference_rank=preference,
            ranking_mode="shape_first",
            object_kind="cylinder",
        ))

    if orientation_class == "upright":
        grasp_pose = (
            center[0],
            center[1],
            float(local_desk_z) + length * float(side_grasp_height_ratio),
        )
        radial_directions = (
            ("pos_x", (1.0, 0.0, 0.0)),
            ("neg_x", (-1.0, 0.0, 0.0)),
            ("pos_y", (0.0, 1.0, 0.0)),
            ("neg_y", (0.0, -1.0, 0.0)),
        )
        # Exact horizontal side grasps are the intended primary strategy for
        # an upright bottle/can, regardless of current TCP rotation.
        for label, outward in radial_directions:
            approach = (-outward[0], -outward[1], 0.0)
            closing = (-outward[1], outward[0], 0.0)
            add(
                pose=grasp_pose,
                approach=approach,
                closing=closing,
                tilt=90.0,
                source=f"cylinder_upright_side_90_from_{label}",
                preference=0,
            )
        sine_60 = math.sin(math.radians(60.0))
        cosine_60 = math.cos(math.radians(60.0))
        for label, outward in radial_directions:
            approach = (
                -sine_60 * outward[0],
                -sine_60 * outward[1],
                -cosine_60,
            )
            closing = (-outward[1], outward[0], 0.0)
            add(
                pose=grasp_pose,
                approach=approach,
                closing=closing,
                tilt=60.0,
                source=f"cylinder_upright_tilt_60_from_{label}",
                preference=1,
            )
        top_pose = (
            center[0],
            center[1],
            float(local_desk_z) + length / 2.0,
        )
        for label, closing in (
            ("x", (1.0, 0.0, 0.0)),
            ("y", (0.0, 1.0, 0.0)),
        ):
            add(
                pose=top_pose,
                approach=down,
                closing=closing,
                tilt=0.0,
                source=f"cylinder_upright_top_close_{label}",
                preference=2,
            )
    else:
        horizontal_axis = _normalize_vector(
            (axis[0], axis[1], 0.0), name="lying_horizontal_axis"
        )
        cross_axis = (
            -horizontal_axis[1], horizontal_axis[0], 0.0,
        )
        add(
            pose=center,
            approach=down,
            closing=cross_axis,
            tilt=0.0,
            source="cylinder_lying_top",
            preference=0,
        )
        for tilt, preference in ((60.0, 1), (90.0, 2)):
            sine = math.sin(math.radians(tilt))
            cosine = math.cos(math.radians(tilt))
            for label, side in (("pos", 1.0), ("neg", -1.0)):
                approach = (
                    -sine * side * cross_axis[0],
                    -sine * side * cross_axis[1],
                    -cosine,
                )
                closing = (
                    cosine * side * cross_axis[0],
                    cosine * side * cross_axis[1],
                    -sine,
                )
                add(
                    pose=center,
                    approach=approach,
                    closing=closing,
                    tilt=tilt,
                    source=(
                        f"cylinder_lying_tilt_{int(tilt)}_from_"
                        f"{label}_cross"
                    ),
                    preference=preference,
                )
    return tuple(candidates)
