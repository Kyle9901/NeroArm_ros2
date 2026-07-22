"""Generate deterministic top and tilted grasp candidates for a box.

This module intentionally has no ROS or MoveIt dependency.  It describes the
geometry only; IK, joint-limit, collision and path checks belong to the planner.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from mcp_server.models.grasp import GraspCandidate, Quaternion, Vector3


_EPSILON = 1e-12
def _as_finite_tuple(
    values: Sequence[float], *, length: int, name: str
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _scale(vector: Vector3, factor: float) -> Vector3:
    return tuple(component * factor for component in vector)


def _normalize_vector(vector: Sequence[float], *, name: str) -> Vector3:
    values = _as_finite_tuple(vector, length=3, name=name)
    norm = math.sqrt(_dot(values, values))
    if norm <= _EPSILON:
        raise ValueError(f"{name} norm is too small")
    return tuple(value / norm for value in values)


def normalize_quaternion_xyzw(quaternion: Sequence[float]) -> Quaternion:
    """Return a unit quaternion, using a deterministic sign representation."""

    values = _as_finite_tuple(
        quaternion, length=4, name="quaternion_xyzw"
    )
    norm = math.sqrt(_dot(values, values))
    if norm <= _EPSILON:
        raise ValueError("quaternion_xyzw norm is too small")
    normalized = tuple(value / norm for value in values)

    # q and -q represent the same orientation.  Prefer w >= 0; for a 180
    # degree rotation, use the first significant xyz component as the tie-break.
    sign_component = normalized[3]
    if abs(sign_component) <= _EPSILON:
        sign_component = next(
            (value for value in normalized[:3] if abs(value) > _EPSILON),
            1.0,
        )
    if sign_component < 0.0:
        normalized = tuple(-value for value in normalized)
    return normalized


def quaternion_angular_distance(
    left_xyzw: Sequence[float], right_xyzw: Sequence[float]
) -> float:
    """Shortest SO(3) rotation angle between two quaternions, in radians."""

    left = normalize_quaternion_xyzw(left_xyzw)
    right = normalize_quaternion_xyzw(right_xyzw)
    cosine_half_angle = min(1.0, max(0.0, abs(_dot(left, right))))
    return 2.0 * math.acos(cosine_half_angle)


def sort_by_tcp_rotation(
    candidates: Iterable[GraspCandidate],
    current_tcp_quat_xyzw: Sequence[float],
) -> tuple[GraspCandidate, ...]:
    """Sort candidates by increasing TCP rotation from the current pose.

    Python's stable sort preserves generator order when distances are equal.
    This function only ranks geometry; feasibility filters must run before a
    candidate is executed.
    """

    current = normalize_quaternion_xyzw(current_tcp_quat_xyzw)
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                quaternion_angular_distance(
                    current, candidate.pose_quat_xyzw
                ),
                candidate.source,
            ),
        )
    )


def _matrix_to_quaternion_xyzw(
    x_axis: Vector3, y_axis: Vector3, z_axis: Vector3
) -> Quaternion:
    """Convert an orthonormal rotation matrix (given by columns) to xyzw."""

    m00, m10, m20 = x_axis
    m01, m11, m21 = y_axis
    m02, m12, m22 = z_axis
    trace = m00 + m11 + m22

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m21 - m12) / scale
        y = (m02 - m20) / scale
        z = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / scale
        x = 0.25 * scale
        y = (m01 + m10) / scale
        z = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / scale
        x = (m01 + m10) / scale
        y = 0.25 * scale
        z = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / scale
        x = (m02 + m20) / scale
        y = (m12 + m21) / scale
        z = 0.25 * scale
    return normalize_quaternion_xyzw((x, y, z, w))


def _orientation_for_axes(
    approach_vector: Vector3,
    unsigned_edge_axis: Vector3,
    current_tcp_quat_xyzw: Quaternion,
) -> tuple[Quaternion, Vector3]:
    """Choose the physically equivalent +/- closing axis nearest current TCP."""

    approach = _normalize_vector(approach_vector, name="approach_vector")
    edge = _normalize_vector(unsigned_edge_axis, name="edge_axis")
    if abs(_dot(approach, edge)) > 1e-8:
        raise ValueError("edge_axis must be perpendicular to approach_vector")

    options: list[tuple[float, Quaternion, Vector3]] = []
    for signed_edge in (edge, _scale(edge, -1.0)):
        x_axis = _normalize_vector(
            _cross(signed_edge, approach), name="tcp_x_axis"
        )
        quaternion = _matrix_to_quaternion_xyzw(
            x_axis, signed_edge, approach
        )
        distance = quaternion_angular_distance(
            current_tcp_quat_xyzw, quaternion
        )
        options.append((distance, quaternion, signed_edge))
    _, quaternion, signed_edge = min(options, key=lambda option: option[0])
    return quaternion, signed_edge


def _candidate(
    *,
    center: Vector3,
    approach: Vector3,
    edge_axis: Vector3,
    width: float,
    tilt_deg: float,
    source: str,
    current_quaternion: Quaternion,
    pregrasp_distance: float,
    retreat_distance: float,
    preference_rank: int = 0,
    ranking_mode: str = "rotation_first",
    object_kind: str = "block",
) -> GraspCandidate:
    quaternion, signed_edge = _orientation_for_axes(
        approach, edge_axis, current_quaternion
    )
    angle = quaternion_angular_distance(current_quaternion, quaternion)
    return GraspCandidate(
        pose_xyz=center,
        pose_quat_xyzw=quaternion,
        approach_vector=_normalize_vector(approach, name="approach_vector"),
        pregrasp_distance=pregrasp_distance,
        retreat_vector=_scale(
            _normalize_vector(approach, name="approach_vector"), -1.0
        ),
        retreat_distance=retreat_distance,
        gripper_width=width,
        tilt_deg=tilt_deg,
        edge_axis=signed_edge,
        score=1.0 - angle / math.pi,
        source=source,
        preference_rank=preference_rank,
        ranking_mode=ranking_mode,
        object_kind=object_kind,
    )


def generate_block_grasp_candidates(
    block_center_xyz: Sequence[float],
    block_size_xyz: Sequence[float],
    block_yaw_rad: float,
    current_tcp_quat_xyzw: Sequence[float],
    *,
    pregrasp_distance: float = 0.08,
    retreat_distance: float = 0.05,
    tilt_angles_deg: Sequence[float] = (0.0, 30.0, 60.0),
) -> tuple[GraspCandidate, ...]:
    """Generate two top, four 30-degree and four 60-degree box grasps.

    ``block_yaw_rad`` describes the block X edge in the planning-frame XY
    plane.  Every candidate's TCP origin is the supplied block center.  Tilt is
    measured from vertical: 0 degrees is straight down and 90 degrees would be
    a horizontal side grasp (which this generator intentionally omits).

    The returned candidates are ordered by the TCP orientation change from the
    current pose.  ``preference_rank`` separately preserves the physical grasp
    priority requested by the operator: 90, 60, then 30 degrees relative to
    the table (0, 30, then 60 degrees away from vertical).  Each candidate
    first chooses the closer of the equivalent ``+Y`` and ``-Y`` parallel-jaw
    orientations.
    """

    center = _as_finite_tuple(
        block_center_xyz, length=3, name="block_center_xyz"
    )
    size = _as_finite_tuple(block_size_xyz, length=3, name="block_size_xyz")
    if any(dimension <= 0.0 for dimension in size):
        raise ValueError("block_size_xyz dimensions must be positive")
    if not math.isfinite(block_yaw_rad):
        raise ValueError("block_yaw_rad must be finite")
    if not math.isfinite(pregrasp_distance) or pregrasp_distance <= 0.0:
        raise ValueError("pregrasp_distance must be finite and positive")
    if not math.isfinite(retreat_distance) or retreat_distance <= 0.0:
        raise ValueError("retreat_distance must be finite and positive")
    tilts = tuple(float(value) for value in tilt_angles_deg)
    if (
        tilts != (0.0, 30.0, 60.0)
        or not all(math.isfinite(value) for value in tilts)
    ):
        raise ValueError("tilt_angles_deg must be exactly (0, 30, 60)")

    current = normalize_quaternion_xyzw(current_tcp_quat_xyzw)
    cosine = math.cos(block_yaw_rad)
    sine = math.sin(block_yaw_rad)
    block_x: Vector3 = (cosine, sine, 0.0)
    block_y: Vector3 = (-sine, cosine, 0.0)
    down: Vector3 = (0.0, 0.0, -1.0)
    candidates: list[GraspCandidate] = []

    # Two top grasps: close along either pair of parallel block faces.
    candidates.append(
        _candidate(
            center=center,
            approach=down,
            edge_axis=block_x,
            width=size[0],
            tilt_deg=0.0,
            source="block_top_close_x",
            current_quaternion=current,
            pregrasp_distance=pregrasp_distance,
            retreat_distance=retreat_distance,
            preference_rank=0,
        )
    )
    candidates.append(
        _candidate(
            center=center,
            approach=down,
            edge_axis=block_y,
            width=size[1],
            tilt_deg=0.0,
            source="block_top_close_y",
            current_quaternion=current,
            pregrasp_distance=pregrasp_distance,
            retreat_distance=retreat_distance,
            preference_rank=0,
        )
    )

    # Four approach sides at each tilt.  Approaching normal to one block edge
    # uses the other edge as the parallel-jaw closing axis.
    sides = (
        ("pos_x", block_x, block_y, size[1]),
        ("neg_x", _scale(block_x, -1.0), block_y, size[1]),
        ("pos_y", block_y, block_x, size[0]),
        ("neg_y", _scale(block_y, -1.0), block_x, size[0]),
    )
    for tilt_deg in tilts[1:]:
        tilt_rad = math.radians(tilt_deg)
        horizontal_scale = math.sin(tilt_rad)
        vertical = -math.cos(tilt_rad)
        for side_name, outward, closing_axis, width in sides:
            approach = (
                -horizontal_scale * outward[0],
                -horizontal_scale * outward[1],
                vertical,
            )
            candidates.append(
                _candidate(
                    center=center,
                    approach=approach,
                    edge_axis=closing_axis,
                    width=width,
                    tilt_deg=tilt_deg,
                    source=f"block_tilt_{int(tilt_deg)}_from_{side_name}",
                    current_quaternion=current,
                    pregrasp_distance=pregrasp_distance,
                    retreat_distance=retreat_distance,
                    preference_rank=int(tilt_deg // 30.0),
                )
            )

    return sort_by_tcp_rotation(candidates, current)
