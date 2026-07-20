import math
from collections import Counter

import pytest

from mcp_server.models import GraspCandidate
from mcp_server.grasping.block_candidates import (
    generate_block_grasp_candidates,
    quaternion_angular_distance,
    sort_by_tcp_rotation,
)


def _rotate_vector(quaternion, vector):
    x, y, z, w = quaternion
    q_vector = (x, y, z)

    def cross(left, right):
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    uv = cross(q_vector, vector)
    uuv = cross(q_vector, uv)
    return tuple(
        component + 2.0 * (w * first + second)
        for component, first, second in zip(vector, uv, uuv)
    )


def _norm(vector):
    return math.sqrt(sum(component * component for component in vector))


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


@pytest.fixture
def candidates():
    return generate_block_grasp_candidates(
        block_center_xyz=(-0.35, -0.10, 0.04),
        block_size_xyz=(0.04, 0.06, 0.05),
        block_yaw_rad=math.radians(20.0),
        current_tcp_quat_xyzw=(0.0, 1.0, 0.0, 0.0),
    )


def test_generates_requested_top_and_tilt_groups(candidates):
    assert len(candidates) == 10
    assert Counter(candidate.tilt_deg for candidate in candidates) == {
        0.0: 2,
        30.0: 4,
        60.0: 4,
    }
    assert len({candidate.source for candidate in candidates}) == 10


def test_tcp_axes_match_approach_and_parallel_jaw_edge(candidates):
    for candidate in candidates:
        tcp_z = _rotate_vector(candidate.pose_quat_xyzw, (0.0, 0.0, 1.0))
        tcp_y = _rotate_vector(candidate.pose_quat_xyzw, (0.0, 1.0, 0.0))

        assert _norm(candidate.pose_quat_xyzw) == pytest.approx(1.0)
        assert tcp_z == pytest.approx(candidate.approach_vector)
        assert tcp_y == pytest.approx(candidate.edge_axis)
        assert _dot(candidate.approach_vector, candidate.edge_axis) == pytest.approx(
            0.0, abs=1e-9
        )
        assert candidate.retreat_vector == pytest.approx(
            tuple(-value for value in candidate.approach_vector)
        )


def test_tilt_is_measured_from_vertical_and_has_four_approach_sides(candidates):
    for tilt_deg in (0.0, 30.0, 60.0):
        group = [
            candidate
            for candidate in candidates
            if candidate.tilt_deg == tilt_deg
        ]
        for candidate in group:
            assert candidate.approach_vector[2] == pytest.approx(
                -math.cos(math.radians(tilt_deg))
            )
            assert math.hypot(*candidate.approach_vector[:2]) == pytest.approx(
                math.sin(math.radians(tilt_deg))
            )

        if tilt_deg:
            horizontal_directions = {
                tuple(round(value, 8) for value in candidate.approach_vector[:2])
                for candidate in group
            }
            assert len(horizontal_directions) == 4


def test_width_tracks_the_block_dimension_along_tcp_closing_axis(candidates):
    widths_by_source = {
        candidate.source: candidate.gripper_width for candidate in candidates
    }
    assert widths_by_source["block_top_close_x"] == pytest.approx(0.04)
    assert widths_by_source["block_top_close_y"] == pytest.approx(0.06)
    assert widths_by_source["block_tilt_30_from_pos_x"] == pytest.approx(0.06)
    assert widths_by_source["block_tilt_30_from_pos_y"] == pytest.approx(0.04)


def test_pregrasp_and_retreat_positions_follow_the_declared_vectors(candidates):
    candidate = candidates[0]
    expected_pregrasp = tuple(
        position - direction * candidate.pregrasp_distance
        for position, direction in zip(
            candidate.pose_xyz, candidate.approach_vector
        )
    )
    expected_retreat = tuple(
        position + direction * candidate.retreat_distance
        for position, direction in zip(
            candidate.pose_xyz, candidate.retreat_vector
        )
    )
    assert candidate.pregrasp_xyz == pytest.approx(expected_pregrasp)
    assert candidate.retreat_xyz == pytest.approx(expected_retreat)
    assert candidate.candidate_id == candidate.source
    assert candidate.position_xyz == candidate.pose_xyz
    assert candidate.quat_xyzw == candidate.pose_quat_xyzw
    assert candidate.to_dict()["candidate_id"] == candidate.source
    assert GraspCandidate.from_dict(candidate.to_dict()) == candidate


def test_output_and_explicit_sort_use_shortest_tcp_rotation(candidates):
    current = (0.0, 1.0, 0.0, 0.0)
    distances = [
        quaternion_angular_distance(current, candidate.pose_quat_xyzw)
        for candidate in candidates
    ]
    assert distances == sorted(distances)
    assert [candidate.score for candidate in candidates] == pytest.approx(
        [1.0 - distance / math.pi for distance in distances]
    )

    reversed_candidates = tuple(reversed(candidates))
    assert sort_by_tcp_rotation(reversed_candidates, current) == candidates


def test_quaternion_distance_treats_opposite_signs_as_same_rotation():
    quaternion = (0.2, -0.3, 0.4, 0.5)
    opposite = tuple(-value for value in quaternion)
    assert quaternion_angular_distance(quaternion, opposite) == pytest.approx(0.0)


def test_equivalent_closing_axis_sign_is_chosen_nearest_current_pose():
    first = generate_block_grasp_candidates(
        (0.0, 0.0, 0.0),
        (0.05, 0.05, 0.05),
        0.0,
        (0.0, 1.0, 0.0, 0.0),
    )
    seed = next(candidate for candidate in first if candidate.source == "block_top_close_x")
    rotated_current = _rotate_quaternion_about_local_z(
        seed.pose_quat_xyzw, math.pi
    )
    second = generate_block_grasp_candidates(
        (0.0, 0.0, 0.0),
        (0.05, 0.05, 0.05),
        0.0,
        rotated_current,
    )
    flipped = next(
        candidate for candidate in second if candidate.source == "block_top_close_x"
    )
    assert flipped.edge_axis == pytest.approx(
        tuple(-value for value in seed.edge_axis)
    )


def _rotate_quaternion_about_local_z(quaternion, angle):
    half = angle / 2.0
    local_rotation = (0.0, 0.0, math.sin(half), math.cos(half))
    x1, y1, z1, w1 = quaternion
    x2, y2, z2, w2 = local_rotation
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


@pytest.mark.parametrize(
    ("center", "size", "yaw", "current"),
    [
        ((0.0, 0.0), (0.05, 0.05, 0.05), 0.0, (0.0, 0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.05, 0.0, 0.05), 0.0, (0.0, 0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.05, 0.05, 0.05), math.inf, (0.0, 0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.05, 0.05, 0.05), 0.0, (0.0, 0.0, 0.0, 0.0)),
    ],
)
def test_invalid_geometry_is_rejected(center, size, yaw, current):
    with pytest.raises(ValueError):
        generate_block_grasp_candidates(center, size, yaw, current)
