from collections import Counter
import math

import pytest

from mcp_server.grasping.cylinder_candidates import (
    generate_cylinder_grasp_candidates,
)
from mcp_server.models import GraspCandidate


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def test_upright_prefers_four_exact_horizontal_side_grasps_at_45_percent():
    candidates = generate_cylinder_grasp_candidates(
        (-0.4, -0.1, 0.097),
        (0.0, 0.0, 1.0),
        0.064,
        0.22,
        "upright",
        (0.0, 0.0, 0.0, 1.0),
        local_desk_z=-0.013,
        side_grasp_height_ratio=0.45,
    )
    assert len(candidates) == 10
    assert Counter(candidate.tilt_deg for candidate in candidates) == {
        90.0: 4,
        60.0: 4,
        0.0: 2,
    }
    primary = [
        candidate for candidate in candidates
        if candidate.preference_rank == 0
    ]
    assert len(primary) == 4
    assert all(candidate.ranking_mode == "shape_first" for candidate in primary)
    assert all(candidate.pose_xyz[2] == pytest.approx(
        -0.013 + 0.22 * 0.45
    ) for candidate in primary)
    assert all(candidate.approach_vector[2] == pytest.approx(0.0)
               for candidate in primary)
    restored = GraspCandidate.from_dict(primary[0].to_dict())
    assert restored.object_kind == "cylinder"
    assert restored.preference_rank == 0
    assert restored.ranking_mode == "shape_first"


def test_lying_candidates_close_across_cross_section_not_along_axis():
    yaw = math.radians(25.0)
    axis = (math.cos(yaw), math.sin(yaw), 0.0)
    candidates = generate_cylinder_grasp_candidates(
        (-0.4, -0.1, 0.019),
        axis,
        0.064,
        0.22,
        "lying",
        (0.0, 0.0, 0.0, 1.0),
        local_desk_z=-0.013,
    )
    assert len(candidates) == 5
    assert candidates[0].candidate_id == "cylinder_lying_top"
    assert candidates[0].preference_rank == 0
    for candidate in candidates:
        assert _dot(candidate.edge_axis, axis) == pytest.approx(0.0, abs=1e-8)
        assert _dot(
            candidate.edge_axis, candidate.approach_vector
        ) == pytest.approx(0.0, abs=1e-8)
        assert candidate.gripper_width == pytest.approx(0.064)
