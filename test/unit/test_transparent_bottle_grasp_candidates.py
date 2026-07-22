import pytest

from mcp_server.grasping import (
    generate_transparent_bottle_grasp_candidates,
)


_CURRENT_QUAT = (0.0, 1.0, 0.0, 0.0)
_CENTER = (-0.38, 0.02, 0.071)


def test_upright_bottle_has_eight_horizontal_label_grasps():
    candidates = generate_transparent_bottle_grasp_candidates(
        _CENTER,
        (0.0, 0.0, 1.0),
        0.060,
        "upright",
        _CURRENT_QUAT,
        upright_axis_overtravel_m=0.020,
    )

    assert len(candidates) == 8
    for candidate in candidates:
        expected = tuple(
            _CENTER[index] + 0.020 * candidate.approach_vector[index]
            for index in range(3)
        )
        assert candidate.pose_xyz == pytest.approx(expected)
    assert all(candidate.tilt_deg == 90.0 for candidate in candidates)
    assert all(
        abs(candidate.approach_vector[2]) < 1e-9
        for candidate in candidates
    )
    assert all(
        candidate.source.startswith("transparent_bottle_upright_side_")
        for candidate in candidates
    )
    approach_xy = {
        tuple(round(value, 3) for value in candidate.approach_vector[:2])
        for candidate in candidates
    }
    assert len(approach_xy) == 8


def test_upright_bottle_zero_overtravel_preserves_axis_center():
    candidates = generate_transparent_bottle_grasp_candidates(
        _CENTER,
        (0.0, 0.0, 1.0),
        0.060,
        "upright",
        _CURRENT_QUAT,
        upright_axis_overtravel_m=0.0,
    )

    assert all(candidate.pose_xyz == _CENTER for candidate in candidates)


@pytest.mark.parametrize("overtravel", [-0.001, 0.030, float("nan")])
def test_upright_bottle_rejects_unsafe_axis_overtravel(overtravel):
    with pytest.raises(ValueError, match="upright_axis_overtravel_m"):
        generate_transparent_bottle_grasp_candidates(
            _CENTER,
            (0.0, 0.0, 1.0),
            0.060,
            "upright",
            _CURRENT_QUAT,
            upright_axis_overtravel_m=overtravel,
        )


def test_lying_bottle_has_only_one_vertical_cross_axis_grasp():
    candidates = generate_transparent_bottle_grasp_candidates(
        _CENTER,
        (1.0, 0.0, 0.0),
        0.060,
        "lying",
        _CURRENT_QUAT,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.pose_xyz == _CENTER
    assert candidate.approach_vector == pytest.approx((0.0, 0.0, -1.0))
    assert candidate.edge_axis[0] == pytest.approx(0.0, abs=1e-9)
    assert abs(candidate.edge_axis[1]) == pytest.approx(1.0)
    assert candidate.tilt_deg == 0.0
    assert candidate.source == "transparent_bottle_lying_vertical_top"
