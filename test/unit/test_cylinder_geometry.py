import math

import numpy as np
import pytest

from mcp_server.models import aggregate_cylinder_geometries
from mcp_server.perception.cylinder_geometry import fit_cylinder_geometry


def _upright_points(seed=0, *, center=(-0.4, -0.1), diameter=0.064,
                     length=0.22, desk=-0.013):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-math.pi / 2.0, math.pi / 2.0, 2500)
    z = rng.uniform(desk, desk + length, len(theta))
    radius = diameter / 2.0
    return np.column_stack((
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta),
        z,
    ))


def _lying_points(seed=0, *, center=(-0.4, -0.1), diameter=0.064,
                   length=0.22, desk=-0.013, yaw=0.35):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-math.pi / 2.0, math.pi / 2.0, 2500)
    along = rng.uniform(-length / 2.0, length / 2.0, len(theta))
    axis = np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
    cross = np.asarray([-axis[1], axis[0], 0.0])
    radius = diameter / 2.0
    base = np.asarray([center[0], center[1], desk + radius])
    return (
        base
        + along[:, None] * axis
        + (radius * np.cos(theta))[:, None] * cross
        + (radius * np.sin(theta))[:, None] * np.asarray([0.0, 0.0, 1.0])
    )


def test_fits_upright_cylinder_from_live_metric_points():
    geometry = fit_cylinder_geometry(
        _upright_points(), local_desk_z=-0.013,
    )
    assert geometry.shape_kind == "cylinder"
    assert geometry.orientation_class == "upright"
    assert geometry.axis_xyz == pytest.approx((0.0, 0.0, 1.0))
    assert geometry.diameter_m == pytest.approx(0.064, abs=0.002)
    assert geometry.length_m == pytest.approx(0.22, abs=0.008)
    assert geometry.center_xyz[2] == pytest.approx(
        -0.013 + geometry.length_m / 2.0
    )


def test_fits_lying_cylinder_and_recovers_horizontal_axis():
    geometry = fit_cylinder_geometry(
        _lying_points(), local_desk_z=-0.013,
    )
    assert geometry.orientation_class == "lying"
    assert abs(geometry.axis_xyz[2]) < 1e-8
    assert abs(geometry.axis_xyz[0]) == pytest.approx(
        math.cos(0.35), abs=0.03
    )
    assert abs(geometry.axis_xyz[1]) == pytest.approx(
        math.sin(0.35), abs=0.03
    )
    assert geometry.diameter_m == pytest.approx(0.064, abs=0.002)
    assert geometry.length_m == pytest.approx(0.22, abs=0.01)


def test_rejects_diagonal_or_out_of_range_cylinder():
    points = _upright_points()
    angle = math.radians(45.0)
    rotation = np.asarray([
        [math.cos(angle), 0.0, math.sin(angle)],
        [0.0, 1.0, 0.0],
        [-math.sin(angle), 0.0, math.cos(angle)],
    ])
    pivot = np.median(points, axis=0)
    diagonal = (points - pivot) @ rotation.T + pivot
    with pytest.raises(ValueError, match="diagonal"):
        fit_cylinder_geometry(diagonal, local_desk_z=-0.013)

    with pytest.raises(ValueError, match="diameter"):
        fit_cylinder_geometry(
            _upright_points(diameter=0.12), local_desk_z=-0.013,
        )


def test_five_frame_cylinder_aggregation_preserves_shape_fields():
    observations = [
        fit_cylinder_geometry(
            _upright_points(seed=index), local_desk_z=-0.013,
        )
        for index in range(5)
    ]
    result = aggregate_cylinder_geometries(observations)
    assert result.geometry is not None
    assert result.quality.reliable is True
    assert result.geometry.orientation_class == "upright"
    assert result.geometry.diameter_m == pytest.approx(0.064, abs=0.002)

