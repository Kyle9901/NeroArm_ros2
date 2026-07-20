import math
from dataclasses import replace

import pytest

from mcp_server.models.geometry import (
    ObjectGeometry,
    aggregate_object_geometries,
)


def _observation(
    x: float,
    y: float,
    z: float,
    height: float,
    yaw: float,
    depth_mm: float,
) -> ObjectGeometry:
    return ObjectGeometry(
        surface_xyz=(x, y, z),
        center_xyz=(x, y, z - height / 2.0),
        size_xyz=(0.05, 0.05, height),
        local_desk_z=z - height,
        height=height,
        height_source="realtime_depth_local_desk",
        yaw_rad=yaw,
        yaw_period_rad=math.pi / 2.0,
        surface_depth_mm=depth_mm,
    )


def test_five_frame_geometry_uses_robust_median_and_rejects_one_outlier():
    observations = [
        _observation(-0.350, -0.100, 0.051, 0.050, 0.20, 551.0),
        _observation(-0.351, -0.099, 0.050, 0.049, 0.21, 550.0),
        _observation(-0.349, -0.101, 0.052, 0.051, 0.19, 552.0),
        _observation(-0.350, -0.100, 0.050, 0.050, 0.20, 550.0),
        _observation(-0.270, -0.020, 0.120, 0.110, 0.70, 630.0),
    ]

    result = aggregate_object_geometries(observations, requested_frames=5)

    assert result.geometry is not None
    assert result.quality.reliable
    assert result.quality.valid_frames == 5
    assert result.quality.inlier_frames == 4
    assert result.geometry.surface_xyz == pytest.approx(
        (-0.350, -0.100, 0.0505), abs=0.001,
    )
    assert result.geometry.height == pytest.approx(0.05, abs=0.001)
    assert result.geometry.quality["requested_frames"] == 5


def test_geometry_rejects_sequence_without_three_consistent_frames():
    observations = [
        _observation(-0.35, -0.10, 0.05, 0.05, 0.00, 550.0),
        _observation(-0.30, -0.10, 0.08, 0.08, 0.35, 590.0),
        _observation(-0.25, -0.10, 0.11, 0.11, 0.70, 630.0),
    ]

    result = aggregate_object_geometries(observations, requested_frames=5)

    assert result.geometry is None
    assert not result.quality.reliable
    assert result.quality.inlier_frames < 3
    assert "mutually consistent" in result.quality.rejection_reasons[0]


def test_geometry_rejects_too_few_valid_frames():
    observations = [
        _observation(-0.35, -0.10, 0.05, 0.05, 0.0, 550.0),
        _observation(-0.35, -0.10, 0.05, 0.05, 0.0, 550.0),
    ]

    result = aggregate_object_geometries(observations, requested_frames=5)

    assert result.geometry is None
    assert result.quality.valid_frames == 2
    assert "only 2/5 valid" in result.quality.rejection_reasons[0]


def test_square_yaw_aggregation_treats_ninety_degrees_as_equivalent():
    observations = [
        _observation(-0.35, -0.10, 0.05, 0.05, 0.02, 550.0),
        _observation(-0.35, -0.10, 0.05, 0.05, math.pi / 2.0 + 0.01, 550.0),
        _observation(-0.35, -0.10, 0.05, 0.05, -0.01, 550.0),
    ]

    result = aggregate_object_geometries(observations, requested_frames=5)

    assert result.geometry is not None
    assert result.quality.yaw_spread_rad < math.radians(2.0)


def test_geometry_rejects_inconsistent_planar_sizes():
    base = _observation(-0.35, -0.10, 0.05, 0.05, 0.0, 550.0)
    observations = [
        replace(base, size_xyz=(size, 0.05, 0.05))
        for size in (0.03, 0.05, 0.07)
    ]

    result = aggregate_object_geometries(
        observations,
        requested_frames=3,
        min_valid_frames=3,
        max_size_deviation_m=0.01,
    )

    assert result.geometry is None
    assert result.quality.inlier_frames == 1


def test_geometry_rejects_inconsistent_local_desk_height():
    base = _observation(-0.35, -0.10, 0.05, 0.05, 0.0, 550.0)
    observations = [
        replace(base, local_desk_z=desk)
        for desk in (-0.02, 0.0, 0.02)
    ]

    result = aggregate_object_geometries(
        observations,
        requested_frames=3,
        min_valid_frames=3,
        max_desk_deviation_m=0.01,
    )

    assert result.geometry is None
    assert result.quality.inlier_frames == 1
