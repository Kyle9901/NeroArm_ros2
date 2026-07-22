import numpy as np
import pytest

from mcp_server.components.base import ImageFrame
from mcp_server.components import perception
from mcp_server.perception.transparent_bottle import (
    aggregate_transparent_bottle_measurements,
    analyze_transparent_bottle_points,
    classify_pose_from_height_percentiles,
)


_THRESHOLDS = {
    "upright_min_p90_m": 0.075,
    "upright_min_p95_m": 0.095,
    "lying_min_p90_m": 0.040,
    "lying_min_p95_m": 0.045,
    "lying_max_p90_m": 0.065,
    "lying_max_p95_m": 0.080,
}


class _DownwardNode:
    @staticmethod
    def get_color_info():
        return {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0}

    @staticmethod
    def transform_to_base(x, y, z, **_kwargs):
        return {"x": x, "y": y, "z": -z}


class _BottleBridge:
    node = _DownwardNode()

    @staticmethod
    def get_transparent_bottle_profile():
        return {
            "transparent_bottle_diameter_m": 0.060,
            "transparent_bottle_height_m": 0.170,
            "transparent_bottle_label_bottom_m": 0.056,
            "transparent_bottle_label_height_m": 0.055,
            "transparent_bottle_min_height_m": 0.008,
            "transparent_bottle_min_label_points": 25,
            "transparent_bottle_tcp_max_spread_m": 0.010,
            "transparent_bottle_upright_min_p90_m": 0.075,
            "transparent_bottle_upright_min_p95_m": 0.095,
            "transparent_bottle_lying_min_p90_m": 0.040,
            "transparent_bottle_lying_min_p95_m": 0.045,
            "transparent_bottle_lying_max_p90_m": 0.065,
            "transparent_bottle_lying_max_p95_m": 0.080,
        }

    @staticmethod
    def get_desk_surface_z():
        # Deliberately differs from the visual plane (-0.600 m) so the test
        # proves that only the measured relative height is transferred.
        return -0.613

    @staticmethod
    def get_desk_measurement_max_error():
        return 0.05


def _analyze(pixels, depths, points, heights):
    return analyze_transparent_bottle_points(
        pixels,
        depths,
        points,
        heights,
        local_desk_z=0.0,
        minimum_height_m=0.008,
        maximum_height_m=0.20,
        label_axis_fraction=0.055 / 0.170,
        minimum_label_points=25,
        **_THRESHOLDS,
    )


def test_height_percentiles_have_an_ambiguous_safety_gap():
    assert classify_pose_from_height_percentiles(
        0.09, 0.14, **_THRESHOLDS,
    ) == "upright"
    assert classify_pose_from_height_percentiles(
        0.055, 0.065, **_THRESHOLDS,
    ) == "lying"
    assert classify_pose_from_height_percentiles(
        0.022, 0.024, **_THRESHOLDS,
    ) == "ambiguous"
    assert classify_pose_from_height_percentiles(
        0.070, 0.087, **_THRESHOLDS,
    ) == "ambiguous"


def test_consensus_excludes_low_label_upright_ghost_frames():
    weak = [
        {
            "orientation": "lying",
            "label_depth_points": count,
            "reliable_heights_m": [0.020, 0.023, 0.024] * 20,
            "tcp_xyz": [-0.39, 0.01, -0.004],
        }
        for count in (32, 64)
    ]
    strong = [
        {
            "orientation": "upright",
            "label_depth_points": 650 + index,
            "reliable_heights_m": [0.080, 0.098, 0.101] * 200,
            "tcp_xyz": [-0.408 + index * 0.0003, 0.016, 0.070],
        }
        for index in range(3)
    ]

    result = aggregate_transparent_bottle_measurements(
        weak + strong,
        minimum_frames=3,
        minimum_label_points=100,
        maximum_tcp_spread_m=0.010,
        **_THRESHOLDS,
    )

    assert result.ready
    assert result.orientation == "upright"
    assert result.inlier_indices == (2, 3, 4)
    assert result.label_gate >= 455
    assert result.tcp_spread_m < 0.001


def test_consensus_waits_for_three_quality_frames():
    result = aggregate_transparent_bottle_measurements(
        [{
            "orientation": "upright",
            "label_depth_points": 669,
            "reliable_heights_m": [0.080, 0.098, 0.101] * 200,
            "tcp_xyz": [-0.408, 0.016, 0.070],
        }],
        minimum_frames=3,
        minimum_label_points=100,
        maximum_tcp_spread_m=0.010,
        **_THRESHOLDS,
    )

    assert not result.ready
    assert "only 1 quality frames" in result.reason


def test_lying_consensus_uses_absolute_measured_support_gate():
    measurements = [
        {
            "orientation": "lying",
            "label_depth_points": count,
            "reliable_heights_m": [0.050, 0.056, 0.059] * 20,
            "tcp_xyz": [-0.400 + index * 0.0005, -0.10, 0.015],
        }
        for index, count in enumerate((25, 31, 120))
    ]

    result = aggregate_transparent_bottle_measurements(
        measurements,
        minimum_frames=3,
        minimum_label_points=25,
        maximum_tcp_spread_m=0.010,
        **_THRESHOLDS,
    )

    assert result.ready
    assert result.orientation == "lying"
    assert result.inlier_indices == (0, 1, 2)
    assert result.label_gate == 25


def test_upright_uses_cap_axis_xy_and_measured_label_z():
    label_u, label_v = np.meshgrid(
        np.arange(292, 313, 2),
        np.arange(204, 237, 2),
    )
    label_pixels = np.column_stack((label_u.ravel(), label_v.ravel()))
    label_count = len(label_pixels)
    label_heights = np.linspace(0.060, 0.110, label_count)
    label_points = np.column_stack((
        np.full(label_count, -0.370),
        np.full(label_count, -0.120),
        label_heights,
    ))
    label_depths = 585.0 - label_heights * 100.0

    cap_u, cap_v = np.meshgrid(
        np.arange(302, 311, 2),
        np.arange(174, 183, 2),
    )
    cap_pixels = np.column_stack((cap_u.ravel(), cap_v.ravel()))
    cap_count = len(cap_pixels)
    cap_heights = np.linspace(0.150, 0.170, cap_count)
    cap_points = np.column_stack((
        np.full(cap_count, -0.400),
        np.full(cap_count, -0.100),
        cap_heights,
    ))
    cap_depths = 565.0 - cap_heights * 30.0

    result = _analyze(
        np.vstack((label_pixels, cap_pixels)),
        np.concatenate((label_depths, cap_depths)),
        np.vstack((label_points, cap_points)),
        np.concatenate((label_heights, cap_heights)),
    )

    assert result.orientation == "upright"
    assert result.label_count >= 25
    assert result.tcp_xyz[0] == pytest.approx(-0.400, abs=0.003)
    assert result.tcp_xyz[1] == pytest.approx(-0.100, abs=0.003)
    assert 0.060 <= result.tcp_xyz[2] <= 0.110


def test_lying_uses_measured_surface_to_desk_midpoint():
    label_u, label_v = np.meshgrid(
        np.arange(220, 301, 2),
        np.arange(246, 257, 2),
    )
    pixels = np.column_stack((label_u.ravel(), label_v.ravel()))
    count = len(pixels)
    x = np.linspace(-0.45, -0.35, count)
    surface_z = np.linspace(0.054, 0.060, count)
    points = np.column_stack((
        x,
        np.full(count, -0.10),
        surface_z,
    ))
    depths = 560.0 + (pixels[:, 0] - np.median(pixels[:, 0])) * 0.03

    result = _analyze(pixels, depths, points, surface_z)

    assert result.orientation == "lying"
    assert result.label_count >= 25
    assert result.tcp_xyz[0] == pytest.approx(-0.40, abs=0.01)
    assert result.tcp_xyz[1] == pytest.approx(-0.10, abs=0.003)
    assert result.tcp_xyz[2] == pytest.approx(
        np.quantile(surface_z, 0.90) * 0.5,
        abs=1e-6,
    )
    assert abs(result.horizontal_axis_xy[0]) > 0.99


def test_lying_uses_long_axis_center_when_central_depth_is_empty():
    left_u, left_v = np.meshgrid(
        np.arange(200, 220, 2), np.arange(245, 257, 2)
    )
    right_u, right_v = np.meshgrid(
        np.arange(300, 320, 2), np.arange(245, 257, 2)
    )
    pixels = np.vstack((
        np.column_stack((left_u.ravel(), left_v.ravel())),
        np.column_stack((right_u.ravel(), right_v.ravel())),
    ))
    count = len(pixels)
    points = np.column_stack((
        np.concatenate((
            np.linspace(-0.46, -0.44, count // 2),
            np.linspace(-0.36, -0.34, count // 2),
        )),
        np.full(count, -0.10),
        np.linspace(0.050, 0.060, count),
    ))
    depths = np.linspace(550.0, 558.0, count)
    heights = points[:, 2]

    result = _analyze(pixels, depths, points, heights)

    assert result.orientation == "lying"
    assert result.label_count >= 25
    assert result.horizontal_span_m > 0.10
    assert result.tcp_xyz[0] == pytest.approx(-0.40, abs=0.01)
    assert result.tcp_xyz[1] == pytest.approx(-0.10, abs=0.003)
    assert result.tcp_xyz[2] == pytest.approx(
        np.quantile(heights, 0.90) * 0.5,
        abs=1e-6,
    )


def test_live_component_filters_see_through_desk_and_measures_label(
        monkeypatch):
    depth = np.full((480, 640), 600, dtype=np.uint16)
    # Transparent pixels continue to see the 600 mm desk. Only the central
    # label and the small cap return object depth.
    depth[205:266, 292:329] = 510
    depth[170:190, 302:319] = 440
    frame = ImageFrame(
        frame_id=1,
        color=np.zeros((480, 640, 3), dtype=np.uint8),
        depth=depth,
        timestamp_s=0.0,
    )
    detection = {
        "bbox": [280, 155, 340, 335],
        "center_2d": [310, 245],
    }
    monkeypatch.setattr(
        perception,
        "_save_transparent_bottle_debug",
        lambda *_args, **_kwargs: {},
    )

    result = perception.transparent_bottle_detection_to_3d(
        _BottleBridge(), frame, detection,
    )

    assert result.ok, result.error
    assert result.data["orientation"] == "upright"
    assert result.data["height_p95_m"] == pytest.approx(0.160, abs=0.003)
    assert result.data["label_surface_xyz"][2] == pytest.approx(
        -0.510, abs=0.003,
    )
    assert result.data["local_desk_z"] == pytest.approx(-0.600, abs=0.003)
    assert result.data["measured_surface_height_m"] == pytest.approx(
        0.090, abs=0.004,
    )
    assert result.data["tcp_xyz"][2] == pytest.approx(-0.523, abs=0.004)
    assert result.data["valid_depth_points"] < (
        (335 - 155) * (340 - 280)
    )
    assert len(result.data["reliable_heights_m"]) == result.data[
        "valid_depth_points"
    ]


def test_live_lying_tcp_xy_uses_yolo_long_axis_center_ray(monkeypatch):
    depth = np.full((480, 640), 600, dtype=np.uint16)
    depth[220:260, 200:400] = 550
    frame = ImageFrame(
        frame_id=2,
        color=np.zeros((480, 640, 3), dtype=np.uint8),
        depth=depth,
        timestamp_s=0.0,
    )
    detection = {
        "bbox": [190, 210, 410, 270],
        # Deliberately offset from the sparse-depth midpoint so the assertion
        # proves that the semantic bottle center, not partial depth extent,
        # defines lying-grasp XY.
        "center_2d": [360, 240],
    }
    monkeypatch.setattr(
        perception,
        "_save_transparent_bottle_debug",
        lambda *_args, **_kwargs: {},
    )

    result = perception.transparent_bottle_detection_to_3d(
        _BottleBridge(), frame, detection,
    )

    assert result.ok, result.error
    assert result.data["orientation"] == "lying"
    assert result.data["height_p90_m"] == pytest.approx(0.050, abs=0.003)
    # Camera ray x=(360-320)/600, intersected at the measured TCP-center
    # plane ~0.575m from this synthetic camera origin.
    assert result.data["tcp_xyz"][0] == pytest.approx(0.0383, abs=0.004)
    assert result.data["tcp_xyz"][1] == pytest.approx(0.0, abs=0.003)
    assert result.data["tcp_xyz"][2] == pytest.approx(-0.588, abs=0.004)
