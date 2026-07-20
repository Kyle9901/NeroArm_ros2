import math

import cv2
import numpy as np
import pytest

from mcp_server.components.base import ImageFrame
from mcp_server.components.perception import bbox_to_3d, color_detection_to_3d


class _FakeNode:
    @staticmethod
    def get_color_info():
        return {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0}

    @staticmethod
    def transform_to_base(x, y, z, **_kwargs):
        return {"x": x, "y": y, "z": z}

    @staticmethod
    def compute_3d(_u, _v, _depth):
        return None


class _FakeBridge:
    node = _FakeNode()


class _DownwardFakeNode(_FakeNode):
    @staticmethod
    def transform_to_base(x, y, z, **_kwargs):
        # Minimal downward-looking camera model: increasing optical depth
        # points toward decreasing base Z.
        return {"x": x, "y": y, "z": -z}


class _DownwardFakeBridge:
    node = _DownwardFakeNode()


class _DeskGuardDownwardBridge(_DownwardFakeBridge):
    @staticmethod
    def get_desk_surface_z():
        return -0.40

    @staticmethod
    def get_desk_measurement_max_error():
        return 0.05


class _TiltedDownwardNode(_FakeNode):
    angle = np.deg2rad(20.0)
    rotation = np.column_stack((
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -np.cos(angle), -np.sin(angle)]),
        np.array([0.0, np.sin(angle), -np.cos(angle)]),
    ))
    translation = np.array([0.0, 0.0, 0.60])

    @classmethod
    def transform_to_base(cls, x, y, z, **_kwargs):
        point = cls.rotation @ np.array([x, y, z]) + cls.translation
        return {"x": point[0], "y": point[1], "z": point[2]}


class _TiltedDownwardBridge:
    node = _TiltedDownwardNode()


class _StrongTiltedDownwardNode(_TiltedDownwardNode):
    angle = np.deg2rad(40.0)
    rotation = np.column_stack((
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -np.cos(angle), -np.sin(angle)]),
        np.array([0.0, np.sin(angle), -np.cos(angle)]),
    ))


class _StrongTiltedDownwardBridge:
    node = _StrongTiltedDownwardNode()


def _detection_from_mask(mask: np.ndarray) -> dict:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    return {
        "bbox": [x, y, x + width, y + height],
        "center_2d": [x + width // 2, y + height // 2],
        "contour_2d": contour.reshape(-1, 2).astype(int).tolist(),
    }


def _periodic_difference(left: float, right: float, period: float) -> float:
    return abs((left - right + period / 2.0) % period - period / 2.0)


def _frame_with_sloped_desk_and_empty_object():
    height, width = 480, 640
    yy, xx = np.indices((height, width))
    # Smooth, tilted table in millimetres.
    depth = (550.0 - 0.03 * yy + 0.01 * xx).astype(np.uint16)
    depth[276:332, 184:240] = 0
    return ImageFrame(
        frame_id=1,
        color=np.zeros((height, width, 3), dtype=np.uint8),
        depth=depth,
        timestamp_s=0.0,
    )


def test_color_geometry_measures_live_surface_height_and_axes():
    height, width = 480, 640
    depth = np.full((height, width), 600, dtype=np.uint16)
    depth[190:291, 270:371] = 550
    frame = ImageFrame(
        frame_id=1,
        color=np.zeros((height, width, 3), dtype=np.uint8),
        depth=depth,
        timestamp_s=0.0,
    )
    detection = {
        "bbox": [270, 190, 370, 290],
        "center_2d": [320, 240],
        "rotated_center_2d": [320.0, 240.0],
        "contour_2d": [[270, 190], [370, 190], [370, 290], [270, 290]],
        "edge_axes_2d": [
            {"direction": [1.0, 0.0], "length_px": 100.0},
            {"direction": [0.0, 1.0], "length_px": 100.0},
        ],
        "yaw_period_rad": np.pi / 2.0,
    }

    result = color_detection_to_3d(_DownwardFakeBridge(), frame, detection)

    assert result.ok, result.error
    assert result.data["method"] == "color_top_plane_local_desk_3d_ransac"
    assert result.data["depth_is_estimated"] is False
    geometry = result.data["geometry"]
    assert geometry["height_source"] == "realtime_depth_local_desk"
    assert geometry["height"] == pytest.approx(0.05, abs=0.002)
    assert geometry["surface"]["z"] - geometry["local_desk_z"] == pytest.approx(
        geometry["height"],
    )
    assert geometry["center"]["z"] - geometry["local_desk_z"] == pytest.approx(
        geometry["height"] / 2.0,
    )
    assert geometry["size"]["x"] == pytest.approx(0.0917, abs=0.003)
    assert geometry["size"]["y"] == pytest.approx(0.0917, abs=0.003)
    assert geometry["yaw_rad"] == pytest.approx(0.0, abs=1e-6)


def test_unknown_object_never_expands_into_surrounding_desk():
    result = bbox_to_3d(
        _FakeBridge(),
        _frame_with_sloped_desk_and_empty_object(),
        [184, 276, 239, 331],
    )

    assert not result.ok
    assert "No valid depth in bbox" in result.error


def test_color_geometry_rejects_desk_height_inconsistent_with_config():
    height, width = 480, 640
    depth = np.full((height, width), 600, dtype=np.uint16)
    depth[190:291, 270:371] = 550
    frame = ImageFrame(
        frame_id=10,
        color=np.zeros((height, width, 3), dtype=np.uint8),
        depth=depth,
        timestamp_s=0.0,
    )
    detection = {
        "bbox": [270, 190, 370, 290],
        "center_2d": [320, 240],
        "contour_2d": [[270, 190], [370, 190], [370, 290], [270, 290]],
    }

    result = color_detection_to_3d(
        _DeskGuardDownwardBridge(), frame, detection,
    )

    assert not result.ok
    assert "configured desk" in result.error


def test_color_geometry_fits_metric_planes_under_perspective():
    height, width = 480, 640
    yy, xx = np.indices((height, width))
    ray_x = (xx - 320.0) / 600.0
    ray_y = (yy - 240.0) / 600.0
    rays = np.stack((ray_x, ray_y, np.ones_like(ray_x)), axis=-1)
    base_ray_z = rays @ _TiltedDownwardNode.rotation[2]
    desk_depth_m = -_TiltedDownwardNode.translation[2] / base_ray_z
    top_depth_m = (
        0.05 - _TiltedDownwardNode.translation[2]
    ) / base_ray_z
    depth = np.rint(desk_depth_m * 1000.0).astype(np.uint16)
    depth[200:281, 280:361] = np.rint(
        top_depth_m[200:281, 280:361] * 1000.0
    ).astype(np.uint16)
    frame = ImageFrame(
        frame_id=2,
        color=np.zeros((height, width, 3), dtype=np.uint8),
        depth=depth,
        timestamp_s=0.0,
    )
    detection = {
        "bbox": [280, 200, 360, 280],
        "center_2d": [320, 240],
        "rotated_center_2d": [320.0, 240.0],
        "contour_2d": [[280, 200], [360, 200], [360, 280], [280, 280]],
        "yaw_period_rad": np.pi / 2.0,
    }

    result = color_detection_to_3d(_TiltedDownwardBridge(), frame, detection)

    assert result.ok, result.error
    geometry = result.data["geometry"]
    assert geometry["surface"]["z"] == pytest.approx(0.05, abs=0.003)
    assert geometry["local_desk_z"] == pytest.approx(0.0, abs=0.003)
    assert geometry["height"] == pytest.approx(0.05, abs=0.003)
    assert geometry["size"]["x"] > 0.06
    assert geometry["size"]["y"] > 0.06


def test_metric_rectangle_is_fitted_after_perspective_recovery():
    height, width = 480, 640
    yy, xx = np.indices((height, width))
    rays = np.stack((
        (xx - 320.0) / 600.0,
        (yy - 240.0) / 600.0,
        np.ones((height, width)),
    ), axis=-1)
    base_ray_z = rays @ _StrongTiltedDownwardNode.rotation[2]
    desk_depth_m = -_StrongTiltedDownwardNode.translation[2] / base_ray_z
    top_z = 0.05
    top_depth_m = (
        top_z - _StrongTiltedDownwardNode.translation[2]
    ) / base_ray_z
    top_points_base = (
        (rays * top_depth_m[..., None])
        @ _StrongTiltedDownwardNode.rotation.T
        + _StrongTiltedDownwardNode.translation
    )

    center_xy = np.array([0.050, top_points_base[240, 320, 1]])
    expected_size = (0.100, 0.045)
    expected_yaw = math.radians(35.0)
    primary = np.array([math.cos(expected_yaw), math.sin(expected_yaw)])
    secondary = np.array([-primary[1], primary[0]])
    relative = top_points_base[..., :2] - center_xy
    local_primary = relative @ primary
    local_secondary = relative @ secondary
    top_mask = (
        (np.abs(local_primary) <= expected_size[0] / 2.0)
        & (np.abs(local_secondary) <= expected_size[1] / 2.0)
    )
    assert int(np.count_nonzero(top_mask)) > 1000

    depth = np.rint(desk_depth_m * 1000.0).astype(np.uint16)
    depth[top_mask] = np.rint(
        top_depth_m[top_mask] * 1000.0
    ).astype(np.uint16)
    frame = ImageFrame(
        frame_id=3,
        color=np.zeros((height, width, 3), dtype=np.uint8),
        depth=depth,
        timestamp_s=0.0,
    )
    detection = _detection_from_mask(top_mask)

    result = color_detection_to_3d(
        _StrongTiltedDownwardBridge(), frame, detection,
    )

    assert result.ok, result.error
    geometry = result.data["geometry"]
    assert geometry["surface"]["x"] == pytest.approx(center_xy[0], abs=0.003)
    assert geometry["surface"]["y"] == pytest.approx(center_xy[1], abs=0.003)
    assert geometry["size"]["x"] == pytest.approx(expected_size[0], abs=0.004)
    assert geometry["size"]["y"] == pytest.approx(expected_size[1], abs=0.004)
    assert _periodic_difference(
        geometry["yaw_rad"], expected_yaw, math.pi,
    ) < math.radians(3.0)


def test_sparse_top_beats_denser_side_bins_and_rejects_high_flying_pixels():
    height, width = 480, 640
    depth = np.full((height, width), 600, dtype=np.uint16)
    x1, x2 = 270, 371
    y1, top_bottom, y2 = 180, 250, 301
    depth[y1:y2, x1:x2] = 0

    # Only about 6% of the top returns valid depth, distributed over its area.
    depth[y1:top_bottom:4, x1:x2:4] = 550

    # Dense vertical-side slices contain more points per height than the top.
    # A "densest bin" selector would choose one of these lower plateaus.
    for start, value in zip(
        range(top_bottom, y2, 10),
        (555, 565, 575, 585, 595, 595),
    ):
        depth[start:min(start + 10, y2), x1:x2] = value

    # A higher but spatially disconnected flying-pixel cluster must not become
    # the top merely because it is the highest non-empty height bin.
    flying_pixels = [
        (v, u)
        for v in range(y1 + 2, top_bottom - 2, 14)
        for u in range(x1 + 3, x2 - 3, 16)
    ]
    for v, u in flying_pixels:
        depth[v, u] = 530

    contour_mask = np.zeros((height, width), dtype=np.uint8)
    contour_mask[y1:y2, x1:x2] = 255
    frame = ImageFrame(
        frame_id=4,
        color=np.zeros((height, width, 3), dtype=np.uint8),
        depth=depth,
        timestamp_s=0.0,
    )
    detection = _detection_from_mask(contour_mask)

    result = color_detection_to_3d(
        _DownwardFakeBridge(), frame, detection,
    )

    assert result.ok, result.error
    geometry = result.data["geometry"]
    assert geometry["height"] == pytest.approx(0.050, abs=0.003)
    quality = geometry["quality"]
    assert quality["top_connected_ratio"] >= 0.65
    assert quality["top_occupancy_ratio"] >= 0.015
    assert 0.20 <= quality["top_extent_ratio"] <= 1.15
    assert quality["top_plane_coverage"] == pytest.approx(
        quality["top_occupancy_ratio"],
    )
