import numpy as np
import pytest

from mcp_server.components.base import ComponentResult, ImageFrame
from mcp_server.models.geometry import ObjectGeometry
from mcp_server.skills import perception as skill_perception


class _Bridge:
    @staticmethod
    def get_block_depth_frames():
        return 5

    @staticmethod
    def get_block_depth_max_spread():
        return 0.012

    @staticmethod
    def get_block_xy_max_spread():
        return 0.015

    @staticmethod
    def get_block_yaw_max_spread_deg():
        return 12.0

    @staticmethod
    def get_cylinder_depth_frames():
        return 5

    @staticmethod
    def get_cylinder_depth_max_spread():
        return 0.015

    @staticmethod
    def get_cylinder_position_max_spread():
        return 0.015

    @staticmethod
    def get_cylinder_axis_max_spread_deg():
        return 12.0

    @staticmethod
    def get_transparent_bottle_depth_frames():
        return 5

    @staticmethod
    def get_transparent_bottle_max_capture_frames():
        return 30

    @staticmethod
    def get_transparent_bottle_profile():
        return {
            "transparent_bottle_diameter_m": 0.060,
            "transparent_bottle_height_m": 0.170,
            "transparent_bottle_label_bottom_m": 0.056,
            "transparent_bottle_label_height_m": 0.055,
            "transparent_bottle_min_height_m": 0.008,
            "transparent_bottle_min_label_points": 100,
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
        return -0.013


def _geometry(index: int) -> ObjectGeometry:
    jitter = (index - 2) * 0.0002
    return ObjectGeometry(
        surface_xyz=(-0.35 + jitter, -0.10, 0.05 + jitter),
        center_xyz=(-0.35 + jitter, -0.10, 0.025 + jitter),
        size_xyz=(0.05, 0.05, 0.05),
        local_desk_z=0.0,
        height=0.05,
        height_source="realtime_depth_local_desk",
        yaw_rad=0.2 + jitter,
        yaw_period_rad=np.pi / 2.0,
        surface_depth_mm=550.0 + index * 0.2,
        quality={"reliable": True},
    )


def test_color_locate_captures_configured_five_frames_and_fuses(monkeypatch):
    calls = {"capture": 0, "measure": 0}

    def capture(_bridge, timeout):
        calls["capture"] += 1
        frame = ImageFrame(
            frame_id=calls["capture"],
            color=np.zeros((8, 8, 3), dtype=np.uint8),
            depth=np.ones((8, 8), dtype=np.uint16),
            timestamp_s=float(calls["capture"]),
        )
        return ComponentResult.success(frame=frame)

    def detect(_frame, color_name, _location_hint):
        return ComponentResult.success(
            found=True,
            color=color_name,
            source="CV",
            bbox=[1, 1, 6, 6],
            center_2d=[4, 4],
        )

    def measure(_bridge, _frame, _detection):
        geometry = _geometry(calls["measure"])
        calls["measure"] += 1
        return ComponentResult.success(
            geometry=geometry.to_dict(),
            valid_depth_points=40,
        )

    monkeypatch.setattr(skill_perception.perception, "capture_image", capture)
    monkeypatch.setattr(skill_perception.perception, "detect_by_color", detect)
    monkeypatch.setattr(
        skill_perception.perception, "color_detection_to_3d", measure,
    )

    result = skill_perception._locate_color_multiframe(
        _Bridge(), "红色物块", "red", "",
    )

    assert result.ok, result.error
    assert calls == {"capture": 5, "measure": 5}
    assert result.data["method"] == "realtime_depth_5frame_median"
    assert result.data["geometry_quality"]["requested_frames"] == 5
    assert result.data["geometry_quality"]["inlier_frames"] == 5
    assert result.data["geometry"]["height_source"] == (
        "realtime_depth_5frame_median"
    )


def _cylinder_geometry(index: int) -> ObjectGeometry:
    jitter = (index - 2) * 0.0002
    return ObjectGeometry(
        surface_xyz=(-0.35 + jitter, -0.10, 0.207),
        center_xyz=(-0.35 + jitter, -0.10, 0.097),
        size_xyz=(0.064, 0.064, 0.22),
        local_desk_z=-0.013,
        height=0.22,
        height_source="realtime_depth_cylinder_fit",
        yaw_rad=0.0,
        yaw_period_rad=np.pi,
        surface_depth_mm=550.0 + index * 0.2,
        shape_kind="cylinder",
        axis_xyz=(0.0, 0.0, 1.0),
        diameter_m=0.064 + jitter,
        length_m=0.22 + jitter,
        orientation_class="upright",
        quality={"reliable": True},
    )


def test_cylinder_locate_uses_yolo_and_fuses_five_depth_fits(monkeypatch):
    calls = {"capture": 0, "detect": 0, "measure": 0}
    events = []

    class _LazyYolo:
        @staticmethod
        def ensure_loaded():
            events.append("yolo_ready")

    def capture(_bridge, timeout):
        assert events == ["yolo_ready"]
        calls["capture"] += 1
        return ComponentResult.success(frame=ImageFrame(
            frame_id=calls["capture"],
            color=np.zeros((8, 8, 3), dtype=np.uint8),
            depth=np.ones((8, 8), dtype=np.uint16),
            timestamp_s=float(calls["capture"]),
        ))

    def detect(_yolo, _frame, target, _location_hint):
        calls["detect"] += 1
        return ComponentResult.success(
            found=True,
            target=target,
            bbox=[1, 1, 6, 6],
            center_2d=[4, 4],
            source="YOLO",
        )

    def measure(_bridge, _frame, _detection, *, color_name):
        assert color_name == "red"
        geometry = _cylinder_geometry(calls["measure"])
        calls["measure"] += 1
        return ComponentResult.success(
            geometry=geometry.to_dict(),
            valid_depth_points=100,
        )

    monkeypatch.setattr(skill_perception.perception, "capture_image", capture)
    monkeypatch.setattr(skill_perception.perception, "detect_by_yolo", detect)
    monkeypatch.setattr(
        skill_perception.perception, "cylinder_detection_to_3d", measure,
    )
    result = skill_perception._locate_cylinder_multiframe(
        _Bridge(), "红色水瓶", _LazyYolo(), "", "red",
    )
    assert result.ok, result.error
    assert events == ["yolo_ready"]
    assert calls == {"capture": 5, "detect": 5, "measure": 5}
    assert result.data["cylinder_orientation"] == "upright"
    assert result.data["geometry"]["shape_kind"] == "cylinder"
    assert result.data["geometry_quality"]["inlier_frames"] == 5


def test_transparent_bottle_locate_fuses_label_depth_into_formal_geometry(
        monkeypatch):
    calls = {"capture": 0, "detect": 0, "measure": 0}
    events = []

    class _LazyYolo:
        @staticmethod
        def ensure_loaded():
            events.append("yolo_ready")

    def capture(_bridge, timeout):
        assert events == ["yolo_ready"]
        calls["capture"] += 1
        return ComponentResult.success(frame=ImageFrame(
            frame_id=calls["capture"],
            color=np.zeros((8, 8, 3), dtype=np.uint8),
            depth=np.ones((8, 8), dtype=np.uint16),
            timestamp_s=float(calls["capture"]),
        ))

    def detect(_yolo, _frame, target, _location_hint):
        calls["detect"] += 1
        return ComponentResult.success(
            found=True,
            target=target,
            bbox=[1, 1, 6, 6],
            center_2d=[4, 4],
            source="YOLO",
        )

    def measure(_bridge, _frame, _detection):
        index = calls["measure"]
        calls["measure"] += 1
        jitter = (index - 2) * 0.0003
        return ComponentResult.success(
            orientation="lying",
            height_p90_m=0.056,
            height_p95_m=0.057,
            local_desk_z=0.002,
            configured_desk_z=-0.013,
            measured_surface_height_m=0.056,
            label_surface_xyz=[-0.39 + jitter, 0.025, 0.058],
            cap_or_axis_xy=[-0.39, 0.025],
            horizontal_axis_xy=[1.0, 0.0],
            tcp_xyz=[-0.39 + jitter, 0.025, 0.015],
            valid_depth_points=120,
            label_depth_points=180 + index,
            reliable_heights_m=np.linspace(0.01, 0.06, 120).tolist(),
            debug_image="pose.png",
            height_debug_image="height.png",
        )

    monkeypatch.setattr(skill_perception.perception, "capture_image", capture)
    monkeypatch.setattr(skill_perception.perception, "detect_by_yolo", detect)
    monkeypatch.setattr(
        skill_perception.perception,
        "transparent_bottle_detection_to_3d",
        measure,
    )

    result = skill_perception._locate_transparent_bottle_multiframe(
        _Bridge(), "水瓶", _LazyYolo(), "",
    )

    assert result.ok, result.error
    assert calls == {"capture": 3, "detect": 3, "measure": 3}
    assert result.data["cylinder_orientation"] == "lying"
    assert result.data["z"] == pytest.approx(0.015)
    assert result.data["geometry"]["center"]["z"] == pytest.approx(0.015)
    assert result.data["geometry"]["local_desk_z"] == pytest.approx(-0.013)
    assert result.data["geometry"]["axis_xyz"] == pytest.approx(
        [1.0, 0.0, 0.0],
    )
    assert result.data["geometry_quality"]["inlier_frames"] == 3
    assert result.data["geometry_quality"]["captured_frames"] == 3


def test_transparent_bottle_exhausted_adaptive_capture_is_not_retried(
        monkeypatch):
    calls = {"capture": 0}

    class _ShortBudgetBridge(_Bridge):
        @staticmethod
        def get_transparent_bottle_max_capture_frames():
            return 4

    class _LazyYolo:
        @staticmethod
        def ensure_loaded():
            return None

    def capture(_bridge, timeout):
        assert timeout > 0.0
        calls["capture"] += 1
        return ComponentResult.success(frame=ImageFrame(
            frame_id=calls["capture"],
            color=np.zeros((8, 8, 3), dtype=np.uint8),
            depth=np.ones((8, 8), dtype=np.uint16),
            timestamp_s=float(calls["capture"]),
        ))

    monkeypatch.setattr(skill_perception.perception, "capture_image", capture)
    monkeypatch.setattr(
        skill_perception.perception,
        "detect_by_yolo",
        lambda *_args: ComponentResult.success(
            found=True, bbox=[1, 1, 6, 6], center_2d=[4, 4]
        ),
    )
    monkeypatch.setattr(
        skill_perception.perception,
        "transparent_bottle_detection_to_3d",
        lambda *_args: ComponentResult.failure("no usable depth"),
    )

    result = skill_perception._locate_transparent_bottle_multiframe(
        _ShortBudgetBridge(), "水瓶", _LazyYolo(), "",
    )

    assert result.ok is False
    assert result.retryable is False
    assert calls["capture"] == 4


def test_colored_bottle_never_enters_hsv_block_geometry(monkeypatch):
    calls = []
    monkeypatch.setattr(
        skill_perception,
        "_locate_transparent_bottle_multiframe",
        lambda *_args, **_kwargs: (
            calls.append("transparent_bottle")
            or skill_perception.SkillResult.success(
                x=0.0, y=0.0, z=0.1, geometry={}
            )
        ),
    )
    monkeypatch.setattr(
        skill_perception,
        "_locate_color_multiframe",
        lambda *_args, **_kwargs: (
            calls.append("color")
            or skill_perception.SkillResult.failure("wrong route")
        ),
    )
    result = skill_perception.locate_object(
        _Bridge(), object(), "红色水瓶", yolo=object(),
    )
    assert result.ok
    assert calls == ["transparent_bottle"]
