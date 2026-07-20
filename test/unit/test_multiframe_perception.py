import numpy as np

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


def test_colored_bottle_never_enters_hsv_block_geometry(monkeypatch):
    calls = []
    monkeypatch.setattr(
        skill_perception,
        "_locate_cylinder_multiframe",
        lambda *_args, **_kwargs: (
            calls.append("cylinder")
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
    assert calls == ["cylinder"]
