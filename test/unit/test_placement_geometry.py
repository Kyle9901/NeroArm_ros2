from types import SimpleNamespace

import pytest

from mcp_server.components.base import ComponentResult
from mcp_server.skills import manipulation, placement


def _geometry(
    *,
    center=(-0.40, -0.10, 0.025),
    size=(0.05, 0.05, 0.05),
    desk=0.0,
    yaw=0.0,
    reliable=True,
):
    return {
        "center": {"x": center[0], "y": center[1], "z": center[2]},
        "surface": {
            "x": center[0],
            "y": center[1],
            "z": desk + size[2],
        },
        "size": {"x": size[0], "y": size[1], "z": size[2]},
        "local_desk_z": desk,
        "height": size[2],
        "yaw_rad": yaw,
        "quality": {"reliable": reliable},
    }


def _candidate():
    return {
        "candidate_id": "block_tilt_30_from_pos_x",
        "pose_xyz": [-0.40, -0.10, 0.030],
        "pose_quat_xyzw": [0.0, 1.0, 0.0, 0.0],
        "approach_vector": [0.0, 0.0, -1.0],
        "pregrasp_distance": 0.08,
        "retreat_vector": [0.0, 0.0, 1.0],
        "retreat_distance": 0.08,
        "gripper_width": 0.05,
        "tilt_deg": 30.0,
        "edge_axis": [0.0, 1.0, 0.0],
        "score": 1.0,
        "source": "block_tilt_30_from_pos_x",
        "preference_rank": 0,
        "ranking_mode": "rotation_first",
        "object_kind": "block",
    }


class GeometryBridge:
    @staticmethod
    def get_stack_clearance_m():
        return 0.003

    @staticmethod
    def get_relative_placement_clearance_m():
        return 0.020

    @staticmethod
    def get_stack_max_overhang_m():
        return 0.010

    @staticmethod
    def get_placement_verify_xy_tolerance_m():
        return 0.025

    @staticmethod
    def get_placement_verify_z_tolerance_m():
        return 0.025


def test_stack_uses_both_geometries_and_preserves_object_tcp_transform():
    source = _geometry()
    support = _geometry(center=(-0.30, 0.10, 0.025))
    result = placement.stack_on(
        GeometryBridge(),
        source,
        support,
        _candidate(),
    )

    assert result.ok
    assert result.data["z"] == pytest.approx(0.053)
    assert result.data["object_center"] == pytest.approx({
        "x": -0.30,
        "y": 0.10,
        "z": 0.078,
    })
    # Object center translated by (+0.10, +0.20, +0.053), so the selected
    # grasp TCP receives exactly the same translation and keeps its quaternion.
    translated = result.data["placement_candidate"]
    assert translated["pose_xyz"] == pytest.approx([-0.30, 0.10, 0.083])
    assert translated["pose_quat_xyzw"] == _candidate()["pose_quat_xyzw"]


def test_stack_height_changes_with_source_and_support_geometry():
    source = _geometry(
        center=(-0.40, -0.10, 0.04),
        size=(0.05, 0.05, 0.08),
    )
    support = _geometry(
        center=(-0.30, 0.10, 0.03),
        size=(0.06, 0.06, 0.06),
    )
    result = placement.stack_on(
        GeometryBridge(), source, support, _candidate()
    )

    assert result.ok
    assert result.data["z"] == pytest.approx(0.063)
    assert result.data["object_center"]["z"] == pytest.approx(0.103)


def test_stack_rejects_source_footprint_with_excessive_overhang():
    source = _geometry(size=(0.09, 0.05, 0.05))
    support = _geometry(size=(0.05, 0.05, 0.05))

    result = placement.stack_on(
        GeometryBridge(), source, support, _candidate()
    )

    assert result.ok is False
    assert result.failed_step == "stack_on"
    assert "not sufficiently supported" in result.error


def test_stack_reports_support_margin_and_expected_top_surface():
    result = placement.stack_on(
        GeometryBridge(), _geometry(size=(0.04, 0.04, 0.05)),
        _geometry(size=(0.05, 0.05, 0.05)), _candidate()
    )

    assert result.ok
    assert result.data["support_margin"] == pytest.approx({"x": 0.005, "y": 0.005})
    assert result.data["overhang"] == pytest.approx({"x": 0.0, "y": 0.0})
    assert result.data["expected_surface_z"] == pytest.approx(0.100)


def test_relative_right_uses_projected_footprints_plus_clearance():
    source = _geometry(size=(0.04, 0.06, 0.05))
    reference = _geometry(
        center=(-0.30, 0.10, 0.025),
        size=(0.08, 0.04, 0.05),
    )
    result = placement.offset_from(
        GeometryBridge(),
        source,
        reference,
        _candidate(),
        "right_of",
    )

    # base_link right is -Y: reference 0.02 + source 0.03 + gap 0.02.
    assert result.ok
    assert result.data["separation"] == pytest.approx(0.07)
    assert result.data["x"] == pytest.approx(-0.30)
    assert result.data["y"] == pytest.approx(0.03)
    assert result.data["z"] == pytest.approx(0.0)


@pytest.mark.parametrize("broken", [
    _geometry(reliable=False),
    {**_geometry(), "center": None},
    {**_geometry(), "size": {"x": 0.0, "y": 0.05, "z": 0.05}},
])
def test_geometry_placement_fails_closed_on_untrusted_geometry(broken):
    result = placement.stack_on(
        GeometryBridge(), broken, _geometry(), _candidate()
    )
    assert result.ok is False
    assert result.retryable is False
    assert result.failed_step == "stack_on"


def test_relation_placement_rejects_cylinder_geometry_until_axes_are_supported():
    cylinder = {**_geometry(), "shape_kind": "cylinder"}
    result = placement.stack_on(
        GeometryBridge(), cylinder, _geometry(), _candidate()
    )
    assert result.ok is False
    assert "cylinder relation placement is not supported" in result.error


def test_post_place_verification_accepts_measured_pose_inside_tolerance():
    observed = _geometry(
        center=(-0.306, 0.108, 0.078),
        size=(0.05, 0.05, 0.05),
        desk=0.053,
    )
    result = placement.verify_placement(
        GeometryBridge(), observed,
        expected_x=-0.300,
        expected_y=0.100,
        expected_surface_z=0.103,
    )

    assert result.ok
    assert result.data["verified"] is True
    assert result.data["xy_error"] == pytest.approx(0.010)
    assert result.data["surface_z_error"] == pytest.approx(0.0)


def test_post_place_verification_fails_closed_outside_tolerance():
    observed = _geometry(
        center=(-0.34, 0.10, 0.078),
        size=(0.05, 0.05, 0.05),
        desk=0.053,
    )
    result = placement.verify_placement(
        GeometryBridge(), observed,
        expected_x=-0.300,
        expected_y=0.100,
        expected_surface_z=0.103,
    )

    assert result.ok is False
    assert result.failed_step == "post_place_verification"
    assert result.holding is False
    assert "xy_error=0.040m" in result.error


def _plan():
    state = SimpleNamespace(
        joint_state=SimpleNamespace(name=["joint7"], position=[0.0]),
    )
    trajectory = SimpleNamespace(
        joint_trajectory=SimpleNamespace(
            joint_names=["joint7"],
            points=[SimpleNamespace(positions=[0.0])],
        ),
    )
    return SimpleNamespace(
        start_state=state,
        end_state=state,
        trajectory=trajectory,
    )


class MotionBridge:
    def __init__(self):
        self.holding = True

    def get_holding(self):
        return self.holding

    def set_holding(self, value):
        self.holding = value

    @staticmethod
    def get_desk_surface_z():
        return -0.013

    @staticmethod
    def get_grasp_candidate_timeout():
        return 5.0

    @staticmethod
    def get_joint7_soft_limit_deg():
        return 85.0

    @staticmethod
    def get_fingertip_depth():
        return 0.04

    @staticmethod
    def get_approach_height():
        return 0.0867

    @staticmethod
    def get_safe_height():
        return 0.2267

    @staticmethod
    def get_gripper_open_width():
        return 0.10

    @staticmethod
    def get_gripper_close_width():
        return 0.02

    @staticmethod
    def get_descent_velocity_scaling():
        return 0.2

    @staticmethod
    def get_descent_accel_scaling():
        return 0.1


def test_translated_place_plans_complete_release_path_before_motion(monkeypatch):
    events = []
    plans = [_plan(), _plan(), _plan()]

    def plan_to_pose(*args, **kwargs):
        events.append(("plan_to", args[1:4]))
        return ComponentResult.success(plan=plans[0])

    def plan_cartesian(*args, **kwargs):
        index = 1 + sum(event[0] == "plan_cartesian" for event in events)
        events.append(("plan_cartesian", args[1:4]))
        return ComponentResult.success(plan=plans[index])

    monkeypatch.setattr(manipulation.motion, "workspace_check", lambda *args: ComponentResult.success())
    monkeypatch.setattr(manipulation.motion, "plan_to_pose", plan_to_pose)
    monkeypatch.setattr(manipulation.motion, "plan_cartesian", plan_cartesian)
    monkeypatch.setattr(
        manipulation.motion,
        "execute_planned",
        lambda *args: events.append(("execute", None)) or ComponentResult.success(),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "control_gripper",
        lambda *args, **kwargs: events.append(("open", None)) or ComponentResult.success(),
    )

    candidate = _candidate()
    candidate["pose_xyz"] = [-0.30, 0.10, 0.083]
    result = manipulation.place_object(
        MotionBridge(),
        -0.30,
        0.10,
        0.053,
        placement_candidate=candidate,
    )

    assert result.ok
    assert [event[0] for event in events[:3]] == [
        "plan_to", "plan_cartesian", "plan_cartesian",
    ]
    assert [event[0] for event in events[3:]] == [
        "execute", "execute", "open", "execute",
    ]


def test_translated_place_does_not_move_when_complete_plan_fails(monkeypatch):
    executed = []
    monkeypatch.setattr(manipulation.motion, "workspace_check", lambda *args: ComponentResult.success())
    monkeypatch.setattr(
        manipulation.motion,
        "plan_to_pose",
        lambda *args, **kwargs: ComponentResult.failure("no preplace path"),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "execute_planned",
        lambda *args: executed.append(True) or ComponentResult.success(),
    )

    result = manipulation.place_object(
        MotionBridge(),
        -0.30,
        0.10,
        0.053,
        placement_candidate=_candidate(),
    )
    assert result.ok is False
    assert result.failed_step == "translated_place_planning"
    assert executed == []
