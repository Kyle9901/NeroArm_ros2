import math
from types import SimpleNamespace

import pytest

from mcp_server.components.base import ComponentResult
from mcp_server.grasping.pipeline import (
    _joint7_path_limit_error,
    _minimum_joint7_margin_deg,
    plan_block_grasp,
    plan_cylinder_grasp,
)
from mcp_server.models import ObjectGeometry


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


class _Node:
    grasp_candidate_markers = None

    @staticmethod
    def workspace_check(_x, _y):
        return True

    @staticmethod
    def robot_state_with_gripper_width(width):
        return {"planned_gripper_width": width}


class _Bridge:
    node = _Node()

    @staticmethod
    def get_current_tcp_pose(timeout=1.0):
        assert timeout > 0
        return {"quaternion": [0.0, 1.0, 0.0, 0.0]}

    @staticmethod
    def get_joint_state():
        return {
            "joints": {
                f"joint{index}": 0.0 for index in range(1, 8)
            }
        }

    @staticmethod
    def get_grasp_pregrasp_distance():
        return 0.08

    @staticmethod
    def get_grasp_retreat_distance():
        return 0.08

    @staticmethod
    def get_grasp_tilt_angles_deg():
        return [0, 30, 60]

    @staticmethod
    def get_cylinder_tilt_angles_deg():
        return [0, 60, 90]

    @staticmethod
    def get_cylinder_side_grasp_height_ratio():
        return 0.45

    @staticmethod
    def get_gripper_open_width():
        return 0.10

    @staticmethod
    def get_desk_surface_z():
        return -0.013

    @staticmethod
    def get_carry_joints_deg():
        return [0.0, -20.0, 0.0, 80.0, 0.0, 0.0, 50.0]

    @staticmethod
    def get_grasp_full_plan_candidates():
        return 2

    @staticmethod
    def get_grasp_candidate_timeout():
        return 8.0

    @staticmethod
    def get_joint7_soft_limit_deg():
        return 75.0

    @staticmethod
    def get_joint7_min_margin_deg():
        return 15.0


def _geometry():
    return ObjectGeometry(
        surface_xyz=(-0.35, -0.10, 0.05),
        center_xyz=(-0.35, -0.10, 0.025),
        size_xyz=(0.05, 0.05, 0.05),
        local_desk_z=0.0,
        height=0.05,
        height_source="realtime_depth_5frame_median",
        yaw_rad=0.2,
        quality={"reliable": True},
    )


def _cylinder_geometry():
    return ObjectGeometry(
        surface_xyz=(-0.35, -0.10, 0.207),
        center_xyz=(-0.35, -0.10, 0.097),
        size_xyz=(0.064, 0.064, 0.22),
        local_desk_z=-0.013,
        height=0.22,
        height_source="realtime_cylinder_5frame_median",
        yaw_rad=0.0,
        quality={"reliable": True},
        shape_kind="cylinder",
        axis_xyz=(0.0, 0.0, 1.0),
        diameter_m=0.064,
        length_m=0.22,
        orientation_class="upright",
    )


def test_pipeline_checks_ten_and_fully_plans_two_without_execution(monkeypatch):
    from mcp_server.grasping import pipeline

    calls = {"ik": 0, "pose": 0, "cartesian": 0, "joints": 0}
    planning_seeds = []

    def solve(*_args, **_kwargs):
        calls["ik"] += 1
        planning_seeds.append(_kwargs.get("seed_state"))
        return ComponentResult.success(
            joints={f"joint{index}": 0.0 for index in range(1, 8)}
        )

    def pose(*_args, **_kwargs):
        calls["pose"] += 1
        planning_seeds.append(_kwargs.get("start_state"))
        return ComponentResult.success(plan=_plan())

    def cartesian(*_args, **_kwargs):
        calls["cartesian"] += 1
        return ComponentResult.success(plan=_plan())

    def joints(*_args, **_kwargs):
        calls["joints"] += 1
        return ComponentResult.success(plan=_plan())

    monkeypatch.setattr(pipeline.motion, "solve_pose_ik", solve)
    monkeypatch.setattr(pipeline.motion, "plan_to_pose", pose)
    monkeypatch.setattr(pipeline.motion, "plan_cartesian", cartesian)
    monkeypatch.setattr(pipeline.motion, "plan_joints", joints)

    result = plan_block_grasp(_Bridge(), _geometry())
    assert result.ok
    assert len(result.candidates) == 10
    assert calls == {"ik": 10, "pose": 2, "cartesian": 4, "joints": 2}
    assert planning_seeds == [{"planned_gripper_width": 0.10}] * 12
    assert result.selected_path is not None
    assert len(result.selected_path.stages) == 4


def test_unreliable_geometry_is_rejected_before_moveit(monkeypatch):
    from mcp_server.grasping import pipeline

    geometry = _geometry().to_dict()
    geometry["quality"] = {"reliable": False}
    monkeypatch.setattr(
        pipeline.motion,
        "solve_pose_ik",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("IK must not run")
        ),
    )
    result = plan_block_grasp(_Bridge(), geometry)
    assert not result.ok
    assert "not reliable" in result.error


def test_upright_cylinder_pipeline_fully_plans_horizontal_candidates(monkeypatch):
    from mcp_server.grasping import pipeline

    calls = {"ik": 0, "pose": 0, "cartesian": 0, "joints": 0}

    def solve(*_args, **_kwargs):
        calls["ik"] += 1
        return ComponentResult.success(
            joints={f"joint{index}": 0.0 for index in range(1, 8)}
        )

    def planned(kind):
        def call(*_args, **_kwargs):
            calls[kind] += 1
            return ComponentResult.success(plan=_plan())
        return call

    monkeypatch.setattr(pipeline.motion, "solve_pose_ik", solve)
    monkeypatch.setattr(pipeline.motion, "plan_to_pose", planned("pose"))
    monkeypatch.setattr(
        pipeline.motion, "plan_cartesian", planned("cartesian")
    )
    monkeypatch.setattr(pipeline.motion, "plan_joints", planned("joints"))

    result = plan_cylinder_grasp(_Bridge(), _cylinder_geometry())
    assert result.ok
    assert len(result.candidates) == 10
    assert result.selected_path.candidate.tilt_deg == 90.0
    assert result.selected_path.candidate.ranking_mode == "shape_first"
    assert calls == {"ik": 10, "pose": 2, "cartesian": 4, "joints": 2}


def test_joint7_path_margin_checks_limit_but_allows_safe_observation_detour():
    def plan(points_deg, start_deg=None):
        points = [
            SimpleNamespace(positions=[math.radians(value)])
            for value in points_deg
        ]
        return SimpleNamespace(
            start_state=SimpleNamespace(
                joint_state=SimpleNamespace(
                    name=["joint7"],
                    position=[math.radians(
                        points_deg[0] if start_deg is None else start_deg
                    )],
                ),
            ),
            trajectory=SimpleNamespace(
                joint_trajectory=SimpleNamespace(
                    joint_names=["joint7"],
                    points=points,
                )
            )
        )

    plans = (
        plan([75.0, 65.0, 55.0], start_deg=80.0),
        plan([55.0, 50.0]),
        plan([50.0, 45.0]),
        plan([45.0, 50.0]),
    )
    assert _minimum_joint7_margin_deg(plans, 85.0) == pytest.approx(5.0)
    unsafe_retreat = plans[:2] + (plan([50.0, 65.0]), plans[3])
    assert _minimum_joint7_margin_deg(
        unsafe_retreat, 85.0
    ) == pytest.approx(5.0)
    limit_exceeded = (
        plan([85.2, 55.0], start_deg=80.0),
        *plans[1:],
    )
    exceeded_margin = _minimum_joint7_margin_deg(limit_exceeded, 85.0)
    assert exceeded_margin == pytest.approx(-0.2)
    assert _joint7_path_limit_error(
        exceeded_margin,
        85.0,
        path_label="complete grasp path",
    ) == (
        "complete grasp path reaches |joint7|=85.2deg, exceeding "
        "the +/-85.0deg software limit"
    )
    cumulative_observation_detour = (
        plan([80.8, 81.6, 55.0], start_deg=80.0),
        *plans[1:],
    )
    assert _minimum_joint7_margin_deg(
        cumulative_observation_detour, 85.0
    ) == pytest.approx(3.4)
    assert _minimum_joint7_margin_deg(
        (plan([50.0, 65.0], start_deg=50.0),),
        75.0,
    ) == pytest.approx(10.0)
