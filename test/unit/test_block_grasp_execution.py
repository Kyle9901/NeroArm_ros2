import math
from types import SimpleNamespace

from mcp_server.components.base import ComponentResult
from mcp_server.grasping.pipeline import PlannedGraspPath
from mcp_server.models import GraspCandidate
from mcp_server.skills import manipulation
from mcp_server.skills.base import GraspGeometry
from mcp_server.visualization import CandidateMarkerStatus


def _candidate():
    return GraspCandidate(
        pose_xyz=(-0.35, -0.10, 0.025),
        pose_quat_xyzw=(0.0, 1.0, 0.0, 0.0),
        approach_vector=(0.0, 0.0, -1.0),
        pregrasp_distance=0.08,
        retreat_vector=(0.0, 0.0, 1.0),
        retreat_distance=0.08,
        gripper_width=0.05,
        tilt_deg=0.0,
        edge_axis=(1.0, 0.0, 0.0),
        score=1.0,
        source="candidate-a",
    )


def _path():
    return PlannedGraspPath(
        candidate=_candidate(),
        to_pregrasp="pregrasp-plan",
        approach="approach-plan",
        retreat="retreat-plan",
        to_carry="carry-plan",
        total_joint_motion_rad=1.0,
    )


def _geometry():
    return GraspGeometry(
        fingertip_depth=0.04,
        approach_height=0.08,
        safe_height=0.22,
        gripper_open=0.10,
        gripper_close=0.02,
        hold_margin=0.005,
        descent_vel=0.2,
        descent_accel=0.05,
    )


class _Node:
    @staticmethod
    def _get_param(name):
        assert name == "planned_start_tolerance_rad"
        return 0.02

    @staticmethod
    def robot_state_with_gripper_width(width, *, base_state=None):
        return {
            "planned_gripper_width": width,
            "base_state": base_state,
        }


class _Bridge:
    def __init__(self):
        self.context = {}
        self.holding = False
        self.node = _Node()

    def get_task_context(self):
        return dict(self.context)

    def update_task_context(self, **changes):
        self.context.update(changes)

    def set_holding(self, value):
        self.holding = value

    def get_holding(self):
        return self.holding

    def get_current_tcp_pose(self, timeout=1.0):
        assert timeout > 0
        return {
            "position": [-0.35, -0.10, 0.025],
            "quaternion": [0.0, 1.0, 0.0, 0.0],
        }

    def get_block_xy_max_spread(self):
        return 0.015

    def get_grasp_candidate_timeout(self):
        return 8.0

    def get_joint7_soft_limit_deg(self):
        return 75.0

    def get_joint7_min_margin_deg(self):
        return 15.0

    def get_reverse_branch_tolerance_rad(self):
        return 0.10

    def get_fingertip_depth(self):
        return 0.04

    def get_approach_height(self):
        return 0.08

    def get_safe_height(self):
        return 0.22

    def get_gripper_open_width(self):
        return 0.10

    def get_gripper_close_width(self):
        return 0.02

    def get_descent_velocity_scaling(self):
        return 0.2

    def get_descent_accel_scaling(self):
        return 0.05


def _planning():
    evaluation = SimpleNamespace(
        candidate_id="candidate-a",
        status=CandidateMarkerStatus.SELECTED,
        reason="selected",
        joint7_margin_deg=30.0,
    )
    batch = SimpleNamespace(
        cheap_checks=10,
        full_plans=2,
        elapsed_s=2.0,
        evaluations=[evaluation],
    )
    return SimpleNamespace(
        ok=True,
        selected_path=_path(),
        batch=batch,
        candidates=tuple([_candidate()] * 10),
        error=None,
    )


def _joint_feedback(width):
    sequence = 0

    def read(_bridge, **_kwargs):
        nonlocal sequence
        sequence += 1
        return ComponentResult.success(
            gripper_width=width,
            sequence=sequence,
        )

    return read


def test_success_executes_selected_path_then_stays_holding_in_carry(monkeypatch):
    bridge = _Bridge()
    calls = []
    monkeypatch.setattr(
        manipulation,
        "plan_block_grasp",
        lambda *_args, **_kwargs: _planning(),
    )
    monkeypatch.setattr(manipulation.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        manipulation.motion,
        "control_gripper",
        lambda _bridge, width, **_kwargs: (
            calls.append(("gripper", width)) or ComponentResult.success()
        ),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "execute_planned",
        lambda _bridge, plan, **_kwargs: (
            calls.append(("execute", plan)) or ComponentResult.success()
        ),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "read_joint_state",
        _joint_feedback(0.05),
    )

    result = manipulation._grasp_block(
        bridge, -0.35, -0.10, 0.05, {"quality": {"reliable": True}}, _geometry()
    )
    assert result.ok
    assert result.holding is True
    assert bridge.holding is True
    assert calls == [
        ("gripper", 0.10),
        ("execute", "pregrasp-plan"),
        ("execute", "approach-plan"),
        ("gripper", 0.02),
        ("execute", "retreat-plan"),
        ("execute", "carry-plan"),
    ]


def test_upright_bottle_logging_accepts_serialized_geometry(monkeypatch, capsys):
    bridge = _Bridge()
    monkeypatch.setattr(
        manipulation,
        "plan_transparent_bottle_grasp",
        lambda *_args, **_kwargs: _planning(),
    )
    monkeypatch.setattr(manipulation.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        manipulation.motion,
        "control_gripper",
        lambda *_args, **_kwargs: ComponentResult.success(),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "execute_planned",
        lambda *_args, **_kwargs: ComponentResult.success(),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "read_joint_state",
        _joint_feedback(0.05),
    )
    geometry = {
        "surface": {"x": -0.35, "y": -0.10, "z": 0.025},
        "center": {"x": -0.35, "y": -0.10, "z": 0.025},
        "orientation_class": "upright",
        "quality": {"reliable": True},
    }

    result = manipulation._grasp_transparent_bottle(
        bridge, -0.35, -0.10, 0.025, geometry, _geometry(),
    )

    assert result.ok
    output = capsys.readouterr().out
    assert "visual_axis=(-0.3500,-0.1000,0.0250)" in output
    assert "reached_tcp=(-0.3500,-0.1000,0.0250)" in output


def test_unknown_holding_preserves_gripper_and_disables_retry(monkeypatch):
    bridge = _Bridge()
    gripper_commands = []
    monkeypatch.setattr(
        manipulation,
        "plan_block_grasp",
        lambda *_args, **_kwargs: _planning(),
    )
    monkeypatch.setattr(manipulation.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        manipulation.motion,
        "control_gripper",
        lambda _bridge, width, **_kwargs: (
            gripper_commands.append(width) or ComponentResult.success()
        ),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "execute_planned",
        lambda *_args, **_kwargs: ComponentResult.success(),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "read_joint_state",
        _joint_feedback(None),
    )

    result = manipulation._grasp_block(
        bridge, -0.35, -0.10, 0.05, {"quality": {"reliable": True}}, _geometry()
    )
    assert not result.ok
    assert result.holding is None
    assert result.retryable is False
    assert bridge.holding is None
    assert gripper_commands == [0.10, 0.02]


def test_empty_grasp_opens_retreats_and_excludes_candidate(monkeypatch):
    bridge = _Bridge()
    gripper_commands = []
    monkeypatch.setattr(
        manipulation,
        "plan_block_grasp",
        lambda *_args, **_kwargs: _planning(),
    )
    monkeypatch.setattr(manipulation.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        manipulation.motion,
        "control_gripper",
        lambda _bridge, width, **_kwargs: (
            gripper_commands.append(width) or ComponentResult.success()
        ),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "execute_planned",
        lambda *_args, **_kwargs: ComponentResult.success(),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "read_joint_state",
        _joint_feedback(0.02),
    )

    result = manipulation._grasp_block(
        bridge, -0.35, -0.10, 0.05, {"quality": {"reliable": True}}, _geometry()
    )
    assert not result.ok
    assert result.holding is False
    assert result.retryable is True
    assert bridge.context["rejected_candidate_ids"] == ["candidate-a"]
    assert gripper_commands == [0.10, 0.02, 0.10]


def test_retreat_fallback_replans_carry_instead_of_using_stale_plan(monkeypatch):
    bridge = _Bridge()
    calls = []

    def _execute(_bridge, plan, **_kwargs):
        calls.append(("execute", plan))
        if plan == "retreat-plan":
            return ComponentResult.failure("retreat execution failed")
        return ComponentResult.success()

    monkeypatch.setattr(manipulation.motion, "execute_planned", _execute)
    monkeypatch.setattr(
        manipulation.motion,
        "move_to_pose",
        lambda *_args, **_kwargs: (
            calls.append(("fallback_retreat", None))
            or ComponentResult.success()
        ),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "go_carry",
        lambda _bridge: (
            calls.append(("go_carry", None))
            or ComponentResult.success()
        ),
    )

    assert manipulation._retreat_with_gripper_state(bridge, _path()) is True
    assert calls == [
        ("execute", "retreat-plan"),
        ("fallback_retreat", None),
        ("go_carry", None),
    ]


def test_execution_timeout_does_not_send_a_second_arm_goal(monkeypatch):
    bridge = _Bridge()
    calls = []
    monkeypatch.setattr(
        manipulation,
        "plan_block_grasp",
        lambda *_args, **_kwargs: _planning(),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "control_gripper",
        lambda *_args, **_kwargs: ComponentResult.success(),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "execute_planned",
        lambda _bridge, plan, **_kwargs: (
            calls.append(("execute", plan))
            or ComponentResult.failure(
                "execute_trajectory result did not arrive in time",
                motion_state_unknown=True,
            )
        ),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "go_home",
        lambda _bridge: (
            calls.append(("go_home", None))
            or ComponentResult.success()
        ),
    )

    result = manipulation._grasp_block(
        bridge, -0.35, -0.10, 0.05,
        {"quality": {"reliable": True}},
        _geometry(),
    )

    assert not result.ok
    assert result.retryable is False
    assert result.data["motion_state_unknown"] is True
    assert calls == [("execute", "pregrasp-plan")]


def test_reverse_place_finishes_all_planning_before_first_execution(monkeypatch):
    bridge = _Bridge()
    calls = []

    def planned(name, joint7_deg=0.0):
        state = SimpleNamespace(
            joint_state=SimpleNamespace(
                name=["joint7"],
                position=[math.radians(joint7_deg)],
            ),
        )
        return SimpleNamespace(
            end_state=f"{name}-end",
            start_state=state,
            terminal_joints={
                f"joint{index}": 0.0 for index in range(1, 8)
            },
            trajectory=SimpleNamespace(
                joint_trajectory=SimpleNamespace(
                    joint_names=["joint7"],
                    points=[
                        SimpleNamespace(
                            positions=[math.radians(joint7_deg)]
                        )
                    ],
                )
            ),
        )

    monkeypatch.setattr(
        manipulation.motion,
        "plan_to_pose",
        lambda *_args, **_kwargs: (
            calls.append(("plan", "to-retreat"))
            or ComponentResult.success(plan=planned("to-retreat"))
        ),
    )
    cartesian_plans = iter(("to-grasp", "to-pregrasp"))
    cartesian_start_states = []

    def plan_cartesian(*_args, **_kwargs):
        name = next(cartesian_plans)
        calls.append(("plan", name))
        cartesian_start_states.append(_kwargs.get("start_state"))
        return ComponentResult.success(plan=planned(name))

    monkeypatch.setattr(
        manipulation.motion, "plan_cartesian", plan_cartesian,
    )
    monkeypatch.setattr(
        manipulation.motion,
        "execute_planned",
        lambda _bridge, plan, **_kwargs: (
            calls.append(("execute", plan.end_state))
            or ComponentResult.success()
        ),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "control_gripper",
        lambda *_args, **_kwargs: (
            calls.append(("gripper", "open"))
            or ComponentResult.success()
        ),
    )

    result = manipulation._place_reverse(
        bridge,
        _candidate().to_dict(),
        -0.35,
        -0.10,
        0.05,
    )

    assert result.ok
    assert calls[:3] == [
        ("plan", "to-retreat"),
        ("plan", "to-grasp"),
        ("plan", "to-pregrasp"),
    ]
    assert calls[3:] == [
        ("execute", "to-retreat-end"),
        ("execute", "to-grasp-end"),
        ("gripper", "open"),
        ("execute", "to-pregrasp-end"),
    ]
    assert cartesian_start_states[1] == {
        "planned_gripper_width": 0.10,
        "base_state": "to-grasp-end",
    }


def _reverse_plan(joint7_deg=0.0, joint1_rad=0.0):
    state = SimpleNamespace(
        joint_state=SimpleNamespace(
            name=["joint7"],
            position=[math.radians(joint7_deg)],
        ),
    )
    return SimpleNamespace(
        end_state=state,
        start_state=state,
        terminal_joints={
            "joint1": float(joint1_rad),
            **{f"joint{index}": 0.0 for index in range(2, 7)},
            "joint7": math.radians(joint7_deg),
        },
        trajectory=SimpleNamespace(
            joint_trajectory=SimpleNamespace(
                joint_names=["joint7"],
                points=[
                    SimpleNamespace(positions=[math.radians(joint7_deg)])
                ],
            )
        ),
    )


def test_reverse_place_reenters_saved_retreat_joint_branch(monkeypatch):
    bridge = _Bridge()
    calls = []
    monkeypatch.setattr(
        manipulation.motion,
        "plan_joints",
        lambda _bridge, target, **_kwargs: (
            calls.append(("joint_target", target))
            or ComponentResult.success(plan=_reverse_plan())
        ),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "plan_to_pose",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("saved branch must use a joint target")
        ),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "plan_cartesian",
        lambda *_args, **_kwargs: ComponentResult.success(
            plan=_reverse_plan()
        ),
    )
    targets = {
        name: [0.0] * 7
        for name in ("pregrasp", "grasp", "retreat")
    }

    plans, error = manipulation._plan_reverse_place(
        bridge,
        _candidate(),
        8.0,
        targets,
    )

    assert error is None
    assert plans is not None
    assert calls == [("joint_target", [0.0] * 7)]


def test_reverse_place_accepts_small_same_branch_endpoint_drift(monkeypatch):
    bridge = _Bridge()
    monkeypatch.setattr(
        manipulation.motion,
        "plan_joints",
        lambda *_args, **_kwargs: ComponentResult.success(
            plan=_reverse_plan()
        ),
    )
    cartesian = iter((
        _reverse_plan(joint1_rad=0.05),
        _reverse_plan(joint1_rad=0.05),
    ))
    monkeypatch.setattr(
        manipulation.motion,
        "plan_cartesian",
        lambda *_args, **_kwargs: ComponentResult.success(
            plan=next(cartesian)
        ),
    )
    targets = {
        name: [0.0] * 7
        for name in ("pregrasp", "grasp", "retreat")
    }

    plans, error = manipulation._plan_reverse_place(
        bridge,
        _candidate(),
        8.0,
        targets,
    )

    assert error is None
    assert plans is not None


def test_reverse_retreat_accepts_saved_retreat_branch_at_same_tcp_pose(
        monkeypatch):
    bridge = _Bridge()
    monkeypatch.setattr(
        manipulation.motion,
        "plan_joints",
        lambda *_args, **_kwargs: ComponentResult.success(
            plan=_reverse_plan(15.2)
        ),
    )
    cartesian = iter((
        _reverse_plan(0.0),
        _reverse_plan(15.2),
    ))
    monkeypatch.setattr(
        manipulation.motion,
        "plan_cartesian",
        lambda *_args, **_kwargs: ComponentResult.success(
            plan=next(cartesian)
        ),
    )
    targets = {
        "pregrasp": [0.0] * 6 + [22.7],
        "grasp": [0.0] * 7,
        "retreat": [0.0] * 6 + [15.2],
    }

    plans, error = manipulation._plan_reverse_place(
        bridge,
        _candidate(),
        8.0,
        targets,
    )

    assert error is None
    assert plans is not None


def test_reverse_place_reports_joint_that_exceeds_branch_tolerance(monkeypatch):
    bridge = _Bridge()
    monkeypatch.setattr(
        manipulation.motion,
        "plan_joints",
        lambda *_args, **_kwargs: ComponentResult.success(
            plan=_reverse_plan()
        ),
    )
    cartesian = iter((
        _reverse_plan(joint1_rad=0.05),
        _reverse_plan(joint1_rad=0.15),
    ))
    monkeypatch.setattr(
        manipulation.motion,
        "plan_cartesian",
        lambda *_args, **_kwargs: ComponentResult.success(
            plan=next(cartesian)
        ),
    )
    targets = {
        name: [0.0] * 7
        for name in ("pregrasp", "grasp", "retreat")
    }

    plans, error = manipulation._plan_reverse_place(
        bridge,
        _candidate(),
        8.0,
        targets,
    )

    assert plans is None
    assert "reverse retreat" in error
    assert "joint1 differs by 8.6deg" in error
    assert "limit 5.7deg" in error


def test_reverse_place_rejects_joint7_path_without_motion(monkeypatch):
    bridge = _Bridge()
    monkeypatch.setattr(
        manipulation.motion,
        "plan_to_pose",
        lambda *_args, **_kwargs: ComponentResult.success(
            plan=_reverse_plan(50.0)
        ),
    )
    cartesian = iter((_reverse_plan(76.0), _reverse_plan(50.0)))
    monkeypatch.setattr(
        manipulation.motion,
        "plan_cartesian",
        lambda *_args, **_kwargs: ComponentResult.success(
            plan=next(cartesian)
        ),
    )

    plans, error = manipulation._plan_reverse_place(
        bridge,
        _candidate(),
        8.0,
    )

    assert plans is None
    assert "reaches |joint7|=76.0deg" in error
    assert "exceeding the +/-75.0deg software limit" in error


def test_legacy_descent_timeout_never_sends_recovery_motion(monkeypatch):
    bridge = _Bridge()
    monkeypatch.setattr(
        manipulation.motion,
        "workspace_check",
        lambda *_args, **_kwargs: ComponentResult.success(),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "control_gripper",
        lambda *_args, **_kwargs: ComponentResult.success(),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "move_to_pose",
        lambda *_args, **_kwargs: ComponentResult.success(),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "move_cartesian",
        lambda *_args, **_kwargs: ComponentResult.failure(
            "execute_trajectory result did not arrive in time",
            motion_state_unknown=True,
        ),
    )
    monkeypatch.setattr(
        manipulation,
        "recover_to_safe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unknown arm state must not trigger recovery")
        ),
    )

    result = manipulation._grasp_legacy(
        bridge,
        -0.35,
        -0.10,
        0.05,
        [0.0, 1.0, 0.0, 0.0],
        {"local_desk_z": 0.0},
    )

    assert not result.ok
    assert result.retryable is False
    assert result.data["motion_state_unknown"] is True


def test_legacy_place_descent_timeout_never_sends_recovery_motion(monkeypatch):
    bridge = _Bridge()
    bridge.holding = True
    monkeypatch.setattr(
        manipulation.motion,
        "workspace_check",
        lambda *_args, **_kwargs: ComponentResult.success(),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "move_to_pose",
        lambda *_args, **_kwargs: ComponentResult.success(),
    )
    monkeypatch.setattr(
        manipulation.motion,
        "move_cartesian",
        lambda *_args, **_kwargs: ComponentResult.failure(
            "execute_trajectory result did not arrive in time",
            motion_state_unknown=True,
        ),
    )
    monkeypatch.setattr(
        manipulation,
        "recover_to_safe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unknown arm state must not trigger recovery")
        ),
    )

    result = manipulation._place_legacy(
        bridge,
        -0.35,
        -0.10,
        0.05,
        [0.0, 1.0, 0.0, 0.0],
    )

    assert not result.ok
    assert result.retryable is False
    assert result.holding is True
    assert result.data["motion_state_unknown"] is True


def test_place_requires_confirmed_holding():
    bridge = _Bridge()
    bridge.holding = None

    result = manipulation.place_object(
        bridge,
        -0.35,
        -0.10,
        0.05,
    )

    assert not result.ok
    assert result.failed_step == "holding_guard"
    assert result.holding is None


def test_grasp_router_enables_shape_candidates_for_block_and_cylinder(monkeypatch):
    bridge = _Bridge()
    calls = []
    monkeypatch.setattr(
        manipulation,
        "_grasp_block",
        lambda *_args, **_kwargs: (
            calls.append("block") or ComponentResult.success()
        ),
    )
    monkeypatch.setattr(
        manipulation,
        "_grasp_cylinder",
        lambda *_args, **_kwargs: (
            calls.append("cylinder") or ComponentResult.success()
        ),
    )
    monkeypatch.setattr(
        manipulation,
        "_grasp_transparent_bottle",
        lambda *_args, **_kwargs: (
            calls.append("transparent_bottle") or ComponentResult.success()
        ),
    )

    manipulation.grasp_object(
        bridge, -0.35, -0.1, 0.05,
        geometry={"quality": {"reliable": True}},
        target="红色物块",
    )
    manipulation.grasp_object(
        bridge, -0.35, -0.1, 0.05,
        geometry={"quality": {"reliable": True}},
        target="红色瓶子",
    )
    manipulation.grasp_object(
        bridge, -0.35, -0.1, 0.05,
        geometry={"quality": {"reliable": True}},
        target="矿泉水瓶",
    )

    assert calls == ["block", "cylinder", "transparent_bottle"]
