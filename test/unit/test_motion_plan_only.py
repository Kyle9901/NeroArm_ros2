import math
from types import SimpleNamespace

import pytest
from moveit_msgs.msg import MoveItErrorCodes, RobotState, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from mcp_server.ros.motion import MotionControllerMixin, PlannedMotion


class Future:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class Handle:
    accepted = True

    def __init__(self, action_result):
        self._result = Future(action_result)

    def get_result_async(self):
        return self._result


class ActionClient:
    def __init__(self, action_result):
        self.action_result = action_result
        self.last_goal = None

    def send_goal_async(self, goal):
        self.last_goal = goal
        return Future(Handle(self.action_result))


class ServiceClient:
    def __init__(self, response):
        self.response = response
        self.last_request = None

    def call_async(self, request):
        self.last_request = request
        return Future(self.response)


class IKServiceClient(ServiceClient):
    def __init__(self, response, *, ready=True):
        super().__init__(response)
        self.ready = ready
        self.wait_timeout = None

    def service_is_ready(self):
        return self.ready

    def wait_for_service(self, timeout_sec):
        self.wait_timeout = timeout_sec
        return True


class FakeController(MotionControllerMixin):
    def __init__(self):
        self.params = {
            "grasp_quat": [0.0, 0.0, 0.0, 1.0],
            "planning_group": "agx_arm",
            "num_planning_attempts": 1,
            "planning_time": 2.0,
            "velocity_scaling": 0.3,
            "accel_scaling": 0.2,
            "base_frame": "base_link",
            "tcp_link": "tcp_link",
            "pos_tolerance": 0.005,
            "ori_tolerance": 0.02,
            "cartesian_eef_step": 0.01,
            "cartesian_jump_threshold": 0.0,
            "cartesian_min_fraction": 0.95,
            "planned_start_tolerance_rad": 0.02,
        }
        self.tracked = []

    def _get_param(self, name):
        return self.params[name]

    def _build_robot_state(self):
        state = RobotState()
        state.joint_state.name = [f"joint{index}" for index in range(1, 8)]
        state.joint_state.position = [0.0] * 7
        return state

    def workspace_check(self, _x, _y):
        return True

    def _spin_until(self, _future, _timeout):
        return True

    def _track_goal(self, future):
        self.tracked.append(future)


def _trajectory():
    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.joint_names = [
        f"joint{index}" for index in range(1, 8)
    ]
    point = JointTrajectoryPoint()
    point.positions = [0.1 * index for index in range(1, 8)]
    trajectory.joint_trajectory.points.append(point)
    return trajectory


def test_pose_plan_only_sets_move_group_flag_and_returns_terminal_state():
    controller = FakeController()
    trajectory = _trajectory()
    action_result = SimpleNamespace(
        result=SimpleNamespace(
            error_code=SimpleNamespace(val=MoveItErrorCodes.SUCCESS),
            planned_trajectory=trajectory,
            trajectory_start=controller._build_robot_state(),
            planning_time=0.12,
        )
    )
    controller._move_group_ac = ActionClient(action_result)

    ok, message, plan = controller.plan_to_pose(
        -0.35, 0.0, 0.2, timeout=1.0
    )

    assert ok is True
    assert message == "ok"
    assert controller._move_group_ac.last_goal.planning_options.plan_only is True
    assert plan.trajectory is trajectory
    assert plan.planning_time == 0.12
    assert plan.terminal_joints["joint7"] == pytest.approx(0.7)


def test_cartesian_plan_returns_solution_without_execution_action():
    controller = FakeController()
    trajectory = _trajectory()
    response = SimpleNamespace(
        error_code=SimpleNamespace(val=MoveItErrorCodes.SUCCESS),
        fraction=1.0,
        solution=trajectory,
    )
    controller._cartesian_cli = ServiceClient(response)

    ok, message, plan = controller.plan_cartesian(
        -0.35, 0.0, 0.2, timeout=1.0
    )

    assert ok is True
    assert message == "ok"
    assert controller._cartesian_cli.last_request.avoid_collisions is True
    assert plan.trajectory is trajectory
    assert plan.fraction == 1.0
    assert plan.terminal_joints["joint7"] == pytest.approx(0.7)


def test_joint_plan_only_uses_explicit_start_state_and_never_executes():
    controller = FakeController()
    trajectory = _trajectory()
    explicit_start = controller._build_robot_state()
    explicit_start.joint_state.position = [0.2] * 7
    action_result = SimpleNamespace(
        result=SimpleNamespace(
            error_code=SimpleNamespace(val=MoveItErrorCodes.SUCCESS),
            planned_trajectory=trajectory,
            trajectory_start=explicit_start,
            planning_time=0.08,
        )
    )
    controller._move_group_ac = ActionClient(action_result)

    ok, message, plan = controller.plan_joints(
        [0.0, -20.0, 0.0, 80.0, 0.0, 0.0, 50.0],
        timeout=1.0,
        start_state=explicit_start,
    )

    goal = controller._move_group_ac.last_goal
    assert ok is True
    assert message == "ok"
    assert goal.planning_options.plan_only is True
    assert list(goal.request.start_state.joint_state.position) == [0.2] * 7
    assert [
        constraint.position
        for constraint in goal.request.goal_constraints[0].joint_constraints
    ] == pytest.approx(
        [0.0, math.radians(-20.0), 0.0, math.radians(80.0),
         0.0, 0.0, math.radians(50.0)]
    )
    assert plan.terminal_joints["joint7"] == pytest.approx(0.7)


def test_execute_planned_is_a_separate_explicit_step():
    controller = FakeController()
    trajectory = _trajectory()
    action_result = SimpleNamespace(
        result=SimpleNamespace(
            error_code=SimpleNamespace(val=MoveItErrorCodes.SUCCESS)
        )
    )
    controller._execute_ac = ActionClient(action_result)

    ok, message = controller.execute_planned(trajectory, timeout=1.0)

    assert ok is True
    assert message == "ok"
    assert controller._execute_ac.last_goal.trajectory is trajectory


def test_ready_ik_service_keeps_the_short_check_timeout_budget():
    controller = FakeController()
    response = SimpleNamespace(
        error_code=SimpleNamespace(val=MoveItErrorCodes.SUCCESS),
        solution=controller._build_robot_state(),
    )
    controller._ik_cli = IKServiceClient(response, ready=True)

    ok, message, joints = controller.solve_pose_ik(
        -0.35, 0.0, 0.2, timeout=0.05
    )

    request = controller._ik_cli.last_request
    requested_timeout = (
        request.ik_request.timeout.sec
        + request.ik_request.timeout.nanosec / 1e9
    )
    assert ok is True
    assert message == "ok"
    assert joints is not None
    assert requested_timeout > 0.04
    assert controller._ik_cli.wait_timeout is None


def test_planning_state_overrides_gripper_to_execution_open_width():
    controller = FakeController()
    state = controller._build_robot_state()
    state.joint_state.name.extend(["gripper_joint1", "gripper_joint2"])
    state.joint_state.position.extend([0.01, -0.01])
    controller._build_robot_state = lambda: state

    planning_state = controller.robot_state_with_gripper_width(0.10)
    positions = dict(zip(
        planning_state.joint_state.name,
        planning_state.joint_state.position,
    ))

    assert positions["gripper_joint1"] == pytest.approx(0.05)
    assert positions["gripper_joint2"] == pytest.approx(-0.05)
    assert planning_state.is_diff is False
    assert list(state.joint_state.position[-2:]) == [0.01, -0.01]


def test_planning_state_can_override_gripper_on_an_explicit_future_arm_state():
    controller = FakeController()
    future_state = controller._build_robot_state()
    future_state.joint_state.position[3] = 0.4

    planning_state = controller.robot_state_with_gripper_width(
        0.10,
        base_state=future_state,
    )
    positions = dict(zip(
        planning_state.joint_state.name,
        planning_state.joint_state.position,
    ))

    assert positions["joint4"] == pytest.approx(0.4)
    assert positions["gripper_joint1"] == pytest.approx(0.05)
    assert "gripper_joint1" not in future_state.joint_state.name


def test_execute_planned_rejects_stale_start_state_before_sending():
    controller = FakeController()
    actual = controller._build_robot_state()
    actual.joint_state.position[3] = 0.2
    controller._build_robot_state = lambda: actual
    controller._execute_ac = ActionClient(
        SimpleNamespace(
            result=SimpleNamespace(
                error_code=SimpleNamespace(val=MoveItErrorCodes.SUCCESS)
            )
        )
    )
    expected = RobotState()
    expected.joint_state.name = [f"joint{index}" for index in range(1, 8)]
    expected.joint_state.position = [0.0] * 7
    plan = PlannedMotion(
        trajectory=_trajectory(),
        start_state=expected,
        end_state=expected,
    )

    ok, message = controller.execute_planned(plan, timeout=1.0)

    assert ok is False
    assert "start state mismatch" in message
    assert controller._execute_ac.last_goal is None
