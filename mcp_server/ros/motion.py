"""MoveIt planning, execution, workspace checks, and goal tracking."""

import copy
import math
import time
from dataclasses import dataclass
from typing import Any

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs import action as _ma
from moveit_msgs import srv as _ms
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, RobotState
from rclpy.action import ActionClient
from shape_msgs.msg import SolidPrimitive

from .hardware import ARM_JOINT_NAMES, GripperController
from .futures import wait_for_future


@dataclass(frozen=True)
class PlannedMotion:
    """A trajectory returned by MoveIt without executing it.

    ``end_state`` makes consecutive plan-only checks composable: the next
    segment can use it as its explicit start state instead of incorrectly
    planning every segment from the live robot state.
    """

    trajectory: Any
    start_state: RobotState
    end_state: RobotState
    planning_time: float = 0.0
    fraction: float | None = None

    @property
    def terminal_joints(self) -> dict[str, float]:
        message = self.end_state.joint_state
        return {
            name: float(position)
            for name, position in zip(message.name, message.position)
        }


_ERROR_HINTS = {
    -1: "Planning failed; target may be unreachable or in collision",
    -2: "Planning timed out",
    -3: "Planner returned an invalid trajectory",
    -4: "Motion was aborted during execution",
    -5: "Robot is already at the target pose",
}

def _friendly_error(prefix: str, code: int) -> str:
    hint = _ERROR_HINTS.get(code)
    return f"{prefix} (code={code}). Hint: {hint}" if hint else f"{prefix} (code={code})"


class MotionControllerMixin:
    def _init_moveit_clients(self):
        self.gripper = GripperController(self, self.cb_group)
        self._move_group_ac = ActionClient(self, _ma.MoveGroup, "/move_action",
                                           callback_group=self.cb_group)
        self._execute_ac = ActionClient(self, _ma.ExecuteTrajectory, "/execute_trajectory",
                                         callback_group=self.cb_group)
        self._cartesian_cli = self.create_client(_ms.GetCartesianPath, "/compute_cartesian_path",
                                                  callback_group=self.cb_group)
        self._ik_cli = self.create_client(
            _ms.GetPositionIK, "/compute_ik", callback_group=self.cb_group
        )

    # ─────────────────────────── wait helpers ─────────────────────
    def wait_servers(self, timeout=15.0):
        ok = True
        for name, ac in [("gripper", self.gripper.client), ("move_group", self._move_group_ac),
                         ("execute_traj", self._execute_ac)]:
            if not ac.wait_for_server(timeout):
                self.get_logger().error(f"{name} action server not available")
                ok = False
        if not self._cartesian_cli.wait_for_service(timeout):
            self.get_logger().error("cartesian_path service not available")
            ok = False
        if ok:
            self.get_logger().info("all MoveIt / gripper servers available")

    # ─────────────────────────── desk collision object ──────────────
    def get_joint_state(self) -> dict:
        return self.joint_states.as_dict()

    def wait_for_joint_state_after(self, sequence: int, timeout: float) -> dict | None:
        if not self.joint_states.wait_for_newer(sequence, timeout):
            return None
        return self.joint_states.as_dict()

    def workspace_check(self, x, y) -> bool:
        return (self._workspace_x_min <= x <= self._workspace_x_max and
                self._workspace_y_min <= y <= self._workspace_y_max)

    # ─────────────────────────── motion implementation ────────────
    def _spin_until(self, future, timeout_sec):
        """Compatibility wrapper for ROS adapters sharing the node."""
        return wait_for_future(future, timeout_sec, context=self.context)

    @property
    def _workspace_x_min(self):
        return self.get_parameter("workspace_x_min").value

    @property
    def _workspace_x_max(self):
        return self.get_parameter("workspace_x_max").value

    @property
    def _workspace_y_min(self):
        return self.get_parameter("workspace_y_min").value

    @property
    def _workspace_y_max(self):
        return self.get_parameter("workspace_y_max").value

    def _get_param(self, name):
        return self.get_parameter(name).value

    def _build_robot_state(self) -> RobotState:
        js = self.joint_states.latest_message()
        rs = RobotState()
        if js is not None:
            rs.joint_state = js
        else:
            rs.is_diff = True
        return rs

    def robot_state_with_gripper_width(
        self,
        width: float,
        *,
        base_state: RobotState | None = None,
    ) -> RobotState:
        """Return the live arm state with an explicit parallel-jaw opening.

        Block grasp planning happens before the physical gripper opens.  MoveIt
        must nevertheless collision-check the wider, execution-time gripper
        geometry, so both gripper joints are overridden in the planning seed.
        """
        if not math.isfinite(width) or width < 0.0:
            raise ValueError("gripper width must be finite and non-negative")
        state = copy.deepcopy(
            base_state if base_state is not None else self._build_robot_state()
        )
        names = list(state.joint_state.name)
        positions = list(state.joint_state.position)
        targets = {
            "gripper_joint1": float(width) * 0.5,
            "gripper_joint2": -float(width) * 0.5,
        }
        by_name = dict(zip(names, positions))
        by_name.update(targets)
        for name in targets:
            if name not in names:
                names.append(name)
        state.joint_state.name = names
        state.joint_state.position = [by_name[name] for name in names]
        state.is_diff = False
        return state

    def _trajectory_end_state(self, start_state: RobotState, trajectory) -> RobotState:
        """Build a complete seed state from a planned trajectory's last point."""
        end_state = copy.deepcopy(start_state)
        joint_trajectory = getattr(trajectory, "joint_trajectory", None)
        points = getattr(joint_trajectory, "points", ()) if joint_trajectory else ()
        names = list(getattr(joint_trajectory, "joint_names", ())) if joint_trajectory else []
        if not points or not names:
            return end_state

        terminal = {
            name: float(position)
            for name, position in zip(names, points[-1].positions)
        }
        state_names = list(end_state.joint_state.name)
        state_positions = list(end_state.joint_state.position)
        if state_names:
            by_name = dict(zip(state_names, state_positions))
            by_name.update(terminal)
            end_state.joint_state.position = [by_name[name] for name in state_names]
            missing = [name for name in names if name not in state_names]
            end_state.joint_state.name.extend(missing)
            end_state.joint_state.position.extend(terminal[name] for name in missing)
        else:
            end_state.joint_state.name = names
            end_state.joint_state.position = [
                terminal[name] for name in names
            ]
        end_state.is_diff = False
        return end_state

    def _normalize_quat(self, quat):
        norm = math.sqrt(sum(v * v for v in quat))
        if norm < 1e-6:
            raise ValueError("quaternion norm too small")
        return [v / norm for v in quat]

    def _make_pose_constraints(self, x, y, z, quat) -> Constraints:
        c = Constraints()
        c.name = "target_pose"
        ptol = self._get_param("pos_tolerance")
        otol = self._get_param("ori_tolerance")
        base = self._get_param("base_frame")
        tcp = self._get_param("tcp_link")
        from geometry_msgs.msg import Vector3
        from moveit_msgs.msg import PositionConstraint as PC, OrientationConstraint as OC

        pc = PC()
        pc.header.frame_id = base
        pc.link_name = tcp
        pc.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        prim = SolidPrimitive(); prim.type = SolidPrimitive.SPHERE; prim.dimensions = [ptol]
        pc.constraint_region.primitives.append(prim)
        rp = Pose(); rp.position.x = float(x); rp.position.y = float(y); rp.position.z = float(z)
        rp.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(rp)
        pc.weight = 1.0
        c.position_constraints.append(pc)

        oc = OC()
        oc.header.frame_id = base; oc.link_name = tcp
        oc.orientation.x = float(quat[0]); oc.orientation.y = float(quat[1])
        oc.orientation.z = float(quat[2]); oc.orientation.w = float(quat[3])
        oc.absolute_x_axis_tolerance = otol; oc.absolute_y_axis_tolerance = otol
        oc.absolute_z_axis_tolerance = otol
        oc.weight = 1.0
        c.orientation_constraints.append(oc)
        return c

    def _request_move_group_plan(
        self,
        goal,
        timeout: float,
        *,
        label: str,
    ):
        """Submit a plan-only MoveGroup goal within one wall-clock budget."""
        started = time.monotonic()
        sent = self._move_group_ac.send_goal_async(goal)
        self._track_goal(sent)
        if not self._spin_until(sent, min(2.0, timeout)):
            return False, f"{label} did not receive a response", None
        handle = sent.result()
        if handle is None or not handle.accepted:
            return False, f"{label} rejected", None
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0.0:
            return False, f"{label} timed out after {timeout:.2f}s", None
        result_future = handle.get_result_async()
        if not self._spin_until(result_future, remaining):
            return False, f"{label} timed out after {timeout:.2f}s", None
        wrapped_result = result_future.result()
        if wrapped_result is None:
            return False, f"{label} returned no result", None
        result = wrapped_result.result
        code = result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            return False, _friendly_error(f"{label} failed", code), None
        return True, "ok", result

    # ── motion primitives ──

    def move_joints(self, joint_angles_deg: list[float], timeout=20.0,
                    velocity_override: float | None = None,
                    accel_override: float | None = None) -> tuple[bool, str]:
        """Move to joint-space target.  joint_angles_deg: 7 floats in degrees."""
        if len(joint_angles_deg) != len(ARM_JOINT_NAMES):
            return False, f"expected {len(ARM_JOINT_NAMES)} joint angles, got {len(joint_angles_deg)}"

        goal = _ma.MoveGroup.Goal()
        req = goal.request
        req.group_name = self._get_param("planning_group")
        req.num_planning_attempts = self._get_param("num_planning_attempts")
        req.allowed_planning_time = self._get_param("planning_time")
        req.max_velocity_scaling_factor = velocity_override if velocity_override is not None else self._get_param("velocity_scaling")
        req.max_acceleration_scaling_factor = accel_override if accel_override is not None else self._get_param("accel_scaling")
        req.start_state = self._build_robot_state()

        c = Constraints()
        c.name = "joint_target"
        for name, deg in zip(ARM_JOINT_NAMES, joint_angles_deg):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = math.radians(float(deg))
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints.append(c)
        goal.planning_options.plan_only = False

        sf = self._move_group_ac.send_goal_async(goal)
        self._track_goal(sf)
        if not self._spin_until(sf, timeout):
            return False, ("MoveGroup action server did not respond in time. "
                                "Check that MoveIt 2 (move_group) is running. "
                                "Run: ros2 run moveit_ros_move_group move_group --ros-args --params-file <config>")
        gh = sf.result()
        if gh is None or not gh.accepted:
            return False, ("MoveGroup goal rejected — the planner refused the request. "
                                "Possibly: (1) target is unreachable or in collision, "
                                "(2) robot is already at the target, "
                                "(3) start state is invalid. Verify the live robot state "
                                "before submitting another motion")
        rf = gh.get_result_async()
        max_plan = self._get_param("num_planning_attempts") * self._get_param("planning_time")
        if not self._spin_until(rf, timeout + max_plan):
            return False, ("MoveGroup result did not arrive in time "
                                f"(timeout={timeout + max_plan:.0f}s). "
                                "The motion may still be executing — check the robot's position. "
                                "If the robot is moving, wait for it to stop then call arm_get_status")
        code = rf.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            return False, _friendly_error("MoveGroup planning failed", code)
        return True, "ok"

    def plan_joints(
        self,
        joint_angles_deg: list[float],
        timeout: float = 3.0,
        *,
        start_state: RobotState | None = None,
        velocity_override: float | None = None,
        accel_override: float | None = None,
    ) -> tuple[bool, str, PlannedMotion | None]:
        """Plan a seven-joint target without executing it."""
        if timeout <= 0.0:
            return False, "plan_joints timeout must be positive", None
        if len(joint_angles_deg) != len(ARM_JOINT_NAMES):
            return False, (
                f"expected {len(ARM_JOINT_NAMES)} joint angles, "
                f"got {len(joint_angles_deg)}"
            ), None

        seed = (
            copy.deepcopy(start_state)
            if start_state is not None
            else self._build_robot_state()
        )
        goal = _ma.MoveGroup.Goal()
        request = goal.request
        request.group_name = self._get_param("planning_group")
        request.num_planning_attempts = self._get_param("num_planning_attempts")
        request.allowed_planning_time = min(
            float(self._get_param("planning_time")), max(0.05, float(timeout))
        )
        request.max_velocity_scaling_factor = (
            velocity_override
            if velocity_override is not None
            else self._get_param("velocity_scaling")
        )
        request.max_acceleration_scaling_factor = (
            accel_override
            if accel_override is not None
            else self._get_param("accel_scaling")
        )
        request.start_state = seed
        constraints = Constraints()
        constraints.name = "joint_target"
        for name, degrees in zip(ARM_JOINT_NAMES, joint_angles_deg):
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = math.radians(float(degrees))
            joint.tolerance_above = 0.01
            joint.tolerance_below = 0.01
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        request.goal_constraints.append(constraints)
        goal.planning_options.plan_only = True

        ok, message, result = self._request_move_group_plan(
            goal, timeout, label="MoveGroup joint plan-only"
        )
        if not ok or result is None:
            return False, message, None
        trajectory = result.planned_trajectory
        plan = PlannedMotion(
            trajectory=trajectory,
            start_state=copy.deepcopy(result.trajectory_start),
            end_state=self._trajectory_end_state(
                result.trajectory_start, trajectory
            ),
            planning_time=float(result.planning_time),
        )
        return True, "ok", plan

    def move_to_pose(self, x: float, y: float, z: float, quat: list[float] | None = None,
                     timeout=60.0, velocity_override: float | None = None,
                     accel_override: float | None = None) -> tuple[bool, str]:
        """Move to Cartesian pose via MoveGroup.  quat defaults to grasp_quat.
        timeout covers the full planning + execution cycle."""
        if quat is None:
            quat = list(self._get_param("grasp_quat"))
        quat = self._normalize_quat(quat)

        if not self.workspace_check(x, y):
            return False, (f"pose ({x:.3f},{y:.3f}) outside workspace "
                           f"[x: {self._workspace_x_min:.2f} to {self._workspace_x_max:.2f}, "
                           f"y: {self._workspace_y_min:.2f} to {self._workspace_y_max:.2f}]. "
                           f"Use arm_get_status to see workspace bounds, then pick a target inside them")

        goal = _ma.MoveGroup.Goal()
        req = goal.request
        req.group_name = self._get_param("planning_group")
        req.num_planning_attempts = self._get_param("num_planning_attempts")
        req.allowed_planning_time = self._get_param("planning_time")
        req.max_velocity_scaling_factor = velocity_override if velocity_override is not None else self._get_param("velocity_scaling")
        req.max_acceleration_scaling_factor = accel_override if accel_override is not None else self._get_param("accel_scaling")
        req.start_state = self._build_robot_state()
        req.goal_constraints.append(self._make_pose_constraints(x, y, z, quat))
        goal.planning_options.plan_only = False

        sf = self._move_group_ac.send_goal_async(goal)
        self._track_goal(sf)
        if not self._spin_until(sf, 5.0):
            return False, ("MoveGroup action server did not respond in time. "
                                "Check that MoveIt 2 (move_group) is running. "
                                "Run: ros2 run moveit_ros_move_group move_group --ros-args --params-file <config>")
        gh = sf.result()
        if gh is None or not gh.accepted:
            return False, ("MoveGroup goal rejected — the planner refused the request. "
                                "Possibly: (1) target is unreachable or in collision, "
                                "(2) robot is already at the target, "
                                "(3) start state is invalid. Verify the live robot state "
                                "before submitting another motion")
        rf = gh.get_result_async()
        # result timeout: planning (attempts * planning_time) + execution + buffer
        max_plan = self._get_param("num_planning_attempts") * self._get_param("planning_time")
        if not self._spin_until(rf, timeout + max_plan):
            return False, ("MoveGroup result did not arrive in time "
                                f"(timeout={timeout + max_plan:.0f}s). "
                                "The motion may still be executing — check the robot's position. "
                                "If the robot is moving, wait for it to stop then call arm_get_status")
        code = rf.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            return False, _friendly_error("MoveGroup planning failed", code)
        return True, "ok"

    def plan_to_pose(
        self,
        x: float,
        y: float,
        z: float,
        quat: list[float] | None = None,
        timeout: float = 3.0,
        *,
        start_state: RobotState | None = None,
        velocity_override: float | None = None,
        accel_override: float | None = None,
    ) -> tuple[bool, str, PlannedMotion | None]:
        """Plan a pose goal and return the trajectory without executing it."""
        if timeout <= 0.0:
            return False, "plan_to_pose timeout must be positive", None
        if quat is None:
            quat = list(self._get_param("grasp_quat"))
        quat = self._normalize_quat(quat)
        if not self.workspace_check(x, y):
            return False, (
                f"pose ({x:.3f},{y:.3f}) outside workspace "
                f"[x: {self._workspace_x_min:.2f} to {self._workspace_x_max:.2f}, "
                f"y: {self._workspace_y_min:.2f} to {self._workspace_y_max:.2f}]"
            ), None

        seed = copy.deepcopy(start_state) if start_state is not None else self._build_robot_state()
        goal = _ma.MoveGroup.Goal()
        req = goal.request
        req.group_name = self._get_param("planning_group")
        req.num_planning_attempts = self._get_param("num_planning_attempts")
        req.allowed_planning_time = min(
            float(self._get_param("planning_time")), max(0.05, float(timeout))
        )
        req.max_velocity_scaling_factor = (
            velocity_override
            if velocity_override is not None
            else self._get_param("velocity_scaling")
        )
        req.max_acceleration_scaling_factor = (
            accel_override
            if accel_override is not None
            else self._get_param("accel_scaling")
        )
        req.start_state = seed
        req.goal_constraints.append(self._make_pose_constraints(x, y, z, quat))
        goal.planning_options.plan_only = True

        ok, message, result = self._request_move_group_plan(
            goal, timeout, label="MoveGroup pose plan-only"
        )
        if not ok or result is None:
            return False, message, None
        trajectory = result.planned_trajectory
        plan = PlannedMotion(
            trajectory=trajectory,
            start_state=copy.deepcopy(result.trajectory_start),
            end_state=self._trajectory_end_state(result.trajectory_start, trajectory),
            planning_time=float(result.planning_time),
        )
        return True, "ok", plan

    def solve_pose_ik(
        self,
        x: float,
        y: float,
        z: float,
        quat: list[float] | None = None,
        timeout: float = 0.25,
        *,
        seed_state: RobotState | None = None,
        avoid_collisions: bool = True,
    ) -> tuple[bool, str, dict[str, float] | None]:
        """Run MoveIt's real IK/collision check without planning or execution.

        A missing ``/compute_ik`` service is a failed check, never an assumed
        success. Returned joint values are radians and include all arm joints.
        """
        if timeout <= 0.0:
            return False, "IK timeout must be positive", None
        if quat is None:
            quat = list(self._get_param("grasp_quat"))
        quat = self._normalize_quat(quat)
        if not self.workspace_check(x, y):
            return False, "pose outside workspace", None
        started = time.monotonic()
        if not self._ik_cli.service_is_ready():
            service_wait = min(0.1, timeout)
            if not self._ik_cli.wait_for_service(timeout_sec=service_wait):
                return False, "/compute_ik service unavailable", None
        seconds = float(timeout) - (time.monotonic() - started)
        if seconds <= 0.0:
            return False, f"/compute_ik timed out after {timeout:.2f}s", None

        pose = PoseStamped()
        pose.header.frame_id = self._get_param("base_frame")
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])

        request = _ms.GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = self._get_param("planning_group")
        ik.robot_state = (
            copy.deepcopy(seed_state)
            if seed_state is not None
            else self._build_robot_state()
        )
        ik.avoid_collisions = bool(avoid_collisions)
        ik.ik_link_name = self._get_param("tcp_link")
        ik.pose_stamped = pose
        ik.timeout = Duration(
            sec=int(seconds),
            nanosec=int((seconds % 1.0) * 1e9),
        )

        future = self._ik_cli.call_async(request)
        remaining = float(timeout) - (time.monotonic() - started)
        if remaining <= 0.0 or not self._spin_until(future, remaining):
            return False, f"/compute_ik timed out after {timeout:.2f}s", None
        response = future.result()
        if response is None:
            return False, "/compute_ik returned no result", None
        code = response.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            return False, _friendly_error("IK check failed", code), None
        all_joints = {
            name: float(position)
            for name, position in zip(
                response.solution.joint_state.name,
                response.solution.joint_state.position,
            )
        }
        missing = [name for name in ARM_JOINT_NAMES if name not in all_joints]
        if missing:
            return False, f"IK result missing arm joints: {', '.join(missing)}", None
        return True, "ok", {
            name: all_joints[name] for name in ARM_JOINT_NAMES
        }

    def move_cartesian(self, x: float, y: float, z: float, quat: list[float] | None = None,
                       timeout=30.0, velocity_override: float | None = None,
                       accel_override: float | None = None) -> tuple[bool, str]:
        """Straight-line Cartesian motion. timeout covers planning + execution.
        velocity_override / accel_override: if set, use these instead of the
        default velocity_scaling / accel_scaling params (e.g. for slow descent)."""
        ok, message, plan = self.plan_cartesian(
            x, y, z, quat, timeout,
            velocity_override=velocity_override,
            accel_override=accel_override,
        )
        if not ok or plan is None:
            return False, message
        self.get_logger().info(
            f"Cartesian path: target_z={z:.4f}, fraction={plan.fraction:.1%}, "
            f"required={self._get_param('cartesian_min_fraction'):.1%}"
        )
        return self._execute_trajectory(plan.trajectory, timeout)

    def plan_cartesian(
        self,
        x: float,
        y: float,
        z: float,
        quat: list[float] | None = None,
        timeout: float = 3.0,
        *,
        start_state: RobotState | None = None,
        velocity_override: float | None = None,
        accel_override: float | None = None,
        minimum_fraction: float | None = None,
    ) -> tuple[bool, str, PlannedMotion | None]:
        """Compute a collision-aware Cartesian path without executing it."""
        if timeout <= 0.0:
            return False, "plan_cartesian timeout must be positive", None
        if quat is None:
            quat = list(self._get_param("grasp_quat"))
        quat = self._normalize_quat(quat)
        if not self.workspace_check(x, y):
            return False, "pose outside workspace", None

        target = Pose()
        target.position.x = float(x)
        target.position.y = float(y)
        target.position.z = float(z)
        target.orientation.x = float(quat[0])
        target.orientation.y = float(quat[1])
        target.orientation.z = float(quat[2])
        target.orientation.w = float(quat[3])

        seed = copy.deepcopy(start_state) if start_state is not None else self._build_robot_state()
        request = _ms.GetCartesianPath.Request()
        request.header.frame_id = self._get_param("base_frame")
        request.start_state = seed
        request.group_name = self._get_param("planning_group")
        request.link_name = self._get_param("tcp_link")
        request.waypoints = [target]
        request.max_step = self._get_param("cartesian_eef_step")
        request.jump_threshold = self._get_param("cartesian_jump_threshold")
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = (
            velocity_override
            if velocity_override is not None
            else self._get_param("velocity_scaling")
        )
        request.max_acceleration_scaling_factor = (
            accel_override
            if accel_override is not None
            else self._get_param("accel_scaling")
        )

        future = self._cartesian_cli.call_async(request)
        if not self._spin_until(future, timeout):
            return False, (
                "cartesian_path service did not respond. "
                "Check that /compute_cartesian_path is available in MoveIt"
            ), None
        response = future.result()
        if response is None:
            return False, "cartesian_path service returned no result", None
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return False, _friendly_error(
                "cartesian_path planning failed", response.error_code.val
            ), None
        required = (
            float(minimum_fraction)
            if minimum_fraction is not None
            else float(self._get_param("cartesian_min_fraction"))
        )
        if response.fraction < required:
            return False, (
                f"cartesian path only {response.fraction:.0%} reachable "
                f"(needs {required:.0%}). The target is too far or blocked"
            ), None
        plan = PlannedMotion(
            trajectory=response.solution,
            start_state=seed,
            end_state=self._trajectory_end_state(seed, response.solution),
            fraction=float(response.fraction),
        )
        return True, "ok", plan

    def _execute_trajectory(self, robot_traj, timeout=20.0) -> tuple[bool, str]:
        goal = _ma.ExecuteTrajectory.Goal()
        goal.trajectory = robot_traj
        sf = self._execute_ac.send_goal_async(goal)
        self._track_goal(sf)
        if not self._spin_until(sf, timeout):
            return False, ("execute_trajectory action server did not respond. "
                                "Check that /execute_trajectory is available in MoveIt")
        gh = sf.result()
        if gh is None or not gh.accepted:
            return False, ("execute_trajectory goal rejected — "
                                "the executor refused the trajectory. "
                                "The trajectory may be invalid or the controller is busy")
        rf = gh.get_result_async()
        if not self._spin_until(rf, timeout):
            return False, ("execute_trajectory result did not arrive in time. "
                                "The arm motion state is unknown: do not submit another "
                                "motion. Use the physical/driver emergency stop if needed, "
                                "then verify that the robot has stopped")
        code = rf.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            return False, _friendly_error("execute_trajectory failed", code)
        return True, "ok"

    def execute_planned(
        self,
        plan: PlannedMotion | Any,
        timeout: float = 20.0,
    ) -> tuple[bool, str]:
        """Explicitly execute a previously accepted plan.

        Planning methods never call this function themselves. The caller must
        select a candidate first and then opt in to execution.
        """
        if isinstance(plan, PlannedMotion):
            live = self._build_robot_state()
            expected = dict(zip(
                plan.start_state.joint_state.name,
                plan.start_state.joint_state.position,
            ))
            actual = dict(zip(
                live.joint_state.name,
                live.joint_state.position,
            ))
            missing = [
                name for name in ARM_JOINT_NAMES
                if name not in expected or name not in actual
            ]
            if missing:
                return False, (
                    "cannot validate planned trajectory start state; missing "
                    + ", ".join(missing)
                )
            errors = {
                name: abs(math.atan2(
                    math.sin(float(actual[name]) - float(expected[name])),
                    math.cos(float(actual[name]) - float(expected[name])),
                ))
                for name in ARM_JOINT_NAMES
            }
            worst_name = max(errors, key=errors.get)
            tolerance = float(self._get_param("planned_start_tolerance_rad"))
            if errors[worst_name] > tolerance:
                return False, (
                    "planned trajectory start state mismatch: "
                    f"{worst_name} differs by "
                    f"{math.degrees(errors[worst_name]):.1f}deg; "
                    "trajectory was not sent"
                )
            trajectory = plan.trajectory
        else:
            trajectory = plan
        if trajectory is None:
            return False, "planned trajectory is missing"
        return self._execute_trajectory(trajectory, timeout)

    def control_gripper(self, width: float, duration=1.5, timeout=5.0) -> tuple[bool, str]:
        """Open/close gripper.  width in metres (e.g. 0.10 = open, 0.02 = close)."""
        return self.gripper.control(width, duration, timeout)

    def go_home(self, timeout=60.0) -> tuple[bool, str]:
        """Move to the camera observation joint configuration."""
        observation_deg = list(self._get_param("observation_joints_deg"))
        return self.move_joints(observation_deg, timeout)

    def go_carry(self, timeout=60.0) -> tuple[bool, str]:
        """Move to the configured safe pose while keeping the object held."""
        carry_deg = list(self._get_param("carry_joints_deg"))
        return self.move_joints(carry_deg, timeout)

    # ── goal tracking / cooperative software stop diagnostics ──
    def _track_goal(self, send_future):
        """Track accepted goals for diagnostics and lifecycle cleanup."""
        if hasattr(send_future, 'add_done_callback'):
            def _on_done(f):
                gh = f.result()
                if gh is not None:
                    with self._gh_lock:
                        self._active_goal_handles.append(gh)
            send_future.add_done_callback(_on_done)

    def emergency_stop(self) -> tuple[bool, str]:
        """Clear active goal tracking. Does NOT cancel goals — avoids MoveIt Jazzy crash bug."""
        with self._gh_lock:
            count = len(self._active_goal_handles)
            self._active_goal_handles.clear()
        if count > 0:
            self.get_logger().warn(
                f"SOFTWARE STOP REQUEST: cleared tracking for {count} goal(s); "
                "active controllers were not cancelled"
            )
            return True, (
                f"cleared tracking for {count} goal(s); "
                "active controllers were not cancelled"
            )
        return True, "no active goals"
