"""MoveIt planning, execution, workspace checks, and goal tracking."""

import math
import time

from geometry_msgs.msg import Pose
from moveit_msgs import action as _ma
from moveit_msgs import srv as _ms
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, RobotState
from rclpy.action import ActionClient
from shape_msgs.msg import SolidPrimitive

from .hardware import ARM_JOINT_NAMES, GripperController

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

    def workspace_check(self, x, y) -> bool:
        return (self._workspace_x_min <= x <= self._workspace_x_max and
                self._workspace_y_min <= y <= self._workspace_y_max)

    # ─────────────────────────── motion implementation ────────────
    @staticmethod
    def _spin_until(future, timeout_sec):
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > timeout_sec:
                return False
            time.sleep(0.02)
        return True

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
                                "(3) start state is invalid. Try arm_go_home and retry")
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
                                "(3) start state is invalid. Try arm_go_home and retry")
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

    def move_cartesian(self, x: float, y: float, z: float, quat: list[float] | None = None,
                       timeout=30.0, velocity_override: float | None = None,
                       accel_override: float | None = None) -> tuple[bool, str]:
        """Straight-line Cartesian motion. timeout covers planning + execution.
        velocity_override / accel_override: if set, use these instead of the
        default velocity_scaling / accel_scaling params (e.g. for slow descent)."""
        if quat is None:
            quat = list(self._get_param("grasp_quat"))
        quat = self._normalize_quat(quat)

        target = Pose()
        target.position.x = float(x)
        target.position.y = float(y)
        target.position.z = float(z)
        target.orientation.x = float(quat[0])
        target.orientation.y = float(quat[1])
        target.orientation.z = float(quat[2])
        target.orientation.w = float(quat[3])

        req = _ms.GetCartesianPath.Request()
        req.header.frame_id = self._get_param("base_frame")
        req.start_state = self._build_robot_state()
        req.group_name = self._get_param("planning_group")
        req.link_name = self._get_param("tcp_link")
        req.waypoints = [target]
        req.max_step = self._get_param("cartesian_eef_step")
        req.jump_threshold = self._get_param("cartesian_jump_threshold")
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = velocity_override if velocity_override is not None else self._get_param("velocity_scaling")
        req.max_acceleration_scaling_factor = accel_override if accel_override is not None else self._get_param("accel_scaling")

        sf = self._cartesian_cli.call_async(req)
        if not self._spin_until(sf, timeout):
            return False, ("cartesian_path service did not respond. "
                           "Check that /compute_cartesian_path is available in MoveIt")
        resp = sf.result()
        if resp is None:
            return False, ("cartesian_path service returned no result. "
                           "This is a MoveIt internal error — try arm_go_home to reset")
        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            return False, _friendly_error("cartesian_path planning failed", resp.error_code.val)
        if resp.fraction < self._get_param("cartesian_min_fraction"):
            return False, (f"cartesian path only {resp.fraction:.0%} reachable "
                           f"(needs {self._get_param('cartesian_min_fraction'):.0%}). "
                           f"The target is too far or blocked by an obstacle. "
                           f"Try arm_go_home first or move to a closer waypoint")

        return self._execute_trajectory(resp.solution, timeout)

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
                                "The arm may still be moving — check the robot. "
                                "If it's moving, call arm_stop then arm_go_home")
        code = rf.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            return False, _friendly_error("execute_trajectory failed", code)
        return True, "ok"

    def control_gripper(self, width: float, duration=1.5, timeout=5.0) -> tuple[bool, str]:
        """Open/close gripper.  width in metres (e.g. 0.10 = open, 0.02 = close)."""
        return self.gripper.control(width, duration, timeout)

    def go_home(self, timeout=60.0) -> tuple[bool, str]:
        """Move to home joint configuration. 60s timeout for first plan with Octomap."""
        home_deg = list(self._get_param("home_joints_deg"))
        return self.move_joints(home_deg, timeout)

    # ── goal tracking for emergency stop ──
    def _track_goal(self, send_future):
        """Track a send_goal_async future so emergency_stop can cancel it."""
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
            self.get_logger().warn(f"EMERGENCY STOP: cleared {count} tracked goal(s)")
            return True, f"cleared {count} tracked goal(s)"
        return True, "no active goals"
