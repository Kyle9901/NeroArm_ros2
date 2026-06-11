"""
Thread-safe ROS 2 bridge for robot arm MCP tools.

Runs a single rclpy Node in a background thread.  All public methods are
callable from any thread — they communicate with the ROS node via Queue /
Event primitives.

Usage (from MCP server main thread):
    bridge = RobotBridge()
    bridge.start()                     # launches rclpy spin thread
    img = bridge.capture_image()       # blocking call, returns dict
    pos  = bridge.get_3d_position(320, 240)
    ok   = bridge.go_home()
    bridge.shutdown()
"""

import math
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped, PointStamped
from moveit_msgs import action as _ma
from moveit_msgs import srv as _ms
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, RobotState
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene as _ApplyPlanningScene
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from shape_msgs.msg import SolidPrimitive
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener

from .vlm_client import VlmClient

# ─────────────────────────── constants ───────────────────────────
ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
GRIPPER_JOINT_NAMES = ["gripper_joint1", "gripper_joint2"]


# ═══════════════════════════════════════════════════════════════════════════════════════════
class RobotBridgeNode(Node):
    """rclpy Node that does all the ROS communication."""

    def __init__(self):
        super().__init__("robot_bridge_node")

        self.bridge = CvBridge()
        self.cb_group = ReentrantCallbackGroup()

        # ── parameters (hard-wired defaults, override via ROS params if needed) ──
        self._declare_params()

        # ── state ──
        self._lock = threading.Lock()

        # --- vision ---
        self._color_img: Optional[np.ndarray] = None
        self._depth_img: Optional[np.ndarray] = None    # 16UC1, mm, hardware-aligned
        self._color_info: Optional[dict] = None
        self._img_ready = threading.Event()

        # --- motion ---
        self._joint_state: Optional[JointState] = None

        # ── subscribers ──
        self.create_subscription(Image, "/camera/color/image_raw", self._color_cb, 10)
        self.create_subscription(Image, "/camera/depth/image_raw", self._depth_cb, 10)
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self._cinfo_cb, 10)
        self.create_subscription(JointState, "/feedback/joint_states", self._joint_cb, 10,
                                 callback_group=self.cb_group)

        # ── tf ──
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── action / service clients ──
        self._init_moveit_clients()

        self.get_logger().info("RobotBridgeNode ready")

    # ─────────────────────────── parameters ───────────────────────
    def _declare_params(self):
        p = self.declare_parameter
        p("planning_group", "arm")
        p("tcp_link", "tcp_link")
        p("base_frame", "base_link")
        p("grasp_quat", [0.476, 0.523, -0.523, 0.476])
        p("workspace_x_min", -0.55)
        p("workspace_x_max", 0.25)
        p("workspace_y_min", -0.55)
        p("workspace_y_max", 0.2)
        p("approach_height", 0.26)
        p("safe_height", 0.40)
        p("grasp_depth", 0.155)     # 0.175-0.02, 预留2cm夹取空间
        p("place_x", -0.40)
        p("place_y", -0.25)
        p("place_z", 0.20)
        p("gripper_open_width", 0.10)
        p("gripper_close_width", 0.02)
        p("planning_time", 5.0)
        p("num_planning_attempts", 20)
        p("velocity_scaling", 0.05)
        p("accel_scaling", 0.05)
        p("cartesian_eef_step", 0.005)
        p("cartesian_min_fraction", 0.5)
        p("cartesian_jump_threshold", 2.0)
        p("pos_tolerance", 0.01)
        p("ori_tolerance", 0.1)
        p("home_joints_deg", [0.0, -20.0, 0.0, 80.0, 0.0, 0.0, 80.0])
        # Desk collision object
        p("desk_z_surface", 0.0)               # desk surface in base_link frame
        p("desk_size", [2.0, 2.0, 0.02])       # x, y, thickness

    def _init_moveit_clients(self):
        from control_msgs.action import FollowJointTrajectory as FJT
        self._gripper_ac = ActionClient(self, FJT, "/gripper_controller/follow_joint_trajectory",
                                        callback_group=self.cb_group)
        self._move_group_ac = ActionClient(self, _ma.MoveGroup, "/move_action",
                                           callback_group=self.cb_group)
        self._execute_ac = ActionClient(self, _ma.ExecuteTrajectory, "/execute_trajectory",
                                         callback_group=self.cb_group)
        self._cartesian_cli = self.create_client(_ms.GetCartesianPath, "/compute_cartesian_path",
                                                  callback_group=self.cb_group)
        self._planning_scene_cli = self.create_client(
            _ApplyPlanningScene, "/apply_planning_scene", callback_group=self.cb_group)
        self._desk_added = False

    # ─────────────────────────── subscribers ──────────────────────
    def _color_cb(self, msg):
        with self._lock:
            self._color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self._depth_img is not None:
                self._img_ready.set()

    def _depth_cb(self, msg):
        with self._lock:
            self._depth_img = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            if self._color_img is not None:
                self._img_ready.set()

    def _cinfo_cb(self, msg):
        with self._lock:
            self._color_info = {
                "fx": msg.k[0], "fy": msg.k[4],
                "cx": msg.k[2], "cy": msg.k[5],
                "width": msg.width, "height": msg.height,
            }

    def _joint_cb(self, msg):
        with self._lock:
            self._joint_state = msg

    # ─────────────────────────── wait helpers ─────────────────────
    def wait_servers(self, timeout=15.0):
        ok = True
        for name, ac in [("gripper", self._gripper_ac), ("move_group", self._move_group_ac),
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
    def add_desk_collision(self, timeout=5.0) -> bool:
        """Add desk surface as collision object to the planning scene.
        Uses ROS params desk_z_surface and desk_size."""
        if self._desk_added:
            return True
        if not self._planning_scene_cli.wait_for_service(timeout):
            self.get_logger().warn("apply_planning_scene not available — skip desk collision")
            return False

        desk_z = self._get_param("desk_z_surface")
        desk_size = list(self._get_param("desk_size"))

        co = CollisionObject()
        co.id = "desk"
        co.header.frame_id = self._get_param("base_frame")
        co.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = desk_size

        box_pose = Pose()
        box_pose.position.x = 0.0
        box_pose.position.y = 0.0
        box_pose.position.z = desk_z - desk_size[2] / 2.0
        box_pose.orientation.w = 1.0

        co.primitives.append(box)
        co.primitive_poses.append(box_pose)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(co)

        req = _ApplyPlanningScene.Request()
        req.scene = scene

        self.get_logger().info(
            f"Adding desk collision: surface_z={desk_z:.3f}, "
            f"size={desk_size}")

        future = self._planning_scene_cli.call_async(req)
        if not self._spin_until(future, timeout):
            self.get_logger().warn("add desk collision timeout")
            return False
        resp = future.result()
        if resp is None or not resp.success:
            self.get_logger().warn("add desk collision failed")
            return False
        self._desk_added = True
        self.get_logger().info("desk collision object added")
        return True

    # ─────────────────────────── vision helpers (thread-safe via _lock) ──
    def get_latest_images(self, timeout=2.0) -> tuple[np.ndarray, np.ndarray] | None:
        """Wait for fresh color+depth pair, return (color_bgr, depth_16uc1)."""
        self._img_ready.clear()
        if not self._img_ready.wait(timeout):
            return None
        with self._lock:
            if self._color_img is None or self._depth_img is None:
                return None
            return self._color_img.copy(), self._depth_img.copy()

    def get_color_info(self) -> dict | None:
        with self._lock:
            return self._color_info.copy() if self._color_info else None

    # ─────────────────────────── 3D reconstruction ────────────────
    def compute_3d(self, u: int, v: int, depth_img: np.ndarray, margin_px: int = 5) -> dict | None:
        """
        Given 2D pixel (u,v) and aligned depth_img (16UC1, mm),
        compute 3D point in camera optical frame.

        Returns {"x_c", "y_c", "z_c", "depth_mm", "valid_points"}
        """
        cinfo = self.get_color_info()
        if cinfo is None:
            return None
        dh, dw = depth_img.shape[:2]
        if not (0 <= u < dw and 0 <= v < dh):
            return None

        roi = depth_img[
            max(0, v - margin_px): min(dh, v + margin_px + 1),
            max(0, u - margin_px): min(dw, u + margin_px + 1),
        ]
        valid = roi[roi > 0]
        if len(valid) == 0:
            return None

        depth_mm = float(np.median(valid))
        z_c = depth_mm / 1000.0
        x_c = (u - cinfo["cx"]) * z_c / cinfo["fx"]
        y_c = (v - cinfo["cy"]) * z_c / cinfo["fy"]
        return {"x_c": x_c, "y_c": y_c, "z_c": z_c, "depth_mm": depth_mm, "valid_points": len(valid)}

    def transform_to_base(self, x_c: float, y_c: float, z_c: float, timeout=1.0) -> dict | None:
        """
        TF2 transform from camera_color_optical_frame → base_link.

        Returns {"x", "y", "z"} in base_link frame.
        """
        from rclpy.duration import Duration
        from rclpy.time import Time

        pt_cam = PointStamped()
        pt_cam.header.frame_id = "camera_color_optical_frame"
        pt_cam.header.stamp = self.get_clock().now().to_msg()
        pt_cam.point.x = x_c
        pt_cam.point.y = y_c
        pt_cam.point.z = z_c

        tf = self.tf_buffer.lookup_transform(
            "base_link", "camera_color_optical_frame",
            Time(), Duration(seconds=timeout),
        )
        pt_base = do_transform_point(pt_cam, tf)
        return {"x": pt_base.point.x, "y": pt_base.point.y, "z": pt_base.point.z}

    # ─────────────────────────── motion helpers ───────────────────
    def get_joint_state(self) -> dict:
        with self._lock:
            js = self._joint_state
        if js is None:
            return {"joints": {}, "gripper": {}}
        return {
            "joints": {n: float(p) for n, p in zip(js.name, js.position)
                       if n in ARM_JOINT_NAMES},
            "gripper": {n: float(p) for n, p in zip(js.name, js.position)
                        if n in GRIPPER_JOINT_NAMES},
        }

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
        with self._lock:
            js = self._joint_state
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

        from geometry_msgs.msg import Vector3 as V3

        pc = _ma.PositionConstraint() if hasattr(_ma, "PositionConstraint") else __import__("moveit_msgs.msg", fromlist=["PositionConstraint"]).PositionConstraint()
        from moveit_msgs.msg import PositionConstraint as PC, OrientationConstraint as OC
        pc = PC()
        pc.header.frame_id = base
        pc.link_name = tcp
        pc.target_point_offset = V3(x=0.0, y=0.0, z=0.0)
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.SPHERE
        prim.dimensions = [ptol]
        pc.constraint_region.primitives.append(prim)
        rp = Pose()
        rp.position.x = float(x)
        rp.position.y = float(y)
        rp.position.z = float(z)
        rp.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(rp)
        pc.weight = 1.0
        c.position_constraints.append(pc)

        oc = OC()
        oc.header.frame_id = base
        oc.link_name = tcp
        oc.orientation.x = float(quat[0])
        oc.orientation.y = float(quat[1])
        oc.orientation.z = float(quat[2])
        oc.orientation.w = float(quat[3])
        oc.absolute_x_axis_tolerance = otol
        oc.absolute_y_axis_tolerance = otol
        oc.absolute_z_axis_tolerance = otol
        oc.weight = 1.0
        c.orientation_constraints.append(oc)
        return c

    # ── motion primitives ──

    def move_joints(self, joint_angles_deg: list[float], timeout=20.0) -> tuple[bool, str]:
        """Move to joint-space target.  joint_angles_deg: 7 floats in degrees."""
        if len(joint_angles_deg) != len(ARM_JOINT_NAMES):
            return False, f"expected {len(ARM_JOINT_NAMES)} joint angles, got {len(joint_angles_deg)}"

        goal = _ma.MoveGroup.Goal()
        req = goal.request
        req.group_name = self._get_param("planning_group")
        req.num_planning_attempts = 5
        req.allowed_planning_time = self._get_param("planning_time")
        req.max_velocity_scaling_factor = self._get_param("velocity_scaling")
        req.max_acceleration_scaling_factor = self._get_param("accel_scaling")
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
        if not self._spin_until(sf, timeout):
            return False, "MoveGroup send timeout"
        gh = sf.result()
        if gh is None or not gh.accepted:
            return False, "MoveGroup goal rejected"
        rf = gh.get_result_async()
        if not self._spin_until(rf, timeout + 5):
            return False, "MoveGroup result timeout"
        code = rf.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            return False, f"MoveGroup error_code={code}"
        return True, "ok"

    def move_to_pose(self, x: float, y: float, z: float, quat: list[float] | None = None,
                     timeout=20.0) -> tuple[bool, str]:
        """Move to Cartesian pose via MoveGroup.  quat defaults to grasp_quat."""
        if quat is None:
            quat = list(self._get_param("grasp_quat"))
        quat = self._normalize_quat(quat)

        if not self.workspace_check(x, y):
            return False, f"pose ({x:.3f},{y:.3f}) outside workspace"

        goal = _ma.MoveGroup.Goal()
        req = goal.request
        req.group_name = self._get_param("planning_group")
        req.num_planning_attempts = self._get_param("num_planning_attempts")
        req.allowed_planning_time = self._get_param("planning_time")
        req.max_velocity_scaling_factor = self._get_param("velocity_scaling")
        req.max_acceleration_scaling_factor = self._get_param("accel_scaling")
        req.start_state = self._build_robot_state()
        req.goal_constraints.append(self._make_pose_constraints(x, y, z, quat))
        goal.planning_options.plan_only = False

        sf = self._move_group_ac.send_goal_async(goal)
        if not self._spin_until(sf, timeout):
            return False, "MoveGroup send timeout"
        gh = sf.result()
        if gh is None or not gh.accepted:
            return False, "MoveGroup goal rejected"
        rf = gh.get_result_async()
        if not self._spin_until(rf, timeout + self._get_param("planning_time")):
            return False, "MoveGroup result timeout"
        code = rf.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            return False, f"MoveGroup error_code={code}"
        return True, "ok"

    def move_cartesian(self, x: float, y: float, z: float, quat: list[float] | None = None,
                       timeout=20.0) -> tuple[bool, str]:
        """Straight-line Cartesian motion."""
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
        req.max_velocity_scaling_factor = self._get_param("velocity_scaling")
        req.max_acceleration_scaling_factor = self._get_param("accel_scaling")

        sf = self._cartesian_cli.call_async(req)
        if not self._spin_until(sf, timeout):
            return False, "cartesian_path call timeout"
        resp = sf.result()
        if resp is None:
            return False, "cartesian_path no response"
        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            return False, f"cartesian_path error_code={resp.error_code.val}"
        if resp.fraction < self._get_param("cartesian_min_fraction"):
            return False, f"cartesian coverage {resp.fraction:.1%} too low"

        return self._execute_trajectory(resp.solution, timeout)

    def _execute_trajectory(self, robot_traj, timeout=20.0) -> tuple[bool, str]:
        goal = _ma.ExecuteTrajectory.Goal()
        goal.trajectory = robot_traj
        sf = self._execute_ac.send_goal_async(goal)
        if not self._spin_until(sf, timeout):
            return False, "execute_trajectory send timeout"
        gh = sf.result()
        if gh is None or not gh.accepted:
            return False, "execute_trajectory rejected"
        rf = gh.get_result_async()
        if not self._spin_until(rf, timeout):
            return False, "execute_trajectory result timeout"
        code = rf.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            return False, f"execute_trajectory error_code={code}"
        return True, "ok"

    def control_gripper(self, width: float, duration=1.5, timeout=5.0) -> tuple[bool, str]:
        """Open/close gripper.  width in metres (e.g. 0.10 = open, 0.02 = close)."""
        from control_msgs.action import FollowJointTrajectory as FJT
        from trajectory_msgs.msg import JointTrajectoryPoint

        j1 = float(width * 0.5)
        j2 = float(-width * 0.5)
        goal = FJT.Goal()
        goal.trajectory.joint_names = GRIPPER_JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [j1, j2]
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration % 1) * 1e9)
        goal.trajectory.points.append(pt)

        sf = self._gripper_ac.send_goal_async(goal)
        if not self._spin_until(sf, timeout):
            return False, "gripper send timeout"
        gh = sf.result()
        if gh is None or not gh.accepted:
            return False, "gripper rejected"
        rf = gh.get_result_async()
        if not self._spin_until(rf, timeout + duration):
            return False, "gripper result timeout"
        code = rf.result().result.error_code
        if code != 0:
            return False, f"gripper error_code={code}"
        return True, "ok"

    def go_home(self, timeout=20.0) -> tuple[bool, str]:
        """Move to home joint configuration."""
        home_deg = list(self._get_param("home_joints_deg"))
        return self.move_joints(home_deg, timeout)


# ═══════════════════════════════════════════════════════════════════════════════════════════
#   Thread-safe wrapper — public API for MCP tools
# ═══════════════════════════════════════════════════════════════════════════════════════════
class RobotBridge:
    """Thread-safe ROS 2 bridge.  Start the ROS spin thread,
    then call public methods from any thread."""

    def __init__(self):
        self._node: RobotBridgeNode | None = None
        self._executor: rclpy.executors.MultiThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._shutdown_flag = False

    # ── lifecycle ──
    def start(self, wait_servers: bool = False) -> None:
        """Start the ROS 2 bridge.  If wait_servers=False (default), the MCP server
        becomes available immediately — motion tools will fail gracefully if the
        underlying MoveIt / gripper servers are not running."""
        if not rclpy.ok():
            rclpy.init()
        self._node = RobotBridgeNode()
        self._executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()
        time.sleep(1.0)  # give spin thread a moment to start
        if wait_servers:
            self._node.wait_servers()
        # Add desk collision object (best-effort, won't fail if service not ready)
        self._node.add_desk_collision()
        self._ready.set()

    def _spin_loop(self):
        while not self._shutdown_flag and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.05)

    def shutdown(self):
        self._shutdown_flag = True
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=3.0)
        if self._node is not None:
            self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    @property
    def node(self) -> RobotBridgeNode:
        self._ready.wait()
        return self._node

    # ── convenience getters ──
    def get_grasp_quat(self) -> list[float]:
        return list(self.node._get_param("grasp_quat"))

    def get_approach_height(self) -> float:
        return self.node._get_param("approach_height")

    def get_grasp_depth(self) -> float:
        return self.node._get_param("grasp_depth")

    def get_safe_height(self) -> float:
        return self.node._get_param("safe_height")

    def get_place_pose(self) -> dict:
        n = self.node
        return {"x": n._get_param("place_x"), "y": n._get_param("place_y"),
                "z": n._get_param("place_z")}

    def get_gripper_open_width(self) -> float:
        return self.node._get_param("gripper_open_width")

    def get_gripper_close_width(self) -> float:
        return self.node._get_param("gripper_close_width")

    def get_velocity_scaling(self) -> float:
        return self.node._get_param("velocity_scaling")

    def get_base_frame(self) -> str:
        return self.node._get_param("base_frame")