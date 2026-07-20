"""
Thread-safe ROS 2 bridge for robot arm MCP tools.

Runs a single rclpy Node in a background thread.  All public methods are
callable from any thread — they communicate with the ROS node via Queue /
Event primitives.

Usage (from MCP server main thread):
    bridge = RobotBridge()
    bridge.start()                     # launches rclpy spin thread
    color, depth = bridge.node.get_latest_images()
    status = bridge.health_status()
    bridge.shutdown()
"""

import math
import threading
import time
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from .ros import (
    ARM_JOINT_NAMES,
    BringupManager,
    CameraStream,
    JointStateMonitor,
    OctomapControl,
    PlanningSceneService,
    TransformService,
)
from .ros.motion import MotionControllerMixin
from .config import load_robot_parameters
from .visualization import GraspCandidateMarkerPublisher

# ═══════════════════════════════════════════════════════════════════════════════════════════
class RobotBridgeNode(Node, MotionControllerMixin):
    """rclpy Node that does all the ROS communication."""

    def __init__(self):
        super().__init__("robot_bridge_node")

        self.cb_group = ReentrantCallbackGroup()

        # YAML values are declaration defaults; ROS parameter overrides still win.
        self._declare_params()

        # ── state ──
        # --- active goal handles (for cancellation) ---
        self._active_goal_handles: list = []  # ClientGoalHandle
        self._gh_lock = threading.Lock()

        # ── subscribers ──
        self.camera = CameraStream(self)
        self.joint_states = JointStateMonitor(self, self.cb_group)

        # ── tf ──
        self.transforms = TransformService(self)
        # Compatibility aliases for prepare checks and existing callers.
        self.tf_buffer = self.transforms.buffer
        self.tf_listener = self.transforms.listener

        # ── action / service clients ──
        self._init_moveit_clients()
        self.scene = PlanningSceneService(self, self.cb_group)
        self.octomap = OctomapControl(self, self.cb_group)
        try:
            self.grasp_candidate_markers = GraspCandidateMarkerPublisher(
                self,
                frame_id=self._get_param("base_frame"),
            )
        except Exception as error:
            # Visualization is diagnostic only and must never block safety or
            # planning when visualization_msgs is unavailable.
            self.grasp_candidate_markers = None
            self.get_logger().warn(
                f"grasp candidate marker publisher unavailable: {error}"
            )

        self.get_logger().info("RobotBridgeNode ready")

    # ─────────────────────────── parameters ───────────────────────
    def _declare_params(self):
        config_path, parameters = load_robot_parameters()
        for name, default in parameters.items():
            self.declare_parameter(name, default)
        self.get_logger().info(f"Loaded robot config: {config_path}")

    def add_desk_collision(self, timeout=5.0) -> bool:
        """Add desk surface as collision object to the planning scene.
        Uses ROS params desk_z_surface and desk_size."""
        return self.scene.add_desk(timeout)

    def get_latest_images(self, timeout=2.0):
        """Wait for fresh color+depth pair, return (color_bgr, depth_16uc1)."""
        return self.camera.get_latest_images(timeout)

    def get_color_info(self) -> dict | None:
        return self.camera.get_color_info()

    # ─────────────────────────── 3D reconstruction ────────────────
    def compute_3d(self, u: int, v: int, depth_img, margin_px: int = 2) -> dict | None:
        return self.camera.compute_3d(u, v, depth_img, margin_px)

    def transform_to_base(
        self, x_c: float, y_c: float, z_c: float, timeout=1.0,
        *, stamp=None, source_frame: str = "camera_color_optical_frame",
    ) -> dict | None:
        """
        TF2 transform from camera_color_optical_frame → base_link.

        Returns {"x", "y", "z"} in base_link frame.
        """
        return self.transforms.transform_point(
            x_c, y_c, z_c,
            source_frame=source_frame,
            stamp=stamp,
            timeout=timeout,
        )

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
        self._task_stop_requested = threading.Event()
        self._holding = False    # whether gripper is currently holding an object
        self._holding_lock = threading.Lock()
        self._task_context: dict = {
            "grasped_object": None,
            "last_action": None,
            "last_place": None,
            "recent_actions": [],
        }
        self._context_lock = threading.Lock()
        self.bringup = BringupManager(lambda: self.node)

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
        # Planning-scene setup belongs to the prepare skill.  MoveIt may not
        # exist yet when the MCP process starts, so doing it here would add a
        # needless service timeout and could silently miss the desk object.
        self._ready.set()

    def _spin_loop(self):
        try:
            while not self._shutdown_flag and rclpy.ok():
                self._executor.spin_once(timeout_sec=0.05)
        except RuntimeError:
            # Executor/rclpy shutdown can race with spin_once during process
            # teardown.  Suppress that expected lifecycle error only when this
            # bridge is already stopping.
            if not self._shutdown_flag and rclpy.ok():
                raise

    def shutdown(self):
        self._shutdown_flag = True
        # Only processes launched by this bridge are owned and stopped here.
        # Externally launched ROS nodes are intentionally left untouched.
        self.bringup.stop_all()
        if self._executor is not None:
            self._executor.wake()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=3.0)
        if self._executor is not None:
            if self._node is not None:
                self._executor.remove_node(self._node)
            self._executor.shutdown(timeout_sec=3.0)
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

    def get_tcp_offset(self) -> list[float]:
        """Configured link7-to-TCP transform [x, y, z, roll, pitch, yaw]."""
        return list(self.node._get_param("tcp_offset"))

    def get_approach_height(self) -> float:
        return self.node._get_param("approach_height")

    def get_fingertip_depth(self) -> float:
        """Configured distance from the detected object surface to the physical fingertip."""
        return self.node._get_param("fingertip_depth")

    def get_safe_height(self) -> float:
        return self.node._get_param("safe_height")

    def get_block_depth_frames(self) -> int:
        return int(self.node._get_param("block_depth_frames"))

    def get_block_depth_max_spread(self) -> float:
        return float(self.node._get_param("block_depth_max_spread"))

    def get_block_xy_max_spread(self) -> float:
        return float(self.node._get_param("block_xy_max_spread"))

    def get_block_yaw_max_spread_deg(self) -> float:
        return float(self.node._get_param("block_yaw_max_spread_deg"))

    def get_cylinder_depth_frames(self) -> int:
        return int(self.node._get_param("cylinder_depth_frames"))

    def get_cylinder_depth_max_spread(self) -> float:
        return float(self.node._get_param("cylinder_depth_max_spread"))

    def get_cylinder_position_max_spread(self) -> float:
        return float(self.node._get_param("cylinder_position_max_spread"))

    def get_cylinder_axis_max_spread_deg(self) -> float:
        return float(self.node._get_param("cylinder_axis_max_spread_deg"))

    def get_cylinder_lying_axis_max_deviation_deg(self) -> float:
        return float(self.node._get_param(
            "cylinder_lying_axis_max_deviation_deg"
        ))

    def get_cylinder_min_diameter_m(self) -> float:
        return float(self.node._get_param("cylinder_min_diameter_m"))

    def get_cylinder_max_diameter_m(self) -> float:
        return float(self.node._get_param("cylinder_max_diameter_m"))

    def get_cylinder_min_length_m(self) -> float:
        return float(self.node._get_param("cylinder_min_length_m"))

    def get_cylinder_max_length_m(self) -> float:
        return float(self.node._get_param("cylinder_max_length_m"))

    def get_cylinder_side_grasp_height_ratio(self) -> float:
        return float(self.node._get_param("cylinder_side_grasp_height_ratio"))

    def get_cylinder_tilt_angles_deg(self) -> list[int]:
        return [
            int(value)
            for value in self.node._get_param("cylinder_tilt_angles_deg")
        ]

    def get_desk_measurement_max_error(self) -> float:
        return float(self.node._get_param("desk_measurement_max_error"))

    def get_grasp_tilt_angles_deg(self) -> list[int]:
        return [int(value) for value in self.node._get_param("grasp_tilt_angles_deg")]

    def get_grasp_pregrasp_distance(self) -> float:
        return float(self.node._get_param("grasp_pregrasp_distance"))

    def get_grasp_retreat_distance(self) -> float:
        return float(self.node._get_param("grasp_retreat_distance"))

    def get_grasp_candidate_timeout(self) -> float:
        return float(self.node._get_param("grasp_candidate_timeout"))

    def get_grasp_full_plan_candidates(self) -> int:
        return int(self.node._get_param("grasp_full_plan_candidates"))

    def get_joint7_soft_limit_deg(self) -> float:
        return float(self.node._get_param("joint7_soft_limit_deg"))

    def get_joint7_min_margin_deg(self) -> float:
        return float(self.node._get_param("joint7_min_margin_deg"))

    def get_reverse_branch_tolerance_rad(self) -> float:
        return float(self.node._get_param("reverse_branch_tolerance_rad"))

    def get_observation_joints_deg(self) -> list[float]:
        return [float(value) for value in self.node._get_param("observation_joints_deg")]

    def get_carry_joints_deg(self) -> list[float]:
        return [float(value) for value in self.node._get_param("carry_joints_deg")]

    def get_current_tcp_pose(self, timeout: float = 1.0) -> dict:
        return self.node.transforms.lookup_pose(
            self.node._get_param("tcp_link"),
            self.node._get_param("base_frame"),
            timeout,
        )

    def get_desk_surface_z(self) -> float:
        return float(self.node._get_param("desk_z_surface"))

    def get_desk_collision_enabled(self) -> bool:
        return bool(self.node._get_param("desk_collision_enabled"))

    def get_workspace_bounds(self) -> dict[str, float]:
        return {
            "x_min": float(self.node._get_param("workspace_x_min")),
            "x_max": float(self.node._get_param("workspace_x_max")),
            "y_min": float(self.node._get_param("workspace_y_min")),
            "y_max": float(self.node._get_param("workspace_y_max")),
        }

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

    def get_octomap_enabled_on_prepare(self) -> bool:
        return bool(self.node._get_param("octomap_enabled_on_prepare"))

    def get_descent_velocity_scaling(self) -> float:
        return self.node._get_param("descent_velocity_scaling")

    def get_descent_accel_scaling(self) -> float:
        return self.node._get_param("descent_accel_scaling")

    def emergency_stop(self) -> tuple[bool, str]:
        self.request_task_stop()
        return self.node.emergency_stop()

    # ── cooperative task stop ──
    def request_task_stop(self) -> None:
        """Prevent the active task from starting another skill or retry.

        This is deliberately cooperative: it cannot cancel a trajectory that
        MoveIt/controller has already accepted.
        """
        self._task_stop_requested.set()

    def clear_task_stop(self) -> None:
        self._task_stop_requested.clear()

    def is_task_stop_requested(self) -> bool:
        return self._task_stop_requested.is_set()

    def get_joint_state(self) -> dict:
        return self.node.get_joint_state()

    def can_transform(
        self, source_frame: str, target_frame: str | None = None,
        timeout: float = 1.0,
    ) -> bool:
        return self.node.transforms.can_transform(
            source_frame,
            target_frame or self.get_base_frame(),
            timeout,
        )

    def get_node_counts(self, names: set[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node_name, _namespace in self.node.get_node_names_and_namespaces():
            base = node_name.split("/")[-1]
            if base in names:
                counts[base] = counts.get(base, 0) + 1
        return counts

    # ── holding state ──
    def get_holding(self) -> bool | None:
        with self._holding_lock:
            return self._holding

    def set_holding(self, value: bool | None) -> None:
        with self._holding_lock:
            self._holding = value

    def get_base_frame(self) -> str:
        return self.node._get_param("base_frame")

    # ── task context (semantic, in-memory) ──
    def get_task_context(self) -> dict:
        with self._context_lock:
            return dict(self._task_context)

    def update_task_context(self, **changes) -> None:
        with self._context_lock:
            self._task_context.update(changes)

    def add_recent_action(self, action: str) -> None:
        with self._context_lock:
            actions = list(self._task_context.get("recent_actions", []))
            actions.append(action)
            self._task_context["recent_actions"] = actions[-5:]

    def reset_task_context(self) -> None:
        with self._context_lock:
            self._task_context = {
                "grasped_object": None,
                "last_action": None,
                "last_place": None,
                "recent_actions": [],
            }

    def is_at_home(self, tolerance_rad: float = 0.08) -> bool:
        state = self.node.get_joint_state().get("joints", {})
        if not state:
            return False
        observation_deg = self.get_observation_joints_deg()
        for name, deg in zip(ARM_JOINT_NAMES, observation_deg):
            if name not in state:
                return False
            if abs(state[name] - math.radians(float(deg))) > tolerance_rad:
                return False
        return True

    def build_planning_context(self) -> str:
        holding = self.get_holding()
        if not holding:
            self.update_task_context(grasped_object=None)
        ctx = self.get_task_context()
        held = ctx.get("grasped_object") if holding else None
        held_desc = held if held else ("unknown object" if holding else "none")
        ready = self.bringup_status().get("endpoints", {})
        lines = [
            "## 当前机器人状态",
            f"- holding: {holding}",
            f"- grasped_object: {held_desc}",
            f"- at_home: {self.is_at_home()}",
            f"- nodes_ready: move_action={ready.get('move_action')}, "
            f"camera={ready.get('camera_color')}, octomap={ready.get('octomap_cloud')}, "
            f"tf={ready.get('tf')}",
        ]
        recent = ctx.get("recent_actions") or []
        if recent:
            lines.append(f"- recent_actions: {recent}")
        return "\n".join(lines)

    # ── bringup: launch management ──

    def bringup_nodes(self, can_port: str = "can0", calib_name: str = "my_eih_calib_park",
                      octomap_enabled: bool = False) -> dict:
        return self.bringup.start(
            can_port=can_port,
            calib_name=calib_name,
            octomap_enabled=octomap_enabled,
        )

    def bringup_status(self) -> dict:
        return self.bringup.status()

    def health_status(self) -> dict:
        status = self.bringup.status()
        endpoints = status.get("endpoints", {})
        topics = status.get("topics", {})
        camera = self.node.camera.health_status()
        registered = topics.get("/camera/depth_registered/points", {})
        filtered = topics.get("/filtered_cloud", {})
        octomap = topics.get("/octomap_cloud", {})
        try:
            tf_ready = self.can_transform("camera_color_optical_frame", timeout=1.0)
        except Exception:
            tf_ready = False

        octomap_enabled = self.get_octomap_enabled_on_prepare()
        checks = {
            "can_up": bool(status.get("can", {}).get("up")),
            "move_action": bool(endpoints.get("move_action")),
            "planning_scene": bool(
                endpoints.get("planning_scene_apply")
                and endpoints.get("planning_scene_get")
            ),
            "rgbd_pair_fresh": bool(camera.get("pair_fresh")),
            "depth_registered": bool(camera.get("registered_shapes_match")),
            "camera_info": bool(camera.get("camera_info_received")),
            "handeye_publisher": bool(endpoints.get("handeye_publisher")),
            "camera_to_base_tf": tf_ready,
            "pointcloud_pipeline": bool(
                filtered.get("publishers", 0) > 0
                or octomap.get("publishers", 0) > 0
            ),
            "octomap_cloud_publisher": octomap.get("publishers", 0) > 0,
            "move_group_octomap_subscriber": "move_group" in octomap.get("subscriber_nodes", []),
            "octomap_control": bool(endpoints.get("octomap_control")),
        }
        required = {
            "can_up", "move_action", "planning_scene", "rgbd_pair_fresh",
            "depth_registered", "camera_info", "handeye_publisher",
            "camera_to_base_tf",
        }
        if octomap_enabled:
            required.update({
                "pointcloud_pipeline", "octomap_cloud_publisher",
                "move_group_octomap_subscriber", "octomap_control",
            })
        failures = [name for name in required if not checks[name]]
        warnings = []
        if registered.get("publishers", 0) == 0:
            warnings.append("registered_cloud_publisher_not_visible")
        return {
            "ready": not failures,
            "failures": failures,
            "warnings": warnings,
            "checks": checks,
            "octomap_required": octomap_enabled,
            "observations": {
                "registered_cloud_publishers": registered.get("publishers", 0),
                "filtered_cloud_publishers": filtered.get("publishers", 0),
                "octomap_cloud_publishers": octomap.get("publishers", 0),
            },
            "camera": camera,
            "topics": topics,
            "endpoints": endpoints,
            "processes": status.get("processes", {}),
            "can": status.get("can", {}),
        }

    def set_octomap_enabled(self, enabled: bool) -> dict:
        result = self.node.octomap.set_enabled(enabled)
        if enabled and not result.get("success"):
            started = self.bringup.start_pointcloud_filter(enabled=True)
            if not started.get("success"):
                result["bringup"] = started
                return result
            result = self.node.octomap.set_enabled(True)
            result["bringup"] = started
        return result
