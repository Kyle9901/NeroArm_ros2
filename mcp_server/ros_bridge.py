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
    PlanningSceneService,
    TransformService,
)
from .ros.motion import MotionControllerMixin

# ═══════════════════════════════════════════════════════════════════════════════════════════
class RobotBridgeNode(Node, MotionControllerMixin):
    """rclpy Node that does all the ROS communication."""

    def __init__(self):
        super().__init__("robot_bridge_node")

        self.cb_group = ReentrantCallbackGroup()

        # ── parameters (hard-wired defaults, override via ROS params if needed) ──
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

        self.get_logger().info("RobotBridgeNode ready")

    # ─────────────────────────── parameters ───────────────────────
    def _declare_params(self):
        p = self.declare_parameter
        # ── 机械臂 / 规划 ──
        p("planning_group", "arm")              # MoveIt 规划组名
        p("tcp_link", "tcp_link")               # 末端执行器连杆名
        p("base_frame", "base_link")            # 机器人基坐标系
        p("grasp_quat", [0.503, 0.497, -0.499, 0.501])  # 实测抓取姿态四元数 (x,y,z,w)
        # ── 工作空间 ──
        p("workspace_x_min", -0.55)             # 工作空间 X 下限 (m)
        p("workspace_x_max", 0.25)              # 工作空间 X 上限 (m)
        p("workspace_y_min", -0.55)             # 工作空间 Y 下限 (m)
        p("workspace_y_max", 0.2)               # 工作空间 Y 上限 (m)
        # ── 抓取几何 ──
        p("approach_height", 0.26)              # 预抓位高度: 物体表面上方距离 (m)
        p("safe_height", 0.40)                  # 安全抬起高度 (m)
        p("grasp_depth", 0.03)                  # 指尖从物体表面向下探入的深度 (m), 旧语义: 法兰从预抓位下降距离
        p("flange_to_tip", 0.175)               # 法兰 → 夹爪指尖距离, 固定硬件参数 (m)
        p("fingertip_overlap", 0.02)            # 指尖探入物块表面的深度 (m)
        # ── 放置位置 ──
        p("place_x", -0.40)                     # 默认放置 X (m)
        p("place_y", -0.25)                     # 默认放置 Y (m)
        p("place_z", 0.20)                      # 默认放置 Z (m)
        # ── 夹爪 ──
        p("gripper_open_width", 0.10)           # 夹爪张开宽度 (m)
        p("gripper_close_width", 0.02)          # 夹爪闭合宽度 (m)
        # ── 规划 ──
        p("planning_time", 10.0)                # 单次规划超时 (s), 首次规划需构建碰撞结构
        p("num_planning_attempts", 3)           # 最大规划尝试次数
        # ── 运动速度 (0-1, 1=全速) ──
        p("velocity_scaling", 0.5)             # 普通运动速度倍率 (approach / lift / home)
        p("accel_scaling", 0.3)                # 普通运动加速度倍率
        p("descent_velocity_scaling", 0.2)     # 笛卡尔下降速度倍率 (慢速精确)
        p("descent_accel_scaling", 0.05)        # 笛卡尔下降加速度倍率
        # ── 笛卡尔路径 ──
        p("cartesian_eef_step", 0.005)          # 笛卡尔路径步长 (m)
        p("cartesian_min_fraction", 0.2)        # 笛卡尔路径最低通过率 (0-1)
        p("cartesian_jump_threshold", 2.0)      # 笛卡尔路径跳跃阈值, 0=禁用
        # ── 位姿容差 ──
        p("pos_tolerance", 0.01)                # 位置容差 (m)
        p("ori_tolerance", 0.1)                 # 姿态容差 (rad)
        # ── Home 位置 ──
        p("home_joints_deg", [0.0, -20.0, 0.0, 80.0, 0.0, 0.0, 80.0])  # 7 关节 home 角度 (度)
        # ── 桌面碰撞对象 ──
        p("desk_z_surface", -0.001)             # 桌面在 base_link 中的 Z 坐标 (m)
        p("desk_size", [2.0, 2.0, 0.02])        # 桌面碰撞体: 长, 宽, 厚度 (m)

    def add_desk_collision(self, timeout=5.0) -> bool:
        """Add desk surface as collision object to the planning scene.
        Uses ROS params desk_z_surface and desk_size."""
        return self.scene.add_desk(timeout)

    def add_target_collision(self, x: float, y: float, z: float,
                             object_id: str = "target_object",
                             size: tuple[float, float, float] = (0.06, 0.06, 0.08),
                             shape: str = "BOX",
                             timeout: float = 5.0) -> bool:
        """Add target object as collision object and allow gripper to touch it.

        Args:
            x, y, z: Object center in base_link frame.
            object_id: Unique name for this collision object.
            size: (x, y, z) dimensions in meters. Default 6x6x8cm for blocks/cups.
            shape: 'BOX' or 'CYLINDER'.
        """
        return self.scene.add_target(x, y, z, object_id, size, shape, timeout)

    def remove_target_collision(self, object_id: str = "target_object",
                                timeout: float = 5.0) -> bool:
        """Remove target collision object and clear its ACM entry."""
        return self.scene.remove_target(object_id, timeout)

    def get_latest_images(self, timeout=2.0):
        """Wait for fresh color+depth pair, return (color_bgr, depth_16uc1)."""
        return self.camera.get_latest_images(timeout)

    def get_color_info(self) -> dict | None:
        return self.camera.get_color_info()

    # ─────────────────────────── 3D reconstruction ────────────────
    def compute_3d(self, u: int, v: int, depth_img, margin_px: int = 2) -> dict | None:
        return self.camera.compute_3d(u, v, depth_img, margin_px)

    def transform_to_base(self, x_c: float, y_c: float, z_c: float, timeout=1.0) -> dict | None:
        """
        TF2 transform from camera_color_optical_frame → base_link.

        Returns {"x", "y", "z"} in base_link frame.
        """
        return self.transforms.transform_point(x_c, y_c, z_c, timeout=timeout)

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
        # Add desk collision object (best-effort, won't fail if service not ready)
        self._node.add_desk_collision()
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

    def get_flange_to_tip(self) -> float:
        return self.node._get_param("flange_to_tip")

    def get_fingertip_overlap(self) -> float:
        return self.node._get_param("fingertip_overlap")

    def get_velocity_scaling(self) -> float:
        return self.node._get_param("velocity_scaling")

    def get_descent_velocity_scaling(self) -> float:
        return self.node._get_param("descent_velocity_scaling")

    def get_descent_accel_scaling(self) -> float:
        return self.node._get_param("descent_accel_scaling")

    def emergency_stop(self) -> tuple[bool, str]:
        return self.node.emergency_stop()

    # ── holding state ──
    def get_holding(self) -> bool:
        with self._holding_lock:
            return self._holding

    def set_holding(self, value: bool) -> None:
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
        home_deg = list(self.node._get_param("home_joints_deg"))
        for name, deg in zip(ARM_JOINT_NAMES, home_deg):
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
            f"- nodes_ready: move_action={ready.get('move_action')}, camera={ready.get('camera_color')}, tf={ready.get('tf')}",
        ]
        recent = ctx.get("recent_actions") or []
        if recent:
            lines.append(f"- recent_actions: {recent}")
        return "\n".join(lines)

    # ── bringup: launch management ──

    def bringup_nodes(self, can_port: str = "can0", calib_name: str = "my_eih_calib_v6") -> dict:
        return self.bringup.start(can_port=can_port, calib_name=calib_name)

    def bringup_status(self) -> dict:
        return self.bringup.status()
