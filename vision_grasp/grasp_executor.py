#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grasp Executor Node (ROS 2 Jazzy)

控制方式:
  机械臂规划 → MoveGroup action (/move_action)
  直线运动   → /compute_cartesian_path 服务 + /execute_trajectory action
  夹爪       → gripper_controller/follow_joint_trajectory action

抓取姿态: 使用实测四元数 (从 tf2_echo base_link tcp_link 得到)，
         不再使用欧拉角，避免 IK 无解。

抓取序列:
  1. 开夹爪，同时运动到预抓位
  2. 直线下降到抓取位
  3. 合夹爪
  4. 直线抬起到安全位
  5. 运动到固定放置位上方
  6. 直线下降放置并开夹爪
  7. 回初始位
"""

import time
import threading
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped, Pose, Vector3
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.msg import (
    Constraints, PositionConstraint, OrientationConstraint,
    JointConstraint, MoveItErrorCodes, RobotState,
)


class GraspExecutorNode(Node):
    def __init__(self):
        super().__init__('grasp_executor_node')

        # ── 参数 ──
        self.declare_parameter('planning_group', 'arm')
        self.declare_parameter('tcp_link', 'tcp_link')
        self.declare_parameter('base_frame', 'base_link')

        # 实测抓取姿态四元数 (x, y, z, w)
        # 来自 tf2_echo base_link tcp_link，垂直向下抓取
        self.declare_parameter('grasp_quat', [0.476, 0.523, -0.523, 0.476])

        self.declare_parameter('table_z_threshold', 0.05)
        self.declare_parameter('workspace_x_min', -0.55)
        self.declare_parameter('workspace_x_max', 0.25)
        self.declare_parameter('workspace_y_min', -0.55)
        self.declare_parameter('workspace_y_max', 0.2)
        self.declare_parameter('approach_height', 0.26)       # 预抓位: 物块上方26cm (法兰位置)
        self.declare_parameter('safe_height', 0.40)            # 安全位法兰 Z
        self.declare_parameter('grasp_depth', 0.175 - 0.02)          # 抓取位: 法兰在物块上方17.5cm
        self.declare_parameter('place_x', -0.40)               # 放置位法兰点 X
        self.declare_parameter('place_y', -0.25)               # 放置位法兰点 Y
        self.declare_parameter('place_z', 0.20)                # 放置位法兰点 Z
        self.declare_parameter('gripper_open_width', 0.10)
        self.declare_parameter('gripper_close_width', 0.02)
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('num_planning_attempts', 20)
        self.declare_parameter('velocity_scaling', 0.05)
        self.declare_parameter('accel_scaling', 0.05)
        self.declare_parameter('cartesian_eef_step', 0.005)
        self.declare_parameter('cartesian_min_fraction', 0.5)
        self.declare_parameter('cartesian_jump_threshold', 2.0)
        self.declare_parameter('pos_tolerance', 0.01)
        self.declare_parameter('ori_tolerance', 0.1)
        self.declare_parameter('home_joints_deg', [0.0, -20.0, 0.0, 80.0, 0.0, 0.0, 80.0])

        g = self.get_parameter
        self.planning_group = g('planning_group').value
        self.tcp_link = g('tcp_link').value
        self.base_frame = g('base_frame').value
        self.grasp_quat = self._normalize_quat(list(g('grasp_quat').value))
        self.table_z_threshold = g('table_z_threshold').value
        self.workspace_x_min = g('workspace_x_min').value
        self.workspace_x_max = g('workspace_x_max').value
        self.workspace_y_min = g('workspace_y_min').value
        self.workspace_y_max = g('workspace_y_max').value
        self.approach_height = g('approach_height').value
        self.safe_height = g('safe_height').value
        self.grasp_depth = g('grasp_depth').value
        self.place_x = g('place_x').value
        self.place_y = g('place_y').value
        self.place_z = g('place_z').value
        self.gripper_open_width = g('gripper_open_width').value
        self.gripper_close_width = g('gripper_close_width').value
        self.planning_time = g('planning_time').value
        self.num_planning_attempts = g('num_planning_attempts').value
        self.velocity_scaling = g('velocity_scaling').value
        self.accel_scaling = g('accel_scaling').value
        self.cartesian_eef_step = g('cartesian_eef_step').value
        self.cartesian_min_fraction = g('cartesian_min_fraction').value
        self.cartesian_jump_threshold = g('cartesian_jump_threshold').value
        self.pos_tolerance = g('pos_tolerance').value
        self.ori_tolerance = g('ori_tolerance').value
        self.home_joints_deg = list(g('home_joints_deg').value)

        # ── 关节名 ──
        self.arm_joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        self.home_joints = self._make_home_joints(self.home_joints_deg)
        self.gripper_joint_names = ['gripper_joint1', 'gripper_joint2']

        # ── 回调组 (允许并发) ──
        self.cb_group = ReentrantCallbackGroup()

        # ── Action / Service Clients ──
        self.gripper_ac = ActionClient(
            self, FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory',
            callback_group=self.cb_group)
        self.move_group_ac = ActionClient(
            self, MoveGroup, '/move_action',
            callback_group=self.cb_group)
        self.execute_ac = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory',
            callback_group=self.cb_group)
        self.cartesian_cli = self.create_client(
            GetCartesianPath, '/compute_cartesian_path',
            callback_group=self.cb_group)

        # ── 关节状态订阅 ──
        self._joint_lock = threading.Lock()
        self._latest_joint_state = None
        self.create_subscription(
            JointState, '/feedback/joint_states', self._joint_cb, 10,
            callback_group=self.cb_group)

        # ── 目标订阅 ──
        self.create_subscription(
            PoseStamped, '/grasp/target', self._target_cb, 10,
            callback_group=self.cb_group)

        # ── 忙状态 ──
        self._busy_lock = threading.Lock()
        self.is_busy = False
        self._done_event = threading.Event()

        self._wait_servers()

        self.get_logger().info('抓取执行节点已启动')
        self.get_logger().info(
            f'等待 /grasp/target，frame={self.base_frame}，Z_min={self.table_z_threshold:.3f}m')

    # ─────────────────────────── 参数校验 ───────────────────────────
    def _normalize_quat(self, quat):
        if len(quat) != 4:
            raise ValueError(f'grasp_quat 必须包含 4 个数，当前为 {len(quat)} 个')
        quat = [float(v) for v in quat]
        norm = math.sqrt(sum(v * v for v in quat))
        if norm < 1e-6:
            raise ValueError('grasp_quat 范数接近 0，无法使用')
        normalized = [v / norm for v in quat]
        if abs(norm - 1.0) > 1e-3:
            self.get_logger().warn(f'grasp_quat 已归一化，原范数={norm:.4f}')
        return normalized

    def _make_home_joints(self, home_joints_deg):
        if len(home_joints_deg) != len(self.arm_joint_names):
            raise ValueError(
                f'home_joints_deg 必须包含 {len(self.arm_joint_names)} 个数，'
                f'当前为 {len(home_joints_deg)} 个')
        return {
            name: math.radians(float(deg))
            for name, deg in zip(self.arm_joint_names, home_joints_deg)
        }

    def _target_in_workspace(self, x, y) -> bool:
        return (self.workspace_x_min <= x <= self.workspace_x_max and
                self.workspace_y_min <= y <= self.workspace_y_max)

    # ─────────────────────────── 启动等待 ───────────────────────────
    def _wait_servers(self):
        self.get_logger().info('检查 MoveIt/夹爪服务...')
        if not self.gripper_ac.wait_for_server(timeout_sec=15.0):
            raise RuntimeError('gripper_controller/follow_joint_trajectory 不可用')
        if not self.move_group_ac.wait_for_server(timeout_sec=15.0):
            raise RuntimeError('/move_action 不可用')
        if not self.execute_ac.wait_for_server(timeout_sec=15.0):
            raise RuntimeError('/execute_trajectory 不可用')
        if not self.cartesian_cli.wait_for_service(timeout_sec=15.0):
            raise RuntimeError('/compute_cartesian_path 服务不可用')
        self.get_logger().info('服务检查通过')

    # ─────────────────────────── 回调 ───────────────────────────
    def _joint_cb(self, msg: JointState):
        with self._joint_lock:
            self._latest_joint_state = msg

    def _build_robot_state(self) -> RobotState:
        """从最新 joint_state 构造规划起点。"""
        with self._joint_lock:
            js = self._latest_joint_state
        rs = RobotState()
        if js is not None:
            rs.joint_state = js
        else:
            rs.is_diff = True
        return rs

    def _target_cb(self, msg: PoseStamped):
        if msg.header.frame_id and msg.header.frame_id != self.base_frame:
            self.get_logger().error(
                f'坐标系不匹配: {msg.header.frame_id} != {self.base_frame}，拒绝。')
            return

        p = msg.pose.position
        with self._busy_lock:
            if self.is_busy:
                self.get_logger().warn('正在执行抓取，忽略新目标。')
                return
            if p.z < self.table_z_threshold:
                self.get_logger().error(
                    f'目标过低: z={p.z:.4f} < {self.table_z_threshold:.3f}，拒绝。')
                return
            if not self._target_in_workspace(p.x, p.y):
                self.get_logger().error(
                    f'目标超出工作区: x={p.x:.4f}, y={p.y:.4f}，拒绝。')
                return
            if not self._target_in_workspace(self.place_x, self.place_y):
                self.get_logger().error(
                    f'放置点超出工作区: x={self.place_x:.4f}, y={self.place_y:.4f}，拒绝。')
                return
            self.is_busy = True

        # 使用实测抓取姿态四元数
        quat = self.grasp_quat

        self.get_logger().info(
            f'收到目标: x={p.x:.4f}, y={p.y:.4f}, z={p.z:.4f}')
        try:
            confirm = input('按回车开始执行抓取；输入 n 后回车取消: ').strip().lower()
        except EOFError:
            confirm = 'n'
        if confirm in ('n', 'no', 'q', 'quit', 'cancel'):
            self.get_logger().warn('用户取消本次抓取。')
            with self._busy_lock:
                self.is_busy = False
            return

        threading.Thread(
            target=self._run_sequence,
            args=(p.x, p.y, p.z, quat),
            daemon=True).start()

    # ─────────────── 同步等待 future 的工具 ───────────────
    @staticmethod
    def _spin_until(future, timeout):
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.02)
        return True

    # ─────────────────────────── 夹爪 ───────────────────────────
    def _send_gripper(self, width, duration=1.5, timeout=5.0) -> bool:
        j1 = float(width * 0.5)
        j2 = float(-width * 0.5)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.gripper_joint_names
        pt = JointTrajectoryPoint()
        pt.positions = [j1, j2]
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration % 1) * 1e9)
        goal.trajectory.points.append(pt)

        self.get_logger().debug(f'gripper → {width * 1000:.1f} mm')

        send_future = self.gripper_ac.send_goal_async(goal)
        if not self._spin_until(send_future, timeout):
            self.get_logger().warn('gripper send timeout')
            return False
        gh = send_future.result()
        if gh is None or not gh.accepted:
            self.get_logger().warn('gripper goal rejected')
            return False

        res_future = gh.get_result_async()
        if not self._spin_until(res_future, timeout + duration):
            self.get_logger().warn('gripper result timeout')
            return False

        code = res_future.result().result.error_code
        if code != 0:
            self.get_logger().warn(f'gripper error_code={code}')
            return False
        return True

    # ──────────────── 姿态约束 (用四元数) ────────────────
    def _make_pose_constraints(self, x, y, z, quat) -> Constraints:
        """quat: [x, y, z, w]"""
        c = Constraints()
        c.name = 'target_pose'

        # 位置约束
        pc = PositionConstraint()
        pc.header.frame_id = self.base_frame
        pc.link_name = self.tcp_link
        pc.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.SPHERE
        prim.dimensions = [self.pos_tolerance]
        pc.constraint_region.primitives.append(prim)

        region_pose = Pose()
        region_pose.position.x = float(x)
        region_pose.position.y = float(y)
        region_pose.position.z = float(z)
        region_pose.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(region_pose)
        pc.weight = 1.0
        c.position_constraints.append(pc)

        # 姿态约束 (直接用四元数)
        oc = OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = self.tcp_link
        oc.orientation.x = float(quat[0])
        oc.orientation.y = float(quat[1])
        oc.orientation.z = float(quat[2])
        oc.orientation.w = float(quat[3])
        oc.absolute_x_axis_tolerance = self.ori_tolerance
        oc.absolute_y_axis_tolerance = self.ori_tolerance
        oc.absolute_z_axis_tolerance = self.ori_tolerance
        oc.weight = 1.0
        c.orientation_constraints.append(oc)
        return c

    # ──────────────── MoveGroup 规划+执行 (关节空间) ────────────────
    def _move_to_pose(self, x, y, z, quat, timeout=20.0) -> bool:
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.planning_group
        req.num_planning_attempts = self.num_planning_attempts
        req.allowed_planning_time = self.planning_time
        req.max_velocity_scaling_factor = self.velocity_scaling
        req.max_acceleration_scaling_factor = self.accel_scaling
        req.start_state = self._build_robot_state()
        req.goal_constraints.append(
            self._make_pose_constraints(x, y, z, quat))
        goal.planning_options.plan_only = False

        self.get_logger().debug(
            f'MoveIt → ({x:.4f}, {y:.4f}, {z:.4f})')

        send_future = self.move_group_ac.send_goal_async(goal)
        if not self._spin_until(send_future, timeout):
            self.get_logger().error('MoveGroup send timeout')
            return False
        gh = send_future.result()
        if gh is None or not gh.accepted:
            self.get_logger().error('MoveGroup goal rejected')
            return False

        res_future = gh.get_result_async()
        if not self._spin_until(res_future, timeout + self.planning_time):
            self.get_logger().error('MoveGroup result timeout')
            return False

        wrapped = res_future.result()
        code = wrapped.result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f'MoveGroup 失败 status={wrapped.status}, error_code={code}')
            return False
        self.get_logger().debug('MoveIt 运动完成')
        return True

    # ──────────────── 笛卡尔直线运动 ────────────────
    def _move_cartesian(self, x, y, z, quat, timeout=20.0) -> bool:
        target = Pose()
        target.position.x = float(x)
        target.position.y = float(y)
        target.position.z = float(z)
        target.orientation.x = float(quat[0])
        target.orientation.y = float(quat[1])
        target.orientation.z = float(quat[2])
        target.orientation.w = float(quat[3])

        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.start_state = self._build_robot_state()
        req.group_name = self.planning_group
        req.link_name = self.tcp_link
        req.waypoints = [target]
        req.max_step = self.cartesian_eef_step
        req.jump_threshold = self.cartesian_jump_threshold
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = self.velocity_scaling
        req.max_acceleration_scaling_factor = self.accel_scaling

        self.get_logger().debug(
            f'Cartesian → ({x:.4f}, {y:.4f}, {z:.4f})')

        srv_future = self.cartesian_cli.call_async(req)
        if not self._spin_until(srv_future, timeout):
            self.get_logger().error('compute_cartesian_path timeout')
            return False

        resp = srv_future.result()
        if resp is None:
            self.get_logger().error('compute_cartesian_path 无响应')
            return False
        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f'笛卡尔规划失败 error_code={resp.error_code.val}')
            return False
        if resp.fraction < self.cartesian_min_fraction:
            self.get_logger().error(
                f'笛卡尔覆盖率 {resp.fraction:.1%} < {self.cartesian_min_fraction:.0%}')
            return False

        self.get_logger().debug(f'笛卡尔覆盖率 {resp.fraction:.1%}')
        return self._execute_trajectory(resp.solution, timeout)

    def _execute_trajectory(self, robot_traj, timeout=20.0) -> bool:
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = robot_traj

        send_future = self.execute_ac.send_goal_async(goal)
        if not self._spin_until(send_future, timeout):
            self.get_logger().error('ExecuteTrajectory send timeout')
            return False
        gh = send_future.result()
        if gh is None or not gh.accepted:
            self.get_logger().error('ExecuteTrajectory rejected')
            return False

        res_future = gh.get_result_async()
        if not self._spin_until(res_future, timeout):
            self.get_logger().error('ExecuteTrajectory result timeout')
            return False

        code = res_future.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f'执行失败 error_code={code}')
            return False
        self.get_logger().debug('直线运动完成')
        return True

    # ──────────────── 回初始位 ────────────────
    def _go_home(self, timeout=20.0) -> bool:
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = self.planning_group
        req.num_planning_attempts = 5
        req.allowed_planning_time = self.planning_time
        req.max_velocity_scaling_factor = self.velocity_scaling
        req.max_acceleration_scaling_factor = self.accel_scaling
        req.start_state = self._build_robot_state()

        c = Constraints()
        c.name = 'home'
        for n in self.arm_joint_names:
            jc = JointConstraint()
            jc.joint_name = n
            jc.position = self.home_joints[n]
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints.append(c)
        goal.planning_options.plan_only = False

        self.get_logger().debug('回初始位')
        send_future = self.move_group_ac.send_goal_async(goal)
        if not self._spin_until(send_future, timeout):
            return False
        gh = send_future.result()
        if gh is None or not gh.accepted:
            return False
        res_future = gh.get_result_async()
        if not self._spin_until(res_future, timeout + self.planning_time):
            return False
        return (res_future.result().result.error_code.val
                == MoveItErrorCodes.SUCCESS)

    # ─────────────────────────── 抓取序列 ───────────────────────────
    def _run_sequence(self, tx, ty, tz, quat):
        try:
            self.get_logger().info('开始抓取')
            self.get_logger().info(f'目标: x={tx:.4f}, y={ty:.4f}, z={tz:.4f}')

            # [1] 开夹爪，同时移动到预抓位
            self.get_logger().info('[1/7] 开夹爪 + 到预抓位')
            gripper_thread = threading.Thread(
                target=self._send_gripper,
                args=(self.gripper_open_width,),
                daemon=True)
            gripper_thread.start()
            if not self._move_to_pose(tx, ty, tz + self.approach_height, quat):
                gripper_thread.join(timeout=0.1)
                self.get_logger().error('预抓位失败，终止。')
                return
            gripper_thread.join(timeout=3.0)

            # [2] 下降到抓取位置
            self.get_logger().info('[2/7] 下降抓取')
            if not self._move_cartesian(tx, ty, tz + self.grasp_depth, quat):
                self.get_logger().error('下降抓取失败，终止。')
                return

            # [3] 闭合夹爪
            self.get_logger().info('[3/7] 合夹爪')
            if not self._send_gripper(self.gripper_close_width, duration=2.0):
                self.get_logger().error('合夹爪失败，终止。')
                return

            # [4] 抬起到安全位
            self.get_logger().info('[4/7] 抬起到安全位')
            if not self._move_cartesian(tx, ty, self.safe_height, quat):
                if not self._move_to_pose(tx, ty, self.safe_height, quat):
                    self.get_logger().error('抬起失败，终止。')
                    return

            # [5] 移动到放置位上方
            self.get_logger().info(
                f'[5/7] 到放置位上方: x={self.place_x:.3f}, '
                f'y={self.place_y:.3f}, z={self.safe_height:.3f}')
            px = self.place_x
            py = self.place_y
            if not self._move_to_pose(px, py, self.safe_height, quat):
                self.get_logger().error('到放置位上方失败，终止。请检查放置点是否可达。')
                return

            # [6] 直接移动到放置位并开夹爪
            self.get_logger().info('[6/7] 放置')
            if not self._move_cartesian(px, py, self.place_z, quat):
                self.get_logger().error('移动到放置位失败，终止。')
                return
            if not self._send_gripper(self.gripper_open_width):
                self.get_logger().error('放置开夹爪失败，终止。')
                return

            # [7] 回到 home 点
            self.get_logger().info('[7/7] 回 home')
            if not self._go_home():
                self.get_logger().error('回 home 失败。')
                return

            self.get_logger().info('抓取完成')

        except Exception as e:
            self.get_logger().error(
                f'❌ 抓取序列异常: {type(e).__name__}: {e}')

        finally:
            with self._busy_lock:
                self.is_busy = False
            self._done_event.set()


def main(args=None):
    rclpy.init(args=args)
    node = GraspExecutorNode()

    # 多线程执行器，让子线程发起的 action future 能正常完成
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        while rclpy.ok() and not node._done_event.is_set():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

