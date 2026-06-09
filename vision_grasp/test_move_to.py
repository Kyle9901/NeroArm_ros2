#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoveIt 单点测试脚本 — 逐步调试机械臂定位

用法:
  # 测试运动到指定点 (XYZ, 单位m):
  python3 /home/alkaid/ros2_ws/test_move_to.py 0.0 0.0 0.30

  # 测试笛卡尔直线运动:
  python3 /home/alkaid/ros2_ws/test_move_to.py --cartesian 0.0 0.0 0.30

  # 测试物块上方 (假设物块在 X=-0.45, Y=-0.07, Z=0.10):
  python3 /home/alkaid/ros2_ws/test_move_to.py -0.45 -0.07 0.25

  # 回零位:
  python3 /home/alkaid/ros2_ws/test_move_to.py --home

功能:
  - 默认使用抓取姿态 quat=[0.476, 0.523, -0.523, 0.476]
  - --quat x y z w: 自定义姿态
  - --cartesian: 笛卡尔直线 (而不是关节空间)
  - --velocity 0.05: 速度比例
  - --home: 回零位
"""

import sys
import time
import argparse
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Pose, Vector3, TransformStamped
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath, ApplyPlanningScene
from moveit_msgs.msg import (
    Constraints, PositionConstraint, OrientationConstraint,
    JointConstraint, MoveItErrorCodes, RobotState,
    PlanningScene, CollisionObject, AttachedCollisionObject,
)


class TestMoveNode(Node):
    def __init__(self):
        super().__init__('test_move_node')

        self.arm_joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7']

        self.cb_group = ReentrantCallbackGroup()

        # TF2 用于查询当前姿态
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Action clients
        self.move_group_ac = ActionClient(
            self, MoveGroup, '/move_action', callback_group=self.cb_group)
        self.execute_ac = ActionClient(
            self, ExecuteTrajectory, '/execute_trajectory', callback_group=self.cb_group)
        self.cartesian_cli = self.create_client(
            GetCartesianPath, '/compute_cartesian_path', callback_group=self.cb_group)

        # Planning Scene — 用于添加桌面等 collision objects
        self.planning_scene_cli = self.create_client(
            ApplyPlanningScene, '/apply_planning_scene', callback_group=self.cb_group)

        # Joint state (remap to match MoveIt)
        self._latest_js = None
        self._js_lock = threading.Lock()
        self.create_subscription(
            JointState, '/feedback/joint_states', self._js_cb, 10,
            callback_group=self.cb_group)

        self.get_logger().info('等待服务器...')
        self.move_group_ac.wait_for_server(timeout_sec=10)
        self.execute_ac.wait_for_server(timeout_sec=10)
        self.cartesian_cli.wait_for_service(timeout_sec=10)
        self.planning_scene_cli.wait_for_service(timeout_sec=10)
        self.get_logger().info('✅ 所有服务器就绪')

        # 桌面 collision object 已添加标志
        self._table_added = False

        # 注意: 不在 __init__ 中等待 joint_states！
        # executor.spin() 还没启动，callback 不会被调用，等待只会白白超时。
        # 等待逻辑移到 main() 中 spin_thread.start() 之后。

    def _js_cb(self, msg):
        with self._js_lock:
            self._latest_js = msg

    def _build_robot_state(self):
        rs = RobotState()
        with self._js_lock:
            if self._latest_js is not None:
                rs.joint_state = self._latest_js
            else:
                rs.is_diff = True
        return rs

    def _spin_until(self, future, timeout):
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.02)
        return True

    def move_to_pose(self, x, y, z, quat, velocity=0.05, accel=0.05):
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = 'arm'
        req.num_planning_attempts = 20
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = velocity
        req.max_acceleration_scaling_factor = accel
        req.start_state = self._build_robot_state()
        req.goal_constraints.append(self._make_constraints(x, y, z, quat))
        goal.planning_options.plan_only = False

        self.get_logger().info(f'➡ MoveIt → ({x:.4f}, {y:.4f}, {z:.4f})')

        send_future = self.move_group_ac.send_goal_async(goal)
        if not self._spin_until(send_future, 30):
            self.get_logger().error('❌ MoveGroup send timeout')
            return False
        gh = send_future.result()
        if gh is None or not gh.accepted:
            self.get_logger().error('❌ MoveGroup goal rejected')
            return False

        res_future = gh.get_result_async()
        if not self._spin_until(res_future, 30):
            self.get_logger().error('❌ MoveGroup result timeout')
            return False

        code = res_future.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f'❌ 失败 error_code={code}')
            return False

        self.get_logger().info('✅ 运动完成')
        return True

    def move_cartesian(self, x, y, z, quat, velocity=0.05, accel=0.05):
        target = Pose()
        target.position.x = float(x)
        target.position.y = float(y)
        target.position.z = float(z)
        target.orientation.x = float(quat[0])
        target.orientation.y = float(quat[1])
        target.orientation.z = float(quat[2])
        target.orientation.w = float(quat[3])

        req = GetCartesianPath.Request()
        req.header.frame_id = 'base_link'
        req.start_state = self._build_robot_state()
        req.group_name = 'arm'
        req.link_name = 'tcp_link'
        req.waypoints = [target]
        req.max_step = 0.005
        req.jump_threshold = 2.0
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = velocity
        req.max_acceleration_scaling_factor = accel

        self.get_logger().info(f'➡ 笛卡尔直线 → ({x:.4f}, {y:.4f}, {z:.4f})')

        srv_future = self.cartesian_cli.call_async(req)
        if not self._spin_until(srv_future, 20):
            self.get_logger().error('❌ compute_cartesian_path timeout')
            return False

        resp = srv_future.result()
        if resp is None:
            self.get_logger().error('❌ compute_cartesian_path 无响应')
            return False
        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f'❌ 笛卡尔规划失败 error_code={resp.error_code.val}')
            return False

        self.get_logger().info(f'  覆盖率 {resp.fraction:.1%}')

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = resp.solution

        send_future = self.execute_ac.send_goal_async(goal)
        if not self._spin_until(send_future, 20):
            self.get_logger().error('❌ ExecuteTrajectory send timeout')
            return False
        gh = send_future.result()
        if gh is None or not gh.accepted:
            self.get_logger().error('❌ ExecuteTrajectory rejected')
            return False

        res_future = gh.get_result_async()
        if not self._spin_until(res_future, 20):
            self.get_logger().error('❌ ExecuteTrajectory result timeout')
            return False

        code = res_future.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f'❌ 执行失败 error_code={code}')
            return False

        self.get_logger().info('✅ 直线运动完成')
        return True

    def go_home(self):
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = 'arm'
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = 0.05
        req.max_acceleration_scaling_factor = 0.05
        req.start_state = self._build_robot_state()

        c = Constraints()
        c.name = 'home'
        for n in self.arm_joint_names:
            jc = JointConstraint()
            jc.joint_name = n
            jc.position = 0.0
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints.append(c)
        goal.planning_options.plan_only = False

        self.get_logger().info('➡ 回零位')

        send_future = self.move_group_ac.send_goal_async(goal)
        if not self._spin_until(send_future, 20):
            return False
        gh = send_future.result()
        if gh is None or not gh.accepted:
            return False
        res_future = gh.get_result_async()
        if not self._spin_until(res_future, 20):
            return False
        ok = res_future.result().result.error_code.val == MoveItErrorCodes.SUCCESS
        if ok:
            self.get_logger().info('✅ 回零位完成')
        return ok

    def add_table_collision_object(self, table_z=0.0, table_size=(2.0, 2.0, 0.02)):
        """将桌面添加为 collision object 到 planning scene。

        Args:
            table_z: 桌面表面高度 (m)，默认 0.0 表示 base_link 所在平面
            table_size: 桌面尺寸 (x, y, z厚度)，默认 2m x 2m x 2cm
        """
        if self._table_added:
            return True

        co = CollisionObject()
        co.id = 'table'
        co.header.frame_id = 'base_link'
        co.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(table_size)

        box_pose = Pose()
        box_pose.position.x = 0.0
        box_pose.position.y = 0.0
        # 桌面中心在表面下方半个厚度处
        box_pose.position.z = table_z - table_size[2] / 2.0
        box_pose.orientation.w = 1.0

        co.primitives.append(box)
        co.primitive_poses.append(box_pose)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(co)

        req = ApplyPlanningScene.Request()
        req.scene = scene

        self.get_logger().info(
            f'添加桌面 collision object: z_surface={table_z:.3f}, '
            f'size={table_size}, center_z={box_pose.position.z:.3f}')

        future = self.planning_scene_cli.call_async(req)
        if not self._spin_until(future, 5.0):
            self.get_logger().error('❌ apply_planning_scene timeout')
            return False

        resp = future.result()
        if resp is None or not resp.success:
            self.get_logger().error('❌ 添加桌面失败')
            return False

        self.get_logger().info('✅ 桌面已加入 planning scene')
        self._table_added = True
        return True

    def remove_table_collision_object(self):
        """从 planning scene 移除桌面。"""
        co = CollisionObject()
        co.id = 'table'
        co.header.frame_id = 'base_link'
        co.operation = CollisionObject.REMOVE

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(co)

        req = ApplyPlanningScene.Request()
        req.scene = scene

        future = self.planning_scene_cli.call_async(req)
        if self._spin_until(future, 5.0):
            self._table_added = False
            self.get_logger().info('桌面已移除')

    def get_current_quat(self):
        """查询当前 tcp_link 相对于 base_link 的四元数，返回 [x, y, z, w]"""
        try:
            t = self.tf_buffer.lookup_transform('base_link', 'tcp_link', rclpy.time.Time())
            q = t.transform.rotation
            return [q.x, q.y, q.z, q.w]
        except Exception as e:
            self.get_logger().warn(f'TF 查询失败: {e}')
            return None

    def _make_constraints(self, x, y, z, quat):
        c = Constraints()
        c.name = 'target'

        pc = PositionConstraint()
        pc.header.frame_id = 'base_link'
        pc.link_name = 'tcp_link'
        pc.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.SPHERE
        prim.dimensions = [0.01]
        pc.constraint_region.primitives.append(prim)
        rp = Pose()
        rp.position.x = float(x)
        rp.position.y = float(y)
        rp.position.z = float(z)
        rp.orientation.w = 1.0
        pc.constraint_region.primitive_poses.append(rp)
        pc.weight = 1.0
        c.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header.frame_id = 'base_link'
        oc.link_name = 'tcp_link'
        oc.orientation.x = float(quat[0])
        oc.orientation.y = float(quat[1])
        oc.orientation.z = float(quat[2])
        oc.orientation.w = float(quat[3])
        oc.absolute_x_axis_tolerance = 0.1
        oc.absolute_y_axis_tolerance = 0.1
        oc.absolute_z_axis_tolerance = 0.1
        oc.weight = 1.0
        c.orientation_constraints.append(oc)
        return c


def main():
    parser = argparse.ArgumentParser(description='MoveIt 单点测试')
    parser.add_argument('x', nargs='?', type=float, default=0.0)
    parser.add_argument('y', nargs='?', type=float, default=0.0)
    parser.add_argument('z', nargs='?', type=float, default=0.30)
    parser.add_argument('--cartesian', action='store_true',
                        help='使用笛卡尔直线运动')
    parser.add_argument('--quat', nargs=4, type=float,
                        default=[0.476, 0.523, -0.523, 0.476],
                        help='目标姿态四元数 xyzw')
    parser.add_argument('--velocity', type=float, default=0.05,
                        help='速度比例 (0.01~1.0)')
    parser.add_argument('--accel', type=float, default=0.05,
                        help='加速度比例 (0.01~1.0)')
    parser.add_argument('--home', action='store_true',
                        help='回零位 (忽略XYZ)')
    parser.add_argument('--get-quat', action='store_true',
                        help='读取当前 tcp_link 姿态四元数并退出')
    args = parser.parse_args()

    rclpy.init()
    node = TestMoveNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    # 在后台 spin，让 callback 能正常处理
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # 等待 joint_states (必须等 spin 启动后才有数据)
    node.get_logger().info('等待 joint_states...')
    timeout = 10.0
    t0 = time.time()
    while node._latest_js is None and time.time() - t0 < timeout:
        time.sleep(0.1)
    if node._latest_js is None:
        node.get_logger().error('❌ 未收到 joint_states，无法规划。请确认 CAN 驱动节点正常运行。')
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(1)
    else:
        node.get_logger().info(f'✅ 收到 joint_states ({len(node._latest_js.name)} joints)')

    try:
        if args.get_quat:
            # 等待 TF 数据
            time.sleep(0.5)
            quat = node.get_current_quat()
            if quat:
                node.get_logger().info(f'当前 tcp_link 四元数: [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]')
            else:
                node.get_logger().error('无法获取当前姿态')
        elif args.home:
            node.get_logger().info('=== 回零位 ===')
            node.go_home()
        else:
            x, y, z = args.x, args.y, args.z
            quat = args.quat

            # 添加桌面 collision object (桌面高度 0.0，即 base_link 平面)
            # 如果桌面实际高度不同，请修改 table_z 参数
            node.add_table_collision_object(table_z=0.0, table_size=(2.0, 2.0, 0.02))

            node.get_logger().info(f'=== 目标: ({x:.4f}, {y:.4f}, {z:.4f}) ===')
            node.get_logger().info(f'=== 姿态: quat={quat} ===')
            node.get_logger().info(f'=== 速度: {args.velocity}, 加速: {args.accel} ===')
            node.get_logger().info(f'=== 模式: {"笛卡尔直线" if args.cartesian else "关节空间"} ===')

            if args.cartesian:
                node.move_cartesian(x, y, z, quat, args.velocity, args.accel)
            else:
                node.move_to_pose(x, y, z, quat, args.velocity, args.accel)

    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()