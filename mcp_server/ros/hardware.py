"""Joint feedback and gripper-controller adapters."""

import threading

from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


ARM_JOINT_NAMES = [f"joint{index}" for index in range(1, 8)]
GRIPPER_JOINT_NAMES = ["gripper_joint1", "gripper_joint2"]


class JointStateMonitor:
    def __init__(self, node, callback_group):
        self._lock = threading.Lock()
        self._message = None
        self._subscription = node.create_subscription(
            JointState,
            "/feedback/joint_states",
            self._callback,
            10,
            callback_group=callback_group,
        )

    def _callback(self, message):
        with self._lock:
            self._message = message

    def latest_message(self):
        with self._lock:
            return self._message

    def as_dict(self) -> dict:
        message = self.latest_message()
        if message is None:
            return {"joints": {}, "gripper": {}, "gripper_width": None}
        joints = {
            name: float(position)
            for name, position in zip(message.name, message.position)
            if name in ARM_JOINT_NAMES
        }
        gripper = {
            name: float(position)
            for name, position in zip(message.name, message.position)
            if name in GRIPPER_JOINT_NAMES
        }
        width = None
        if all(name in gripper for name in GRIPPER_JOINT_NAMES):
            width = abs(gripper["gripper_joint1"] - gripper["gripper_joint2"])
        return {"joints": joints, "gripper": gripper, "gripper_width": width}


class GripperController:
    def __init__(self, node, callback_group):
        self._node = node
        self.client = ActionClient(
            node,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
            callback_group=callback_group,
        )

    def control(self, width: float, duration: float = 1.5, timeout: float = 5.0):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = GRIPPER_JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = [float(width * 0.5), float(-width * 0.5)]
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1) * 1e9)
        goal.trajectory.points.append(point)

        sent = self.client.send_goal_async(goal)
        self._node._track_goal(sent)
        if not self._node._spin_until(sent, timeout):
            return False, "gripper action server did not respond"
        handle = sent.result()
        if handle is None or not handle.accepted:
            return False, "gripper goal rejected; controller may be busy"
        result = handle.get_result_async()
        if not self._node._spin_until(result, timeout + duration):
            return False, "gripper result did not arrive in time"
        error_code = result.result().result.error_code
        if error_code != 0:
            return False, f"gripper failed with error_code={error_code}"
        return True, "ok"
