from mcp_server.components import motion


class _Node:
    def __getattr__(self, name):
        raise AssertionError(f"robot command {name} must not be called")


class _Bridge:
    node = _Node()

    @staticmethod
    def is_task_stop_requested():
        return True


def test_stop_guard_blocks_motion_planning_execution_and_gripper_commands():
    bridge = _Bridge()
    calls = (
        lambda: motion.move_to_pose(bridge, 0.0, 0.0, 0.1),
        lambda: motion.move_cartesian(bridge, 0.0, 0.0, 0.1),
        lambda: motion.move_joints(bridge, [0.0] * 7),
        lambda: motion.execute_planned(bridge, object()),
        lambda: motion.control_gripper(bridge, 0.1),
        lambda: motion.plan_to_pose(bridge, 0.0, 0.0, 0.1),
        lambda: motion.plan_cartesian(bridge, 0.0, 0.0, 0.1),
        lambda: motion.plan_joints(bridge, [0.0] * 7),
        lambda: motion.solve_pose_ik(bridge, 0.0, 0.0, 0.1),
        lambda: motion.go_home(bridge),
        lambda: motion.go_carry(bridge),
    )

    for call in calls:
        result = call()
        assert not result.ok
        assert result.data["stop_requested"] is True
        assert result.data["motion_state_unknown"] is False
