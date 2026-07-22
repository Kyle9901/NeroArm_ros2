from types import SimpleNamespace

import pytest

from mcp_server import task_server


class _Executor:
    def __init__(self):
        self.calls = []

    def execute_task(self, task, params=None):
        self.calls.append((task, params))
        return SimpleNamespace(
            status="completed",
            messages=["ok"],
            user_output={"target": "水瓶"},
            error=None,
            session_id=None,
            question=None,
            options=None,
        )


class _StatusBridge:
    health_freshness_requests = []
    @staticmethod
    def get_joint_state():
        return {"joints": {"joint1": 0.0}}

    @staticmethod
    def get_holding():
        return False

    @staticmethod
    def get_workspace_bounds():
        return {"x_min": -0.55, "x_max": 0.25, "y_min": -0.55, "y_max": 0.20}

    @staticmethod
    def get_safe_height():
        return 0.2267

    @staticmethod
    def get_desk_surface_z():
        return -0.013

    @staticmethod
    def get_tcp_offset():
        return [0.1733, 0.0, -0.0235, -1.5708, 0.0, -1.5708]

    @staticmethod
    def get_fingertip_depth():
        return 0.04

    @staticmethod
    def get_grasp_tilt_angles_deg():
        return [0, 30, 60]

    @staticmethod
    def get_cylinder_tilt_angles_deg():
        return [0, 60, 90]

    @staticmethod
    def get_cylinder_side_grasp_height_ratio():
        return 0.45

    @staticmethod
    def get_transparent_bottle_upright_axis_overtravel_m():
        return 0.020

    @staticmethod
    def get_cylinder_min_diameter_m():
        return 0.035

    @staticmethod
    def get_cylinder_max_diameter_m():
        return 0.085

    @staticmethod
    def get_cylinder_min_length_m():
        return 0.08

    @staticmethod
    def get_cylinder_max_length_m():
        return 0.30

    @staticmethod
    def get_grasp_pregrasp_distance():
        return 0.08

    @staticmethod
    def get_grasp_retreat_distance():
        return 0.08

    @staticmethod
    def get_reverse_branch_tolerance_rad():
        return 0.10

    @staticmethod
    def get_observation_joints_deg():
        return [0.0, -66.0, 0.0, 120.0, 0.0, 0.0, 60.0]

    @staticmethod
    def get_carry_joints_deg():
        return [0.0, -20.0, 0.0, 80.0, 0.0, 0.0, 50.0]

    @staticmethod
    def health_status(*, require_fresh_camera=True):
        _StatusBridge.health_freshness_requests.append(require_fresh_camera)
        return {"ok": True}


def test_formal_mcp_execute_task_passes_water_bottle_to_graph_executor():
    executor = _Executor()

    result = task_server._call_tool(
        "arm_execute_task",
        {"task": "抓取水瓶"},
        bridge=object(),
        vlm=object(),
        yolo=object(),
        executor=executor,
        sessions={},
    )

    assert executor.calls == [("抓取水瓶", {})]
    assert result["status"] == "completed"
    assert result["user_output"] == {"target": "水瓶"}


def test_formal_mcp_status_exposes_upright_bottle_overtravel():
    _StatusBridge.health_freshness_requests = []
    result = task_server._call_tool(
        "arm_get_status",
        {},
        bridge=_StatusBridge(),
        vlm=object(),
        yolo=object(),
        executor=_Executor(),
        sessions={},
    )

    assert result["success"] is True
    assert result["grasp_geometry"][
        "transparent_bottle_upright_axis_overtravel_m"
    ] == pytest.approx(0.020)
    assert _StatusBridge.health_freshness_requests == [False]


def test_formal_mcp_tool_description_lists_water_bottle_example():
    execute_tool = next(
        item for item in task_server.TOOL_DEFINITIONS
        if item["name"] == "arm_execute_task"
    )
    assert "抓取水瓶" in execute_tool["description"]
