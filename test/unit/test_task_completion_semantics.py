from mcp_server.components.base import ComponentResult
from mcp_server.orchestrator import graph as graph_module


class _Bridge:
    def __init__(self, holding):
        self._holding = holding
        self.home_calls = 0
        self.stop_requested = False

    def get_holding(self):
        return self._holding

    def get_task_context(self):
        return {}

    def get_place_pose(self):
        return {"x": -0.4, "y": -0.25, "z": 0.2}

    def is_task_stop_requested(self):
        return self.stop_requested


def test_grasp_only_route_does_not_append_place():
    pipeline = graph_module._fast_route("抓取红色物块", _Bridge(False))
    assert [step["skill"] for step in pipeline] == [
        "go_home", "locate_object", "grasp_object",
    ]
    assert pipeline[-1]["args"]["target"] == "红色物块"


def test_new_grasp_is_rejected_before_any_pipeline_motion_when_holding():
    pipeline = graph_module._fast_route("抓取红色物块", _Bridge(True))
    assert graph_module._grasp_start_error(pipeline, True) is not None
    assert graph_module._grasp_start_error(pipeline, None) is not None
    assert graph_module._grasp_start_error(pipeline, False) is None


def test_original_place_receives_selected_candidate_for_reverse_path():
    pipeline = graph_module._fast_route(
        "抓取红色物块并放回原位置", _Bridge(False)
    )
    place = pipeline[-1]
    assert place["skill"] == "place_object"
    assert place["args"]["reverse_candidate"] == "$grasp.selected_candidate"
    assert place["args"]["x"] == "$grasp.pick_x"


def test_cylinder_original_place_uses_selected_reverse_candidate():
    pipeline = graph_module._fast_route(
        "抓取水瓶并放回原位置", _Bridge(False)
    )
    place = pipeline[-1]
    assert place["skill"] == "place_object"
    assert place["args"]["x"] == "$grasp.pick_x"
    assert place["args"]["reverse_candidate"] == "$grasp.selected_candidate"


def test_cleanup_keeps_carry_pose_while_holding(monkeypatch):
    bridge = _Bridge(True)

    def _unexpected_home(_bridge):
        bridge.home_calls += 1
        return ComponentResult.success()

    monkeypatch.setattr(graph_module.motion, "go_home", _unexpected_home)
    result = graph_module._make_cleanup_node(bridge)({"messages": []})
    assert bridge.home_calls == 0
    assert "carry" in result["messages"][-1]


def test_cleanup_returns_to_observation_after_release(monkeypatch):
    bridge = _Bridge(False)

    def _home(_bridge):
        bridge.home_calls += 1
        return ComponentResult.success()

    monkeypatch.setattr(graph_module.motion, "go_home", _home)
    result = graph_module._make_cleanup_node(bridge)({"messages": []})
    assert bridge.home_calls == 1
    assert "观察位" in result["messages"][-1]


def test_cleanup_reports_failure_when_observation_pose_is_not_reached(monkeypatch):
    bridge = _Bridge(False)
    monkeypatch.setattr(
        graph_module.motion,
        "go_home",
        lambda _bridge: ComponentResult.failure("observation unreachable"),
    )
    result = graph_module._make_cleanup_node(bridge)({
        "status": "completed",
        "messages": [],
    })
    assert result["status"] == "failed"
    assert "observation unreachable" in result["error"]


def test_cleanup_does_not_move_when_holding_is_unknown(monkeypatch):
    bridge = _Bridge(None)

    def _unexpected_home(_bridge):
        bridge.home_calls += 1
        return ComponentResult.success()

    monkeypatch.setattr(graph_module.motion, "go_home", _unexpected_home)
    result = graph_module._make_cleanup_node(bridge)({"messages": []})
    assert bridge.home_calls == 0
    assert "不确定" in result["messages"][-1]


def test_cleanup_does_not_move_after_cooperative_stop(monkeypatch):
    bridge = _Bridge(False)
    bridge.stop_requested = True

    def _unexpected_home(_bridge):
        bridge.home_calls += 1
        return ComponentResult.success()

    monkeypatch.setattr(graph_module.motion, "go_home", _unexpected_home)
    result = graph_module._make_cleanup_node(bridge)({"messages": []})
    assert bridge.home_calls == 0
    assert result["status"] == "failed"
    assert "停止" in result["messages"][-1]


def test_stop_request_prevents_next_skill_and_all_retries(monkeypatch):
    bridge = _Bridge(False)
    bridge.stop_requested = True
    calls = []

    def _unexpected_skill(_bridge):
        calls.append(True)
        return graph_module.SkillResult.success()

    monkeypatch.setitem(graph_module._SKILL_FNS, "go_home", _unexpected_skill)
    node = graph_module._make_step_node(
        {"name": "home", "skill": "go_home", "args": {}},
        bridge,
        vlm=None,
        yolo=None,
        max_retries=3,
    )
    result = node({"step_outputs": {}})

    assert calls == []
    output = result["step_outputs"]["home"]
    assert output["_ok"] is False
    assert output["_retryable"] is False
    assert output["stop_requested"] is True


def test_plain_put_down_does_not_reuse_original_reverse_candidate():
    class Bridge(_Bridge):
        def get_task_context(self):
            return {
                "pick_x": -0.3,
                "pick_y": 0.0,
                "pick_z": 0.05,
                "selected_candidate": {"candidate_id": "old"},
            }

    pipeline = graph_module._fast_route("放下", Bridge(True))
    assert pipeline[0]["args"] == {"x": -0.4, "y": -0.25, "z": 0.2}
