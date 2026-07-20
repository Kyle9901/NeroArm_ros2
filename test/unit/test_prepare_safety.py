from mcp_server.components.base import ComponentResult
from mcp_server.skills import prepare as prepare_skill


class _Node:
    def __init__(self, desk_ok=True):
        self.desk_ok = desk_ok
        self.desk_calls = 0

    def add_desk_collision(self):
        self.desk_calls += 1
        return self.desk_ok


class _Bridge:
    def __init__(self, *, desk_ok=True):
        self.node = _Node(desk_ok)
        self.octomap_calls = []

    def get_octomap_enabled_on_prepare(self):
        return False

    def get_node_counts(self, _names):
        return {}

    def set_octomap_enabled(self, enabled):
        self.octomap_calls.append(enabled)
        return {"success": True, "enabled": enabled}

    def get_desk_collision_enabled(self):
        return True

    def health_status(self):
        return {"ready": True, "failures": [], "camera": {}}


def _ready_status(*_args, **_kwargs):
    return ComponentResult.success(
        endpoints={
            "move_action": True,
            "camera_color": True,
            "handeye_publisher": True,
            "planning_scene_apply": True,
            "planning_scene_get": True,
            "tf": True,
        }
    )


def test_prepare_defaults_octomap_off_and_adds_only_desk(monkeypatch):
    bridge = _Bridge()
    monkeypatch.setattr(prepare_skill.infra, "bringup_status", _ready_status)
    monkeypatch.setattr(
        prepare_skill.perception,
        "capture_image",
        lambda *_args, **_kwargs: ComponentResult.success(),
    )

    result = prepare_skill.prepare(bridge)

    assert result.ok
    assert bridge.octomap_calls == [False]
    assert bridge.node.desk_calls == 1


def test_prepare_desk_failure_is_a_safety_failure_before_capture(monkeypatch):
    bridge = _Bridge(desk_ok=False)
    captured = []
    monkeypatch.setattr(prepare_skill.infra, "bringup_status", _ready_status)
    monkeypatch.setattr(
        prepare_skill.perception,
        "capture_image",
        lambda *_args, **_kwargs: (
            captured.append(True) or ComponentResult.success()
        ),
    )

    result = prepare_skill.prepare(bridge)

    assert not result.ok
    assert result.failed_step == "desk_collision"
    assert captured == []
