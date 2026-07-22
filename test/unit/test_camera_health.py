import threading
from types import SimpleNamespace

import numpy as np

from mcp_server.ros.camera import CameraStream
from mcp_server.ros_bridge import RobotBridge


class _Stamp:
    sec = 1
    nanosec = 0


def _stream(color_shape=(480, 640), depth_shape=(480, 640), skew_ns=20_000_000):
    stream = CameraStream.__new__(CameraStream)
    stream._lock = threading.Lock()
    stream._ready = threading.Event()
    stream._color_sample = (
        1_000_000_000, _Stamp(), "camera_color_optical_frame",
        np.zeros((*color_shape, 3), dtype=np.uint8),
    )
    stream._depth_sample = (
        1_000_000_000 + skew_ns, _Stamp(), "camera_depth_optical_frame",
        np.zeros(depth_shape, dtype=np.uint16),
    )
    stream._paired_images = None
    stream._color_info = {"fx": 1.0}
    stream._max_pair_skew_ns = 80_000_000
    stream._last_pair_monotonic = None
    stream._last_pair_skew_ms = None
    stream._last_rejection = "waiting_for_frames"
    return stream


def test_health_accepts_registered_synchronized_pair():
    stream = _stream()
    stream._try_pair_locked()
    health = stream.health_status()
    assert health["pair_received"] is True
    assert health["pair_fresh"] is True
    assert health["registered_shapes_match"] is True
    assert health["pair_skew_ms"] == 20.0
    assert health["last_rejection"] is None


def test_health_rejects_timestamp_skew():
    stream = _stream(skew_ns=100_000_000)
    stream._try_pair_locked()
    health = stream.health_status()
    assert health["pair_received"] is False
    assert health["last_rejection"] == "timestamp_skew"


def test_health_rejects_unregistered_shape_mismatch():
    stream = _stream(depth_shape=(400, 640))
    stream._try_pair_locked()
    health = stream.health_status()
    assert health["pair_received"] is False
    assert health["registered_shapes_match"] is False
    assert health["last_rejection"] == "shape_mismatch"


def _robot_health_bridge(*, camera_publisher=True):
    camera = SimpleNamespace(
        health_status=lambda: {
            "pair_fresh": False,
            "registered_shapes_match": False,
            "camera_info_received": True,
            "pair_age_s": 9.56,
        }
    )
    status = {
        "can": {"up": True},
        "endpoints": {
            "move_action": True,
            "camera_color": camera_publisher,
            "planning_scene_apply": True,
            "planning_scene_get": True,
            "handeye_publisher": True,
            "octomap_control": False,
        },
        "topics": {},
        "processes": {},
    }
    return SimpleNamespace(
        node=SimpleNamespace(camera=camera),
        bringup=SimpleNamespace(status=lambda: status),
        get_octomap_enabled_on_prepare=lambda: False,
        can_transform=lambda *_args, **_kwargs: True,
    )


def test_idle_robot_status_keeps_stale_on_demand_pair_as_diagnostic():
    health = RobotBridge.health_status(
        _robot_health_bridge(), require_fresh_camera=False,
    )

    assert health["ready"] is True
    assert health["failures"] == []
    assert health["checks"]["rgbd_pair_fresh"] is False
    assert health["checks"]["depth_registered"] is False
    assert health["camera"]["pair_age_s"] == 9.56
    assert health["camera_capture_mode"] == "on_demand"
    assert health["fresh_camera_required"] is False


def test_prepare_health_still_rejects_stale_or_unregistered_pair():
    health = RobotBridge.health_status(
        _robot_health_bridge(), require_fresh_camera=True,
    )

    assert health["ready"] is False
    assert set(health["failures"]) == {
        "rgbd_pair_fresh", "depth_registered",
    }
    assert health["fresh_camera_required"] is True


def test_idle_robot_status_still_requires_camera_publisher():
    health = RobotBridge.health_status(
        _robot_health_bridge(camera_publisher=False),
        require_fresh_camera=False,
    )

    assert health["ready"] is False
    assert health["failures"] == ["camera_publisher"]
