import threading

import numpy as np

from mcp_server.ros.camera import CameraStream


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
