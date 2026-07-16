import threading
import time
from concurrent.futures import Future

from mcp_server.ros.futures import wait_for_future


class _Context:
    def __init__(self, active=True):
        self.active = active

    def ok(self):
        return self.active


def test_wait_returns_when_future_completes():
    future = Future()
    threading.Timer(0.02, lambda: future.set_result("done")).start()
    assert wait_for_future(future, 1.0)
    assert future.result() == "done"


def test_wait_uses_timeout_without_cancelling_future():
    future = Future()
    started = time.monotonic()
    assert not wait_for_future(future, 0.03)
    assert time.monotonic() - started < 0.2
    assert not future.cancelled()


def test_wait_stops_when_ros_context_closes():
    future = Future()
    context = _Context(active=False)
    assert not wait_for_future(future, 10.0, context=context)
