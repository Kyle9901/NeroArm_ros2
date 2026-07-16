"""Common waiting policy for ROS and action futures."""

import threading
import time


def wait_for_future(future, timeout: float, *, context=None) -> bool:
    """Wait without busy polling, using a monotonic deadline.

    The future is deliberately not cancelled on timeout: cancelling an action
    send future is not equivalent to cancelling the accepted robot goal.
    """
    if future.done():
        return True

    completed = threading.Event()
    future.add_done_callback(lambda _future: completed.set())
    deadline = time.monotonic() + max(0.0, float(timeout))

    while not future.done():
        if context is not None and not context.ok():
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return future.done()
        completed.wait(min(remaining, 0.1))
    return True
