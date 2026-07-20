import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

from mcp_server import task_server


def _app():
    return SimpleNamespace(
        bridge=object(),
        vlm=object(),
        yolo=object(),
        executor=object(),
        sessions={},
    )


def test_robot_task_runs_off_the_mcp_event_loop(monkeypatch):
    event_loop_thread = threading.get_ident()
    worker_threads = []

    def fake_call(*_args):
        worker_threads.append(threading.get_ident())
        return {"status": "completed"}

    monkeypatch.setattr(task_server, "_call_tool", fake_call)

    async def scenario():
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await task_server._dispatch_tool_async(
                "arm_execute_task",
                {"task": "test"},
                _app(),
                asyncio.Lock(),
                executor,
            )

    result = asyncio.run(scenario())
    assert result["status"] == "completed"
    assert worker_threads and worker_threads[0] != event_loop_thread


def test_busy_task_rejects_new_task_but_allows_stop(monkeypatch):
    calls = []

    def fake_call(name, *_args):
        calls.append(name)
        return {"success": True}

    monkeypatch.setattr(task_server, "_call_tool", fake_call)

    async def scenario():
        lock = asyncio.Lock()
        executor = ThreadPoolExecutor(max_workers=1)
        await lock.acquire()
        try:
            duplicate = await task_server._dispatch_tool_async(
                "arm_execute_task", {"task": "second"}, _app(), lock, executor
            )
            stop = await task_server._dispatch_tool_async(
                "arm_stop", {}, _app(), lock, executor
            )
            octomap = await task_server._dispatch_tool_async(
                "arm_configure_octomap",
                {"enabled": True},
                _app(),
                lock,
                executor,
            )
        finally:
            lock.release()
            executor.shutdown()
        return duplicate, stop, octomap

    duplicate, stop, octomap = asyncio.run(scenario())
    assert duplicate["status"] == "failed"
    assert stop["success"] is True
    assert calls == ["arm_stop"]
    assert octomap["success"] is False
