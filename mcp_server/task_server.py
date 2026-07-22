#!/usr/bin/env python3
"""
MCP Server for AGX Robot Arm — new agent/orchestration entry point.
Usage: python -m mcp_server.task_server

OpenClaw only sees the high-level tools defined in this module.
"""

import asyncio
from concurrent.futures import Executor, ThreadPoolExecutor
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any

_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from .ros_bridge import RobotBridge
from .vlm_client import VlmClient
from .yolo_detector import LazyYoloDetector, YoloDetector
from .orchestrator.graph import GraphExecutor
from .skills.prepare import prepare as prepare_skill
from . import api_services


# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Tool definitions — OpenClaw-visible API
# ═══════════════════════════════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    {
        "name": "arm_execute_task",
        "description": (
            "Execute a robot task using LLM-powered pipeline planning + LangGraph execution. "
            "This is the ONLY tool needed for pick-and-place, visual grasping, and scene scanning. "
            "The Planning LLM internally generates a skill sequence, then LangGraph deterministically executes it. "
            "All z-offset handling, retry/recovery, and parameter passing is automatic.\n\n"
            "Just describe the task naturally. Examples:\n"
            "- '把蓝色方块放到右边'\n"
            "- '直接抓黄色方块'\n"
            "- '抓取水瓶'\n"
            "- '扫描桌面'\n"
            "- 'pick the red block and place it on the left'\n\n"
            "OPTIONAL: provide target (object description) and place (location word) to override LLM extraction.\n"
            "Place words: right/右边, left/左边, center/中间, front/前面, back/后面."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Natural language task, e.g. '把蓝色方块放到右边', '直接抓黄色方块', '扫描桌面'.",
                },
                "target": {
                    "type": "string",
                    "description": "Object to grasp, e.g. 'blue block', '红色方块'. Required for pick_and_place tasks.",
                },
                "place": {
                    "type": "string",
                    "description": "Where to place. Use right/left/center/front/back or 右边/左边/中间/前面/后面.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Resume a paused task from a previous needs_input response.",
                },
                "answer": {
                    "type": "object",
                    "description": "User's answer to a needs_input question.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "arm_prepare",
        "description": (
            "Prepare the robot system for task execution. Starts or checks arm, camera, hand-eye TF, "
            "MoveIt planning scene, and the configured desk collision box. "
            "Point-cloud filtering and OctoMap remain optional and disabled by default. "
            "arm_execute_task already runs this preparation internally, so normally call the task directly. "
            "Use arm_prepare for an explicit preflight check. If everything is already ready, it returns immediately."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "can_port": {"type": "string", "description": "CAN interface name. Default: can0."},
                "calib_name": {"type": "string", "description": "Hand-eye calibration name. Default: my_eih_calib_park."},
            },
            "required": [],
        },
    },
    {
        "name": "arm_get_status",
        "description": (
            "Get the current robot status: joint angles, gripper state, holding state, workspace bounds, "
            "safe height, desk surface Z, and grasp geometry parameters. RGB-D capture is on demand: "
            "pair_age_s describes the last cached frame and may grow while idle; it is diagnostic and "
            "does not require arm_prepare. Each task captures and validates a fresh pair before perception."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "arm_configure_octomap",
        "description": (
            "Enable or disable MoveIt OctoMap collision mapping at runtime. "
            "Disabling stops cloud updates and clears existing voxels; enabling resumes updates."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "true to enable OctoMap, false to disable and clear it.",
                },
            },
            "required": ["enabled"],
        },
    },
    {
        "name": "arm_stop",
        "description": (
            "Cooperatively block subsequent skills, retries, plans, trajectories, "
            "and gripper commands, then clear tracked goals. Due to a MoveIt Jazzy "
            "cancellation crash workaround, this does not cancel a trajectory already "
            "accepted by the controller and is not an emergency stop."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "arm_configure_vlm",
        "description": (
            "Configure VLM and YOLO settings at runtime. "
            "Use this to set/change VLM API key, API URL, model, YOLO confidence threshold, "
            "or toggle VLM fallback without restarting the MCP server. "
            "Leave a field empty to keep its current value."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "api_key": {"type": "string", "description": "VLM API key. Leave empty to keep current."},
                "api_url": {"type": "string", "description": "VLM API endpoint URL. Leave empty to keep current."},
                "model": {"type": "string", "description": "VLM model name. Leave empty to keep current."},
                "yolo_confidence": {"type": "number", "description": "YOLO confidence threshold (0.0-1.0). Leave empty to keep current."},
                "vlm_fallback": {"type": "boolean", "description": "Enable/disable VLM fallback when YOLO fails."},
            },
            "required": [],
        },
    },
    {
        "name": "arm_reset_context",
        "description": (
            "Reset the robot's semantic context (grasped object, action history). "
            "Call this when the user says 'reset', '重新开始', '清空状态', "
            "or when the robot's physical state may differ from the software's memory."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
]


# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Tool dispatch
# ═══════════════════════════════════════════════════════════════════════════════════════════

def _call_tool(name: str, args: dict, bridge: RobotBridge, vlm: VlmClient,
               yolo: YoloDetector | None, executor: GraphExecutor, sessions: dict) -> dict:
    try:
        if name == "arm_execute_task":
            return _execute_task(args, executor, sessions)
        if name == "arm_prepare":
            return _skill_result_to_dict(prepare_skill(
                bridge,
                can_port=args.get("can_port", "can0"),
                calib_name=args.get("calib_name", "my_eih_calib_park"),
            ))
        if name == "arm_get_status":
            return api_services.get_status(bridge)
        if name == "arm_configure_octomap":
            return api_services.configure_octomap(bridge, args["enabled"])
        if name == "arm_stop":
            return api_services.stop(bridge)
        if name == "arm_reset_context":
            bridge.reset_task_context()
            return {"success": True, "message": "task context reset"}
        if name == "arm_configure_vlm":
            return api_services.configure_runtime(
                vlm, yolo,
                args.get("api_key"), args.get("api_url"), args.get("model"),
                args.get("yolo_confidence"), args.get("vlm_fallback"),
            )
        return {"success": False, "error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


def _skill_result_to_dict(result) -> dict:
    return {
        "success": result.ok,
        "status": "completed" if result.ok else "failed",
        "holding": result.holding,
        "recovered": result.recovered,
        "retryable": result.retryable,
        "failed_step": result.failed_step,
        "data": result.data,
        "error": result.error,
    }


def _execute_task(args: dict, executor: GraphExecutor,
                  sessions: dict) -> dict:
    session_id = args.get("session_id")
    answer = args.get("answer")

    if session_id and session_id in sessions:
        return _resume_task(session_id, answer, sessions)

    task = args.get("task", "")
    if not task:
        return {"status": "failed", "error": "Missing task", "messages": ["缺少 task 参数"]}

    params = {}
    if args.get("target"):
        params["target"] = args["target"]
    if args.get("place"):
        params["place"] = args["place"]

    try:
        result = executor.execute_task(task, params=params)
    except Exception as e:
        return {
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }

    return {
        "status": result.status,
        "messages": result.messages,
        "user_output": result.user_output,
        "error": result.error,
        "session_id": result.session_id,
        "question": result.question,
        "options": result.options,
    }


def _resume_task(session_id: str, answer: dict | None,
                 sessions: dict) -> dict:
    return {
        "status": "failed",
        "error": "needs_input resume not yet implemented",
        "session_id": session_id,
        "messages": ["会话恢复功能尚未实现"],
    }


# ═══════════════════════════════════════════════════════════════════════════════════════════
#  MCP server
# ═══════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class AppContext:
    bridge: RobotBridge
    vlm: VlmClient
    yolo: YoloDetector | None
    executor: GraphExecutor
    sessions: dict[str, dict] = field(default_factory=dict)


_app: AppContext | None = None


async def _dispatch_tool_async(
    name: str,
    arguments: dict[str, Any],
    app: AppContext,
    task_lock: asyncio.Lock,
    task_executor: Executor,
) -> dict:
    """Dispatch MCP tools while preserving single-task robot semantics."""
    if name == "arm_execute_task":
        if task_lock.locked():
            return {
                "status": "failed",
                "error": "Another robot task is already running",
            }
        # Task execution contains blocking ROS waits.  Keep it off the MCP
        # event loop so arm_stop can set the cooperative stop flag while a
        # task is in progress.
        async with task_lock:
            clear_stop = getattr(app.bridge, "clear_task_stop", None)
            if callable(clear_stop):
                # Clear before queueing the worker. A stop arriving after this
                # point must remain visible to the active task.
                clear_stop()
            return await asyncio.get_running_loop().run_in_executor(
                task_executor,
                _call_tool,
                name, arguments, app.bridge, app.vlm, app.yolo,
                app.executor, app.sessions,
            )
    if task_lock.locked() and name in {
        "arm_prepare",
        "arm_configure_octomap",
        "arm_reset_context",
    }:
        return {
            "success": False,
            "error": (
                f"{name} is unavailable while a robot task is running; "
                "use arm_stop to prevent subsequent task steps"
            ),
        }
    return _call_tool(
        name, arguments, app.bridge, app.vlm, app.yolo,
        app.executor, app.sessions,
    )


async def _async_main():
    global _app
    print("[robot-arm] Starting MCP server (orchestration mode)...", file=sys.stderr)
    vlm = VlmClient()
    if not vlm.api_key:
        print("[robot-arm] WARNING: VLM_API_KEY not set.", file=sys.stderr)

    yolo = LazyYoloDetector()
    print("[robot-arm] YOLO will load on first use.", file=sys.stderr)

    bridge = RobotBridge()
    print("[robot-arm] Connecting to ROS 2...", file=sys.stderr)
    bridge.start()
    print("[robot-arm] ROS 2 bridge ready.", file=sys.stderr)
    _app = AppContext(
        bridge=bridge,
        vlm=vlm,
        yolo=yolo,
        executor=GraphExecutor(bridge, vlm, yolo),
    )

    server = Server("robot-arm")
    task_lock = asyncio.Lock()
    task_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="robot-task",
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [Tool(**t) for t in TOOL_DEFINITIONS]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> dict:
        return await _dispatch_tool_async(
            name, arguments, _app, task_lock, task_executor
        )

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        task_executor.shutdown(wait=True, cancel_futures=True)


def main():
    asyncio.run(_async_main())
    if _app:
        _app.bridge.shutdown()
    print("[robot-arm] Server stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
