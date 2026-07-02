#!/usr/bin/env python3
"""
MCP Server for AGX Robot Arm — new orchestration entry point.
Only exposes 2 tools: arm_execute_task + arm_get_status.
All task planning and skill sequencing is handled by the internal orchestrator.
Usage: python -m mcp_server.task_server
"""

import asyncio
import json
import os
import sys
import traceback
import uuid
from typing import Any

_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .ros_bridge import RobotBridge
from .vlm_client import VlmClient
from .orchestrator.executor import TemplateExecutor
from .orchestrator.router import route_template
from .tools import tools as _tools


# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Tool definitions
# ═══════════════════════════════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    {
        "name": "arm_execute_task",
        "description": (
            "Execute a robot task using the internal orchestration engine. "
            "This is the PRIMARY tool for all pick-and-place, visual grasping, and scene scanning tasks. "
            "The orchestrator internally handles: routing → skill sequencing → parameter passing → "
            "retry/recovery → z-offset calculation → grasp execution. "
            "You do NOT need to call individual tools like arm_detect_vlm, arm_get_3d_position, "
            "arm_execute_grasp, or arm_execute_place — this one tool does everything.\n\n"
            "AVAILABLE TASKS:\n"
            "- pick_and_place: grasp an object and place it somewhere. "
            "Provide target (object description) and place (location word or coordinates).\n"
            "- visual_grasp: grasp an object using visual servoing. "
            "Use when depth is uncertain. Provide target.\n"
            "- scan_scene: detect all visible color blocks on the table. No params needed.\n\n"
            "PLACE LOCATIONS (方位词): right/右边, left/左边, center/中间, front/前面, back/后面\n\n"
            "RETURNS: status (completed/failed/needs_input), messages, user_output with key results "
            "(detection image, grasp coordinates, holding state, etc.).\n"
            "If status=needs_input, the task is paused waiting for user choice — "
            "call again with the same session_id and the user's answer."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Natural language task description. "
                        "Examples: '把蓝色方块放到右边', '直接抓黄色方块', '扫描桌面', "
                        "'pick the red block and place it on the left'"
                    ),
                },
                "target": {
                    "type": "string",
                    "description": (
                        "Object to grasp. Describe in natural language, e.g. 'blue block', '红色方块'. "
                        "Required for pick_and_place and visual_grasp tasks."
                    ),
                },
                "place": {
                    "type": "string",
                    "description": (
                        "Where to place. Can be a location word (right/left/center/front/back or "
                        "右边/左边/中间/前面/后面) or base_link coordinates like '{\"x\":-0.2,\"y\":-0.35,\"z\":0.0}'. "
                        "Required for pick_and_place."
                    ),
                },
                "session_id": {
                    "type": "string",
                    "description": "Resume a paused task. Pass the session_id returned from a previous needs_input response.",
                },
                "answer": {
                    "type": "object",
                    "description": "User's answer to a needs_input question. Pass the selected option, e.g. {\"selected_id\": 0}.",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "arm_get_status",
        "description": (
            "Get the current state of the robot arm: joint angles (degrees), gripper position, "
            "holding state (whether gripper is currently holding an object), "
            "workspace bounds (x/y min/max), safe height, desk surface Z, "
            "and grasp geometry parameters. "
            "Call this to check the arm's condition before or after a task."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Tool dispatch
# ═══════════════════════════════════════════════════════════════════════════════════════════

def _call_tool(name: str, args: dict, bridge: RobotBridge, vlm: VlmClient,
               executor: TemplateExecutor, sessions: dict) -> dict:
    if name == "arm_get_status":
        return _tools.arm_get_status(bridge)

    if name == "arm_execute_task":
        return _execute_task(args, executor, sessions)

    return {"status": "failed", "error": f"Unknown tool: {name}"}


def _execute_task(args: dict, executor: TemplateExecutor,
                  sessions: dict) -> dict:
    session_id = args.get("session_id")
    answer = args.get("answer")

    # ── Resume from needs_input ──
    if session_id and session_id in sessions:
        return _resume_task(session_id, answer, sessions)

    # ── New task ──
    task = args.get("task", "")
    template = route_template(task)
    if template is None:
        return {
            "status": "failed",
            "error": "No matching template",
            "messages": [
                f"未匹配到任务模板: '{task}'",
                "可用任务: pick_and_place (抓+放), visual_grasp (视觉伺服抓取), scan_scene (扫描桌面)",
                "请重新描述任务,或使用旧版 arm_* 工具手动执行",
            ],
        }

    params = {}
    if args.get("target"):
        params["target"] = args["target"]
    if args.get("place"):
        params["place"] = args["place"]

    try:
        result = executor.execute_task(task, params=params, template=template)
    except Exception as e:
        return {
            "status": "failed",
            "template": template.name,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }

    if result.status == "needs_input":
        sid = str(uuid.uuid4())[:8]
        sessions[sid] = {
            "template": template,
            "params": params,
            "outputs": result.outputs,
        }
        result.session_id = sid

    return {
        "status": result.status,
        "template": result.template,
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

_bridge: RobotBridge | None = None
_vlm: VlmClient | None = None
_executor: TemplateExecutor | None = None
_sessions: dict[str, dict] = {}


async def _async_main():
    global _bridge, _vlm, _executor, _sessions
    print("[task-server] Starting MCP server (orchestration mode)...", file=sys.stderr)
    _vlm = VlmClient()
    if not _vlm.api_key:
        print("[task-server] WARNING: VLM_API_KEY not set.", file=sys.stderr)
    _bridge = RobotBridge()
    print("[task-server] Connecting to ROS 2...", file=sys.stderr)
    _bridge.start()
    print("[task-server] ROS 2 bridge ready.", file=sys.stderr)
    _executor = TemplateExecutor(_bridge, _vlm)
    _sessions = {}

    server = Server("robot-arm-task")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [Tool(**t) for t in TOOL_DEFINITIONS]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> dict:
        return _call_tool(name, arguments, _bridge, _vlm, _executor, _sessions)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    print("[task-server] DISABLED — use server.py instead", file=sys.stderr)
    return
    asyncio.run(_async_main())
    if _bridge:
        _bridge.shutdown()
    print("[task-server] Server stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()