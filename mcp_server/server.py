#!/usr/bin/env python3
"""
MCP Server for AGX Robot Arm — launch via OpenClaw / Claude Code.

Usage:
    python -m vision_grasp.mcp_server.server

Environment variables:
    VLM_API_KEY        — required for VLM detection
    VLM_API_URL        — optional, defaults to Qwen dashscope
    VLM_MODEL          — optional, defaults to qwen3.7-plus
    VLM_DEBUG_DIR      — optional, saves debug image
"""

import asyncio
import json
import os
import sys
import threading
import traceback
from typing import Any

# ── Ensure parent package is importable ──
_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# ── Try to import mcp (official SDK) ──
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("WARNING: 'mcp' package not installed. Install with: pip install mcp", file=sys.stderr)
    print("Falling back to raw JSON-RPC stdio mode.", file=sys.stderr)

from .ros_bridge import RobotBridge
from .vlm_client import VlmClient
from .tools import tools as _tools


# ═══════════════════════════════════════════════════════════════════════════════════════════
#   Tool definitions
# ═══════════════════════════════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    {
        "name": "arm_capture_image",
        "description": "Capture the latest colour image from the robot's eye-in-hand camera. "
                       "Returns a base64-encoded JPEG and depth statistics.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "arm_detect_vlm",
        "description": "Detect an object in the current camera frame using the VLM (Vision Language Model). "
                       "Describe the target in natural language, e.g. 'blue block' or 'red cube'. "
                       "The VLM will classify the object and return a bounding box. "
                       "For colour blocks, OpenCV HSV detection is used as a fallback for higher precision. "
                       "For best results, call arm_go_home before this tool to give the camera an unobstructed view.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Natural language description of the object to detect, e.g. 'blue block', 'red cube', 'metal bottle'.",
                },
            },
            "required": ["target"],
        },
    },
    {
        "name": "arm_detect_color",
        "description": "Detect a colour block using OpenCV HSV colour detection (no VLM call). "
                       "Faster than arm_detect_vlm but only works for solid-colour blocks. "
                       "Supported colours: blue, red, green, yellow, purple, orange, cyan.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "color_name": {
                    "type": "string",
                    "description": "Colour name: blue, red, green, yellow, purple, orange, cyan.",
                },
                "location_hint": {
                    "type": "string",
                    "description": "Optional location hint: left, right, top, bottom, center. Defaults to empty (full image).",
                },
            },
            "required": ["color_name"],
        },
    },
    {
        "name": "arm_get_3d_position",
        "description": "Convert a 2D pixel coordinate (u, v) from the camera image to a 3D position "
                       "in the robot's base_link frame. Uses the depth image and TF2 transform.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "u": {"type": "integer", "description": "X pixel coordinate (column)."},
                "v": {"type": "integer", "description": "Y pixel coordinate (row)."},
            },
            "required": ["u", "v"],
        },
    },
    {
        "name": "arm_get_status",
        "description": "Get the current state of the robot arm: joint angles (degrees) and gripper position.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "arm_move_joints",
        "description": "Move the robot arm to a target joint configuration. "
                       "Provide 7 joint angles in degrees: [joint1, joint2, joint3, joint4, joint5, joint6, joint7].",
        "inputSchema": {
            "type": "object",
            "properties": {
                "joint_angles_deg": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 7,
                    "maxItems": 7,
                    "description": "7 joint angles in degrees.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds. Default: 20.",
                },
            },
            "required": ["joint_angles_deg"],
        },
    },
    {
        "name": "arm_move_to_pose",
        "description": "Move the robot TCP to a Cartesian pose (x, y, z in metres) via MoveGroup motion planning. "
                       "The orientation defaults to the pre-configured grasp quaternion.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "X position in base_link frame (metres)."},
                "y": {"type": "number", "description": "Y position in base_link frame (metres)."},
                "z": {"type": "number", "description": "Z position in base_link frame (metres)."},
                "quat": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Optional quaternion [x, y, z, w]. Defaults to grasp quat.",
                },
                "timeout": {"type": "number", "description": "Timeout in seconds. Default: 20."},
            },
            "required": ["x", "y", "z"],
        },
    },
    {
        "name": "arm_move_cartesian",
        "description": "Move the robot TCP along a straight line in Cartesian space. "
                       "Use this for precise vertical descent / ascent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "X position in base_link frame (metres)."},
                "y": {"type": "number", "description": "Y position in base_link frame (metres)."},
                "z": {"type": "number", "description": "Z position in base_link frame (metres)."},
                "quat": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Optional quaternion [x, y, z, w]. Defaults to grasp quat.",
                },
                "timeout": {"type": "number", "description": "Timeout in seconds. Default: 20."},
            },
            "required": ["x", "y", "z"],
        },
    },
    {
        "name": "arm_control_gripper",
        "description": "Open or close the gripper. "
                       "Typical values: 0.10 = fully open, 0.02 = fully closed (in metres).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "width": {"type": "number", "description": "Gripper opening width in metres."},
                "duration": {"type": "number", "description": "Movement duration in seconds. Default: 1.5."},
                "timeout": {"type": "number", "description": "Timeout in seconds. Default: 5."},
            },
            "required": ["width"],
        },
    },
    {
        "name": "arm_go_home",
        "description": "Move the robot arm to its home joint configuration.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "timeout": {"type": "number", "description": "Timeout in seconds. Default: 20."},
            },
            "required": [],
        },
    },
    {
        "name": "arm_stop",
        "description": "Emergency stop — cancel all current motion goals.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "arm_execute_grasp",
        "description": "Pick up an object at (x, y, z) in base_link frame. "
                       "Steps: 1) open gripper + move to approach pose, "
                       "2) Cartesian descent to grasp, "
                       "3) close gripper, "
                       "4) lift to safe height. "
                       "After this call, the object is held at safe height. "
                       "Use arm_execute_place() to put it down, or call arm_execute_swap() to swap two objects. "
                       "⚠️ This will physically move the robot — ensure the workspace is clear!",
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Target X in base_link frame (metres)."},
                "y": {"type": "number", "description": "Target Y in base_link frame (metres)."},
                "z": {"type": "number", "description": "Target Z in base_link frame (metres)."},
                "quat": {
                    "type": "array", "items": {"type": "number"},
                    "minItems": 4, "maxItems": 4,
                    "description": "Optional quaternion [x, y, z, w]. Defaults to grasp quat.",
                },
            },
            "required": ["x", "y", "z"],
        },
    },
    {
        "name": "arm_execute_place",
        "description": "Place the currently held object at (x, y, z) in base_link frame. "
                       "Steps: 1) move to above place pose, "
                       "2) Cartesian descent to place Z, "
                       "3) open gripper. "
                       "After this call, the hand is empty. "
                       "Does NOT go home — caller can chain another grasp or call arm_go_home().",
        "inputSchema": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "Place X in base_link frame (metres)."},
                "y": {"type": "number", "description": "Place Y in base_link frame (metres)."},
                "z": {"type": "number", "description": "Place Z in base_link frame (metres)."},
                "quat": {
                    "type": "array", "items": {"type": "number"},
                    "minItems": 4, "maxItems": 4,
                    "description": "Optional quaternion [x, y, z, w]. Defaults to grasp quat.",
                },
            },
            "required": ["x", "y", "z"],
        },
    },
]

# Lookup table for fast dispatch
TOOL_TABLE = {t["name"]: t for t in TOOL_DEFINITIONS}


# ═══════════════════════════════════════════════════════════════════════════════════════════
#   Tool dispatch
# ═══════════════════════════════════════════════════════════════════════════════════════════

def _call_tool(name: str, args: dict, bridge: RobotBridge, vlm: VlmClient) -> dict:
    """Dispatch tool call to the appropriate handler."""

    handlers = {
        "arm_capture_image":   lambda: _tools.arm_capture_image(bridge),
        "arm_detect_vlm":      lambda: _tools.arm_detect_vlm(bridge, vlm, args["target"]),
        "arm_detect_color":    lambda: _tools.arm_detect_color(bridge, args.get("color_name", ""), args.get("location_hint", "")),
        "arm_get_3d_position": lambda: _tools.arm_get_3d_position(bridge, args["u"], args["v"]),
        "arm_get_status":      lambda: _tools.arm_get_status(bridge),
        "arm_move_joints":     lambda: _tools.arm_move_joints(bridge, args["joint_angles_deg"], args.get("timeout", 20.0)),
        "arm_move_to_pose":    lambda: _tools.arm_move_to_pose(bridge, args["x"], args["y"], args["z"], args.get("quat"), args.get("timeout", 20.0)),
        "arm_move_cartesian":  lambda: _tools.arm_move_cartesian(bridge, args["x"], args["y"], args["z"], args.get("quat"), args.get("timeout", 20.0)),
        "arm_control_gripper": lambda: _tools.arm_control_gripper(bridge, args["width"], args.get("duration", 1.5), args.get("timeout", 5.0)),
        "arm_go_home":         lambda: _tools.arm_go_home(bridge, args.get("timeout", 20.0)),
        "arm_stop":            lambda: _tools.arm_stop(bridge),
        "arm_execute_grasp":   lambda: _tools.arm_execute_grasp(bridge, args["x"], args["y"], args["z"], args.get("quat")),
        "arm_execute_place":   lambda: _tools.arm_execute_place(bridge, args["x"], args["y"], args["z"], args.get("quat")),
    }

    if name not in handlers:
        return {"success": False, "error": f"Unknown tool: {name}"}

    try:
        return handlers[name]()
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


# ═══════════════════════════════════════════════════════════════════════════════════════════
#   MCP SDK mode
# ═══════════════════════════════════════════════════════════════════════════════════════════

if MCP_AVAILABLE:

    async def _run_mcp_server():
        server = Server("robot-arm")

        @server.list_tools()
        async def list_tools() -> list[Tool]:
            return [Tool(**t) for t in TOOL_DEFINITIONS]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            result = _call_tool(name, arguments, _bridge, _vlm)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())


# ═══════════════════════════════════════════════════════════════════════════════════════════
#   Raw JSON-RPC mode (fallback when mcp package is not installed)
# ═══════════════════════════════════════════════════════════════════════════════════════════

def _run_jsonrpc_loop():
    """Minimal JSON-RPC 2.0 loop over stdin/stdout.  Compatible with MCP clients."""
    import select

    buf = ""
    while True:
        # Read a line
        if sys.stdin.isatty():
            # Non-interactive — just read
            chunk = sys.stdin.readline()
            if not chunk:
                break
        else:
            ready, _, _ = select.select([sys.stdin], [], [], 0.5)
            if not ready:
                continue
            chunk = sys.stdin.readline()
            if not chunk:
                break

        buf += chunk
        # Try to parse complete JSON lines
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue

            resp = _handle_jsonrpc(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()


def _handle_jsonrpc(req: dict) -> dict | None:
    """Handle a single JSON-RPC 2.0 request."""
    req_id = req.get("id")
    method = req.get("method", "")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "robot-arm", "version": "0.0.1"},
                "capabilities": {"tools": {}},
            },
        }
    elif method == "notifications/initialized":
        return None  # No response for notifications
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOL_DEFINITIONS},
        }
    elif method == "tools/call":
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        result = _call_tool(name, arguments, _bridge, _vlm)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
            },
        }
    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    else:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


# ═══════════════════════════════════════════════════════════════════════════════════════════
#   Entry point
# ═══════════════════════════════════════════════════════════════════════════════════════════

_bridge: RobotBridge | None = None
_vlm: VlmClient | None = None


def main():
    global _bridge, _vlm

    print("[robot-arm] Starting MCP server...", file=sys.stderr)

    # Init VLM client
    _vlm = VlmClient()
    if not _vlm.api_key:
        print("[robot-arm] WARNING: VLM_API_KEY not set — VLM detection will fail.", file=sys.stderr)

    # Init ROS bridge
    _bridge = RobotBridge()
    print("[robot-arm] Connecting to ROS 2...", file=sys.stderr)
    _bridge.start()
    print("[robot-arm] ROS 2 bridge ready.", file=sys.stderr)

    if MCP_AVAILABLE:
        print("[robot-arm] Running with MCP SDK (stdio transport)", file=sys.stderr)
        asyncio.run(_run_mcp_server())
    else:
        print("[robot-arm] Running with raw JSON-RPC 2.0 (stdio transport)", file=sys.stderr)
        _run_jsonrpc_loop()

    _bridge.shutdown()
    print("[robot-arm] Server stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()