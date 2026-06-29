#!/usr/bin/env python3
"""
MCP Server for AGX Robot Arm — launch via OpenClaw / Claude Code.
Usage: python -m mcp_server.server
"""

import asyncio
import json
import os
import sys
import traceback
from typing import Any

# Ensure parent package is importable
_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .ros_bridge import RobotBridge
from .vlm_client import VlmClient
from .tools import tools as _tools


# ═══════════════════════════════════════════════════════════════════════════════════════════
#   Tool definitions
# ═══════════════════════════════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    {
        "name": "arm_bringup_nodes",
        "description": "Start the robot arm, camera, and handeye TF nodes. "
                       "Checks CAN interface first — if CAN is down, the arm component will fail "
                       "and the returned hint will tell the user to configure CAN manually. "
                       "Waits for each component to become ready (MoveIt 10s, camera 3s, TF 3s). "
                       "Returns status per component: 'ready', 'started_but_not_ready', or 'failed'. "
                       "Call this once at the start of a session before any motion or vision tools.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "can_port": {
                    "type": "string",
                    "description": "CAN interface name. Default: can0.",
                },
                "calib_name": {
                    "type": "string",
                    "description": "Handeye calibration name. Default: my_eih_calib_v6.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "arm_bringup_status",
        "description": "Check the current bringup status: CAN interface state, "
                       "which managed processes are running, and whether key ROS endpoints "
                       "(/move_action, /camera/color/image_raw, /tf) are available. "
                       "Use this to diagnose why motion or vision tools are not working.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "arm_configure_vlm",
        "description": "Configure VLM API settings at runtime. Use this to set or change the VLM API key, "
                       "API URL, and model name without restarting the MCP server. "
                       "Leave a field empty to keep its current value. "
                       "Call this first if VLM detection fails with auth errors.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "api_key": {
                    "type": "string",
                    "description": "VLM API key. Leave empty to keep current.",
                },
                "api_url": {
                    "type": "string",
                    "description": "VLM API endpoint URL (OpenAI-compatible). Leave empty to keep current.",
                },
                "model": {
                    "type": "string",
                    "description": "VLM model name. Leave empty to keep current.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "arm_capture_image",
        "description": "Capture the latest colour image from the robot's eye-in-hand camera. "
                       "Returns a base64-encoded JPEG and depth statistics. "
                       "⚠️ Do NOT use this to verify whether the gripper is holding an object — "
                       "the camera cannot reliably see the gripper at safe height. "
                       "Use arm_get_status.holding instead.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "arm_detect_vlm",
        "description": "Detect ONE specified object in the current camera frame using the VLM. "
                       "Internally captures the current camera frame — do NOT call arm_capture_image before this. "
                       "⚠️ Do NOT use this to verify whether the gripper is holding an object after grasp — "
                       "the gripper is at safe height and the camera cannot reliably see it. "
                       "Use arm_get_status.holding instead. "
                       "Use when the user specifies a target, e.g. 'blue block', 'red cube', 'metal bottle'. "
                       "For generic requests like '识别桌面上的物块' / 'detect blocks on the table', use arm_detect_blocks instead. "
                       "For colour blocks, OpenCV HSV detection is used as a fallback for higher precision. "
                       "Typical workflow: arm_go_home → arm_detect_vlm(target) → arm_get_3d_position(center_2d) → arm_execute_grasp.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Natural language description of ONE object to detect, e.g. 'blue block', 'red cube', 'metal bottle'.",
                },
            },
            "required": ["target"],
        },
    },
    {
        "name": "arm_detect_color",
        "description": "Detect ONE specified colour block using OpenCV HSV colour detection (no VLM call). "
                       "Use this ONLY when the user explicitly specifies a colour, e.g. 'blue block'. "
                       "Do NOT use this tool to search for all blocks or enumerate colours. "
                       "For generic requests like '识别桌面上的物块', use arm_detect_blocks. "
                       "Not-found is returned as success=true, found=false (not a tool error). "
                       "Supported colours: blue, red, green, yellow, purple, orange, cyan.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "color_name": {
                    "type": "string",
                    "description": "Explicit colour name: blue, red, green, yellow, purple, orange, cyan.",
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
        "name": "arm_detect_blocks",
        "description": "Detect ALL visible solid-colour blocks on the table in ONE tool call using OpenCV HSV detection. "
                       "Use this for generic requests like '识别桌面上的物块', 'detect blocks on the table', or 'what blocks are visible'. "
                       "Do NOT call arm_detect_color repeatedly to enumerate colours. "
                       "Returns count and a blocks list with color, bbox, center_2d, area_px. "
                       "Internally captures the camera frame — do NOT call arm_capture_image before this. "
                       "Typical workflow: arm_go_home → arm_detect_blocks → choose a block → arm_get_3d_position(center_2d).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "location_hint": {
                    "type": "string",
                    "description": "Optional location hint: left, right, top, bottom, center. Defaults to empty (full image).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "arm_get_3d_position",
        "description": "Convert a 2D pixel coordinate (u, v) from the camera image to a 3D position "
                       "in the robot's base_link frame. Uses the depth image and TF2 transform. "
                       "Returns x, y, z — NOTE: z is the OBJECT SURFACE height, not the flange height. "
                       "The flange must be ~0.175m ABOVE the surface to avoid collision. "
                       "Do NOT pass the returned z directly to arm_move_to_pose or arm_move_cartesian — "
                       "always add an offset (approach_height=0.26m or safe_height=0.40m). "
                       "For pick-and-place, use arm_execute_grasp and arm_execute_place which handle offsets automatically.",
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
        "description": "Get the current state of the robot arm: joint angles (degrees), gripper position, "
                       "holding state (whether gripper is currently holding an object), "
                       "workspace bounds (x/y min/max), safe height, desk surface Z, "
                       "and grasp_geometry (flange_to_tip, fingertip_overlap, grasp_depth). "
                       "Call this before any motion sequence to check the arm's current condition "
                       "and understand the grasp geometry parameters.",
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
                       "⚠️ CRITICAL: z is the FLANGE (TCP) height, NOT the gripper tip height! "
                       "The gripper tip is ~0.175m BELOW the flange. "
                       "The z from arm_get_3d_position is the OBJECT SURFACE height — you MUST add an offset "
                       "(at least +0.26m approach_height or +0.40m safe_height) before passing to this tool. "
                       "Example: if arm_get_3d_position returns z=0.05, use z=0.31 (0.05 + 0.26) to approach above the object. "
                       "NEVER pass raw surface z to this tool — the flange would go below the desk and planning will fail. "
                       "The orientation defaults to the pre-configured grasp quaternion. "
                       "Timeout covers planning (5 attempts × 3s) + execution. "
                       "For typical pick-and-place, use arm_execute_grasp and arm_execute_place instead — "
                       "they handle z-offsets automatically.",
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
                "timeout": {"type": "number", "description": "Timeout in seconds. Default: 60."},
            },
            "required": ["x", "y", "z"],
        },
    },
    {
        "name": "arm_move_cartesian",
        "description": "Move the robot TCP along a straight line in Cartesian space. "
                       "⚠️ CRITICAL: z is the FLANGE (TCP) height, NOT the gripper tip! "
                       "Gripper tip is ~0.175m below flange. Never pass raw surface z directly — "
                       "if arm_get_3d_position returned z=0.05, the flange must be at least z+0.175=0.225 "
                       "to keep the tip above the desk. Planning will fail if z is too low. "
                       "Use this for precise vertical descent / ascent. "
                       "For typical pick-and-place, use arm_execute_grasp and arm_execute_place instead — "
                       "they handle z-offsets automatically. "
                       "Default timeout: 30s.",
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
                "timeout": {"type": "number", "description": "Timeout in seconds. Default: 30."},
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
                       "⚠️ z = OBJECT SURFACE HEIGHT (from arm_get_3d_position), NOT flange height. "
                       "The tool internally handles all z-offsets: flange descends to z + grasp_depth (0.155m), "
                       "fingertip ends up at z - fingertip_overlap (z - 0.02m = 2cm BELOW surface to grip the object). "
                       "You do NOT need to add any offset to z — just pass the surface height directly. "
                       "Steps: 1) open gripper + move to approach pose (z + 0.26m), "
                       "2) SLOW Cartesian descent to grasp (z + 0.155m → fingertip at z - 0.02m), "
                       "3) close gripper + MEASURE actual gripper width, "
                       "4) lift to safe height (0.40m). "
                       "RETURNS: 'holding' field (bool) — TRUE if gripper stayed open (object inside), "
                       "FALSE if gripper fully closed (nothing grabbed). "
                       "TRUST holding — it's measured from hardware, not estimated. "
                       "Do NOT call arm_capture_image or arm_detect_vlm to verify. "
                       "If holding=false: retry grasp or tell user. "
                       "If holding=true: use arm_execute_place() to put it down. "
                       "Auto-recovers to safe height on failure. "
                       "⚠️ Physically moves the robot — ensure workspace is clear!",
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
                       "z is the SURFACE height (e.g. desk top z=0.0) — the arm internally adds "
                       "the same flange offset (0.155m) used during grasping, so the gripper tip "
                       "stays safely above the surface. Use the same z from arm_get_3d_position. "
                       "Steps: 1) move to above place pose, "
                       "2) Cartesian descent to place Z, "
                       "3) open gripper. "
                       "After this call, the hand is empty (arm_get_status will show holding=false). "
                       "Does NOT go home — caller can chain another grasp or call arm_go_home(). "
                       "On failure, automatically attempts recovery to safe height.",
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
    {
        "name": "arm_visual_grasp",
        "description": "Grasp an object using VLM detection + CSRT visual tracking. "
                       "You only need to describe the target, e.g. 'blue block'. "
                       "No pre-computed 3D coordinates required — the tool handles everything: "
                       "go_home → VLM detect → CSRT tracker init → iterative look-and-move descent → grasp → lift. "
                       "The tracker re-detects the object at each step to correct position errors. "
                       "If the tracker loses the target, it falls back to VLM re-detection. "
                       "Use this instead of arm_execute_grasp when depth accuracy is uncertain "
                       "or when you want real-time position correction during the approach.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Natural language description of the object to grasp, e.g. 'blue block', 'red cube'.",
                },
            },
            "required": ["target"],
        },
    },
]

# ── Tool dispatch ──

def _call_tool(name: str, args: dict, bridge: RobotBridge, vlm: VlmClient) -> dict:
    handlers = {
        "arm_bringup_nodes":   lambda: _tools.arm_bringup_nodes(bridge, args.get("can_port", "can0"), args.get("calib_name", "my_eih_calib_v6")),
        "arm_bringup_status":  lambda: _tools.arm_bringup_status(bridge),
        "arm_configure_vlm":   lambda: _tools.arm_configure_vlm(bridge, vlm, args.get("api_key"), args.get("api_url"), args.get("model")),
        "arm_capture_image":   lambda: _tools.arm_capture_image(bridge),
        "arm_detect_vlm":      lambda: _tools.arm_detect_vlm(bridge, vlm, args["target"]),
        "arm_detect_color":    lambda: _tools.arm_detect_color(bridge, args.get("color_name", ""), args.get("location_hint", "")),
        "arm_detect_blocks":   lambda: _tools.arm_detect_blocks(bridge, args.get("location_hint", "")),
        "arm_get_3d_position": lambda: _tools.arm_get_3d_position(bridge, args["u"], args["v"]),
        "arm_get_status":      lambda: _tools.arm_get_status(bridge),
        "arm_move_joints":     lambda: _tools.arm_move_joints(bridge, args["joint_angles_deg"], args.get("timeout", 20.0)),
        "arm_move_to_pose":    lambda: _tools.arm_move_to_pose(bridge, args["x"], args["y"], args["z"], args.get("quat"), args.get("timeout", 60.0)),
        "arm_move_cartesian":  lambda: _tools.arm_move_cartesian(bridge, args["x"], args["y"], args["z"], args.get("quat"), args.get("timeout", 30.0)),
        "arm_control_gripper": lambda: _tools.arm_control_gripper(bridge, args["width"], args.get("duration", 1.5), args.get("timeout", 5.0)),
        "arm_go_home":         lambda: _tools.arm_go_home(bridge, args.get("timeout", 20.0)),
        "arm_stop":            lambda: _tools.arm_stop(bridge),
        "arm_execute_grasp":   lambda: _tools.arm_execute_grasp(bridge, args["x"], args["y"], args["z"], args.get("quat")),
        "arm_execute_place":   lambda: _tools.arm_execute_place(bridge, args["x"], args["y"], args["z"], args.get("quat")),
        "arm_visual_grasp":    lambda: _tools.arm_visual_grasp(bridge, vlm, args["target"]),
    }
    if name not in handlers:
        return {"success": False, "error": f"Unknown tool: {name}"}
    try:
        return handlers[name]()
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}


# ── MCP server ──

_bridge: RobotBridge | None = None
_vlm: VlmClient | None = None


async def _async_main():
    global _bridge, _vlm
    print("[robot-arm] Starting MCP server...", file=sys.stderr)
    _vlm = VlmClient()
    if not _vlm.api_key:
        print("[robot-arm] WARNING: VLM_API_KEY not set.", file=sys.stderr)
    _bridge = RobotBridge()
    print("[robot-arm] Connecting to ROS 2...", file=sys.stderr)
    _bridge.start()
    print("[robot-arm] ROS 2 bridge ready.", file=sys.stderr)

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


def main():
    asyncio.run(_async_main())
    if _bridge:
        _bridge.shutdown()
    print("[robot-arm] Server stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()