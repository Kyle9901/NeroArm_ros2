#!/usr/bin/env python3
"""
CLI test harness for MCP task server.
Usage: source scripts/run_mcp.sh && python -m test.task_test "抓取红色物块"
       or set VLM_API_KEY, PLANNING_LLM_API_KEY etc. before running
"""

import argparse
import json
import os
import sys
import time

_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)


def _load_env():
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "scripts", "run_mcp.sh")
    if not os.path.exists(script):
        return
    with open(script) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
                if "=" in line:
                    key, _, val = line.partition("=")
                    val = val.strip().strip('"').strip("'")
                    val = val.replace("${PYTHONPATH}", os.environ.get("PYTHONPATH", ""))
                    os.environ[key.strip()] = val

_load_env()  # must run before imports that read env vars

from mcp_server.ros_bridge import RobotBridge
from mcp_server.vlm_client import VlmClient
from mcp_server.yolo_detector import LazyYoloDetector
from mcp_server.orchestrator.graph import GraphExecutor
from mcp_server.orchestrator.planner import SKILL_SCHEMA, FEW_SHOT_EXAMPLES


def main():
    parser = argparse.ArgumentParser(description="Test MCP task server")
    parser.add_argument("task", nargs="?", default="")
    parser.add_argument("--target", default=None)
    parser.add_argument("--place", default=None)
    parser.add_argument(
        "--calib-name", default=None,
        help="easy_handeye2 calibration name used by prepare",
    )
    parser.add_argument(
        "--octomap", action=argparse.BooleanOptionalAction, default=None,
        help="enable or disable OctoMap for this test run",
    )
    parser.add_argument("--list", action="store_true", help="List skills and examples")
    args = parser.parse_args()

    if args.list:
        print("=== 技能 ===")
        for name, schema in SKILL_SCHEMA.items():
            args_str = ", ".join(f"{k}: {v}" for k, v in schema["args"].items()) or "无"
            print(f"  {name}({args_str}): {schema['description']}")
        print("\n=== Few-shot ===")
        for ex in FEW_SHOT_EXAMPLES:
            print(f"  {ex['task']}: {' → '.join(s['name'] for s in ex['pipeline'])}")
        return

    vlm = VlmClient()
    bridge = RobotBridge()
    bridge.start()
    print("[init] ready", file=sys.stderr, flush=True)

    try:
        yolo = LazyYoloDetector()
        print("[init] YOLO lazy loader ready", file=sys.stderr, flush=True)

        params = {}
        if args.target:
            params["target"] = args.target
        if args.place:
            params["place"] = args.place
        if args.calib_name:
            params["calib_name"] = args.calib_name
        if args.octomap is not None:
            params["octomap_enabled"] = args.octomap

        executor = GraphExecutor(bridge, vlm, yolo)
        t0 = time.monotonic()
        result = executor.execute_task(args.task, params=params)
        elapsed = time.monotonic() - t0

        print(f"status: {result.status}")
        print(f"time: {elapsed:.1f}s")
        print(f"holding: {bridge.get_holding()}")
        if result.error:
            print(f"error: {result.error}")
        print(f"output: {json.dumps(result.user_output, ensure_ascii=False)}")
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
