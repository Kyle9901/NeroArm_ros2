#!/usr/bin/env python3
"""
Direct CLI test harness for the new orchestration stack.
Usage:  python -m mcp_server.test_task "把蓝色方块放到右边"
"""

import argparse
import json
import os
import sys
import time

_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from mcp_server.ros_bridge import RobotBridge
from mcp_server.vlm_client import VlmClient
from mcp_server.orchestrator.executor import TemplateExecutor
from mcp_server.orchestrator.router import route_template


def main():
    parser = argparse.ArgumentParser(description="Test orchestration stack")
    parser.add_argument("task", nargs="?", default="把蓝色方块放到右边")
    parser.add_argument("--target", default="blue block", help="Object to grasp")
    parser.add_argument("--place", default="right", help="Place target")
    parser.add_argument("--list", action="store_true", help="List available templates")
    args = parser.parse_args()

    if args.list:
        from mcp_server.orchestrator.templates import TEMPLATES
        for t in TEMPLATES:
            print(f"  {t.name}: {t.description}")
            print(f"    patterns: {t.match_patterns}")
            print(f"    params: {[p.name for p in t.required_params]}")
            print()
        return

    # ── Route ──
    template = route_template(args.task)
    if template is None:
        print(f"❌ 未匹配到模板: {args.task}")
        print("   可用模板: pick_and_place, visual_grasp, scan_scene")
        return
    print(f"🔀 路由: {template.name} ({template.description})")

    # ── Init bridge + VLM ──
    print("[init] 启动 VLM client ...")
    vlm = VlmClient()
    if not vlm.api_key:
        print("[init] ⚠️  VLM_API_KEY 未设置,VLM 检测可能失败")

    print("[init] 连接 ROS 2 ...")
    bridge = RobotBridge()
    bridge.start()
    print("[init] ROS 2 bridge 就绪")

    # ── Execute ──
    params = {"target": args.target, "place": args.place}
    print(f"\n🚀 执行任务: {args.task}")
    print(f"   参数: {json.dumps(params, ensure_ascii=False)}")
    print(f"   模板: {template.name}")
    print(f"   pipeline: {' → '.join(s.name for s in template.pipeline)}")
    print()

    executor = TemplateExecutor(bridge, vlm)
    t0 = time.monotonic()
    result = executor.execute_task(args.task, params=params, template=template)
    elapsed = time.monotonic() - t0

    print(f"\n{'='*60}")
    print(f"结果: {result.status}")
    print(f"耗时: {elapsed:.1f}s")
    if result.messages:
        for m in result.messages:
            print(f"  📍 {m}")
    if result.error:
        print(f"  ❌ 错误: {result.error}")
    print(f"\n用户可见输出:")
    print(json.dumps(result.user_output, indent=2, ensure_ascii=False, default=str))
    print(f"\n完整输出:")
    print(json.dumps(result.outputs, indent=2, ensure_ascii=False, default=str,
                     cls=_CompactEncoder))

    bridge.shutdown()
    print("\n[shutdown] 完成")


class _CompactEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


if __name__ == "__main__":
    main()