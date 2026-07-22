#!/bin/bash
# Wrapper for robot-arm MCP server — copy to run_mcp.sh and fill in your keys
set -e
exec 2>/tmp/robot_arm_mcp_stderr.log

# ---- 必填 ----
export VLM_API_KEY="sk-xxx"
export PLANNING_LLM_API_KEY="sk-xxx"

# ---- 可选 (有默认值) ----
export VLM_API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
export VLM_MODEL="qwen3.6-plus"
export PLANNING_LLM_API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
export PLANNING_LLM_MODEL="deepseek-v4-flash"

# ROS 2 environment
[ -f /opt/ros/jazzy/setup.bash ] && source /opt/ros/jazzy/setup.bash
[ -f /home/[User_name]/ros2_ws/install/setup.bash ] && source /home/[User_name]/ros2_ws/install/setup.bash

# Keep the development source and YAML ahead of stale install/ copies.
export PYTHONPATH="/home/[User_name]/ros2_ws/src/vision_grasp:${PYTHONPATH}"
export VISION_GRASP_ROBOT_CONFIG="/home/[User_name]/ros2_ws/src/vision_grasp/config/robot.yaml"

exec /home/[User_name]/ros2_ws/src/vision_grasp/.venv/bin/python -m mcp_server.task_server "$@"
