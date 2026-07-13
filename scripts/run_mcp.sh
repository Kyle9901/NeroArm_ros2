#!/bin/bash
# Wrapper for robot-arm MCP server
# OpenClaw blocks PYTHONPATH in env, so set it here
set -e
exec 2>/tmp/robot_arm_mcp_stderr.log

export PYTHONPATH="/home/alkaid/ros2_ws/src/vision_grasp:${PYTHONPATH}"

# VLM + Planning LLM (same key, same endpoint)
export VLM_API_KEY="sk-ws-H.RPMHLMD.eNJE.MEUCIBm07fqNtBjq6uFXj5r4XPa_kAGB8mEE0UKuLLZGbc2HAiEAkavdsqSmb1KgE9ZsQUArWKdVpYKSL5Q1QF_PmM_xzsY"
export VLM_API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
export VLM_MODEL="qwen3.6-plus"

# Planning LLM
export PLANNING_LLM_API_KEY="sk-ws-H.RPMHLMD.eNJE.MEUCIBm07fqNtBjq6uFXj5r4XPa_kAGB8mEE0UKuLLZGbc2HAiEAkavdsqSmb1KgE9ZsQUArWKdVpYKSL5Q1QF_PmM_xzsY"
export PLANNING_LLM_API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
export PLANNING_LLM_MODEL="deepseek-v4-flash"

# YOLO detection
export YOLO_MODEL_PATH="/home/alkaid/ros2_ws/src/vision_grasp/models/yolov8s.pt"
export YOLO_CONFIDENCE="0.35"
export YOLO_DEVICE="cuda"
export VLM_FALLBACK="1"

# ROS 2 environment
[ -f /opt/ros/jazzy/setup.bash ] && source /opt/ros/jazzy/setup.bash
[ -f /home/alkaid/ros2_ws/install/setup.bash ] && source /home/alkaid/ros2_ws/install/setup.bash

exec /home/alkaid/ros2_ws/src/vision_grasp/.venv/bin/python -m mcp_server.task_server "$@"
