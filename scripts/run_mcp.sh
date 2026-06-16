#!/bin/bash
# Wrapper for robot-arm MCP server
# OpenClaw blocks PYTHONPATH in env, so set it here
set -e

export PYTHONPATH="/home/alkaid/ros2_ws/src/vision_grasp:${PYTHONPATH}"
export VLM_API_KEY="sk-ztI…nXH1"

# ROS 2 environment
[ -f /opt/ros/jazzy/setup.bash ] && source /opt/ros/jazzy/setup.bash
[ -f /home/alkaid/ros2_ws/install/setup.bash ] && source /home/alkaid/ros2_ws/install/setup.bash

exec /home/alkaid/ros2_ws/src/vision_grasp/.venv/bin/python -m mcp_server.server "$@"
