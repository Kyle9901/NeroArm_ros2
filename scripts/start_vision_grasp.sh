#!/bin/bash
# Vision Grasp Startup Script
# This script starts all necessary nodes for vision-based grasping

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  Vision Grasp System Startup${NC}"
echo -e "${GREEN}=====================================${NC}"

# Source ROS2 environment
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down all nodes...${NC}"
    kill 0
    exit 0
}
trap cleanup SIGINT SIGTERM

# Check if camera is connected
echo -e "${YELLOW}Checking camera connection...${NC}"
if ! lsusb | grep -i "orbbec\|dabai" > /dev/null; then
    echo -e "${RED}Warning: Orbbec camera not detected!${NC}"
    echo -e "${RED}Please connect the camera and try again.${NC}"
    exit 1
fi
echo -e "${GREEN}Camera detected.${NC}"

# Check if CAN interface is up
echo -e "${YELLOW}Checking CAN interface...${NC}"
if ! ip link show can0 > /dev/null 2>&1; then
    echo -e "${RED}Warning: CAN interface can0 not found!${NC}"
    echo -e "${RED}Please configure CAN interface first:${NC}"
    echo -e "${RED}  sudo ip link set can0 type can bitrate 1000000${NC}"
    echo -e "${RED}  sudo ip link set can0 up${NC}"
    exit 1
fi

if ! ip link show can0 | grep -q "state UP"; then
    echo -e "${RED}Warning: CAN interface can0 is down!${NC}"
    echo -e "${RED}Please bring it up:${NC}"
    echo -e "${RED}  sudo ip link set can0 up${NC}"
    exit 1
fi
echo -e "${GREEN}CAN interface is up.${NC}"

# Start camera
echo -e "${YELLOW}Starting camera...${NC}"
ros2 launch orbbec_camera dabai.launch.py &
CAMERA_PID=$!
sleep 5

# Start YOLO with 3D detection
echo -e "${YELLOW}Starting YOLO detection with 3D...${NC}"
ros2 launch yolo_bringup yolo.launch.py \
  use_tracking:=False \
  use_3d:=True \
  input_image_topic:=/camera/color/image_raw \
  input_depth_topic:=/camera/depth/image_raw \
  input_depth_info_topic:=/camera/depth/camera_info \
  target_frame:=base_link &
YOLO_PID=$!
sleep 10

# Start arm control
echo -e "${YELLOW}Starting arm control...${NC}"
ros2 launch agx_arm_ctrl start_single_agx_arm.launch.py \
  can_port:=can0 \
  arm_type:=nero \
  effector_type:=agx_gripper &
ARM_PID=$!
sleep 5

# Start vision grasp
echo -e "${YELLOW}Starting vision grasp node...${NC}"
ros2 launch vision_grasp vision_grasp.launch.py &
GRASP_PID=$!

# Wait for all background processes
wait
