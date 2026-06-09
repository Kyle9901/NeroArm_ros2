# 视觉抓取快速启动指南

## 环境准备

```bash
# 每次新开终端都需要执行
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
```

## 快速启动 (4个终端)

### 终端1: 相机
```bash
ros2 launch orbbec_camera dabai.launch.py
```

### 终端2: YOLO + 3D检测
```bash
ros2 launch yolo_bringup yolo.launch.py \
  use_tracking:=False \
  use_3d:=True \
  input_image_topic:=/camera/color/image_raw \
  input_depth_topic:=/camera/depth/image_raw \
  input_depth_info_topic:=/camera/depth/camera_info \
  target_frame:=base_link
```

### 终端3: 机械臂
```bash
ros2 launch agx_arm_ctrl start_single_agx_arm.launch.py \
  can_port:=can0 \
  arm_type:=nero \
  effector_type:=agx_gripper
```

### 终端4: 视觉抓取
```bash
ros2 launch vision_grasp vision_grasp.launch.py
```

## 一键启动

```bash
~/ros2_ws/src/vision_grasp/scripts/start_vision_grasp.sh
```

## 关键话题

| 话题 | 类型 | 说明 |
|------|------|------|
| `/camera/color/image_raw` | sensor_msgs/Image | RGB图像 |
| `/camera/depth/image_raw` | sensor_msgs/Image | 深度图像 |
| `/yolo/detections_3d` | yolo_msgs/DetectionArray | 3D检测结果 |
| `/control/move_p` | geometry_msgs/PoseStamped | 位置控制 |
| `/control/joint_states` | sensor_msgs/JointState | 关节控制(夹爪) |
| `/feedback/tcp_pose` | geometry_msgs/PoseStamped | TCP位姿反馈 |

## 调试命令

```bash
# 查看话题列表
ros2 topic list

# 查看3D检测结果
ros2 topic echo /yolo/detections_3d

# 查看TF树
ros2 run tf2_tools view_frames

# 查看特定TF变换
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame

# 查看图像
ros2 run image_view image_view --ros-args -r image:=/yolo/image
```

## 手动控制测试

```bash
# 移动到Home位置
ros2 topic pub --once /control/move_p geometry_msgs/PoseStamped '{
  header: {frame_id: "base_link"},
  pose: {
    position: {x: 0.056, y: 0.0, z: 0.213},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}'

# 打开夹爪
ros2 topic pub --once /control/joint_states sensor_msgs/JointState '{
  name: ["gripper"],
  position: [0.1],
  effort: [1.0]
}'

# 闭合夹爪
ros2 topic pub --once /control/joint_states sensor_msgs/JointState '{
  name: ["gripper"],
  position: [0.0],
  effort: [1.5]
}'
```

## 常见问题

### CAN接口未启动
```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

### 重新编译
```bash
cd ~/ros2_ws
colcon build --packages-select vision_grasp
source install/setup.bash
```

### 相机未识别
```bash
# 检查USB连接
lsusb | grep -i orbbec

# 检查视频设备
ls /dev/video*
```
