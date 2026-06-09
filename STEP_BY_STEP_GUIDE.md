# 视觉抓取完整方案 - 分步指南

## 您已完成的前三步 ✅

1. ✅ 启动相机: `ros2 launch orbbec_camera dabai.launch.py`
2. ✅ 启动YOLO: `ros2 launch yolo_bringup yolov8.launch.py use_tracking:=False`
3. ⬅️ **当前步骤**: 启动3D检测

---

## 第四步: 启动3D检测节点

### 4.1 检查当前话题

在**新的终端**中运行:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

# 查看当前所有话题
ros2 topic list

# 应该看到:
# /camera/color/image_raw
# /camera/depth/image_raw
# /camera/depth/camera_info
# /yolo/detections
# /yolo/image
```

### 4.2 启动3D检测

**方式1: 使用yolo.launch.py (推荐)**

停止之前的YOLO节点,然后重新启动带3D检测的:

```bash
# 终端2 (先停止之前的,然后运行这个)
ros2 launch yolo_bringup yolo.launch.py \
  use_tracking:=False \
  use_3d:=True \
  input_image_topic:=/camera/color/image_raw \
  input_depth_topic:=/camera/depth/image_raw \
  input_depth_info_topic:=/camera/depth/camera_info \
  target_frame:=base_link
```

**方式2: 单独启动3D检测节点**

如果YOLO已经在运行,可以单独启动3D检测:

```bash
# 终端2b
ros2 run yolo_ros detect_3d_node \
  --ros-args \
  -p target_frame:=base_link \
  -p depth_image_units_divisor:=1000 \
  --remap depth_image:=/camera/depth/image_raw \
  --remap depth_info:=/camera/depth/camera_info \
  --remap detections:=/yolo/detections
```

### 4.3 验证3D检测

```bash
# 终端3
ros2 topic echo /yolo/detections_3d

# 应该看到3D检测结果,包含:
# - class_name: 检测到的类别
# - bbox3d.center.position: 3D中心点坐标
# - bbox3d.size: 3D尺寸
```

---

## 第五步: 配置TF变换 (相机→机械臂基座)

### 5.1 查看当前TF树

```bash
# 终端4
ros2 run tf2_tools view_frames

# 等待几秒,会生成 frames.pdf
# 查看文件了解当前TF结构
```

### 5.2 配置静态TF变换

**测量相机安装位置**:
- 相机在机械臂基座前方多远? (x方向,单位:米)
- 相机在机械臂基座左侧/右侧多远? (y方向,单位:米)
- 相机在机械臂基座上方多高? (z方向,单位:米)

**编辑 launch 文件**:

```bash
nano ~/ros2_ws/src/vision_grasp/launch/vision_grasp.launch.py
```

修改 `tf_node` 的参数:

```python
tf_node = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='camera_to_base_tf',
    arguments=[
        '0.3',    # x: 根据实际测量修改
        '0.0',    # y: 根据实际测量修改
        '0.5',    # z: 根据实际测量修改
        '0.0',    # roll: 根据实际测量修改
        '0.0',    # pitch: 根据实际测量修改
        '0.0',    # yaw: 根据实际测量修改
        'base_link',
        'camera_color_optical_frame'
    ]
)
```

### 5.3 验证TF变换

```bash
# 终端5
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame

# 应该显示变换矩阵
```

---

## 第六步: 启动机械臂控制

### 6.1 确认CAN接口

```bash
# 检查CAN接口状态
ip link show can0

# 应该显示 state UP
# 如果不是,执行:
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

### 6.2 启动机械臂

```bash
# 终端6
ros2 launch agx_arm_ctrl start_single_agx_arm.launch.py \
  can_port:=can0 \
  arm_type:=nero \
  effector_type:=agx_gripper
```

### 6.3 验证机械臂状态

```bash
# 终端7
ros2 topic echo /feedback/arm_status

# 检查机械臂是否就绪
```

---

## 第七步: 启动视觉抓取节点

### 7.1 编译工作空间

```bash
cd ~/ros2_ws
colcon build --packages-select vision_grasp
source install/setup.bash
```

### 7.2 启动视觉抓取

```bash
# 终端8
ros2 launch vision_grasp vision_grasp.launch.py
```

### 7.3 监控运行状态

```bash
# 终端9: 查看日志
ros2 topic echo /rosout

# 终端10: 查看TF树
ros2 run rqt_tf_tree rqt_tf_tree
```

---

## 第八步: 测试抓取

### 8.1 手动触发测试

放置一个目标物体在相机视野内,查看是否能自动检测并抓取。

### 8.2 调试话题

```bash
# 查看3D检测结果
ros2 topic echo /yolo/detections_3d

# 查看机械臂控制指令
ros2 topic echo /control/move_p

# 查看夹爪控制
ros2 topic echo /control/joint_states
```

### 8.3 手动控制测试

```bash
# 测试移动到Home位置
ros2 topic pub --once /control/move_p geometry_msgs/PoseStamped '{
  header: {frame_id: "base_link"},
  pose: {
    position: {x: 0.056, y: 0.0, z: 0.213},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}'

# 测试夹爪
ros2 topic pub --once /control/joint_states sensor_msgs/JointState '{
  name: ["gripper"],
  position: [0.1],
  effort: [1.0]
}'
```

---

## 故障排除

### 问题1: 3D检测没有输出

**检查**:
```bash
ros2 topic hz /camera/depth/image_raw
ros2 topic hz /camera/color/image_raw
```

**解决**:
- 确保深度图像和RGB图像分辨率匹配
- 检查相机内参话题是否正确

### 问题2: TF变换失败

**检查**:
```bash
ros2 run tf2_tools view_frames
```

**解决**:
- 确认 `base_link` 和 `camera_color_optical_frame` 都存在
- 检查静态TF发布参数是否正确

### 问题3: 机械臂不运动

**检查**:
```bash
ros2 topic echo /feedback/arm_status
```

**解决**:
- 确认CAN接口正常
- 确认机械臂已使能
- 检查是否有错误状态

### 问题4: 抓取位置偏差

**解决**:
- 重新测量相机安装位置
- 更新TF变换参数
- 进行手眼标定

---

## 高级配置

### 调整抓取参数

编辑 `launch/vision_grasp.launch.py`:

```python
grasp_node = Node(
    package='vision_grasp',
    executable='grasp_node',
    parameters=[{
        'grasp_offset_z': 0.05,      # 抓取Z偏移
        'grasp_depth': 0.02,          # 抓取深度
        'gripper_open_width': 0.08,   # 夹爪张开宽度
        'gripper_close_width': 0.0,   # 夹爪闭合宽度
        'gripper_force': 1.5,         # 夹爪力度
        'approach_height': 0.10,      # 接近高度
        'lift_height': 0.15,          # 提升高度
        'place_x_offset': 0.15,       # 放置位置偏移
    }]
)
```

### 自定义目标选择逻辑

编辑 `vision_grasp/grasp_node.py` 中的 `detection_callback` 方法:

```python
def detection_callback(self, msg: DetectionArray):
    # 自定义选择逻辑
    # 例如: 选择最大的物体,或特定类别的物体
    for detection in msg.detections:
        if detection.class_name == "cup":  # 只抓取杯子
            # ...
```

---

## 一键启动脚本

使用提供的启动脚本:

```bash
~/ros2_ws/src/vision_grasp/scripts/start_vision_grasp.sh
```

**注意**: 脚本会自动检查相机和CAN接口,如果未连接会提示错误。

---

## 下一步

1. ✅ 完成上述步骤
2. 🔄 测试并调整参数
3. 🔄 优化抓取策略
4. 🔄 添加更多功能(如多目标抓取、动态跟踪等)

有任何问题请随时询问!
