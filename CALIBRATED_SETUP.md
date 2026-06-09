# 已标定相机 setup 指南

## 状态确认

✅ **已完成**: ArUco标记手眼标定 (easy_handeye2)

## 当前TF树状态

从之前的检查可以看到:
- **相机TF树**: `camera_link` → `camera_depth_frame` → `camera_color_frame` → `camera_color_optical_frame`
- **机械臂TF树**: `world` → `base_link` → `link1` → ... → `link7`
- **问题**: 两个树之间**没有连接**!

## 解决方案

### 方案1: 启动 easy_handeye2 发布标定结果 (推荐)

如果您保存了标定结果，使用以下命令启动:

```bash
# 终端A: 发布手眼标定TF
ros2 launch easy_handeye2 publish.launch.py name:=<您的标定名称>
```

**常见的标定名称**:
- `nero_handeye`
- `agx_arm_handeye`
- `camera_handeye`

如果不确定名称，可以查找:
```bash
# 查找标定文件
find ~/.ros -name "*handeye*" -type f

# 或者查看 easy_handeye2 的默认保存位置
ls ~/.ros/easy_handeye2/
```

### 方案2: 在 vision_grasp launch 中自动加载

我已经修改了 `vision_grasp.launch.py`，支持自动加载标定:

```bash
# 使用标定 (默认)
ros2 launch vision_grasp vision_grasp.launch.py \
  use_handeye:=true \
  handeye_name:=nero_handeye

# 如果不使用标定，回退到静态TF
ros2 launch vision_grasp vision_grasp.launch.py \
  use_handeye:=false
```

### 方案3: 手动输入标定结果

如果您知道标定结果（平移+旋转），可以直接编辑 launch 文件:

```bash
nano ~/ros2_ws/src/vision_grasp/launch/vision_grasp.launch.py
```

修改 `static_tf_node` 的参数:
```python
arguments=[
    '0.3',     # x: 根据标定结果修改
    '0.0',     # y: 根据标定结果修改
    '0.5',     # z: 根据标定结果修改
    '0.0',     # roll: 根据标定结果修改 (弧度)
    '0.0',     # pitch: 根据标定结果修改 (弧度)
    '0.0',     # yaw: 根据标定结果修改 (弧度)
    'base_link',
    'camera_color_optical_frame'
]
```

## 验证TF连接

启动后，验证TF是否正确:

```bash
# 方法1: 查看TF树
ros2 run tf2_tools view_frames

# 方法2: 实时查看变换
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame

# 应该输出变换矩阵，而不是错误
```

## 完整启动流程 (4个终端)

### 终端1: 相机
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch orbbec_camera dabai.launch.py
```

### 终端2: YOLO + 3D检测
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch yolo_bringup yolo.launch.py \
  use_tracking:=False \
  use_3d:=True \
  input_image_topic:=/camera/color/image_raw \
  input_depth_topic:=/camera/depth/image_raw \
  input_depth_info_topic:=/camera/depth/camera_info \
  target_frame:=base_link
```

### 终端3: 手眼标定TF发布
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

# 方式A: 使用 easy_handeye2 (如果有保存的标定)
ros2 launch easy_handeye2 publish.launch.py name:=nero_handeye

# 方式B: 手动静态TF (如果知道标定结果)
# ros2 run tf2_ros static_transform_publisher \
#   0.3 0.0 0.5 0.0 0.0 0.0 \
#   base_link camera_color_optical_frame
```

### 终端4: 机械臂控制
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch agx_arm_ctrl start_single_agx_arm.launch.py \
  can_port:=can0 \
  arm_type:=nero \
  effector_type:=agx_gripper
```

### 终端5: 视觉抓取
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

# 如果使用 easy_handeye2 标定
ros2 launch vision_grasp vision_grasp.launch.py \
  use_handeye:=true \
  handeye_name:=nero_handeye

# 或者使用静态TF
# ros2 launch vision_grasp vision_grasp.launch.py use_handeye:=false
```

## 调试步骤

### 1. 检查所有节点是否运行
```bash
ros2 node list
```
应该看到:
- `/orbbec_camera_node`
- `/yolo_node`
- `/detect_3d_node`
- `/handeye_publisher` (如果使用标定)
- `/agx_arm_ctrl_single_node`
- `/vision_grasp_node`

### 2. 检查TF树
```bash
ros2 run tf2_tools view_frames
# 查看生成的 frames.pdf，确认 base_link 和 camera_color_optical_frame 已连接
```

### 3. 测试变换
```bash
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
```

### 4. 检查3D检测
```bash
ros2 topic echo /yolo/detections_3d
# 放置一个物体在相机前，应该看到检测结果
```

### 5. 测试机械臂控制
```bash
# 手动发送Home位置
ros2 topic pub --once /control/move_p geometry_msgs/PoseStamped '{
  header: {frame_id: "base_link"},
  pose: {
    position: {x: 0.056, y: 0.0, z: 0.213},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}'
```

## 常见问题

### Q: easy_handeye2 找不到标定文件
**A**: 检查标定文件位置:
```bash
ls ~/.ros/easy_handeye2/
# 或者
find /home/alkaid -name "*handeye*.json" -o -name "*handeye*.yaml"
```

### Q: TF变换错误
**A**: 确认:
1. 所有节点都已启动
2. 标定名称正确
3. 坐标系名称匹配 (`base_link` 和 `camera_color_optical_frame`)

### Q: 3D检测没有输出
**A**: 检查:
1. 相机是否发布深度图像
2. YOLO是否检测到物体
3. 深度图像和RGB图像是否对齐

## 下一步

1. **确认标定名称** 并启动TF发布
2. **验证TF连接**
3. **测试3D检测**
4. **启动完整系统**
5. **放置测试物体** 并观察抓取

**请告诉我:**
1. 您的 easy_handeye2 标定名称是什么?
2. 标定文件保存在哪里?
3. 是否需要我帮您查找标定结果?
