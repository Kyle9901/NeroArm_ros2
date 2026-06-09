# VLM 智能抓取系统

基于视觉大模型（VLM）的七轴机械臂智能抓取系统。支持自然语言描述目标物体，通过 eye-in-hand 深度相机实现 3D 视觉定位，联动 MoveIt 2 完成抓取-放置全自动化流程。

---

## 系统架构

```
┌─────────────────────┐          ┌─────────────────────┐
│   vlm_picker_node   │          │   grasp_executor    │
│     (视觉感知)       │          │     (运动执行)       │
│                     │          │                     │
│  订阅:              │          │  订阅:              │
│  /camera/color/     │          │  /grasp/target      │
│         image_raw   │          │                     │
│  /camera/depth/     │          │  动作客户端:         │
│         image_raw   │          │  /move_action       │
│  /camera/color/     │          │  /execute_trajectory│
│         camera_info │          │  /compute_cartesian_path
│                     │          │  gripper_controller/│
│  VLM API 推理       │          │         follow_joint_trajectory
│  TF2 坐标变换        │          │                     │
│  OpenCV 颜色检测     │          │  抓取序列:           │
│                     │          │  1.开夹爪+预抓位     │
│  发布:              │          │  2.直线下降抓取      │
│  /grasp/target      │─────────▶│  3.闭合夹爪         │
│  (PoseStamped)      │          │  4.抬起至安全位      │
│                     │          │  5.移动到放置位      │
│                     │          │  6.放置+开夹爪       │
│                     │          │  7.回初始位         │
└─────────────────────┘          └─────────────────────┘
```

---

## 依赖

### 系统与 ROS 2

| 组件 | 版本 | 安装方式 |
|------|------|----------|
| Ubuntu | 24.04 LTS | |
| ROS 2 | Jazzy Jalisco | [官方文档](https://docs.ros.org/en/jazzy/Installation.html) |
| MoveIt 2 | 与 Jazzy 配套 | `sudo apt install ros-jazzy-moveit` |

```bash
sudo apt install ros-jazzy-cv-bridge ros-jazzy-tf2-ros-py \
  ros-jazzy-tf2-geometry-msgs ros-jazzy-message-filters
```

### 第三方 ROS 2 包（需单独克隆到工作空间）

```bash
cd ~/ros2_ws/src
git clone https://github.com/agxbot/agx_arm_ros.git        # 机械臂驱动
git clone https://github.com/marcoesposito1988/easy_handeye2.git  # 手眼标定
```

### Python 依赖

```bash
pip install opencv-python numpy requests
```

---

## 编译

```bash
cd ~/ros2_ws
colcon build --packages-select vision_grasp --symlink-install
source install/setup.bash
```

---

## 使用

### 1. 配置环境变量

```bash
export VLM_API_KEY="sk-xxxxxxxxxxxxxxxx"
export VLM_API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
export VLM_MODEL="qwen3.7-plus"

# Debug 图片保存目录（可选，默认 ~/vlm_grasp_debug）
export VLM_DEBUG_DIR="/tmp/vlm_grasp_debug"
```

### 2. 启动基础服务

```bash
# 终端1: 机械臂 + MoveIt
ros2 launch agx_arm_ros piper.launch.py

# 终端2: 深度相机（确保 RGB-D 硬件对齐）
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

### 3. 启动抓取节点

```bash
# 终端3: VLM 视觉检测
ros2 run vision_grasp vlm_picker

# 终端4: 抓取执行
ros2 run vision_grasp grasp_executor
```

---

## 节点说明

| 节点 | 命令 | 功能 |
|------|------|------|
| `vlm_picker` | `ros2 run vision_grasp vlm_picker` | 接收相机图像，调用 VLM API，发布 `/grasp/target` |
| `grasp_executor` | `ros2 run vision_grasp grasp_executor` | 订阅目标，执行 MoveIt 2 抓取-放置序列 |
| `test_move_to` | `ros2 run vision_grasp test_move_to` | 单点运动测试工具 |

---

## 话题接口

| 话题名 | 类型 | 方向 | 说明 |
|--------|------|------|------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | vlm_picker 订阅 | 彩色图 (BGR8) |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | vlm_picker 订阅 | 对齐深度图 (16UC1, mm) |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | vlm_picker 订阅 | 相机内参 |
| `/grasp/target` | `geometry_msgs/PoseStamped` | vlm_picker 发布 / grasp_executor 订阅 | 抓取目标（`frame_id=base_link`） |
| `/feedback/joint_states` | `sensor_msgs/JointState` | grasp_executor 订阅 | 当前关节状态 |

---

## 参数配置

### grasp_executor

| ROS 参数 | 默认值 | 说明 |
|----------|--------|------|
| `planning_group` | `arm` | MoveIt 规划组名 |
| `tcp_link` | `tcp_link` | TCP 连杆名 |
| `base_frame` | `base_link` | 全局坐标系 |
| `grasp_quat` | `[0.476, 0.523, -0.523, 0.476]` | **实测**抓取姿态四元数 (x,y,z,w) |
| `approach_height` | `0.26` | 预抓位高度（法兰在物块上方 26cm） |
| `grasp_depth` | `0.155` | 抓取位深度（法兰在物块上方 15.5cm） |
| `safe_height` | `0.40` | 安全抬升高度 |
| `place_x/y/z` | `-0.40, -0.25, 0.20` | 固定放置位 |
| `velocity_scaling` | `0.05` | 速度缩放因子 |
| `home_joints_deg` | `[0, -20, 0, 80, 0, 0, 80]` | 初始关节角度（度） |

> **重要**：`grasp_quat` 必须通过 `tf2_echo base_link tcp_link` 在**实际垂直向下抓取姿态**下实测获得，直接套用默认值可能导致 IK 无解或碰撞！

---

## 安全机制

`grasp_executor` 内置多层安全校验：

1. **Workspace 限制**：目标必须在设定的工作空间范围内
2. **高度保护**：`z < table_z_threshold` 的目标会被拒绝
3. **放置点校验**：放置点也必须在 workspace 内
4. **笛卡尔覆盖率检查**：直线运动覆盖率低于 50% 会终止
5. **交互确认**：每次抓取前询问用户确认（可按 `n` 取消）

---

## 调试

VLM 节点的 debug 图片默认保存到 `~/vlm_grasp_debug/vlm_picker_debug.jpg`，可通过环境变量修改：

```bash
export VLM_DEBUG_DIR=/tmp/vlm_grasp_debug
```

图片上会叠加检测框、深度采样 ROI、采样中心点、base_link 坐标值等信息。

---

## License

MIT
