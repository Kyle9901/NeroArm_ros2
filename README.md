# vision_grasp

基于 ROS 2、MoveIt 2、HSV/YOLO/VLM 感知和 LangGraph 编排的七轴机械臂抓取服务。当前项目只保留 MCP 任务服务这一条正式运行链路。

## 项目结构

```text
vision_grasp/
├── mcp_server/
│   ├── components/       # 感知、运动等原子能力
│   ├── config/           # 运行参数与统一路径
│   ├── models/           # 通用结果模型
│   ├── orchestrator/     # 任务规划与 LangGraph 执行
│   ├── perception/       # HSV 检测与调试图绘制
│   ├── ros/              # 相机、TF、运动、规划场景和启动管理
│   ├── skills/           # 定位、抓取、放置等任务技能
│   ├── ros_bridge.py     # ROS 能力门面与生命周期
│   ├── task_server.py    # MCP 服务入口
│   ├── vlm_client.py     # VLM 客户端
│   └── yolo_detector.py  # YOLO 推理封装
├── models/               # 模型权重（默认 models/yolov8s.pt）
├── scripts/              # MCP 启动脚本及配置模板
├── test/
│   ├── depth_test.py     # HSV 物块及周围桌面深度诊断
│   └── task_test.py      # 自然语言任务端到端测试
├── tmp/                  # 运行时调试图，不提交 Git
├── package.xml
├── setup.cfg
└── setup.py
```

## 安装与编译

系统环境为 Ubuntu 24.04、ROS 2 Jazzy 和对应版本的 MoveIt 2。

```bash
sudo apt install ros-jazzy-cv-bridge ros-jazzy-tf2-ros-py ros-jazzy-tf2-geometry-msgs
cd ~/ros2_ws
colcon build --packages-select vision_grasp --symlink-install
source install/setup.bash
```

Python 依赖：

```bash
pip install -r src/vision_grasp/requirements.txt
```

## 配置与启动

复制模板并填写 API Key：

```bash
cd ~/ros2_ws/src/vision_grasp
cp scripts/run_mcp.example.sh scripts/run_mcp.sh
```

启动 MCP 服务：

```bash
scripts/run_mcp.sh
```

也可以直接测试完整任务：

```bash
python -m test.task_test "抓取红色物块并放回原位置"
```

深度诊断：

```bash
python -m test.depth_test 红色物块
```

## 抓取高度调节

当前生效参数位于 `mcp_server/ros_bridge.py` 的 `RobotBridgeNode._declare_params()`：

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `approach_height` | `0.26` m | 物体表面到预抓位置的高度 |
| `flange_to_tip` | `0.175` m | TCP/法兰到夹爪指尖的距离 |
| `grasp_depth` | `0.04` m | 指尖向物体表面下探的距离 |
| `desk_z_surface` | `-0.013` m | `base_link` 下桌面表面高度 |

抓取 TCP 高度计算在 `mcp_server/skills/base.py`：

```text
grasp_z = object_surface_z + flange_to_tip - grasp_depth
```

如果夹爪位置偏低，应减小 `grasp_depth`。建议每次调整 `0.005` m，例如先从 `0.040` 改为 `0.035`；不要直接修改感知得到的物体 Z。若日志出现 `clamping grasp_depth`，实际高度受桌面安全钳制控制，应优先检查 `desk_z_surface` 和相机到 `base_link` 的 TF 标定。

## 调试输出

所有识别调试图默认写入项目目录：

```text
tmp/debug/       # HSV、YOLO、VLM 检测图
tmp/depth_test/  # 深度测试标注图
```

可通过环境变量修改：

```bash
export VISION_GRASP_TMP_DIR=/path/to/runtime-output
# 或仅修改识别调试图目录
export VLM_DEBUG_DIR=/path/to/debug-images
```

`tmp/` 中的运行时文件已被 `.gitignore` 忽略，仅保留目录占位文件。

## 任务技能

- `locate_object`：HSV 优先，YOLO/VLM 补充的目标检测与三维定位
- `scan_scene`：扫描桌面物体
- `grasp_object` / `place_object`：抓取和放置
- `resolve_place` / `stack_on`：方位解析和堆叠位置计算
- `go_home` / `open_gripper` / `close_gripper`：基础动作
- `wave` / `nod` / `handshake`：交互动作

## 安全机制

- 工作空间边界检查
- 桌面和目标物碰撞场景
- 指尖低于桌面时的抓取深度钳制
- 动作失败恢复与回 Home
- 夹爪宽度反馈判断是否持物

## License

MIT
