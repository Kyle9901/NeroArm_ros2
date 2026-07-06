# VLM 智能抓取系统 v1.1.2

基于 LLM 任务编排 + LangGraph 执行引擎的七轴机械臂智能抓取系统。通过 MCP 协议接入 OpenClaw,支持自然语言控制,内部自动完成检测、定位、抓取、放置全流程。

---

## 系统架构

```
OpenClaw (自然语言交互)
    │  MCP 协议 (stdio)
    ▼
task_server.py (MCP 入口, 6 个工具)
    │
    ▼
orchestrator/ (编排层)
    ├── planner.py     → Planning LLM 生成 pipeline
    ├── graph.py       → LangGraph 动态图执行
    └── place_resolver.py → 方位词→坐标
    │
    ▼
skills/ (技能层, 14 个确定性技能)
    ├── perception.py  → locate_object / scan_scene
    ├── manipulation.py → grasp_object / place_object
    ├── prepare.py     → 启动节点
    └── recovery.py    → 故障恢复
    │
    ▼
components/ (原子组件层, 17 个)
    ├── motion.py      → 8 运动组件
    ├── perception.py  → 5 感知组件
    └── infra.py       → 2 基建组件
    │
    ▼
ros_bridge.py / vlm_client.py (硬件层)
```

---

## 依赖

### 系统与 ROS 2

| 组件 | 版本 |
|------|------|
| Ubuntu | 24.04 LTS |
| ROS 2 | Jazzy Jalisco |
| MoveIt 2 | 与 Jazzy 配套 |

```bash
sudo apt install ros-jazzy-cv-bridge ros-jazzy-tf2-ros-py ros-jazzy-tf2-geometry-msgs
```

### Python 依赖

```bash
pip install opencv-python numpy requests langgraph langchain-core mcp
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

### 1. 配置

复制启动脚本模板并填入 API key:

```bash
cp scripts/run_mcp.example.sh scripts/run_mcp.sh
# 编辑 scripts/run_mcp.sh，填入你的 VLM_API_KEY 和 PLANNING_LLM_API_KEY
```

`scripts/run_mcp.sh` 已在 `.gitignore` 中，不会被提交到 git。

VLM 和 Planning LLM 的模型/URL 默认值在 `mcp_server/vlm_client.py` 和 `mcp_server/orchestrator/planner_config.py` 中。

### 2. 启动 MCP Server

```bash
scripts/run_mcp.sh
```

### 3. OpenClaw 配置

```json
{
  "mcpServers": {
    "robot-arm": {
      "command": "/home/alkaid/ros2_ws/src/vision_grasp/scripts/run_mcp.sh"
    }
  }
}
```

---

## MCP 工具

| 工具 | 说明 |
|------|------|
| `arm_execute_task` | 唯一任务入口,LLM 编排 + LangGraph 执行 |
| `arm_prepare` | 启动/检查机械臂和相机节点 |
| `arm_get_status` | 查询关节状态、夹爪、holding 等 |
| `arm_stop` | 急停(同时清除上下文) |
| `arm_configure_vlm` | 运行时配置 VLM |
| `arm_reset_context` | 清除语义上下文 |

---

## 技能列表

| 技能 | 说明 |
|------|------|
| `detect_by_color` | 纯 HSV 颜色检测(纯色物块) |
| `locate_object` | 检测 + 3D 定位(HSV 优先,VLM 兜底) |
| `grasp_object` | 开环抓取(自动 z-offset) |
| `place_object` | 放置 |
| `stack_on` | 叠加高度偏移(放到 X 上方) |
| `scan_scene` | 扫描桌面所有色块 |
| `resolve_place` | 方位词→坐标 |
| `go_home` | 回 home 位置 |
| `open_gripper` | 打开夹爪 |
| `close_gripper` | 闭合夹爪 |
| `prepare` | 启动节点 |
| `wave` | 挥手 |
| `nod` | 点头 |
| `handshake` | 握手 |

---

## 状态管理

系统维护**硬件状态**(实时查询)和**语义状态**(内存):

- `holding`: 硬件真值,夹爪是否持有物体
- `grasped_object`: 语义值,手持物体名称
- `pick_x/y/z`: 上次抓取位置("放下"时用)
- `recent_actions`: 最近动作历史(滚动窗口)

语义状态**硬件真值优先**:`holding=false` 时自动清空 `grasped_object`。

---

## 安全机制

1. **Workspace 限制**: 目标必须在工作空间内
2. **Desk 碰撞**: 桌面自动添加为碰撞对象
3. **故障恢复**: 抓取/放置失败自动 `recover_to_safe`
4. **急停联动**: `arm_stop` 同时清除语义上下文
5. **夹爪硬件反馈**: 抓取后通过硬件宽度判断是否持有

---

## 参数配置

### ros_bridge.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `home_joints_deg` | `[0, -20, 0, 80, 0, 0, 80]` | 初始关节角度 |
| `grasp_quat` | `[0.503, 0.497, -0.499, 0.501]` | 抓取姿态四元数 |
| `approach_height` | `0.26` | 预抓位高度 |
| `grasp_depth` | `0.155` | 抓取时法兰偏移 |
| `safe_height` | `0.40` | 安全抬升高度 |
| `flange_to_tip` | `0.175` | 法兰→指尖距离 |
| `fingertip_overlap` | `0.02` | 指尖探入深度 |
| `velocity_scaling` | `0.15` | 普通运动速度 |
| `descent_velocity_scaling` | `0.05` | 下降阶段速度 |

---

## License

MIT