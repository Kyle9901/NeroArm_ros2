"""Load RobotBridge ROS parameter defaults from YAML."""

import os
from pathlib import Path

import yaml

from .paths import PROJECT_ROOT


_REQUIRED = {
    "planning_group", "tcp_link", "base_frame", "grasp_quat",
    "workspace_x_min", "workspace_x_max", "workspace_y_min", "workspace_y_max",
    "approach_height", "safe_height", "fingertip_depth", "flange_to_tip",
    "color_block_height", "gripper_open_width", "gripper_close_width",
    "planning_time", "num_planning_attempts", "velocity_scaling", "accel_scaling",
    "octomap_enabled_on_prepare", "desk_collision_enabled",
    "descent_velocity_scaling", "descent_accel_scaling", "cartesian_eef_step",
    "cartesian_min_fraction", "home_joints_deg", "desk_z_surface", "desk_size",
}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_robot_parameters(parameters: dict) -> None:
    """Raise one actionable error containing every unsafe configuration value."""
    errors = []
    missing = sorted(_REQUIRED - parameters.keys())
    if missing:
        errors.append(f"missing required parameters: {', '.join(missing)}")

    for low, high in (
        ("workspace_x_min", "workspace_x_max"),
        ("workspace_y_min", "workspace_y_max"),
    ):
        if _is_number(parameters.get(low)) and _is_number(parameters.get(high)):
            if parameters[low] >= parameters[high]:
                errors.append(f"{low} must be smaller than {high}")

    for name in (
        "velocity_scaling", "accel_scaling",
        "descent_velocity_scaling", "descent_accel_scaling",
    ):
        value = parameters.get(name)
        if not _is_number(value) or not 0.0 < value <= 1.0:
            errors.append(f"{name} must be a number in (0, 1]")

    for name in (
        "approach_height", "safe_height", "fingertip_depth", "flange_to_tip",
        "color_block_height", "gripper_open_width", "planning_time",
        "cartesian_eef_step",
    ):
        value = parameters.get(name)
        if not _is_number(value) or value <= 0:
            errors.append(f"{name} must be a positive number")

    close_width = parameters.get("gripper_close_width")
    open_width = parameters.get("gripper_open_width")
    if not _is_number(close_width) or close_width < 0:
        errors.append("gripper_close_width must be a non-negative number")
    elif _is_number(open_width) and close_width >= open_width:
        errors.append("gripper_close_width must be smaller than gripper_open_width")

    fingertip_depth = parameters.get("fingertip_depth")
    block_height = parameters.get("color_block_height")
    if (_is_number(fingertip_depth) and _is_number(block_height)
            and fingertip_depth >= block_height):
        errors.append("fingertip_depth must be smaller than color_block_height")

    fraction = parameters.get("cartesian_min_fraction")
    if not _is_number(fraction) or not 0.0 < fraction <= 1.0:
        errors.append("cartesian_min_fraction must be a number in (0, 1]")

    attempts = parameters.get("num_planning_attempts")
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        errors.append("num_planning_attempts must be an integer >= 1")

    if not isinstance(parameters.get("octomap_enabled_on_prepare"), bool):
        errors.append("octomap_enabled_on_prepare must be a boolean")
    if not isinstance(parameters.get("desk_collision_enabled"), bool):
        errors.append("desk_collision_enabled must be a boolean")

    quat = parameters.get("grasp_quat")
    if not isinstance(quat, list) or len(quat) != 4 or not all(_is_number(v) for v in quat):
        errors.append("grasp_quat must contain four numbers [x, y, z, w]")
    else:
        norm = sum(float(value) ** 2 for value in quat) ** 0.5
        if not 0.95 <= norm <= 1.05:
            errors.append(f"grasp_quat must be normalized (current norm={norm:.4f})")

    home = parameters.get("home_joints_deg")
    if not isinstance(home, list) or len(home) != 7 or not all(_is_number(v) for v in home):
        errors.append("home_joints_deg must contain seven numeric joint angles")

    desk_size = parameters.get("desk_size")
    if (not isinstance(desk_size, list) or len(desk_size) != 3
            or not all(_is_number(v) and v > 0 for v in desk_size)):
        errors.append("desk_size must contain three positive numbers [x, y, z]")

    for name in ("planning_group", "tcp_link", "base_frame"):
        if not isinstance(parameters.get(name), str) or not parameters[name].strip():
            errors.append(f"{name} must be a non-empty string")

    if errors:
        formatted = "\n  - ".join(errors)
        raise ValueError(f"Invalid robot configuration:\n  - {formatted}")


def robot_config_path() -> Path:
    override = os.environ.get("VISION_GRASP_ROBOT_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    source_path = PROJECT_ROOT / "config" / "robot.yaml"
    if source_path.is_file():
        return source_path
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory("vision_grasp")) / "config" / "robot.yaml"


def load_robot_parameters() -> tuple[Path, dict]:
    path = robot_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Robot configuration not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    parameters = document.get("robot_bridge_node", {}).get("ros__parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError(f"Expected robot_bridge_node.ros__parameters in {path}")
    validate_robot_parameters(parameters)
    return path, parameters
