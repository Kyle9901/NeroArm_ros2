"""Load RobotBridge ROS parameter defaults from YAML."""

import os
from pathlib import Path

import yaml

from .paths import PROJECT_ROOT


_REQUIRED = {
    "planning_group", "tcp_link", "base_frame", "tcp_offset", "grasp_quat",
    "workspace_x_min", "workspace_x_max", "workspace_y_min", "workspace_y_max",
    "approach_height", "safe_height", "fingertip_depth",
    "block_depth_frames", "block_depth_max_spread", "block_xy_max_spread",
    "block_yaw_max_spread_deg", "desk_measurement_max_error",
    "cylinder_depth_frames", "cylinder_depth_max_spread",
    "cylinder_position_max_spread", "cylinder_axis_max_spread_deg",
    "cylinder_lying_axis_max_deviation_deg",
    "cylinder_min_diameter_m", "cylinder_max_diameter_m",
    "cylinder_min_length_m", "cylinder_max_length_m",
    "cylinder_side_grasp_height_ratio", "cylinder_tilt_angles_deg",
    "grasp_tilt_angles_deg",
    "grasp_pregrasp_distance", "grasp_retreat_distance",
    "grasp_candidate_timeout", "grasp_full_plan_candidates",
    "joint7_soft_limit_deg", "joint7_min_margin_deg",
    "planned_start_tolerance_rad", "reverse_branch_tolerance_rad",
    "place_x", "place_y", "place_z",
    "gripper_open_width", "gripper_close_width",
    "planning_time", "num_planning_attempts", "velocity_scaling", "accel_scaling",
    "octomap_enabled_on_prepare", "desk_collision_enabled",
    "descent_velocity_scaling", "descent_accel_scaling", "cartesian_eef_step",
    "cartesian_min_fraction", "cartesian_jump_threshold",
    "pos_tolerance", "ori_tolerance",
    "observation_joints_deg", "carry_joints_deg",
    "desk_z_surface", "desk_size",
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
        if not _is_number(parameters.get(low)):
            errors.append(f"{low} must be a number")
        if not _is_number(parameters.get(high)):
            errors.append(f"{high} must be a number")
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
        "approach_height", "safe_height", "fingertip_depth",
        "block_depth_max_spread", "block_xy_max_spread",
        "block_yaw_max_spread_deg", "desk_measurement_max_error",
        "cylinder_depth_max_spread", "cylinder_position_max_spread",
        "cylinder_axis_max_spread_deg", "cylinder_min_diameter_m",
        "cylinder_lying_axis_max_deviation_deg",
        "cylinder_max_diameter_m", "cylinder_min_length_m",
        "cylinder_max_length_m",
        "grasp_pregrasp_distance",
        "grasp_retreat_distance", "grasp_candidate_timeout",
        "joint7_soft_limit_deg", "joint7_min_margin_deg",
        "planned_start_tolerance_rad", "reverse_branch_tolerance_rad",
        "gripper_open_width", "planning_time", "cartesian_eef_step",
        "pos_tolerance", "ori_tolerance",
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

    fraction = parameters.get("cartesian_min_fraction")
    if not _is_number(fraction) or not 0.0 < fraction <= 1.0:
        errors.append("cartesian_min_fraction must be a number in (0, 1]")

    jump_threshold = parameters.get("cartesian_jump_threshold")
    if not _is_number(jump_threshold) or jump_threshold < 0.0:
        errors.append("cartesian_jump_threshold must be a non-negative number")

    for name in ("place_x", "place_y", "place_z", "desk_z_surface"):
        if not _is_number(parameters.get(name)):
            errors.append(f"{name} must be a number")

    attempts = parameters.get("num_planning_attempts")
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        errors.append("num_planning_attempts must be an integer >= 1")

    frames = parameters.get("block_depth_frames")
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 3:
        errors.append("block_depth_frames must be an integer >= 3")
    cylinder_frames = parameters.get("cylinder_depth_frames")
    if (
        not isinstance(cylinder_frames, int)
        or isinstance(cylinder_frames, bool)
        or cylinder_frames < 3
    ):
        errors.append("cylinder_depth_frames must be an integer >= 3")

    full_plans = parameters.get("grasp_full_plan_candidates")
    if (not isinstance(full_plans, int) or isinstance(full_plans, bool)
            or not 1 <= full_plans <= 10):
        errors.append("grasp_full_plan_candidates must be an integer in [1, 10]")

    tilts = parameters.get("grasp_tilt_angles_deg")
    if (not isinstance(tilts, list) or tilts != [0, 30, 60]):
        errors.append("grasp_tilt_angles_deg must be exactly [0, 30, 60]")
    cylinder_tilts = parameters.get("cylinder_tilt_angles_deg")
    if (
        not isinstance(cylinder_tilts, list)
        or cylinder_tilts != [0, 60, 90]
    ):
        errors.append(
            "cylinder_tilt_angles_deg must be exactly [0, 60, 90]"
        )
    lying_axis_deviation = parameters.get(
        "cylinder_lying_axis_max_deviation_deg"
    )
    if (
        _is_number(lying_axis_deviation)
        and float(lying_axis_deviation) > 45.0
    ):
        errors.append(
            "cylinder_lying_axis_max_deviation_deg must not exceed 45"
        )
    side_ratio = parameters.get("cylinder_side_grasp_height_ratio")
    if not _is_number(side_ratio) or not 0.25 <= side_ratio <= 0.75:
        errors.append(
            "cylinder_side_grasp_height_ratio must be in [0.25, 0.75]"
        )
    min_diameter = parameters.get("cylinder_min_diameter_m")
    max_diameter = parameters.get("cylinder_max_diameter_m")
    if (
        _is_number(min_diameter)
        and _is_number(max_diameter)
        and min_diameter >= max_diameter
    ):
        errors.append(
            "cylinder_min_diameter_m must be smaller than "
            "cylinder_max_diameter_m"
        )
    min_length = parameters.get("cylinder_min_length_m")
    max_length = parameters.get("cylinder_max_length_m")
    if (
        _is_number(min_length)
        and _is_number(max_length)
        and min_length >= max_length
    ):
        errors.append(
            "cylinder_min_length_m must be smaller than "
            "cylinder_max_length_m"
        )

    soft_limit = parameters.get("joint7_soft_limit_deg")
    min_margin = parameters.get("joint7_min_margin_deg")
    if (_is_number(soft_limit) and _is_number(min_margin)
            and min_margin >= soft_limit):
        errors.append("joint7_min_margin_deg must be smaller than joint7_soft_limit_deg")
    if _is_number(soft_limit) and soft_limit > 180.0:
        errors.append("joint7_soft_limit_deg must not exceed 180")
    start_tolerance = parameters.get("planned_start_tolerance_rad")
    if _is_number(start_tolerance) and start_tolerance > 0.05:
        errors.append("planned_start_tolerance_rad must not exceed 0.05")
    branch_tolerance = parameters.get("reverse_branch_tolerance_rad")
    if _is_number(branch_tolerance) and branch_tolerance > 0.35:
        errors.append("reverse_branch_tolerance_rad must not exceed 0.35")

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

    tcp_offset = parameters.get("tcp_offset")
    if (not isinstance(tcp_offset, list) or len(tcp_offset) != 6
            or not all(_is_number(v) for v in tcp_offset)):
        errors.append(
            "tcp_offset must contain six numbers [x, y, z, roll, pitch, yaw]"
        )

    for name in ("observation_joints_deg", "carry_joints_deg"):
        joints = parameters.get(name)
        if (not isinstance(joints, list) or len(joints) != 7
                or not all(_is_number(v) for v in joints)):
            errors.append(f"{name} must contain seven numeric joint angles")
    carry = parameters.get("carry_joints_deg")
    if (
        isinstance(carry, list)
        and len(carry) == 7
        and all(_is_number(value) for value in carry)
        and _is_number(soft_limit)
        and _is_number(min_margin)
        and soft_limit - abs(float(carry[6])) < min_margin
    ):
        errors.append(
            "carry_joints_deg joint7 must satisfy the configured soft-limit margin"
        )

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
