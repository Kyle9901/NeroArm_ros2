from copy import deepcopy

import pytest

from mcp_server.config.robot import load_robot_parameters, validate_robot_parameters


@pytest.fixture
def valid_parameters():
    _path, parameters = load_robot_parameters()
    return deepcopy(parameters)


def test_repository_robot_config_is_valid(valid_parameters):
    validate_robot_parameters(valid_parameters)


def test_repository_uses_calibrated_tcp_without_legacy_flange_compensation(
        valid_parameters):
    assert valid_parameters["tcp_offset"] == pytest.approx(
        [0.1733, 0.0, -0.0235, -1.5708, 0.0, -1.5708]
    )
    assert valid_parameters["approach_height"] == pytest.approx(0.0867)
    assert valid_parameters["safe_height"] == pytest.approx(0.2267)
    assert "flange_to_tip" not in valid_parameters
    assert "color_block_height" not in valid_parameters
    assert valid_parameters["observation_joints_deg"][-1] == pytest.approx(80.0)
    assert valid_parameters["carry_joints_deg"] == pytest.approx(
        [0.0, -20.0, 0.0, 80.0, 0.0, 0.0, 50.0]
    )
    assert valid_parameters["desk_collision_enabled"] is True
    assert valid_parameters["grasp_tilt_angles_deg"] == [0, 30, 60]
    assert valid_parameters["planned_start_tolerance_rad"] == pytest.approx(0.02)
    assert valid_parameters["reverse_branch_tolerance_rad"] == pytest.approx(0.10)
    assert valid_parameters["desk_measurement_max_error"] == pytest.approx(0.05)


def test_default_grasp_points_calibrated_tcp_axis_down(valid_parameters):
    x, y, z, w = valid_parameters["grasp_quat"]
    # Rotate the TCP local +Z approach axis by q * v * q^-1.
    approach = (
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    )
    assert approach[0] == pytest.approx(0.0, abs=0.01)
    assert approach[1] == pytest.approx(0.0, abs=0.01)
    assert approach[2] < -0.999


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("velocity_scaling", 1.5, "velocity_scaling"),
        ("block_depth_frames", 2, "integer >= 3"),
        ("grasp_quat", [0.0, 0.0, 0.0, 0.2], "normalized"),
        ("tcp_offset", [0.1, 0.0, 0.0], "six numbers"),
        ("tcp_offset", [0.1, 0.0, "bad", 0.0, 0.0, 0.0], "six numbers"),
        ("observation_joints_deg", [0.0] * 6, "seven"),
        ("carry_joints_deg", [0.0] * 6, "seven"),
        ("desk_size", [2.0, -1.0, 0.02], "three positive"),
        ("num_planning_attempts", 1.5, "integer"),
        ("octomap_enabled_on_prepare", "false", "boolean"),
        ("desk_collision_enabled", "false", "boolean"),
        ("workspace_x_min", "bad", "must be a number"),
        ("desk_z_surface", "bad", "must be a number"),
        ("place_x", None, "must be a number"),
        ("cartesian_jump_threshold", -0.1, "non-negative"),
        ("pos_tolerance", 0.0, "positive"),
        ("carry_joints_deg", [0, -20, 0, 80, 0, 0, 71], "soft-limit"),
        ("planned_start_tolerance_rad", 0.08, "must not exceed"),
        ("reverse_branch_tolerance_rad", 0.40, "must not exceed"),
        ("desk_measurement_max_error", 0.0, "positive"),
    ],
)
def test_rejects_unsafe_values(valid_parameters, name, value, expected):
    valid_parameters[name] = value
    with pytest.raises(ValueError, match=expected):
        validate_robot_parameters(valid_parameters)


def test_reports_multiple_errors_together(valid_parameters):
    valid_parameters["workspace_x_min"] = 1.0
    valid_parameters["workspace_x_max"] = -1.0
    valid_parameters["accel_scaling"] = 0.0
    with pytest.raises(ValueError) as error:
        validate_robot_parameters(valid_parameters)
    message = str(error.value)
    assert "workspace_x_min" in message
    assert "accel_scaling" in message


@pytest.mark.parametrize(
    "name",
    [
        "place_x", "place_y", "place_z",
        "cartesian_jump_threshold", "pos_tolerance", "ori_tolerance",
        "planned_start_tolerance_rad",
        "reverse_branch_tolerance_rad",
        "desk_measurement_max_error",
    ],
)
def test_runtime_parameters_are_required(valid_parameters, name):
    del valid_parameters[name]
    with pytest.raises(ValueError, match=name):
        validate_robot_parameters(valid_parameters)
