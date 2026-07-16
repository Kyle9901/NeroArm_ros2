from copy import deepcopy

import pytest

from mcp_server.config.robot import load_robot_parameters, validate_robot_parameters


@pytest.fixture
def valid_parameters():
    _path, parameters = load_robot_parameters()
    return deepcopy(parameters)


def test_repository_robot_config_is_valid(valid_parameters):
    validate_robot_parameters(valid_parameters)


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("velocity_scaling", 1.5, "velocity_scaling"),
        ("fingertip_depth", 0.06, "smaller than color_block_height"),
        ("grasp_quat", [0.0, 0.0, 0.0, 0.2], "normalized"),
        ("home_joints_deg", [0.0] * 6, "seven"),
        ("desk_size", [2.0, -1.0, 0.02], "three positive"),
        ("num_planning_attempts", 1.5, "integer"),
        ("octomap_enabled_on_prepare", "false", "boolean"),
        ("desk_collision_enabled", "false", "boolean"),
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
