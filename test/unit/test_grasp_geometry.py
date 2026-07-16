import pytest

from mcp_server.skills.base import GraspGeometry


def _geometry(depth=0.04):
    return GraspGeometry(
        flange_to_tip=0.1733,
        fingertip_depth=depth,
        approach_height=0.26,
        safe_height=0.40,
        gripper_open=0.10,
        gripper_close=0.02,
        hold_margin=0.005,
        descent_vel=0.2,
        descent_accel=0.05,
    )


def test_fingertip_depth_is_measured_from_object_surface():
    geometry = _geometry()
    assert geometry.fingertip_z(0.0554) == pytest.approx(0.0154)
    assert geometry.grasp_z(0.0554) == pytest.approx(0.1887)


def test_clamped_depth_updates_fingertip_and_tcp_together():
    geometry = _geometry().with_fingertip_depth(0.03)
    assert geometry.fingertip_depth == pytest.approx(0.03)
    assert geometry.fingertip_z(0.0554) == pytest.approx(0.0254)
    assert geometry.grasp_z(0.0554) == pytest.approx(0.1987)
