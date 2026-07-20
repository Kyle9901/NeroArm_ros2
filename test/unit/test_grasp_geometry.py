import pytest

from mcp_server.skills.base import GraspGeometry


def _geometry(depth=0.04):
    return GraspGeometry(
        fingertip_depth=depth,
        approach_height=0.0867,
        safe_height=0.2267,
        gripper_open=0.10,
        gripper_close=0.02,
        hold_margin=0.005,
        descent_vel=0.2,
        descent_accel=0.05,
    )


def test_fingertip_depth_is_measured_from_object_surface():
    geometry = _geometry()
    assert geometry.grasp_tcp_z(0.0554) == pytest.approx(0.0154)


def test_clamped_depth_updates_grasp_tcp():
    geometry = _geometry().with_fingertip_depth(0.03)
    assert geometry.fingertip_depth == pytest.approx(0.03)
    assert geometry.grasp_tcp_z(0.0554) == pytest.approx(0.0254)


def test_desk_clamp_places_tcp_exactly_on_desk():
    surface_z = 0.0554
    desk_z = 0.009
    geometry = _geometry().with_fingertip_depth(surface_z - desk_z)
    assert geometry.grasp_tcp_z(surface_z) == pytest.approx(desk_z)
