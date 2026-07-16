import numpy as np
import pytest

from mcp_server.components.base import ImageFrame
from mcp_server.components.perception import bbox_to_3d


class _FakeNode:
    @staticmethod
    def get_color_info():
        return {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0}

    @staticmethod
    def transform_to_base(x, y, z, **_kwargs):
        return {"x": x, "y": y, "z": z}

    @staticmethod
    def compute_3d(_u, _v, _depth):
        return None


class _FakeBridge:
    node = _FakeNode()


def _frame_with_sloped_desk_and_empty_object():
    height, width = 480, 640
    yy, xx = np.indices((height, width))
    # Smooth, tilted table in millimetres.
    depth = (550.0 - 0.03 * yy + 0.01 * xx).astype(np.uint16)
    depth[276:332, 184:240] = 0
    return ImageFrame(
        frame_id=1,
        color=np.zeros((height, width, 3), dtype=np.uint8),
        depth=depth,
        timestamp_s=0.0,
    )


def test_known_block_height_uses_local_desk_geometry():
    result = bbox_to_3d(
        _FakeBridge(),
        _frame_with_sloped_desk_and_empty_object(),
        [184, 276, 239, 331],
        known_object_height=0.05,
    )

    assert result.ok
    assert result.data["method"] == "known_height_local_desk_ransac"
    assert result.data["depth_is_estimated"] is True
    geometry = result.data["geometry"]
    assert geometry["height"] == 0.05
    assert geometry["height_source"] == "configured_color_block"
    assert geometry["surface"]["z"] - geometry["local_desk_z"] == pytest.approx(0.05)
    assert geometry["center"]["z"] - geometry["local_desk_z"] == pytest.approx(0.025)
    assert geometry["size"] == {"x": 0.05, "y": 0.05, "z": 0.05}


def test_unknown_object_never_expands_into_surrounding_desk():
    result = bbox_to_3d(
        _FakeBridge(),
        _frame_with_sloped_desk_and_empty_object(),
        [184, 276, 239, 331],
    )

    assert not result.ok
    assert "No valid depth in bbox" in result.error
