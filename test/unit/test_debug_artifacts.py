import numpy as np

from mcp_server.components import perception


def test_yolo_debug_keeps_one_latest_image_and_preserves_other_artifacts(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(perception, "_DEBUG_DIR", str(tmp_path))
    (tmp_path / "yolo_legacy_1.jpg").write_bytes(b"old")
    (tmp_path / "yolo_legacy_2.jpg").write_bytes(b"old")
    unrelated = tmp_path / "detect_old.jpg"
    unrelated.write_bytes(b"keep")
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    first = perception._save_debug(
        image, [[4, 4, 20, 20]], ["bottle:0.80"], prefix="yolo",
    )
    second = perception._save_debug(
        image, [[5, 5, 21, 21]], ["bottle:0.90"], prefix="yolo",
    )

    assert first == second == str(tmp_path / "yolo_latest.jpg")
    assert [path.name for path in tmp_path.glob("yolo_*.jpg")] == [
        "yolo_latest.jpg"
    ]
    assert unrelated.read_bytes() == b"keep"


def test_cylinder_depth_debug_keeps_one_four_panel_png(monkeypatch, tmp_path):
    monkeypatch.setattr(perception, "_DEBUG_DIR", str(tmp_path))
    (tmp_path / "cylinder_depth_legacy.png").write_bytes(b"old")
    depth = np.full((64, 64), 550, dtype=np.uint16)
    raw_u = np.asarray([20, 21, 22, 23])
    raw_v = np.asarray([20, 21, 22, 23])
    desk_fit = {
        "sample_u": np.asarray([5, 6, 7]),
        "sample_v": np.asarray([50, 50, 50]),
        "inliers": np.asarray([True, False, True]),
    }
    connectivity = {
        "gap_px": 5,
        "depth_tolerance_mm": 19.0,
    }

    first = perception._save_cylinder_depth_debug(
        depth,
        [16, 16, 32, 40],
        raw_u,
        raw_v,
        desk_fit,
        raw_u[:3],
        raw_v[:3],
        raw_u[:2],
        raw_v[:2],
        connectivity,
    )
    second = perception._save_cylinder_depth_debug(
        depth,
        [16, 16, 32, 40],
        raw_u,
        raw_v,
        desk_fit,
        raw_u[:3],
        raw_v[:3],
        raw_u[:2],
        raw_v[:2],
        connectivity,
    )

    assert first == second == str(tmp_path / "cylinder_depth_latest.png")
    assert [path.name for path in tmp_path.glob("cylinder_depth_*.png")] == [
        "cylinder_depth_latest.png"
    ]


def test_adaptive_depth_support_bridges_holes_but_separates_far_layer():
    # The two 550 mm strips are five pixels apart and should join for this
    # 60-pixel-wide bbox. The equally close 700 mm strip is another layer.
    vertical = np.arange(10, 16, dtype=np.int32)
    sample_u = np.concatenate((
        np.full(len(vertical), 10),
        np.full(len(vertical), 15),
        np.full(len(vertical), 20),
    ))
    sample_v = np.tile(vertical, 3)
    sample_depths = np.concatenate((
        np.full(len(vertical), 550.0),
        np.full(len(vertical), 552.0),
        np.full(len(vertical), 700.0),
    ))

    support = perception._adaptive_depth_connected_support(
        sample_u,
        sample_v,
        sample_depths,
        [0, 0, 59, 79],
    )

    assert support["gap_px"] == 5
    assert support["connected_count"] == 12
    assert support["connected_ratio"] == 12 / 18
    assert np.all(support["keep_local"][:12])
    assert not np.any(support["keep_local"][12:])
