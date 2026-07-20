from mcp_server.object_types import (
    is_block_target,
    is_cylinder_target,
    uses_candidate_grasp,
)
from mcp_server.yolo_detector import resolve_target_class


def test_candidate_shape_routing_is_explicit():
    assert is_block_target("抓取红色物块")
    assert is_cylinder_target("抓取红色水瓶")
    assert is_cylinder_target("pick the upright cylinder")
    assert is_cylinder_target("pick a can")
    assert not is_cylinder_target("scan the table")
    assert uses_candidate_grasp("易拉罐")
    assert not uses_candidate_grasp("杯子")


def test_can_uses_custom_label_or_stock_coco_bottle_fallback():
    assert resolve_target_class(
        "易拉罐", {0: "bottle", 1: "cup"}
    ) == "bottle"
    assert resolve_target_class(
        "易拉罐", {0: "bottle", 1: "can"}
    ) == "can"
