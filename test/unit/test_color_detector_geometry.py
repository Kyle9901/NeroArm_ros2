import math

import cv2
import numpy as np
import pytest

from mcp_server.perception.color_detector import (
    detect_by_color,
    detect_color_object,
)


def _periodic_difference(a: float, b: float, period: float) -> float:
    return abs((a - b + period / 2.0) % period - period / 2.0)


def test_hsv_detection_exposes_rotated_center_and_edge_axes():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    box = cv2.boxPoints(((160.0, 120.0), (90.0, 50.0), 30.0))
    cv2.fillPoly(image, [box.astype(np.int32)], (0, 0, 255))

    detection = detect_color_object(image, "red")

    assert detection is not None
    assert detection["rotated_center_2d"] == pytest.approx((160.0, 120.0), abs=1.0)
    axes = detection["edge_axes_2d"]
    assert len(axes) == 2
    first = np.asarray(axes[0]["direction"])
    second = np.asarray(axes[1]["direction"])
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(second) == pytest.approx(1.0, abs=1e-6)
    assert float(np.dot(first, second)) == pytest.approx(0.0, abs=1e-6)
    assert axes[0]["length_px"] > axes[1]["length_px"]
    assert _periodic_difference(
        detection["yaw_image_rad"], math.radians(30.0), math.pi,
    ) < math.radians(2.0)
    assert len(detection["contour_2d"]) >= 4
    assert len(detection["rotated_box_2d"]) == 4


def test_legacy_color_bbox_return_type_is_preserved():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(image, (40, 30), (100, 90), (0, 255, 0), -1)

    bbox = detect_by_color(image, "green")

    assert isinstance(bbox, tuple)
    assert len(bbox) == 4
    assert bbox[0] <= 40 and bbox[1] <= 30
    assert bbox[2] >= 100 and bbox[3] >= 90
