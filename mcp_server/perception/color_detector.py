"""OpenCV HSV detection for solid-colour objects.

The axis-aligned bbox remains available for general callers.  Block grasping
also needs the contour's oriented rectangle, so detections expose a pixel
center plus two orthogonal edge directions.  These are image-plane
measurements only; :mod:`mcp_server.components.perception` converts them into
metric base-frame axes using the registered depth image and TF.
"""

import math

import cv2
import numpy as np


COLOR_HSV_RANGES = {
    "blue": [((90, 80, 40), (140, 255, 255))],
    "red": [((0, 100, 50), (10, 255, 255)),
            ((170, 100, 50), (180, 255, 255))],
    "green": [((35, 80, 40), (85, 255, 255))],
    "yellow": [((20, 100, 50), (35, 255, 255))],
    "purple": [((120, 80, 40), (160, 255, 255))],
    "orange": [((10, 100, 50), (20, 255, 255))],
    "cyan": [((80, 80, 40), (100, 255, 255))],
}


def build_color_mask(
    color_img: np.ndarray,
    color_name: str,
) -> np.ndarray | None:
    """Return a cleaned HSV mask without imposing any object-shape model."""

    ranges = COLOR_HSV_RANGES.get(color_name.lower(), ())
    if not ranges:
        return None
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    combined = np.zeros(color_img.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        combined = cv2.bitwise_or(
            combined,
            cv2.inRange(hsv, np.asarray(lower), np.asarray(upper)),
        )
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)


def _filter_by_location(candidates, hint, img_w, img_h):
    if not hint or hint == "unknown":
        return candidates
    hint_l = hint.lower().replace("-", " ").strip()
    result = []
    for candidate in candidates:
        cx, cy = candidate["cx"], candidate["cy"]
        ok = True
        if "left" in hint_l and cx > img_w * 0.6:
            ok = False
        if "right" in hint_l and cx < img_w * 0.4:
            ok = False
        if "top" in hint_l and cy > img_h * 0.6:
            ok = False
        if "bottom" in hint_l and cy < img_h * 0.4:
            ok = False
        if "center" in hint_l and not (
            img_w * 0.25 < cx < img_w * 0.75
            and img_h * 0.25 < cy < img_h * 0.75
        ):
            ok = False
        if ok:
            result.append(candidate)
    return result


def _canonical_direction(vector: np.ndarray) -> np.ndarray:
    """Give an undirected image edge a deterministic sign."""
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    direction = vector.astype(np.float64) / norm
    if direction[0] < -1e-9 or (
        abs(float(direction[0])) <= 1e-9 and direction[1] < 0
    ):
        direction = -direction
    return direction


def _oriented_contour_geometry(contour: np.ndarray) -> dict:
    """Return JSON-friendly geometry from ``cv2.minAreaRect``."""
    rect = cv2.minAreaRect(contour)
    center = np.asarray(rect[0], dtype=np.float64)
    box = cv2.boxPoints(rect).astype(np.float64)

    # boxPoints returns consecutive corners. Pick the longer edge as the
    # primary axis; near-square blocks deliberately use a 90-degree yaw period
    # because either pair of opposite sides is physically equivalent.
    edge_0 = box[1] - box[0]
    edge_1 = box[2] - box[1]
    length_0 = float(np.linalg.norm(edge_0))
    length_1 = float(np.linalg.norm(edge_1))
    if length_1 > length_0:
        primary, primary_length = edge_1, length_1
        secondary_length = length_0
    else:
        primary, primary_length = edge_0, length_0
        secondary_length = length_1

    primary_direction = _canonical_direction(primary)
    secondary_direction = np.array(
        [-primary_direction[1], primary_direction[0]], dtype=np.float64,
    )
    yaw_image_rad = math.atan2(
        float(primary_direction[1]), float(primary_direction[0]),
    )
    aspect_ratio = primary_length / max(secondary_length, 1e-9)
    yaw_period_rad = math.pi / 2.0 if aspect_ratio < 1.15 else math.pi

    contour_points = contour.reshape(-1, 2)
    return {
        "rotated_center_2d": [float(center[0]), float(center[1])],
        "rotated_box_2d": box.tolist(),
        "contour_2d": contour_points.astype(int).tolist(),
        "edge_axes_2d": [
            {
                "direction": primary_direction.tolist(),
                "length_px": primary_length,
            },
            {
                "direction": secondary_direction.tolist(),
                "length_px": secondary_length,
            },
        ],
        "yaw_image_rad": yaw_image_rad,
        "yaw_period_rad": yaw_period_rad,
    }


def _color_candidates(
    color_img: np.ndarray,
    color_name: str,
    include_alternatives: bool = False,
) -> list[dict]:
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    height, width = color_img.shape[:2]
    image_area = height * width

    ranges = []
    color_lower = color_name.lower()
    ranges.extend(COLOR_HSV_RANGES.get(color_lower, []))
    if include_alternatives and color_lower == "blue":
        for alternative in ("purple", "cyan"):
            ranges.extend(COLOR_HSV_RANGES[alternative])
    if not ranges:
        return []

    combined_mask = None
    for lower, upper in ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        combined_mask = mask if combined_mask is None else cv2.bitwise_or(combined_mask, mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200 or area > image_area * 0.3:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        aspect = box_width / max(box_height, 1)
        if not 0.2 <= aspect <= 5.0:
            continue
        oriented = _oriented_contour_geometry(contour)
        oriented_area = (
            float(oriented["edge_axes_2d"][0]["length_px"])
            * float(oriented["edge_axes_2d"][1]["length_px"])
        )
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        rectangularity = area / max(oriented_area, 1e-9)
        solidity = area / max(hull_area, 1e-9)
        if rectangularity < 0.65 or solidity < 0.85:
            continue
        candidates.append({
            "color": color_lower,
            "area": area,
            "x": x,
            "y": y,
            "xmax": x + box_width,
            "ymax": y + box_height,
            "cx": x + box_width / 2,
            "cy": y + box_height / 2,
            "rectangularity": rectangularity,
            "solidity": solidity,
            **oriented,
        })
    return candidates


def detect_color_object(
    color_img: np.ndarray,
    color_name: str,
    location_hint: str = "",
) -> dict | None:
    """Return the best HSV detection with bbox and oriented contour geometry."""
    height, width = color_img.shape[:2]
    candidates = _color_candidates(color_img, color_name, include_alternatives=True)
    if not candidates:
        return None
    filtered = _filter_by_location(candidates, location_hint, width, height)
    if location_hint and location_hint != "unknown" and not filtered:
        return None
    filtered = filtered or candidates
    return max(filtered, key=lambda candidate: candidate["area"])


def detect_by_color(color_img: np.ndarray, color_name: str, location_hint: str = ""):
    """Return the best matching bbox as ``(xmin, ymin, xmax, ymax)``.

    This compatibility wrapper intentionally keeps the original return type.
    New geometry-aware callers should use :func:`detect_color_object`.
    """
    best = detect_color_object(color_img, color_name, location_hint)
    if best is None:
        return None
    return best["x"], best["y"], best["xmax"], best["ymax"]


def detect_all_color_blocks(color_img: np.ndarray, location_hint: str = "") -> list[dict]:
    """Return all visible solid-colour blocks."""
    height, width = color_img.shape[:2]
    blocks = []
    for color_name in COLOR_HSV_RANGES:
        candidates = _color_candidates(color_img, color_name)
        candidates = _filter_by_location(candidates, location_hint, width, height) or candidates
        for candidate in candidates:
            xmin = int(candidate["x"])
            ymin = int(candidate["y"])
            xmax = int(candidate["xmax"])
            ymax = int(candidate["ymax"])
            blocks.append({
                "color": color_name,
                "bbox": [xmin, ymin, xmax, ymax],
                "center_2d": [
                    int(round(candidate["rotated_center_2d"][0])),
                    int(round(candidate["rotated_center_2d"][1])),
                ],
                "area_px": float(candidate["area"]),
                "source": "CV",
                "rotated_center_2d": candidate["rotated_center_2d"],
                "rotated_box_2d": candidate["rotated_box_2d"],
                "contour_2d": candidate["contour_2d"],
                "edge_axes_2d": candidate["edge_axes_2d"],
                "yaw_image_rad": candidate["yaw_image_rad"],
                "yaw_period_rad": candidate["yaw_period_rad"],
            })
    blocks.sort(key=lambda block: block["area_px"], reverse=True)
    return blocks
