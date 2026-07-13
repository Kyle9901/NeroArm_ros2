"""OpenCV HSV detection for solid-colour blocks."""

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
        candidates.append({
            "color": color_lower,
            "area": area,
            "x": x,
            "y": y,
            "xmax": x + box_width,
            "ymax": y + box_height,
            "cx": x + box_width / 2,
            "cy": y + box_height / 2,
        })
    return candidates


def detect_by_color(color_img: np.ndarray, color_name: str, location_hint: str = ""):
    """Return the best matching bbox as ``(xmin, ymin, xmax, ymax)``."""
    height, width = color_img.shape[:2]
    candidates = _color_candidates(color_img, color_name, include_alternatives=True)
    if not candidates:
        return None
    filtered = _filter_by_location(candidates, location_hint, width, height) or candidates
    best = max(filtered, key=lambda candidate: candidate["area"])
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
                "center_2d": [(xmin + xmax) // 2, (ymin + ymax) // 2],
                "area_px": float(candidate["area"]),
                "source": "CV",
            })
    blocks.sort(key=lambda block: block["area_px"], reverse=True)
    return blocks
