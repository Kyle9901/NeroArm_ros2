"""
VLM HTTP client — extracted from vlm_picker_node.py.
Stateless: takes image + target, returns detection result.
"""

import base64
import json
import os

import cv2
import numpy as np
import requests

# ─────────────────────────── Grid overlay (from vlm_picker_node) ───────────────────────────
def _add_coordinate_grid(img: np.ndarray, step: int = 50) -> np.ndarray:
    ann = img.copy()
    h, w = ann.shape[:2]
    for x in range(0, w, step):
        cv2.line(ann, (x, 0), (x, h), (255, 0, 0), 1)
        cv2.putText(ann, str(x), (x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(ann, str(x), (x + 5, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    for y in range(0, h, step):
        cv2.line(ann, (0, y), (w, y), (0, 0, 255), 1)
        cv2.putText(ann, str(y), (5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(ann, str(y), (w - 45, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for x in range(0, w, step):
        for y in range(0, h, step):
            text = f"({x},{y})"
            cv2.putText(ann, text, (x + 2, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 2)
            cv2.putText(ann, text, (x + 2, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
    return ann


# ─────────────────────────── HSV colour detection (from vlm_picker_node) ───────────────────
_COLOR_HSV_RANGES = {
    "blue":   [((90, 80, 40),  (140, 255, 255))],
    "red":    [((0, 100, 50),   (10, 255, 255)),
               ((170, 100, 50), (180, 255, 255))],
    "green":  [((35, 80, 40),   (85, 255, 255))],
    "yellow": [((20, 100, 50),  (35, 255, 255))],
    "purple": [((120, 80, 40),  (160, 255, 255))],
    "orange": [((10, 100, 50),  (20, 255, 255))],
    "cyan":   [((80, 80, 40),   (100, 255, 255))],
}


def _filter_by_location(candidates, hint, img_w, img_h):
    if not hint or hint == "unknown":
        return candidates
    hint_l = hint.lower().replace("-", " ").strip()
    result = []
    for c in candidates:
        cx, cy = c["cx"], c["cy"]
        ok = True
        if "left" in hint_l and cx > img_w * 0.6:
            ok = False
        if "right" in hint_l and cx < img_w * 0.4:
            ok = False
        if "top" in hint_l and cy > img_h * 0.6:
            ok = False
        if "bottom" in hint_l and cy < img_h * 0.4:
            ok = False
        if "center" in hint_l:
            if not (img_w * 0.25 < cx < img_w * 0.75 and img_h * 0.25 < cy < img_h * 0.75):
                ok = False
        if ok:
            result.append(c)
    return result


def detect_by_color(color_img: np.ndarray, color_name: str, location_hint: str = ""):
    """OpenCV HSV colour detection → bbox (xmin,ymin,xmax,ymax) or None."""
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    h, w = color_img.shape[:2]
    img_area = h * w

    ranges = []
    color_lower = color_name.lower()
    if color_lower in _COLOR_HSV_RANGES:
        ranges.extend(_COLOR_HSV_RANGES[color_lower])
    for alt in (["purple", "cyan"] if color_lower == "blue" else []):
        if alt in _COLOR_HSV_RANGES:
            ranges.extend(_COLOR_HSV_RANGES[alt])
    if not ranges:
        return None

    combined_mask = None
    for (lower, upper) in ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        combined_mask = mask if combined_mask is None else cv2.bitwise_or(combined_mask, mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        if area > img_area * 0.3:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cx, cy = x + bw / 2, y + bh / 2
        aspect = bw / max(bh, 1)
        if aspect < 0.2 or aspect > 5.0:
            continue
        candidates.append({"area": area, "x": x, "y": y, "xmax": x + bw, "ymax": y + bh, "cx": cx, "cy": cy})

    if not candidates:
        return None

    filtered = _filter_by_location(candidates, location_hint, w, h) or candidates
    best = max(filtered, key=lambda c: c["area"])
    return (best["x"], best["y"], best["xmax"], best["ymax"])


# ─────────────────────────── Prompt template (from vlm_picker_node) ─────────────────────────
PROMPT_TEMPLATE = (
    "Task: Find the physical object '{target}' in this image.\n"
    "Context: You are a robot eye-in-hand camera. "
    "Ignore the black robotic grippers at the bottom. "
    "Ignore the small holes/indentations on the black foam surface. Focus strictly on the actual 3D physical object.\n"
    "The image resolution is {width}x{height} pixels.\n\n"
    "=== VISUAL COORDINATE GRID HELP ===\n"
    "To help you output highly accurate bounding box coordinates, a visual grid has been overlaid on the image:\n"
    "- BLUE lines are the X-axis (width from 0 to {width}).\n"
    "- RED lines are the Y-axis (height from 0 to {height}).\n"
    "CRITICAL INSTRUCTION: Do NOT just output the exact numbers written on the grid lines! You must INTERPOLATE between the lines. "
    "For example, if an object edge is exactly halfway between the 200 and 250 lines, you MUST output 225. "
    "Ensure the bounding box is EXTREMELY TIGHT, touching the very outer edges of the physical object.\n\n"
    "Step 1: Classify the object into one category:\n"
    '- "color_block": solid single color, simple shape like cube/block (e.g. blue block, red cube)\n'
    '- "textured": complex patterns, non-uniform color, or printed labels\n'
    '- "reflective": glass, metal, bottle, or shiny surface\n'
    '- "other": none of the above\n\n'
    "Step 2: If category is 'color_block', describe its dominant color and approximate location.\n"
    "If category is NOT 'color_block', provide a tight bounding box.\n\n"
    "=== CRITICAL JSON RULES ===\n"
    "1. Return ONLY raw, syntactically valid JSON. No markdown block formatting (```json).\n"
    "2. The 'bbox' object MUST contain EXACTLY 4 key-value pairs: 'xmin', 'ymin', 'xmax', 'ymax'.\n"
    "3. DO NOT omit any keys. DO NOT group multiple numbers under one single key.\n\n"
    "=== EXPECTED OUTPUT FORMAT FOR COLOR_BLOCK ===\n"
    '{{"category": "color_block", "color": "blue", "alternative_colors": [], "location_hint": "center", "found": true}}\n\n'
    "Strictly follow the rules. Do not include any extra chat text, thinking process, or keys."
)


# ─────────────────────────── Public API ─────────────────────────────────────────────────────
class VlmClient:
    """Stateless VLM client — wraps Qwen / GPT-4V API call + OpenCV colour fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        model_name: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("VLM_API_KEY", "")
        self.api_url = api_url or os.environ.get(
            "VLM_API_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
        self.model_name = model_name or os.environ.get("VLM_MODEL", "qwen3.7-plus")

    # ── low-level VLM call ──
    def call(self, image_bgr: np.ndarray, target: str = "", prompt: str | None = None,
             timeout: float = 60.0, max_tokens: int = 300) -> dict | None:
        """Send image + prompt to VLM, return parsed JSON or None.

        If prompt is None, uses the default PROMPT_TEMPLATE with target.
        """
        if not self.api_key:
            raise RuntimeError("VLM_API_KEY not set — cannot call VLM")

        vlm_img = _add_coordinate_grid(image_bgr, step=50)
        ok, buf = cv2.imencode(".jpg", vlm_img)
        if not ok:
            raise RuntimeError("image encode failed")
        b64 = base64.b64encode(buf).decode()

        h, w = image_bgr.shape[:2]
        if prompt is None:
            prompt = PROMPT_TEMPLATE.format(target=target, width=w, height=h)
        else:
            prompt = prompt.format(width=w, height=h)

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        resp = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"VLM API status={resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        text = body["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)

    # ── full detection pipeline ──
    def detect(self, image_bgr: np.ndarray, target: str) -> dict | None:
        """
        Full VLM + OpenCV fallback detection pipeline.

        Returns dict:
          { "found": bool,
            "category": str,
            "bbox": [xmin, ymin, xmax, ymax],
            "center_2d": [cx, cy],
            "color": str | null,
            "source": "VLM" | "CV",
          }
        """
        raw = self.call(image_bgr, target)
        if raw is None:
            return None

        if not raw.get("found", True):
            return {"found": False}

        category = raw.get("category", "other")
        xmin = ymin = xmax = ymax = None
        source = "VLM"

        if category == "color_block":
            color_name = raw.get("color", "")
            location_hint = raw.get("location_hint", "")
            alt_colors = raw.get("alternative_colors", [])

            bbox_cv = detect_by_color(image_bgr, color_name, location_hint)
            if bbox_cv is None and alt_colors:
                for alt in alt_colors:
                    bbox_cv = detect_by_color(image_bgr, alt, location_hint)
                    if bbox_cv is not None:
                        break
            if bbox_cv is not None:
                xmin, ymin, xmax, ymax = bbox_cv
                source = "CV"

        if xmin is None:
            b = raw.get("bbox", {})
            if not all(k in b for k in ("xmin", "ymin", "xmax", "ymax")):
                raise RuntimeError(f"VLM bbox incomplete: {b}")
            xmin = int(b["xmin"])
            ymin = int(b["ymin"])
            xmax = int(b["xmax"])
            ymax = int(b["ymax"])

        h, w = image_bgr.shape[:2]
        xmin, xmax = sorted((max(0, min(xmin, w - 1)), max(0, min(xmax, w - 1))))
        ymin, ymax = sorted((max(0, min(ymin, h - 1)), max(0, min(ymax, h - 1))))

        if xmax <= xmin or ymax <= ymin:
            return None

        cx = int((xmin + xmax) / 2)
        cy = int((ymin + ymax) / 2)

        return {
            "found": True,
            "category": category,
            "bbox": [xmin, ymin, xmax, ymax],
            "center_2d": [cx, cy],
            "color": raw.get("color"),
            "source": source,
        }