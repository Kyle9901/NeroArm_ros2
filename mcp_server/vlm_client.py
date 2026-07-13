"""Stateless VLM HTTP client for image-target detection."""

import base64
import json
import os
import re
import time

import cv2
import numpy as np
import requests

from .config import runtime_dir
from .perception.color_detector import detect_by_color
from .perception.debug import draw_bboxes


def _safe_json_parse(text: str) -> dict:
    """Robust JSON parse for VLM output — handles common LLM formatting issues."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fix single-quoted keys/values (common LLM mistake)
    # Use ast.literal_eval for Python-style dicts
    try:
        import ast
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    # Regex-based fix as fallback
    try:
        fixed = re.sub(r"'([^']*)':", r'"\1":', text)
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
        fixed = re.sub(r":\s*True", r': true', fixed)
        fixed = re.sub(r":\s*False", r': false', fixed)
        fixed = re.sub(r":\s*None", r': null', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # Extract JSON object with regex (supports nested braces)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Last resort: try to fix malformed bbox like {"xmin": 425, 310, 510, 600}
    # This is a common VLM mistake where it outputs positional values without keys
    m = re.search(r'\{[^}]*\}', text)
    if m:
        candidate = m.group()
        # Try to extract 4 numbers as bbox
        nums = re.findall(r'\d+', candidate)
        if len(nums) >= 4:
            return {"found": True, "category": "other", "bbox": [int(n) for n in nums[:4]]}
    raise RuntimeError(f"VLM returned unparseable JSON: {text[:300]}")

# ─────────────────────────── Coordinate grid overlay ───────────────────────────
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


# ─────────────────────────── Detection prompt template ─────────────────────────
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


class VlmClient:
    """Stateless VLM client — wraps Qwen / GPT-4V API call + OpenCV colour fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        model_name: str | None = None,
        debug_dir: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("VLM_API_KEY")
        self.api_url = api_url or os.environ.get("VLM_API_URL")
        self.model_name = model_name or os.environ.get("VLM_MODEL")
        self.debug_dir = debug_dir or os.environ.get("VLM_DEBUG_DIR", str(runtime_dir("debug")))
        os.makedirs(self.debug_dir, exist_ok=True)
        if not self.api_key:
            raise RuntimeError(
                "VLM_API_KEY not set. "
                "Set the environment variable or pass api_key to VlmClient()")

    def _save_debug(self, image_bgr: np.ndarray, bboxes: list[list[int]],
                    labels: list[str] | None = None, prefix: str = "detect") -> str:
        """Save image with bboxes drawn. Returns file path."""
        annotated = draw_bboxes(image_bgr, bboxes, labels)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.debug_dir, f"{prefix}_{ts}.jpg")
        cv2.imwrite(path, annotated)
        return path

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
            raise RuntimeError(
                f"VLM API returned HTTP {resp.status_code}. "
                f"Check VLM_API_KEY is valid and VLM_MODEL is correct. "
                f"Detail: {resp.text[:200]}")
        body = resp.json()
        text = body["choices"][0]["message"]["content"].strip()
        return _safe_json_parse(text)

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
            if isinstance(b, dict):
                # Try standard keys first
                if all(k in b for k in ("xmin", "ymin", "xmax", "ymax")):
                    xmin, ymin, xmax, ymax = int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"])
                # Fallback: VLM sometimes returns {"xmin": v1, v2, v3, v4} as a malformed dict
                elif len(b) == 1 and isinstance(list(b.values())[0], (list, tuple)):
                    vals = list(b.values())[0]
                    if len(vals) == 4:
                        xmin, ymin, xmax, ymax = [int(v) for v in vals]
                # Fallback: try numeric values in order
                elif len(b) >= 4:
                    vals = list(b.values())[:4]
                    if all(isinstance(v, (int, float)) for v in vals):
                        xmin, ymin, xmax, ymax = [int(v) for v in vals]
            elif isinstance(b, (list, tuple)) and len(b) == 4:
                xmin, ymin, xmax, ymax = [int(v) for v in b]
            if xmin is None:
                raise RuntimeError(
                    f"VLM returned incomplete bbox: {b}. "
                    "The model did not follow the JSON output format. "
                    "Try simplifying the target description or retry")

        h, w = image_bgr.shape[:2]
        xmin, xmax = sorted((max(0, min(xmin, w - 1)), max(0, min(xmax, w - 1))))
        ymin, ymax = sorted((max(0, min(ymin, h - 1)), max(0, min(ymax, h - 1))))

        if xmax <= xmin or ymax <= ymin:
            return None

        cx = int((xmin + xmax) / 2)
        cy = int((ymin + ymax) / 2)

        debug_path = self._save_debug(image_bgr, [[xmin, ymin, xmax, ymax]],
                                       labels=[f"{category}:{raw.get('color', '')}"],
                                       prefix="detect")

        return {
            "found": True,
            "category": category,
            "bbox": [xmin, ymin, xmax, ymax],
            "center_2d": [cx, cy],
            "color": raw.get("color"),
            "source": source,
            "debug_image": debug_path,
        }
