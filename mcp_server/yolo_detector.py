#!/usr/bin/env python3
"""
YOLO object detector for robot arm grasping.
Supports YOLOv8/v11 models via ultralytics.

Architecture decisions (see yolo-integration-plan.md):
  - Model: YOLOv11s, PyTorch CUDA, 10-15ms on GTX 1050 Ti
  - Label Mapping: 2-layer (hardcoded dict + fuzzy text similarity)
  - Multi-target: category → location_hint → largest area
  - Fail Fast: OOM → crash loudly, never silently fallback to CPU
"""

import os
import sys
import threading
from typing import Optional

import cv2
import numpy as np

_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Layer 1: Hardcoded label mapping (Chinese → English → COCO class)
# ═══════════════════════════════════════════════════════════════════════════════════════════

_LABEL_MAP: dict[str, str] = {
    # ── 瓶罐类 ──
    "瓶子": "bottle", "水瓶": "bottle", "矿泉水": "bottle",
    "矿泉水瓶": "bottle", "塑料瓶": "bottle", "饮料瓶": "bottle",
    "水杯": "cup", "杯子": "cup", "马克杯": "cup",
    "茶杯": "cup", "咖啡杯": "cup", "酒杯": "cup",
    "易拉罐": "can", "罐子": "can", "可乐罐": "can",
    "瓶": "bottle", "杯": "cup", "罐": "can",
    # ── 水果类 ──
    "苹果": "apple", "香蕉": "banana", "橘子": "orange",
    "橙子": "orange", "梨": "pear",
    # ── 方块/积木 ──
    "方块": "block", "积木": "block", "色块": "block",
    "立方体": "block", "正方体": "block",
    "蓝色方块": "block", "红色方块": "block", "绿色方块": "block",
    "黄色方块": "block", "紫色方块": "block", "橙色方块": "block",
    "物块": "block",
    # ── 日常物品 ──
    "剪刀": "scissors", "手机": "cell phone",
    "遥控器": "remote", "鼠标": "mouse",
    "键盘": "keyboard", "书": "book",
    "碗": "bowl", "叉子": "fork", "勺子": "spoon",
    "刀": "knife",
    # ── 工具类 ──
    "螺丝刀": "screwdriver", "扳手": "wrench",
    "锤子": "hammer", "钳子": "pliers",
}

# COCO classes that YOLO can detect (subset of 80 classes)
_COCO_CLASSES: set[str] = {
    "bottle", "cup", "wine glass", "can", "bowl",
    "apple", "banana", "orange", "pear", "carrot", "broccoli",
    "cell phone", "remote", "keyboard", "mouse", "book",
    "scissors", "fork", "knife", "spoon",
    "chair", "couch", "potted plant", "dining table",
    "tv", "laptop", "clock", "vase", "teddy bear",
    "sports ball", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket",
    "screwdriver", "wrench", "hammer", "pliers",
    "toothbrush", "hair drier", "toaster", "refrigerator",
    "sink", "toilet", "oven", "microwave",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "kite",
    "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket",
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
    "cake", "pizza", "donut", "hot dog", "sandwich",
}

# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Layer 2: Fuzzy text similarity (Levenshtein-based)
# ═══════════════════════════════════════════════════════════════════════════════════════════

def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Return normalized similarity ratio between two strings (0.0 to 1.0)."""
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)
    for i, c1 in enumerate(s1, 1):
        curr[0] = i
        for j, c2 in enumerate(s2, 1):
            curr[j] = prev[j - 1] if c1 == c2 else 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return 1.0 - prev[len2] / max(len1, len2)


def _fuzzy_match_class(target: str, yolo_names: dict[int, str], threshold: float = 0.6) -> Optional[str]:
    """Find the best-matching YOLO class name for a target string.

    Returns the matched COCO class name, or None if no match above threshold.
    """
    target_lower = target.lower().strip()

    # Try exact substring match first (fast path)
    for name in yolo_names.values():
        if target_lower == name.lower() or target_lower in name.lower() or name.lower() in target_lower:
            return name

    # Levenshtein fuzzy match
    best_name, best_score = None, 0.0
    for name in yolo_names.values():
        score = _levenshtein_ratio(target_lower, name.lower())
        if score > best_score:
            best_name, best_score = name, score

    if best_score >= threshold:
        return best_name
    return None


def resolve_target_class(target: str, yolo_names: dict[int, str]) -> Optional[str]:
    """Resolve a user-facing target description to a YOLO COCO class name.

    Two-layer strategy:
      1. Hardcoded LABEL_MAP (Chinese → English → COCO class)
      2. Fuzzy Levenshtein matching against all YOLO class names

    Returns the matched COCO class name, or None.
    """
    target_lower = target.lower().strip()

    # Layer 1: Hardcoded mapping
    for cn_key, en_class in _LABEL_MAP.items():
        if cn_key in target_lower or target_lower in cn_key:
            return en_class

    # Check if target already IS a valid COCO class
    if target_lower in _COCO_CLASSES:
        return target_lower

    # Layer 2: Fuzzy matching
    return _fuzzy_match_class(target, yolo_names)


# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Drawing utilities
# ═══════════════════════════════════════════════════════════════════════════════════════════

_BBOX_COLORS = {
    "HSV":  (0, 255, 0),    # green
    "YOLO": (255, 0, 0),    # blue
    "VLM":  (255, 255, 0),  # cyan
}


def draw_detection(img: np.ndarray, bbox: list[int], label: str,
                   confidence: float | None = None,
                   pos_3d: tuple[float, float, float] | None = None,
                   color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """Draw a single detection bbox with label, confidence, and 3D coordinates."""
    out = img.copy()
    xmin, ymin, xmax, ymax = [int(v) for v in bbox]
    cv2.rectangle(out, (xmin, ymin), (xmax, ymax), color, 2)

    cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
    cv2.circle(out, (cx, cy), 4, color, -1)

    lines = [label]
    if confidence is not None:
        lines.append(f"conf={confidence:.2f}")
    if pos_3d is not None:
        lines.append(f"3D=({pos_3d[0]:.3f},{pos_3d[1]:.3f},{pos_3d[2]:.3f})")

    y0 = max(ymin - 10, 15)
    for i, text in enumerate(lines):
        y = y0 - i * 18
        cv2.putText(out, text, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ═══════════════════════════════════════════════════════════════════════════════════════════
#  YoloDetector
# ═══════════════════════════════════════════════════════════════════════════════════════════

class YoloDetector:
    """Singleton-style YOLO detector. Load once at startup, reuse across all detections.

    Usage:
        detector = YoloDetector(model_path="models/yolov11s.pt")
        result = detector.detect(image_bgr, target="bottle")
        # result: {"found": True, "bbox": [...], "center_2d": [...], "class": "bottle", ...}
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float | None = None,
        device: str | None = None,
    ):
        self._model_path = model_path or os.environ.get(
            "YOLO_MODEL_PATH",
            os.path.join(_pkg_dir, "models", "yolov8s.pt"),
        )
        self.confidence = confidence if confidence is not None else float(
            os.environ.get("YOLO_CONFIDENCE", "0.35")
        )
        self._device = device or os.environ.get("YOLO_DEVICE", "cuda")

        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model. Fail Fast on any error."""
        if not os.path.exists(self._model_path):
            if self._model_path.endswith(".pt"):
                msg = (
                    f"\n{'='*60}\n"
                    f"  YOLO 模型文件不存在: {self._model_path}\n"
                    f"\n"
                    f"  如果你是离线环境，请先在有网络的机器上下载预训练模型：\n"
                    f"    pip install ultralytics\n"
                    f"    python -c \"from ultralytics import YOLO; YOLO('yolov11s.pt')\"\n"
                    f"    cp yolov11s.pt {self._model_path}\n"
                    f"\n"
                    f"  或者设置环境变量 YOLO_MODEL_PATH 指向你的模型文件。\n"
                    f"{'='*60}\n"
                )
                print(msg, file=sys.stderr, flush=True)
                raise FileNotFoundError(f"YOLO model not found: {self._model_path}")
            else:
                raise FileNotFoundError(f"YOLO model not found: {self._model_path}")

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics 未安装。请运行: pip install ultralytics\n"
                "如果已安装，请检查 Python 环境是否与 .venv 一致。"
            )

        print(f"[yolo] Loading model: {self._model_path} (device={self._device})",
              file=sys.stderr, flush=True)

        try:
            self.model = YOLO(self._model_path)
        except Exception as e:
            raise RuntimeError(
                f"YOLO 模型加载失败: {e}\n"
                f"模型路径: {self._model_path}\n"
                f"如果文件已损坏，请重新下载。"
            ) from e

        # Verify GPU availability (Fail Fast if CUDA requested but unavailable)
        if self._device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "YOLO_DEVICE=cuda 但 CUDA 不可用。\n"
                    "请检查: (1) nvidia-smi 是否正常, (2) torch 是否安装了 CUDA 版本\n"
                    "或者设置 YOLO_DEVICE=cpu 使用 CPU 推理（会很慢，不推荐）。"
                )
            self.model.to(self._device)
            print(f"[yolo] Model loaded on CUDA (GPU memory: "
                  f"{torch.cuda.memory_allocated() / 1024**2:.0f} MB allocated)",
                  file=sys.stderr, flush=True)
        else:
            print(f"[yolo] Model loaded on CPU (warning: inference will be slow)",
                  file=sys.stderr, flush=True)

        self._class_names = self.model.names  # {0: 'person', 1: 'bicycle', ...}

        # Warmup: run a dummy inference to compile CUDA kernels
        import numpy as np
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.model(dummy, conf=self.confidence, verbose=False)
        print(f"[yolo] Warmup complete, ready for inference", file=sys.stderr, flush=True)

    # ─────────────────────────── internal helpers ───────────────────────────

    def _filter_by_location(self, detections: list[dict], hint: str,
                            img_w: int, img_h: int) -> list[dict]:
        """Filter detections by location hint (left/right/top/bottom/center)."""
        if not hint or hint == "unknown":
            return detections
        hint_lower = hint.lower().replace("-", " ").strip()
        result = []
        for d in detections:
            cx, cy = d["center_2d"]
            ok = True
            if "left" in hint_lower and cx > img_w * 0.6:
                ok = False
            if "right" in hint_lower and cx < img_w * 0.4:
                ok = False
            if "top" in hint_lower and cy > img_h * 0.6:
                ok = False
            if "bottom" in hint_lower and cy < img_h * 0.4:
                ok = False
            if "center" in hint_lower:
                if not (img_w * 0.25 < cx < img_w * 0.75 and img_h * 0.25 < cy < img_h * 0.75):
                    ok = False
            if ok:
                result.append(d)
        return result

    def _pick_best(self, detections: list[dict]) -> Optional[dict]:
        """Pick the best detection: largest area."""
        if not detections:
            return None
        return max(detections, key=lambda d: d["area"])

    # ─────────────────────────── public API ───────────────────────────

    def detect(self, image_bgr: np.ndarray, target: str,
               location_hint: str = "") -> dict:
        """Detect a single object matching the target description.

        Filtering strategy: category match → location hint → largest area.

        Args:
            image_bgr: BGR image (numpy array, H×W×3).
            target: User-facing target description, e.g. "bottle", "矿泉水瓶".
            location_hint: Optional spatial hint: "left", "right", "center", etc.

        Returns:
            {"found": True, "bbox": [xmin,ymin,xmax,ymax], "center_2d": [cx,cy],
             "class": "bottle", "confidence": 0.92, "area": 12345, "source": "YOLO"}
            or {"found": False}
        """
        results = self.model(image_bgr, conf=self.confidence, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return {"found": False}

        boxes = results[0].boxes
        h, w = image_bgr.shape[:2]

        # Resolve target to COCO class
        matched_class = resolve_target_class(target, self._class_names)

        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            detected_name = self._class_names[cls_id]

            # Filter by class if we have a match
            if matched_class is not None:
                # Fuzzy check: is the detected class close to the matched class?
                if detected_name.lower() != matched_class.lower():
                    # Also check if one contains the other
                    if (matched_class.lower() not in detected_name.lower() and
                        detected_name.lower() not in matched_class.lower()):
                        continue

            xyxy = box.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = [float(v) for v in xyxy]
            area = (xmax - xmin) * (ymax - ymin)
            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)

            detections.append({
                "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                "center_2d": [cx, cy],
                "class": detected_name,
                "confidence": float(box.conf[0]),
                "area": area,
            })

        if not detections:
            return {"found": False}

        # Filter by location hint
        filtered = self._filter_by_location(detections, location_hint, w, h)
        if not filtered:
            filtered = detections

        # Pick largest area
        best = self._pick_best(filtered)
        if best is None:
            return {"found": False}

        best["source"] = "YOLO"
        best["found"] = True
        return best

    def detect_all(self, image_bgr: np.ndarray) -> list[dict]:
        """Detect all objects in the image.

        Returns:
            List of dicts, each with: bbox, center_2d, class, confidence, area, source.
        """
        results = self.model(image_bgr, conf=self.confidence, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            detected_name = self._class_names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = [float(v) for v in xyxy]
            area = (xmax - xmin) * (ymax - ymin)

            detections.append({
                "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                "center_2d": [int((xmin + xmax) / 2), int((ymin + ymax) / 2)],
                "class": detected_name,
                "confidence": float(box.conf[0]),
                "area": area,
                "source": "YOLO",
            })

        detections.sort(key=lambda d: d["area"], reverse=True)
        return detections

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return hasattr(self, "model") and self.model is not None


class LazyYoloDetector:
    """YOLO-compatible proxy which loads and warms the model on first inference."""

    def __init__(self, model_path=None, confidence=None, device=None):
        self._options = (model_path, confidence, device)
        self._confidence = confidence
        self._detector = None
        self._lock = threading.Lock()

    def _get(self) -> YoloDetector:
        if self._detector is None:
            with self._lock:
                if self._detector is None:
                    model_path, _initial_confidence, device = self._options
                    # Runtime configuration may update confidence before the
                    # first inference; preserve that value across lazy load.
                    self._detector = YoloDetector(model_path, self._confidence, device)
        return self._detector

    @property
    def confidence(self) -> float:
        if self._detector is not None:
            return self._detector.confidence
        if self._confidence is not None:
            return self._confidence
        return float(os.environ.get("YOLO_CONFIDENCE", "0.35"))

    @confidence.setter
    def confidence(self, value: float):
        self._confidence = value
        if self._detector is not None:
            self._detector.confidence = value

    def detect(self, *args, **kwargs):
        return self._get().detect(*args, **kwargs)

    def detect_all(self, *args, **kwargs):
        return self._get().detect_all(*args, **kwargs)

    def is_loaded(self) -> bool:
        return self._detector is not None and self._detector.is_loaded()
