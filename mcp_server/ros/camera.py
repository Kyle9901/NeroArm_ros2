"""Thread-safe RGB-D camera subscription and pixel deprojection."""

import threading

import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image


class CameraStream:
    """Own camera subscriptions and the latest aligned RGB-D samples."""

    def __init__(
        self,
        node,
        color_topic: str = "/camera/color/image_raw",
        depth_topic: str = "/camera/depth/image_raw",
        camera_info_topic: str = "/camera/color/camera_info",
    ):
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._color_img = None
        self._depth_img = None
        self._color_info = None

        self._subscriptions = [
            node.create_subscription(Image, color_topic, self._color_callback, 10),
            node.create_subscription(Image, depth_topic, self._depth_callback, 10),
            node.create_subscription(CameraInfo, camera_info_topic, self._info_callback, 10),
        ]

    def _color_callback(self, message):
        image = self._bridge.imgmsg_to_cv2(message, "bgr8")
        with self._lock:
            self._color_img = image
            if self._depth_img is not None:
                self._ready.set()

    def _depth_callback(self, message):
        image = self._bridge.imgmsg_to_cv2(message, "16UC1")
        with self._lock:
            self._depth_img = image
            if self._color_img is not None:
                self._ready.set()

    def _info_callback(self, message):
        info = {
            "fx": message.k[0],
            "fy": message.k[4],
            "cx": message.k[2],
            "cy": message.k[5],
            "width": message.width,
            "height": message.height,
        }
        with self._lock:
            self._color_info = info

    def get_latest_images(self, timeout: float = 2.0):
        """Wait for camera activity and return copies of color and depth."""
        self._ready.clear()
        if not self._ready.wait(timeout):
            return None
        with self._lock:
            if self._color_img is None or self._depth_img is None:
                return None
            return self._color_img.copy(), self._depth_img.copy()

    def get_color_info(self) -> dict | None:
        with self._lock:
            return self._color_info.copy() if self._color_info else None

    def compute_3d(
        self,
        u: int,
        v: int,
        depth_img: np.ndarray,
        margin_px: int = 2,
    ) -> dict | None:
        """Deproject an aligned depth sample into the color optical frame."""
        info = self.get_color_info()
        if info is None:
            return None
        height, width = depth_img.shape[:2]
        if not (0 <= u < width and 0 <= v < height):
            return None

        roi = depth_img[
            max(0, v - margin_px):min(height, v + margin_px + 1),
            max(0, u - margin_px):min(width, u + margin_px + 1),
        ]
        valid = roi[roi > 0]
        if len(valid) == 0:
            return None

        depth_mm = float(np.median(valid))
        z_c = depth_mm / 1000.0
        return {
            "x_c": (u - info["cx"]) * z_c / info["fx"],
            "y_c": (v - info["cy"]) * z_c / info["fy"],
            "z_c": z_c,
            "depth_mm": depth_mm,
            "valid_points": len(valid),
        }
