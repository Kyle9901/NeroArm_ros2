"""Thread-safe RGB-D camera subscription and pixel deprojection."""

import threading
import time
from dataclasses import dataclass

import numpy as np
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


@dataclass
class CameraSample:
    """A registered RGB-D pair with the ROS acquisition metadata intact."""

    color: np.ndarray
    depth: np.ndarray
    stamp: object
    source_frame: str

    def __iter__(self):
        """Keep legacy ``color, depth = sample`` callers working."""
        yield self.color
        yield self.depth


class CameraStream:
    """Own camera subscriptions and the latest aligned RGB-D samples."""

    def __init__(
        self,
        node,
        color_topic: str = "/camera/color/image_raw",
        depth_topic: str = "/camera/depth/image_raw",
        camera_info_topic: str = "/camera/color/camera_info",
    ):
        self._node = node
        self._color_topic = color_topic
        self._depth_topic = depth_topic
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._ready = threading.Event()
        self._color_sample = None
        self._depth_sample = None
        self._paired_images = None
        self._last_paired_stamps = None
        self._color_info = None
        self._max_pair_skew_ns = 80_000_000
        self._last_pair_monotonic = None
        self._last_pair_skew_ms = None
        self._last_rejection = "waiting_for_frames"

        # CameraInfo is cheap and remains available for deprojection. RGB-D
        # images are subscribed only while a caller is actively requesting a
        # frame. This preserves the Orbbec driver's lazy MJPEG decoding and
        # avoids competing with MoveIt/RViz during planning.
        self._info_subscription = node.create_subscription(
            CameraInfo, camera_info_topic, self._info_callback,
            qos_profile_sensor_data,
        )
        self._image_subscriptions = []

    def _start_image_subscriptions(self):
        if self._image_subscriptions:
            return
        self._image_subscriptions = [
            self._node.create_subscription(
                Image, self._color_topic, self._color_callback,
                qos_profile_sensor_data,
            ),
            self._node.create_subscription(
                Image, self._depth_topic, self._depth_callback,
                qos_profile_sensor_data,
            ),
        ]

    def _stop_image_subscriptions(self):
        subscriptions, self._image_subscriptions = self._image_subscriptions, []
        for subscription in subscriptions:
            self._node.destroy_subscription(subscription)

    def _color_callback(self, message):
        image = self._bridge.imgmsg_to_cv2(message, "bgr8")
        with self._lock:
            self._color_sample = (
                self._stamp_ns(message), message.header.stamp,
                message.header.frame_id, image,
            )
            self._try_pair_locked()

    def _depth_callback(self, message):
        image = self._bridge.imgmsg_to_cv2(message, "16UC1")
        with self._lock:
            self._depth_sample = (
                self._stamp_ns(message), message.header.stamp,
                message.header.frame_id, image,
            )
            self._try_pair_locked()

    @staticmethod
    def _stamp_ns(message) -> int:
        return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec

    def _try_pair_locked(self):
        """Publish only spatially compatible frames captured at nearly the same time."""
        if self._color_sample is None or self._depth_sample is None:
            return
        color_stamp, color_ros_stamp, color_frame, color = self._color_sample
        depth_stamp, depth_ros_stamp, _depth_frame, depth = self._depth_sample
        skew_ns = abs(color_stamp - depth_stamp)
        if skew_ns > self._max_pair_skew_ns:
            self._last_rejection = "timestamp_skew"
            return
        if color.shape[:2] != depth.shape[:2]:
            # A color bbox cannot index an unregistered depth image safely.
            self._last_rejection = "shape_mismatch"
            return
        pair_stamps = (color_stamp, depth_stamp)
        if pair_stamps == getattr(self, "_last_paired_stamps", None):
            return
        # Depth is registered into the color optical geometry, so coordinates
        # deprojected with color intrinsics belong to the color frame. Use the
        # depth acquisition stamp because its values produced the 3D point.
        self._paired_images = CameraSample(
            color=color.copy(),
            depth=depth.copy(),
            stamp=depth_ros_stamp,
            source_frame=color_frame or "camera_color_optical_frame",
        )
        self._last_pair_monotonic = time.monotonic()
        self._last_pair_skew_ms = skew_ns / 1_000_000.0
        self._last_paired_stamps = pair_stamps
        self._last_rejection = None
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
        """Wait for a timestamp-matched, depth-to-color registered RGB-D pair."""
        with self._capture_lock:
            with self._lock:
                self._color_sample = None
                self._depth_sample = None
                self._paired_images = None
                self._last_paired_stamps = None
                self._last_rejection = "waiting_for_frames"
                self._ready.clear()
            self._start_image_subscriptions()
            try:
                if not self._ready.wait(timeout):
                    return None
                with self._lock:
                    if self._paired_images is None:
                        return None
                    sample = self._paired_images
                    return CameraSample(
                        color=sample.color.copy(),
                        depth=sample.depth.copy(),
                        stamp=sample.stamp,
                        source_frame=sample.source_frame,
                    )
            finally:
                self._stop_image_subscriptions()

    def get_color_info(self) -> dict | None:
        with self._lock:
            return self._color_info.copy() if self._color_info else None

    def health_status(self, fresh_within: float = 2.0) -> dict:
        with self._lock:
            color_shape = (
                list(self._color_sample[3].shape[:2]) if self._color_sample else None
            )
            depth_shape = (
                list(self._depth_sample[3].shape[:2]) if self._depth_sample else None
            )
            age = (
                time.monotonic() - self._last_pair_monotonic
                if self._last_pair_monotonic is not None else None
            )
            return {
                "color_received": self._color_sample is not None,
                "depth_received": self._depth_sample is not None,
                "camera_info_received": self._color_info is not None,
                "color_shape": color_shape,
                "depth_shape": depth_shape,
                "registered_shapes_match": (
                    color_shape is not None and color_shape == depth_shape
                ),
                "pair_received": self._paired_images is not None,
                "pair_skew_ms": self._last_pair_skew_ms,
                "max_pair_skew_ms": self._max_pair_skew_ns / 1_000_000.0,
                "pair_age_s": age,
                "pair_fresh": age is not None and age <= fresh_within,
                "source_frame": (
                    self._paired_images.source_frame if self._paired_images else None
                ),
                "last_rejection": self._last_rejection,
            }

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
