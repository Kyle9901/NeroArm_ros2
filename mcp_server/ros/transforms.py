"""TF lookup and point transformation services."""

from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener


class TransformService:
    def __init__(self, node):
        self._node = node
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node)

    def can_transform(
        self,
        source_frame: str,
        target_frame: str = "base_link",
        timeout: float = 1.0,
    ) -> bool:
        return self.buffer.can_transform(
            target_frame,
            source_frame,
            Time(),
            Duration(seconds=timeout),
        )

    def transform_point(
        self,
        x: float,
        y: float,
        z: float,
        *,
        source_frame: str = "camera_color_optical_frame",
        target_frame: str = "base_link",
        stamp=None,
        timeout: float = 1.0,
    ) -> dict:
        point = PointStamped()
        point.header.frame_id = source_frame
        point.header.stamp = stamp or self._node.get_clock().now().to_msg()
        point.point.x = x
        point.point.y = y
        point.point.z = z

        query_time = Time.from_msg(stamp) if stamp is not None else Time()
        transform = self.buffer.lookup_transform(
            target_frame,
            source_frame,
            query_time,
            Duration(seconds=timeout),
        )
        transformed = do_transform_point(point, transform)
        return {
            "x": transformed.point.x,
            "y": transformed.point.y,
            "z": transformed.point.z,
        }

    def lookup_pose(
        self,
        source_frame: str,
        target_frame: str = "base_link",
        timeout: float = 1.0,
    ) -> dict:
        """Return the latest source-frame pose expressed in target_frame."""
        transform = self.buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
            Duration(seconds=timeout),
        ).transform
        return {
            "position": [
                float(transform.translation.x),
                float(transform.translation.y),
                float(transform.translation.z),
            ],
            "quaternion": [
                float(transform.rotation.x),
                float(transform.rotation.y),
                float(transform.rotation.z),
                float(transform.rotation.w),
            ],
            "source_frame": source_frame,
            "target_frame": target_frame,
        }
