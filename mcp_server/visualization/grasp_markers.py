"""Color-coded grasp-candidate markers.

The marker specification is deliberately ROS-independent so candidate
evaluation and color mapping remain unit-testable without a running ROS graph.
The publisher adapter imports ``visualization_msgs`` only when instantiated.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence
import zlib


RGBA = tuple[float, float, float, float]


class CandidateMarkerStatus(str, Enum):
    UNCHECKED = "unchecked"
    FEASIBLE = "feasible"
    REJECTED = "rejected"
    SELECTED = "selected"


_STATUS_COLORS: dict[CandidateMarkerStatus, RGBA] = {
    CandidateMarkerStatus.UNCHECKED: (0.55, 0.55, 0.55, 0.55),
    CandidateMarkerStatus.FEASIBLE: (0.10, 0.85, 0.20, 0.90),
    CandidateMarkerStatus.REJECTED: (0.95, 0.10, 0.10, 0.90),
    CandidateMarkerStatus.SELECTED: (1.00, 0.82, 0.05, 1.00),
}


def marker_color(status: CandidateMarkerStatus | str) -> RGBA:
    """Map unchecked/feasible/rejected/selected to gray/green/red/yellow."""
    return _STATUS_COLORS[CandidateMarkerStatus(status)]


def _candidate_value(candidate: Any, *names: str):
    for name in names:
        if hasattr(candidate, name):
            return getattr(candidate, name)
    raise AttributeError(
        f"grasp candidate is missing all expected fields: {', '.join(names)}"
    )


@dataclass(frozen=True, slots=True)
class CandidateMarkerSpec:
    marker_id: int
    candidate_id: str
    frame_id: str
    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    status: CandidateMarkerStatus
    color_rgba: RGBA
    reason: str = ""
    namespace: str = "grasp_candidates"
    scale_xyz: tuple[float, float, float] = (0.065, 0.012, 0.012)


def marker_spec_from_candidate(
    candidate: Any,
    status: CandidateMarkerStatus | str,
    *,
    frame_id: str = "base_link",
    reason: str = "",
) -> CandidateMarkerSpec:
    candidate_id = str(_candidate_value(candidate, "candidate_id", "name", "source"))
    position = tuple(
        float(value)
        for value in _candidate_value(candidate, "pose_xyz", "position_xyz")
    )
    quaternion = tuple(
        float(value)
        for value in _candidate_value(
            candidate, "pose_quat_xyzw", "quat_xyzw", "quaternion_xyzw"
        )
    )
    if len(position) != 3:
        raise ValueError("candidate marker position must have three values")
    if len(quaternion) != 4:
        raise ValueError("candidate marker quaternion must have four values")
    marker_status = CandidateMarkerStatus(status)
    # CRC32 is deterministic across processes, unlike Python's hash().
    marker_id = zlib.crc32(candidate_id.encode("utf-8")) & 0x7FFFFFFF
    return CandidateMarkerSpec(
        marker_id=marker_id,
        candidate_id=candidate_id,
        frame_id=frame_id,
        position_xyz=position,
        quaternion_xyzw=quaternion,
        status=marker_status,
        color_rgba=marker_color(marker_status),
        reason=reason,
    )


def _to_ros_marker(spec: CandidateMarkerSpec, *, stamp=None):
    from visualization_msgs.msg import Marker

    marker = Marker()
    marker.header.frame_id = spec.frame_id
    if stamp is not None:
        marker.header.stamp = stamp
    marker.ns = spec.namespace
    marker.id = spec.marker_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
        spec.position_xyz
    )
    (
        marker.pose.orientation.x,
        marker.pose.orientation.y,
        marker.pose.orientation.z,
        marker.pose.orientation.w,
    ) = spec.quaternion_xyzw
    marker.scale.x, marker.scale.y, marker.scale.z = spec.scale_xyz
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = spec.color_rgba
    marker.text = spec.reason
    return marker


class GraspCandidateMarkerPublisher:
    """Small ROS adapter; publishing changes visualization only, never motion."""

    def __init__(
        self,
        node,
        *,
        topic: str = "/grasp_candidates",
        frame_id: str = "base_link",
        qos_depth: int = 10,
    ):
        from visualization_msgs.msg import MarkerArray

        self._node = node
        self._frame_id = frame_id
        self._marker_array_type = MarkerArray
        self._publisher = node.create_publisher(MarkerArray, topic, qos_depth)

    def publish(
        self,
        candidate: Any,
        status: CandidateMarkerStatus | str,
        reason: str = "",
    ) -> CandidateMarkerSpec:
        spec = marker_spec_from_candidate(
            candidate, status, frame_id=self._frame_id, reason=reason
        )
        message = self._marker_array_type()
        message.markers.append(
            _to_ros_marker(spec, stamp=self._node.get_clock().now().to_msg())
        )
        self._publisher.publish(message)
        return spec

    def publish_many(
        self,
        updates: Sequence[tuple[Any, CandidateMarkerStatus | str, str]],
    ) -> tuple[CandidateMarkerSpec, ...]:
        message = self._marker_array_type()
        stamp = self._node.get_clock().now().to_msg()
        specs = tuple(
            marker_spec_from_candidate(
                candidate, status, frame_id=self._frame_id, reason=reason
            )
            for candidate, status, reason in updates
        )
        message.markers.extend(
            _to_ros_marker(spec, stamp=stamp) for spec in specs
        )
        self._publisher.publish(message)
        return specs

    def clear(self) -> None:
        from visualization_msgs.msg import Marker

        message = self._marker_array_type()
        marker = Marker()
        marker.action = Marker.DELETEALL
        message.markers.append(marker)
        self._publisher.publish(message)
