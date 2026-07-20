"""Domain model for a complete, frame-explicit grasp candidate."""

from __future__ import annotations

from dataclasses import dataclass
import math


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


def _is_finite(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) for value in values)


def _vector_norm(vector: Vector3) -> float:
    return math.sqrt(sum(component * component for component in vector))


@dataclass(frozen=True, slots=True)
class GraspCandidate:
    """A TCP grasp pose plus the linear approach and retreat semantics.

    All vectors and poses are expressed in the same planning frame.  The TCP
    convention is:

    * local ``+Z`` points along ``approach_vector`` (pre-grasp to grasp);
    * local ``+Y`` points along ``edge_axis`` (parallel-jaw closing axis).

    ``gripper_width`` is the expected object span along the closing axis.  It is
    deliberately not an actuator command; execution code may add clearance for
    the pre-grasp opening.
    """

    pose_xyz: Vector3
    pose_quat_xyzw: Quaternion
    approach_vector: Vector3
    pregrasp_distance: float
    retreat_vector: Vector3
    retreat_distance: float
    gripper_width: float
    tilt_deg: float
    edge_axis: Vector3
    score: float
    source: str
    preference_rank: int = 0
    ranking_mode: str = "rotation_first"
    object_kind: str = "block"

    def __post_init__(self) -> None:
        if len(self.pose_xyz) != 3 or not _is_finite(self.pose_xyz):
            raise ValueError("pose_xyz must contain three finite values")
        if len(self.pose_quat_xyzw) != 4 or not _is_finite(self.pose_quat_xyzw):
            raise ValueError("pose_quat_xyzw must contain four finite values")
        if (
            len(self.approach_vector) != 3
            or not _is_finite(self.approach_vector)
        ):
            raise ValueError("approach_vector must contain three finite values")
        if (
            len(self.retreat_vector) != 3
            or not _is_finite(self.retreat_vector)
        ):
            raise ValueError("retreat_vector must contain three finite values")
        if len(self.edge_axis) != 3 or not _is_finite(self.edge_axis):
            raise ValueError("edge_axis must contain three finite values")
        if abs(_vector_norm(self.approach_vector) - 1.0) > 1e-6:
            raise ValueError("approach_vector must be a unit vector")
        if abs(_vector_norm(self.retreat_vector) - 1.0) > 1e-6:
            raise ValueError("retreat_vector must be a unit vector")
        if abs(_vector_norm(self.edge_axis) - 1.0) > 1e-6:
            raise ValueError("edge_axis must be a unit vector")

        quaternion_norm = math.sqrt(
            sum(component * component for component in self.pose_quat_xyzw)
        )
        if abs(quaternion_norm - 1.0) > 1e-6:
            raise ValueError("pose_quat_xyzw must be normalized")
        if self.pregrasp_distance <= 0.0 or not math.isfinite(
            self.pregrasp_distance
        ):
            raise ValueError("pregrasp_distance must be finite and positive")
        if self.retreat_distance <= 0.0 or not math.isfinite(
            self.retreat_distance
        ):
            raise ValueError("retreat_distance must be finite and positive")
        if self.gripper_width <= 0.0 or not math.isfinite(self.gripper_width):
            raise ValueError("gripper_width must be finite and positive")
        if self.tilt_deg < 0.0 or not math.isfinite(self.tilt_deg):
            raise ValueError("tilt_deg must be finite and non-negative")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if not self.source:
            raise ValueError("source must not be empty")
        if (
            not isinstance(self.preference_rank, int)
            or isinstance(self.preference_rank, bool)
            or self.preference_rank < 0
        ):
            raise ValueError("preference_rank must be a non-negative integer")
        if self.ranking_mode not in {"rotation_first", "shape_first"}:
            raise ValueError(
                "ranking_mode must be 'rotation_first' or 'shape_first'"
            )
        if not self.object_kind:
            raise ValueError("object_kind must not be empty")

    @property
    def pregrasp_xyz(self) -> Vector3:
        """Position before moving linearly along the approach vector."""

        return tuple(
            position - direction * self.pregrasp_distance
            for position, direction in zip(self.pose_xyz, self.approach_vector)
        )

    @property
    def candidate_id(self) -> str:
        """Stable identifier used by evaluators and RViz markers."""

        return self.source

    @property
    def name(self) -> str:
        """Human-readable alias for ``candidate_id``."""

        return self.source

    @property
    def position_xyz(self) -> Vector3:
        """Compatibility alias for consumers that call a pose a position."""

        return self.pose_xyz

    @property
    def quat_xyzw(self) -> Quaternion:
        """Compatibility alias for the TCP pose quaternion."""

        return self.pose_quat_xyzw

    @property
    def retreat_xyz(self) -> Vector3:
        """Position reached after moving along the retreat vector."""

        return tuple(
            position + direction * self.retreat_distance
            for position, direction in zip(self.pose_xyz, self.retreat_vector)
        )

    def to_dict(self) -> dict:
        """Return a JSON-compatible representation for diagnostics."""

        return {
            "candidate_id": self.candidate_id,
            "pose_xyz": list(self.pose_xyz),
            "pose_quat_xyzw": list(self.pose_quat_xyzw),
            "approach_vector": list(self.approach_vector),
            "pregrasp_distance": self.pregrasp_distance,
            "pregrasp_xyz": list(self.pregrasp_xyz),
            "retreat_vector": list(self.retreat_vector),
            "retreat_distance": self.retreat_distance,
            "retreat_xyz": list(self.retreat_xyz),
            "gripper_width": self.gripper_width,
            "tilt_deg": self.tilt_deg,
            "edge_axis": list(self.edge_axis),
            "score": self.score,
            "source": self.source,
            "preference_rank": self.preference_rank,
            "ranking_mode": self.ranking_mode,
            "object_kind": self.object_kind,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "GraspCandidate":
        """Rebuild a candidate after it has passed through task JSON state."""
        return cls(
            pose_xyz=tuple(float(v) for v in value["pose_xyz"]),
            pose_quat_xyzw=tuple(float(v) for v in value["pose_quat_xyzw"]),
            approach_vector=tuple(float(v) for v in value["approach_vector"]),
            pregrasp_distance=float(value["pregrasp_distance"]),
            retreat_vector=tuple(float(v) for v in value["retreat_vector"]),
            retreat_distance=float(value["retreat_distance"]),
            gripper_width=float(value["gripper_width"]),
            tilt_deg=float(value["tilt_deg"]),
            edge_axis=tuple(float(v) for v in value["edge_axis"]),
            score=float(value["score"]),
            source=str(value["source"]),
            preference_rank=int(value.get("preference_rank", 0)),
            ranking_mode=str(value.get("ranking_mode", "rotation_first")),
            object_kind=str(value.get("object_kind", "block")),
        )
