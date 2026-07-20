"""Deadline-bounded, non-executing grasp candidate evaluation.

The evaluator knows nothing about ROS actions. Callers inject two checkers:

* ``cheap_check`` must perform a real IK/collision check and return its joint
  solution; lack of an IK service is a rejection, not an assumed success.
* ``full_plan`` must plan the complete candidate path without execution.

This separation keeps selection deterministic and makes it impossible for the
evaluator itself to send a robot motion goal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping, Sequence

from mcp_server.grasping.block_candidates import quaternion_angular_distance
from mcp_server.visualization import CandidateMarkerStatus


ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))


@dataclass(frozen=True, slots=True)
class CheapCheckResult:
    feasible: bool
    joint_positions_rad: Mapping[str, float] | Sequence[float] | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class FullPlanResult:
    feasible: bool
    plan: Any = None
    total_joint_motion_rad: float | None = None
    reason: str = ""


@dataclass(slots=True)
class CandidateEvaluation:
    candidate: Any
    status: CandidateMarkerStatus = CandidateMarkerStatus.UNCHECKED
    reason: str = ""
    orientation_distance_rad: float = math.inf
    joint7_rad: float | None = None
    joint7_margin_deg: float = -math.inf
    total_joint_motion_rad: float = math.inf
    cheap_check: CheapCheckResult | None = None
    full_plan: FullPlanResult | None = None

    @property
    def candidate_id(self) -> str:
        for name in ("candidate_id", "name", "source"):
            if hasattr(self.candidate, name):
                return str(getattr(self.candidate, name))
        return repr(self.candidate)

    @property
    def ranking_key(self) -> tuple[float, float, float, float, str]:
        preference = float(getattr(self.candidate, "preference_rank", 0))
        mode = str(getattr(
            self.candidate, "ranking_mode", "rotation_first"
        ))
        if mode == "shape_first":
            # Used by cylinders: physically appropriate grasp families (for
            # example a horizontal side grasp on a standing bottle) outrank a
            # smaller wrist rotation. Within one family, preserve joint7
            # margin and total arm motion before using rotation as a tie-break.
            return (
                preference,
                -self.joint7_margin_deg,
                self.total_joint_motion_rad,
                self.orientation_distance_rad,
                self.candidate_id,
            )
        return (
            preference,
            self.orientation_distance_rad,
            -self.joint7_margin_deg,
            self.total_joint_motion_rad,
            self.candidate_id,
        )


@dataclass(frozen=True, slots=True)
class EvaluationBatch:
    evaluations: tuple[CandidateEvaluation, ...]
    selected: CandidateEvaluation | None
    elapsed_s: float
    timed_out: bool
    cheap_checks: int
    full_plans: int


CheapChecker = Callable[[Any, float], CheapCheckResult]
FullPlanner = Callable[[Any, CheapCheckResult, float], FullPlanResult]
MarkerSink = Callable[[Any, CandidateMarkerStatus, str], None]


def _joint_map(
    values: Mapping[str, float] | Sequence[float] | None,
) -> dict[str, float] | None:
    if values is None:
        return None
    if isinstance(values, Mapping):
        result = {
            name: float(values[name])
            for name in ARM_JOINT_NAMES
            if name in values
        }
    else:
        sequence = tuple(float(value) for value in values)
        result = dict(zip(ARM_JOINT_NAMES, sequence))
    if len(result) != len(ARM_JOINT_NAMES):
        return None
    if not all(math.isfinite(value) for value in result.values()):
        return None
    return result


def _candidate_quaternion(candidate: Any) -> tuple[float, float, float, float]:
    for name in ("pose_quat_xyzw", "quat_xyzw", "quaternion_xyzw"):
        if hasattr(candidate, name):
            values = tuple(float(value) for value in getattr(candidate, name))
            if len(values) == 4:
                return values
    raise ValueError("candidate has no xyzw quaternion")


def _joint_motion(
    current: Mapping[str, float],
    target: Mapping[str, float],
) -> float:
    return sum(
        abs(float(target[name]) - float(current[name]))
        for name in ARM_JOINT_NAMES
        if name in current and name in target
    )


class CandidateEvaluator:
    """Evaluate every cheap check, then fully plan only the best candidates."""

    def __init__(
        self,
        cheap_check: CheapChecker,
        full_plan: FullPlanner,
        *,
        marker_sink: MarkerSink | None = None,
        deadline_s: float = 8.0,
        full_plan_count: int = 2,
        joint7_limit_deg: float = 75.0,
        joint7_min_margin_deg: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if deadline_s <= 0.0:
            raise ValueError("deadline_s must be positive")
        if full_plan_count <= 0:
            raise ValueError("full_plan_count must be positive")
        if joint7_limit_deg <= 0.0:
            raise ValueError("joint7_limit_deg must be positive")
        if not 0.0 <= joint7_min_margin_deg < joint7_limit_deg:
            raise ValueError("joint7_min_margin_deg must be inside the software limit")
        self._cheap_check = cheap_check
        self._full_plan = full_plan
        self._marker_sink = marker_sink
        self._deadline_s = float(deadline_s)
        self._full_plan_count = int(full_plan_count)
        self._joint7_limit_deg = float(joint7_limit_deg)
        self._joint7_min_margin_deg = float(joint7_min_margin_deg)
        self._clock = clock

    def _publish(
        self,
        evaluation: CandidateEvaluation,
        status: CandidateMarkerStatus,
        reason: str = "",
    ) -> None:
        evaluation.status = status
        evaluation.reason = reason
        if self._marker_sink is not None:
            try:
                self._marker_sink(evaluation.candidate, status, reason)
            except Exception:
                # RViz diagnostics must never change planning decisions.
                pass

    def evaluate(
        self,
        candidates: Sequence[Any],
        *,
        current_tcp_quat_xyzw: Sequence[float],
        current_joint_positions_rad: Mapping[str, float] | Sequence[float],
    ) -> EvaluationBatch:
        started = self._clock()
        deadline = started + self._deadline_s
        current_joints = _joint_map(current_joint_positions_rad)
        if current_joints is None:
            raise ValueError("current_joint_positions_rad must contain joint1..joint7")
        evaluations = [CandidateEvaluation(candidate) for candidate in candidates]
        for evaluation in evaluations:
            self._publish(evaluation, CandidateMarkerStatus.UNCHECKED)

        cheap_count = 0
        timed_out = False
        for index, evaluation in enumerate(evaluations):
            remaining = deadline - self._clock()
            if remaining <= 0.0:
                timed_out = True
                for pending in evaluations[index:]:
                    self._publish(
                        pending,
                        CandidateMarkerStatus.REJECTED,
                        "evaluation deadline exceeded before IK check",
                    )
                break
            try:
                cheap = self._cheap_check(evaluation.candidate, remaining)
            except Exception as error:
                cheap = CheapCheckResult(False, reason=f"IK check error: {error}")
            cheap_count += 1
            evaluation.cheap_check = cheap
            if self._clock() > deadline:
                timed_out = True
                self._publish(
                    evaluation,
                    CandidateMarkerStatus.REJECTED,
                    "evaluation deadline exceeded during IK check",
                )
                for pending in evaluations[index + 1:]:
                    self._publish(
                        pending,
                        CandidateMarkerStatus.REJECTED,
                        "evaluation deadline exceeded before IK check",
                    )
                break
            if not cheap.feasible:
                self._publish(
                    evaluation,
                    CandidateMarkerStatus.REJECTED,
                    cheap.reason or "IK/collision check rejected",
                )
                continue
            joints = _joint_map(cheap.joint_positions_rad)
            if joints is None:
                self._publish(
                    evaluation,
                    CandidateMarkerStatus.REJECTED,
                    "IK checker returned no complete seven-joint solution",
                )
                continue
            joint7 = joints["joint7"]
            joint7_deg = math.degrees(joint7)
            margin_deg = self._joint7_limit_deg - abs(joint7_deg)
            # ``joint7_limit_deg`` is the actual software safety boundary.
            # The configured minimum margin is a ranking preference only:
            # candidates inside the boundary remain usable, while candidates
            # with more clearance are still preferred by ``ranking_key``.
            if margin_deg < -1e-9:
                self._publish(
                    evaluation,
                    CandidateMarkerStatus.REJECTED,
                    (
                        f"joint7 {joint7_deg:.1f}deg exceeds software limit "
                        f"+/-{self._joint7_limit_deg:.1f}deg"
                    ),
                )
                continue
            try:
                orientation_distance = quaternion_angular_distance(
                    current_tcp_quat_xyzw,
                    _candidate_quaternion(evaluation.candidate),
                )
            except (TypeError, ValueError) as error:
                self._publish(
                    evaluation,
                    CandidateMarkerStatus.REJECTED,
                    f"invalid candidate orientation: {error}",
                )
                continue
            evaluation.orientation_distance_rad = orientation_distance
            evaluation.joint7_rad = joint7
            evaluation.joint7_margin_deg = margin_deg
            evaluation.total_joint_motion_rad = _joint_motion(
                current_joints, joints
            )
            preference_note = ""
            if margin_deg < self._joint7_min_margin_deg:
                preference_note = (
                    f"inside joint7 software limit with {margin_deg:.1f}deg "
                    f"margin; preferred margin is "
                    f"{self._joint7_min_margin_deg:.1f}deg"
                )
            self._publish(
                evaluation,
                CandidateMarkerStatus.FEASIBLE,
                preference_note,
            )

        ranked = sorted(
            (
                evaluation
                for evaluation in evaluations
                if evaluation.status is CandidateMarkerStatus.FEASIBLE
            ),
            key=lambda evaluation: evaluation.ranking_key,
        )
        full_count = 0
        fully_feasible: list[CandidateEvaluation] = []
        for evaluation in ranked[: self._full_plan_count]:
            remaining = deadline - self._clock()
            if remaining <= 0.0:
                timed_out = True
                self._publish(
                    evaluation,
                    CandidateMarkerStatus.REJECTED,
                    "evaluation deadline exceeded before full plan",
                )
                continue
            try:
                full = self._full_plan(
                    evaluation.candidate,
                    evaluation.cheap_check,
                    remaining,
                )
            except Exception as error:
                full = FullPlanResult(False, reason=f"full plan error: {error}")
            full_count += 1
            evaluation.full_plan = full
            if self._clock() > deadline:
                timed_out = True
                self._publish(
                    evaluation,
                    CandidateMarkerStatus.REJECTED,
                    "evaluation deadline exceeded during full plan",
                )
                continue
            if not full.feasible:
                self._publish(
                    evaluation,
                    CandidateMarkerStatus.REJECTED,
                    full.reason or "complete plan rejected",
                )
                continue
            if full.total_joint_motion_rad is not None:
                if not math.isfinite(full.total_joint_motion_rad):
                    self._publish(
                        evaluation,
                        CandidateMarkerStatus.REJECTED,
                        "full plan returned non-finite joint motion",
                    )
                    continue
                evaluation.total_joint_motion_rad = float(
                    full.total_joint_motion_rad
                )
            fully_feasible.append(evaluation)

        selected = (
            min(fully_feasible, key=lambda evaluation: evaluation.ranking_key)
            if fully_feasible
            else None
        )
        if selected is not None:
            self._publish(
                selected,
                CandidateMarkerStatus.SELECTED,
                "selected best complete plan",
            )
        elapsed = self._clock() - started
        return EvaluationBatch(
            evaluations=tuple(evaluations),
            selected=selected,
            elapsed_s=elapsed,
            timed_out=timed_out,
            cheap_checks=cheap_count,
            full_plans=full_count,
        )
