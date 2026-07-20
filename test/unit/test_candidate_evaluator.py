from dataclasses import dataclass
import math

import pytest

from mcp_server.grasping.evaluator import (
    CandidateEvaluator,
    CheapCheckResult,
    FullPlanResult,
)
from mcp_server.visualization import CandidateMarkerStatus


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    pose_quat_xyzw: tuple[float, float, float, float]
    pose_xyz: tuple[float, float, float] = (-0.35, 0.0, 0.05)
    preference_rank: int = 0
    ranking_mode: str = "rotation_first"


def _z_rotation(angle_rad):
    return (0.0, 0.0, math.sin(angle_rad / 2.0), math.cos(angle_rad / 2.0))


def _joints(*, joint7_deg=0.0, joint1=0.0):
    values = {f"joint{index}": 0.0 for index in range(1, 8)}
    values["joint1"] = joint1
    values["joint7"] = math.radians(joint7_deg)
    return values


def test_all_ten_get_ik_checks_but_only_top_two_get_full_plans():
    candidates = [
        Candidate(f"c{index}", _z_rotation(0.05 * index))
        for index in range(10)
    ]
    cheap_calls = []
    full_calls = []
    marker_updates = []

    def cheap(candidate, timeout):
        assert timeout > 0.0
        cheap_calls.append(candidate.candidate_id)
        return CheapCheckResult(True, _joints(joint1=0.1))

    def full(candidate, cheap_result, timeout):
        assert cheap_result.feasible
        assert timeout > 0.0
        full_calls.append(candidate.candidate_id)
        return FullPlanResult(True, plan=f"plan-{candidate.candidate_id}")

    result = CandidateEvaluator(
        cheap,
        full,
        marker_sink=lambda candidate, status, reason: marker_updates.append(
            (candidate.candidate_id, status, reason)
        ),
    ).evaluate(
        candidates,
        current_tcp_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        current_joint_positions_rad=_joints(),
    )

    assert cheap_calls == [f"c{index}" for index in range(10)]
    assert full_calls == ["c0", "c1"]
    assert result.cheap_checks == 10
    assert result.full_plans == 2
    assert result.selected.candidate_id == "c0"
    assert result.selected.status is CandidateMarkerStatus.SELECTED
    assert sum(
        status is CandidateMarkerStatus.UNCHECKED
        for _candidate_id, status, _reason in marker_updates
    ) == 10


def test_joint7_preferred_margin_does_not_reject_inside_software_limit():
    candidate = Candidate("inside_limit", _z_rotation(0.0))
    full_calls = []
    result = CandidateEvaluator(
        lambda _candidate, _timeout: CheapCheckResult(
            True, _joints(joint7_deg=60.01)
        ),
        lambda candidate, *_args: (
            full_calls.append(candidate.candidate_id)
            or FullPlanResult(True, plan="complete")
        ),
    ).evaluate(
        [candidate],
        current_tcp_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        current_joint_positions_rad=_joints(),
    )

    evaluation = result.evaluations[0]
    assert evaluation.status is CandidateMarkerStatus.SELECTED
    assert evaluation.joint7_margin_deg == pytest.approx(14.99)
    assert evaluation.reason == "selected best complete plan"
    assert full_calls == ["inside_limit"]
    assert result.selected is evaluation


def test_joint7_rejects_only_when_software_limit_is_exceeded():
    candidate = Candidate("outside_limit", _z_rotation(0.0))
    result = CandidateEvaluator(
        lambda _candidate, _timeout: CheapCheckResult(
            True, _joints(joint7_deg=75.01)
        ),
        lambda *_args: pytest.fail("unsafe candidate must not be fully planned"),
    ).evaluate(
        [candidate],
        current_tcp_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        current_joint_positions_rad=_joints(),
    )

    evaluation = result.evaluations[0]
    assert evaluation.status is CandidateMarkerStatus.REJECTED
    assert "75.0deg exceeds software limit +/-75.0deg" in evaluation.reason
    assert result.full_plans == 0
    assert result.selected is None


def test_ranking_uses_orientation_then_joint7_margin_then_joint_motion():
    candidates = [
        Candidate("more_motion", _z_rotation(0.1)),
        Candidate("less_motion", _z_rotation(0.1)),
        Candidate("larger_margin", _z_rotation(0.1)),
        Candidate("smaller_rotation", _z_rotation(0.05)),
    ]

    solutions = {
        "more_motion": _joints(joint7_deg=40.0, joint1=1.0),
        "less_motion": _joints(joint7_deg=40.0, joint1=0.2),
        "larger_margin": _joints(joint7_deg=20.0, joint1=2.0),
        "smaller_rotation": _joints(joint7_deg=55.0, joint1=2.0),
    }
    full_calls = []

    evaluator = CandidateEvaluator(
        lambda candidate, _timeout: CheapCheckResult(
            True, solutions[candidate.candidate_id]
        ),
        lambda candidate, _cheap, _timeout: (
            full_calls.append(candidate.candidate_id)
            or FullPlanResult(True, plan=candidate.candidate_id)
        ),
    )
    result = evaluator.evaluate(
        candidates,
        current_tcp_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        current_joint_positions_rad=_joints(),
    )

    # Orientation has absolute priority, even with a smaller j7 margin.
    assert full_calls[0] == "smaller_rotation"
    # Among equal orientations, larger j7 margin beats total joint motion.
    assert full_calls[1] == "larger_margin"
    assert result.selected.candidate_id == "smaller_rotation"


def test_second_full_plan_is_selected_when_first_complete_path_fails():
    candidates = [
        Candidate("first", _z_rotation(0.0)),
        Candidate("second", _z_rotation(0.1)),
    ]

    def full(candidate, _cheap, _timeout):
        if candidate.candidate_id == "first":
            return FullPlanResult(False, reason="lift plan failed")
        return FullPlanResult(True, plan="complete")

    result = CandidateEvaluator(
        lambda _candidate, _timeout: CheapCheckResult(True, _joints()),
        full,
    ).evaluate(
        candidates,
        current_tcp_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        current_joint_positions_rad=_joints(),
    )

    assert result.selected.candidate_id == "second"
    assert result.evaluations[0].status is CandidateMarkerStatus.REJECTED
    assert result.evaluations[0].reason == "lift plan failed"


def test_shape_first_prefers_horizontal_family_over_smaller_tcp_rotation():
    candidates = [
        Candidate(
            "top_small_rotation",
            _z_rotation(0.01),
            preference_rank=2,
            ranking_mode="shape_first",
        ),
        Candidate(
            "horizontal_side",
            _z_rotation(1.2),
            preference_rank=0,
            ranking_mode="shape_first",
        ),
    ]
    full_calls = []
    result = CandidateEvaluator(
        lambda candidate, _timeout: CheapCheckResult(
            True,
            _joints(
                joint7_deg=20.0
                if candidate.candidate_id == "horizontal_side" else 0.0
            ),
        ),
        lambda candidate, _cheap, _timeout: (
            full_calls.append(candidate.candidate_id)
            or FullPlanResult(True, plan=candidate.candidate_id)
        ),
    ).evaluate(
        candidates,
        current_tcp_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        current_joint_positions_rad=_joints(),
    )
    assert full_calls[0] == "horizontal_side"
    assert result.selected.candidate_id == "horizontal_side"


def test_deadline_stops_checks_and_never_selects_late_success():
    now = [0.0]
    calls = []
    candidates = [
        Candidate(f"c{index}", _z_rotation(0.0))
        for index in range(10)
    ]

    def clock():
        return now[0]

    def cheap(candidate, _timeout):
        calls.append(candidate.candidate_id)
        now[0] += 3.0
        return CheapCheckResult(True, _joints())

    result = CandidateEvaluator(
        cheap,
        lambda *_args: pytest.fail("deadline must prevent full planning"),
        deadline_s=8.0,
        clock=clock,
    ).evaluate(
        candidates,
        current_tcp_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        current_joint_positions_rad=_joints(),
    )

    assert calls == ["c0", "c1", "c2"]
    assert result.timed_out is True
    assert result.selected is None
    assert all(
        evaluation.status is CandidateMarkerStatus.REJECTED
        for evaluation in result.evaluations
    )
