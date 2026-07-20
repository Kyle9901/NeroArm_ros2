from dataclasses import dataclass

from mcp_server.visualization import (
    CandidateMarkerStatus,
    marker_color,
    marker_spec_from_candidate,
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str = "block_tilt_30_from_pos_x"
    pose_xyz: tuple[float, float, float] = (-0.35, 0.02, 0.05)
    pose_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


def test_status_colors_match_rviz_contract():
    assert marker_color(CandidateMarkerStatus.UNCHECKED)[:3] == (
        0.55, 0.55, 0.55
    )
    assert marker_color(CandidateMarkerStatus.FEASIBLE)[:3] == (
        0.10, 0.85, 0.20
    )
    assert marker_color(CandidateMarkerStatus.REJECTED)[:3] == (
        0.95, 0.10, 0.10
    )
    assert marker_color(CandidateMarkerStatus.SELECTED)[:3] == (
        1.00, 0.82, 0.05
    )


def test_marker_spec_preserves_pose_status_and_reason():
    candidate = Candidate()
    spec = marker_spec_from_candidate(
        candidate,
        CandidateMarkerStatus.SELECTED,
        frame_id="base_link",
        reason="selected best complete plan",
    )

    assert spec.candidate_id == candidate.candidate_id
    assert spec.position_xyz == candidate.pose_xyz
    assert spec.quaternion_xyzw == candidate.pose_quat_xyzw
    assert spec.status is CandidateMarkerStatus.SELECTED
    assert spec.color_rgba == marker_color(CandidateMarkerStatus.SELECTED)
    assert spec.reason == "selected best complete plan"


def test_marker_id_is_stable_for_candidate_name():
    first = marker_spec_from_candidate(
        Candidate(), CandidateMarkerStatus.UNCHECKED
    )
    second = marker_spec_from_candidate(
        Candidate(), CandidateMarkerStatus.REJECTED
    )
    other = marker_spec_from_candidate(
        Candidate(candidate_id="other"), CandidateMarkerStatus.UNCHECKED
    )

    assert first.marker_id == second.marker_id
    assert first.marker_id != other.marker_id
