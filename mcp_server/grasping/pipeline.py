"""Connect shape geometry, non-executing MoveIt checks, and planned paths.

This module is the safety boundary between candidate generation and motion
execution. Block and cylinder entry points perform only IK and plan-only calls.
They never open the gripper or execute a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Iterable

from .block_candidates import generate_block_grasp_candidates
from .bottle_candidates import (
    generate_transparent_bottle_grasp_candidates,
)
from .cylinder_candidates import generate_cylinder_grasp_candidates
from .evaluator import (
    CandidateEvaluator,
    CheapCheckResult,
    EvaluationBatch,
    FullPlanResult,
)
from ..components import motion
from ..models import GraspCandidate, ObjectGeometry


@dataclass(frozen=True, slots=True)
class PlannedGraspPath:
    """Four arm-only trajectories covering the complete grasp geometry."""

    candidate: GraspCandidate
    to_pregrasp: Any
    approach: Any
    retreat: Any
    to_carry: Any
    total_joint_motion_rad: float

    @property
    def stages(self) -> tuple[Any, Any, Any, Any]:
        return (
            self.to_pregrasp,
            self.approach,
            self.retreat,
            self.to_carry,
        )


@dataclass(frozen=True, slots=True)
class BlockGraspPlanning:
    batch: EvaluationBatch | None
    selected_path: PlannedGraspPath | None
    candidates: tuple[GraspCandidate, ...]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.selected_path is not None and self.error is None


def _trajectory_motion_rad(plan: Any) -> float:
    """Sum absolute arm-joint motion along one MoveIt trajectory."""
    trajectory = getattr(plan, "trajectory", plan)
    joint_trajectory = getattr(trajectory, "joint_trajectory", None)
    names = list(getattr(joint_trajectory, "joint_names", ()))
    points = list(getattr(joint_trajectory, "points", ()))
    arm_indices = [
        index for index, name in enumerate(names)
        if name in {f"joint{joint}" for joint in range(1, 8)}
    ]
    if not points or not arm_indices:
        return 0.0

    start = getattr(plan, "start_state", None)
    start_joint_state = getattr(start, "joint_state", None)
    start_map = dict(zip(
        getattr(start_joint_state, "name", ()),
        getattr(start_joint_state, "position", ()),
    ))
    previous = [
        float(start_map.get(names[index], points[0].positions[index]))
        for index in arm_indices
    ]
    total = 0.0
    for point in points:
        current = [float(point.positions[index]) for index in arm_indices]
        total += sum(abs(after - before) for before, after in zip(previous, current))
        previous = current
    return total


def _minimum_joint7_margin_deg(
    plans: Iterable[Any],
    limit_deg: float,
) -> float:
    """Return clearance from the joint7 software limit over complete paths.

    Every available start state and trajectory waypoint is checked.  A
    non-negative result is inside the configured symmetric software limit;
    a negative result is the number of degrees by which the path exceeds it.
    No monotonic-exit rule is applied to the observation transition because a
    redundant arm may need a small, safe detour while remaining inside the
    actual software boundary.
    """
    max_abs_deg = 0.0
    found = False
    for plan in plans:
        trajectory = getattr(plan, "trajectory", plan)
        joint_trajectory = getattr(trajectory, "joint_trajectory", None)
        names = list(getattr(joint_trajectory, "joint_names", ()))
        if "joint7" not in names:
            continue
        index = names.index("joint7")
        points = list(getattr(joint_trajectory, "points", ()))
        start_state = getattr(plan, "start_state", None)
        start_joint_state = getattr(start_state, "joint_state", None)
        start_map = dict(zip(
            getattr(start_joint_state, "name", ()),
            getattr(start_joint_state, "position", ()),
        ))
        if "joint7" in start_map:
            max_abs_deg = max(
                max_abs_deg,
                abs(math.degrees(float(start_map["joint7"]))),
            )
            found = True
        for point in points:
            max_abs_deg = max(
                max_abs_deg,
                abs(math.degrees(float(point.positions[index]))),
            )
            found = True
    return float(limit_deg - max_abs_deg) if found else -math.inf


def _joint7_path_limit_error(
    path_margin_deg: float,
    limit_deg: float,
    *,
    path_label: str,
) -> str | None:
    """Describe missing joint7 data or a true software-limit violation."""
    if not math.isfinite(path_margin_deg):
        return f"{path_label} has no joint7 trajectory data"
    if path_margin_deg < -1e-9:
        max_abs_deg = float(limit_deg) - float(path_margin_deg)
        return (
            f"{path_label} reaches |joint7|={max_abs_deg:.1f}deg, exceeding "
            f"the +/-{float(limit_deg):.1f}deg software limit"
        )
    return None


def _joint7_stage_margins_deg(
    named_plans: Iterable[tuple[str, Any]],
    limit_deg: float,
) -> dict[str, float]:
    """Return joint7 software-limit clearance for every planned stage."""

    return {
        str(name): _minimum_joint7_margin_deg((plan,), limit_deg)
        for name, plan in named_plans
    }


def _geometry_ready(geometry: ObjectGeometry, object_kind: str) -> str | None:
    if geometry.center_xyz is None:
        return f"{object_kind} geometry has no center"
    if geometry.size_xyz is None:
        return f"{object_kind} geometry has no size"
    if geometry.quality and geometry.quality.get("reliable") is not True:
        return f"{object_kind} geometry quality is not reliable"
    if any(not math.isfinite(value) for value in (*geometry.center_xyz, *geometry.size_xyz)):
        return f"{object_kind} geometry contains non-finite values"
    if any(value <= 0.0 for value in geometry.size_xyz):
        return f"{object_kind} geometry dimensions must be positive"
    if object_kind == "block" and geometry.yaw_rad is None:
        return "block geometry has no planar yaw"
    if object_kind in {"cylinder", "transparent_bottle"}:
        if geometry.shape_kind != "cylinder":
            return f"{object_kind} geometry has the wrong shape kind"
        if geometry.axis_xyz is None:
            return f"{object_kind} geometry has no axis"
        if geometry.diameter_m is None or geometry.length_m is None:
            return f"{object_kind} geometry has no diameter/length"
        if (
            any(not math.isfinite(float(value)) for value in geometry.axis_xyz)
            or not math.isfinite(float(geometry.diameter_m))
            or not math.isfinite(float(geometry.length_m))
            or float(geometry.diameter_m) <= 0.0
            or float(geometry.length_m) <= 0.0
        ):
            return f"{object_kind} geometry contains invalid axis/dimensions"
        if geometry.orientation_class not in {"upright", "lying"}:
            return (
                f"{object_kind} geometry has no stable upright/lying class"
            )
        if (
            geometry.local_desk_z is None
            or not math.isfinite(float(geometry.local_desk_z))
        ):
            return f"{object_kind} geometry has no local desk height"
    return None


def _candidate_inside_bounds(bridge, candidate: GraspCandidate) -> bool:
    return all(
        bridge.node.workspace_check(point[0], point[1])
        for point in (
            candidate.pregrasp_xyz,
            candidate.pose_xyz,
            candidate.retreat_xyz,
        )
    )


def _plan_grasp(
    bridge,
    geometry_value: ObjectGeometry | dict,
    *,
    object_kind: str,
    excluded_candidate_ids: Iterable[str] = (),
) -> BlockGraspPlanning:
    """Generate and evaluate shape-specific candidates without moving."""
    geometry = (
        geometry_value
        if isinstance(geometry_value, ObjectGeometry)
        else ObjectGeometry.from_dict(geometry_value)
    )
    geometry_error = _geometry_ready(geometry, object_kind)
    if geometry_error:
        return BlockGraspPlanning(None, None, (), geometry_error)

    try:
        tcp_pose = bridge.get_current_tcp_pose(timeout=1.0)
    except Exception as error:
        return BlockGraspPlanning(
            None, None, (), f"cannot read current TCP orientation: {error}"
        )
    current_quat = tuple(float(value) for value in tcp_pose["quaternion"])
    joint_state = bridge.get_joint_state().get("joints", {})
    current_joints = {
        f"joint{index}": float(joint_state[f"joint{index}"])
        for index in range(1, 8)
        if f"joint{index}" in joint_state
    }
    if len(current_joints) != 7:
        return BlockGraspPlanning(
            None, None, (), "current seven-joint feedback is unavailable"
        )

    if object_kind == "block":
        generated = generate_block_grasp_candidates(
            geometry.center_xyz,
            geometry.size_xyz,
            geometry.yaw_rad,
            current_quat,
            pregrasp_distance=bridge.get_grasp_pregrasp_distance(),
            retreat_distance=bridge.get_grasp_retreat_distance(),
            tilt_angles_deg=bridge.get_grasp_tilt_angles_deg(),
        )
    elif object_kind == "cylinder":
        generated = generate_cylinder_grasp_candidates(
            geometry.center_xyz,
            geometry.axis_xyz,
            geometry.diameter_m,
            geometry.length_m,
            geometry.orientation_class,
            current_quat,
            local_desk_z=geometry.local_desk_z,
            side_grasp_height_ratio=(
                bridge.get_cylinder_side_grasp_height_ratio()
            ),
            pregrasp_distance=bridge.get_grasp_pregrasp_distance(),
            retreat_distance=bridge.get_grasp_retreat_distance(),
            tilt_angles_deg=bridge.get_cylinder_tilt_angles_deg(),
        )
    elif object_kind == "transparent_bottle":
        generated = generate_transparent_bottle_grasp_candidates(
            geometry.center_xyz,
            geometry.axis_xyz,
            geometry.diameter_m,
            geometry.orientation_class,
            current_quat,
            pregrasp_distance=bridge.get_grasp_pregrasp_distance(),
            retreat_distance=bridge.get_grasp_retreat_distance(),
            upright_axis_overtravel_m=(
                bridge.get_transparent_bottle_upright_axis_overtravel_m()
            ),
        )
    else:
        return BlockGraspPlanning(
            None, None, (), f"unsupported grasp object kind: {object_kind}"
        )
    excluded = set(excluded_candidate_ids)
    candidates = tuple(
        candidate for candidate in generated
        if candidate.candidate_id not in excluded
    )
    if not candidates:
        return BlockGraspPlanning(
            None, None, generated,
            f"all {object_kind} grasp candidates were excluded"
        )

    marker_publisher = getattr(bridge.node, "grasp_candidate_markers", None)
    if marker_publisher is not None:
        try:
            marker_publisher.clear()
        except Exception:
            marker_publisher = None

    def marker_sink(candidate, status, reason):
        if marker_publisher is not None:
            marker_publisher.publish(candidate, status, reason)

    gripper_limit = float(bridge.get_gripper_open_width())
    planning_start_state = bridge.node.robot_state_with_gripper_width(
        gripper_limit,
    )

    def cheap_check(candidate: GraspCandidate, remaining: float) -> CheapCheckResult:
        if candidate.gripper_width >= gripper_limit:
            return CheapCheckResult(
                False,
                reason=(
                    f"required width {candidate.gripper_width:.3f}m exceeds "
                    f"open gripper width {gripper_limit:.3f}m"
                ),
            )
        if not _candidate_inside_bounds(bridge, candidate):
            return CheapCheckResult(False, reason="candidate path leaves workspace")
        if candidate.pose_xyz[2] <= bridge.get_desk_surface_z():
            return CheapCheckResult(False, reason="candidate TCP is at or below desk")
        checked = motion.solve_pose_ik(
            bridge,
            *candidate.pose_xyz,
            list(candidate.pose_quat_xyzw),
            timeout=min(0.25, max(0.05, remaining)),
            seed_state=planning_start_state,
            avoid_collisions=True,
        )
        if not checked.ok:
            return CheapCheckResult(False, reason=checked.error or "IK rejected")
        return CheapCheckResult(
            True,
            joint_positions_rad=checked.data["joints"],
        )

    full_plan_count = bridge.get_grasp_full_plan_candidates()
    full_calls_by_preference: dict[int, int] = {}

    def full_plan(
        candidate: GraspCandidate,
        _cheap: CheapCheckResult,
        remaining: float,
    ) -> FullPlanResult:
        preference = int(getattr(candidate, "preference_rank", 0))
        full_calls_by_preference[preference] = (
            full_calls_by_preference.get(preference, 0) + 1
        )
        remaining_candidates = max(
            1,
            full_plan_count - full_calls_by_preference[preference] + 1,
        )
        deadline = time.monotonic() + remaining / remaining_candidates

        def stage_timeout(stages_left: int) -> float:
            available = deadline - time.monotonic()
            return max(0.01, available / max(1, stages_left))

        quaternion = list(candidate.pose_quat_xyzw)
        pregrasp = motion.plan_to_pose(
            bridge,
            *candidate.pregrasp_xyz,
            quaternion,
            timeout=stage_timeout(4),
            start_state=planning_start_state,
        )
        if not pregrasp.ok:
            return FullPlanResult(False, reason=f"pregrasp: {pregrasp.error}")
        pregrasp_plan = pregrasp.data["plan"]

        approach = motion.plan_cartesian(
            bridge,
            *candidate.pose_xyz,
            quaternion,
            timeout=stage_timeout(3),
            start_state=pregrasp_plan.end_state,
        )
        if not approach.ok:
            return FullPlanResult(False, reason=f"approach: {approach.error}")
        approach_plan = approach.data["plan"]

        retreat = motion.plan_cartesian(
            bridge,
            *candidate.retreat_xyz,
            quaternion,
            timeout=stage_timeout(2),
            start_state=approach_plan.end_state,
        )
        if not retreat.ok:
            return FullPlanResult(False, reason=f"retreat: {retreat.error}")
        retreat_plan = retreat.data["plan"]

        carry = motion.plan_joints(
            bridge,
            bridge.get_carry_joints_deg(),
            timeout=stage_timeout(1),
            start_state=retreat_plan.end_state,
        )
        if not carry.ok:
            return FullPlanResult(False, reason=f"carry: {carry.error}")
        carry_plan = carry.data["plan"]
        named_plans = (
            ("pregrasp", pregrasp_plan),
            ("approach", approach_plan),
            ("retreat", retreat_plan),
            ("carry", carry_plan),
        )
        plans = tuple(plan for _, plan in named_plans)
        joint7_limit = bridge.get_joint7_soft_limit_deg()
        stage_margins = _joint7_stage_margins_deg(
            named_plans, joint7_limit,
        )
        stage_summary = ", ".join(
            (
                f"{name}={joint7_limit - margin:.1f}deg"
                if math.isfinite(margin)
                else f"{name}=missing"
            )
            for name, margin in stage_margins.items()
        )
        print(
            f"[grasp-plan] {candidate.source} joint7 max: "
            f"{stage_summary}",
            flush=True,
        )
        for stage_name, stage_margin in stage_margins.items():
            path_error = _joint7_path_limit_error(
                stage_margin,
                joint7_limit,
                path_label=f"{stage_name} stage",
            )
            if path_error:
                return FullPlanResult(False, reason=path_error)
        total_motion = sum(_trajectory_motion_rad(plan) for plan in plans)
        path = PlannedGraspPath(
            candidate=candidate,
            to_pregrasp=pregrasp_plan,
            approach=approach_plan,
            retreat=retreat_plan,
            to_carry=carry_plan,
            total_joint_motion_rad=total_motion,
        )
        return FullPlanResult(
            True,
            plan=path,
            total_joint_motion_rad=total_motion,
        )

    evaluator = CandidateEvaluator(
        cheap_check,
        full_plan,
        marker_sink=marker_sink,
        deadline_s=bridge.get_grasp_candidate_timeout(),
        full_plan_count=full_plan_count,
        joint7_limit_deg=bridge.get_joint7_soft_limit_deg(),
        joint7_min_margin_deg=bridge.get_joint7_min_margin_deg(),
    )
    batch = evaluator.evaluate(
        candidates,
        current_tcp_quat_xyzw=current_quat,
        current_joint_positions_rad=current_joints,
    )
    if batch.timed_out:
        return BlockGraspPlanning(
            batch,
            None,
            generated,
            (
                f"candidate evaluation exceeded "
                f"{bridge.get_grasp_candidate_timeout():.1f}s; no motion allowed"
            ),
        )
    if batch.selected is None or batch.selected.full_plan is None:
        reasons = [
            f"{evaluation.candidate_id}: {evaluation.reason}"
            for evaluation in batch.evaluations
            if evaluation.reason
        ]
        detail = "; ".join(reasons[:4])
        return BlockGraspPlanning(
            batch,
            None,
            generated,
            f"no complete {object_kind} grasp path"
            f"{': ' + detail if detail else ''}",
        )
    selected_path = batch.selected.full_plan.plan
    if not isinstance(selected_path, PlannedGraspPath):
        return BlockGraspPlanning(
            batch, None, generated, "selected planner result has no complete path"
        )
    return BlockGraspPlanning(batch, selected_path, generated)


def plan_block_grasp(
    bridge,
    geometry_value: ObjectGeometry | dict,
    *,
    excluded_candidate_ids: Iterable[str] = (),
) -> BlockGraspPlanning:
    """Generate and evaluate block candidates without moving the robot."""

    return _plan_grasp(
        bridge,
        geometry_value,
        object_kind="block",
        excluded_candidate_ids=excluded_candidate_ids,
    )


def plan_cylinder_grasp(
    bridge,
    geometry_value: ObjectGeometry | dict,
    *,
    excluded_candidate_ids: Iterable[str] = (),
) -> BlockGraspPlanning:
    """Generate and evaluate upright/lying cylinder candidates plan-only."""

    return _plan_grasp(
        bridge,
        geometry_value,
        object_kind="cylinder",
        excluded_candidate_ids=excluded_candidate_ids,
    )


def plan_transparent_bottle_grasp(
    bridge,
    geometry_value: ObjectGeometry | dict,
    *,
    excluded_candidate_ids: Iterable[str] = (),
) -> BlockGraspPlanning:
    """Plan only the measured-label strategy for the transparent bottle."""

    return _plan_grasp(
        bridge,
        geometry_value,
        object_kind="transparent_bottle",
        excluded_candidate_ids=excluded_candidate_ids,
    )
