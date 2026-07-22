"""Pick and place skills."""

import math
import time
from statistics import median
from typing import TYPE_CHECKING

from .base import GraspGeometry, SkillResult
from .recovery import recover_to_safe
from ..components import motion
from ..grasping import (
    PlannedGraspPath,
    plan_block_grasp,
    plan_cylinder_grasp,
    plan_transparent_bottle_grasp,
)
from ..grasping.pipeline import (
    _joint7_path_limit_error,
    _minimum_joint7_margin_deg,
)
from ..models import GraspCandidate
from ..object_types import (
    is_block_target,
    is_cylinder_target,
    is_transparent_bottle_target,
)

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge


def _fail(error: str, step: str, *, recovered: bool = False,
          retryable: bool = True,
          holding: bool | None = None,
          **data) -> SkillResult:
    return SkillResult.failure(
        error,
        failed_step=step,
        recovered=recovered,
        retryable=retryable,
        holding=holding,
        **data,
    )


def _is_block_target(target: str | None) -> bool:
    """Compatibility alias retained for callers outside this module."""

    return is_block_target(target)


def _is_cylinder_target(target: str | None) -> bool:
    return is_cylinder_target(target)


def _candidate_session(bridge: "RobotBridge", x: float, y: float, z: float):
    key = [round(float(value), 3) for value in (x, y, z)]
    context = bridge.get_task_context()
    if context.get("candidate_target_key") != key:
        bridge.update_task_context(
            candidate_target_key=key,
            rejected_candidate_ids=[],
        )
        return key, []
    return key, list(context.get("rejected_candidate_ids") or [])


def _reject_candidate(bridge: "RobotBridge", candidate_id: str) -> None:
    context = bridge.get_task_context()
    rejected = list(context.get("rejected_candidate_ids") or [])
    if candidate_id not in rejected:
        rejected.append(candidate_id)
    bridge.update_task_context(rejected_candidate_ids=rejected)


_ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))


def _terminal_arm_joints_deg(plan) -> list[float] | None:
    terminal = getattr(plan, "terminal_joints", None)
    if not isinstance(terminal, dict):
        return None
    if any(name not in terminal for name in _ARM_JOINT_NAMES):
        return None
    return [
        math.degrees(float(terminal[name]))
        for name in _ARM_JOINT_NAMES
    ]


def _candidate_with_reverse_context(path: PlannedGraspPath) -> dict:
    """Serialize the selected geometry plus the exact forward IK branch."""
    value = path.candidate.to_dict()
    targets = {
        "pregrasp": _terminal_arm_joints_deg(path.to_pregrasp),
        "grasp": _terminal_arm_joints_deg(path.approach),
        "retreat": _terminal_arm_joints_deg(path.retreat),
    }
    if all(target is not None for target in targets.values()):
        value["reverse_joint_targets_deg"] = targets
    return value


def _terminal_target_difference(
    plan,
    target_deg: list[float] | None,
) -> dict | None:
    """Return the largest wrapped arm-joint endpoint difference."""
    if target_deg is None or len(target_deg) != len(_ARM_JOINT_NAMES):
        return None
    terminal = getattr(plan, "terminal_joints", None)
    if not isinstance(terminal, dict):
        return {
            "joint": "unknown",
            "error_rad": math.inf,
            "saved_deg": math.nan,
            "planned_deg": math.nan,
        }
    differences = []
    for name, degrees in zip(_ARM_JOINT_NAMES, target_deg):
        if name not in terminal:
            return {
                "joint": name,
                "error_rad": math.inf,
                "saved_deg": float(degrees),
                "planned_deg": math.nan,
            }
        error = abs(math.atan2(
            math.sin(float(terminal[name]) - math.radians(float(degrees))),
            math.cos(float(terminal[name]) - math.radians(float(degrees))),
        ))
        differences.append({
            "joint": name,
            "error_rad": error,
            "saved_deg": float(degrees),
            "planned_deg": math.degrees(float(terminal[name])),
        })
    return max(differences, key=lambda item: item["error_rad"])


def _branch_check_error(
    stage: str,
    difference: dict | None,
    tolerance_rad: float,
) -> str | None:
    if difference is None:
        return None
    error_rad = float(difference["error_rad"])
    if math.isfinite(error_rad) and error_rad <= tolerance_rad:
        return None
    return (
        f"{stage} changed from the saved IK branch: "
        f"{difference['joint']} differs by {math.degrees(error_rad):.1f}deg "
        f"(limit {math.degrees(tolerance_rad):.1f}deg, "
        f"saved={difference['saved_deg']:.1f}deg, "
        f"planned={difference['planned_deg']:.1f}deg)"
    )


def _closest_saved_branch(
    plan,
    saved_targets: dict[str, list[float] | None],
) -> tuple[str | None, dict | None]:
    """Return the saved endpoint with the smallest worst-joint difference."""
    differences = [
        (label, difference)
        for label, target in saved_targets.items()
        if (difference := _terminal_target_difference(plan, target)) is not None
    ]
    if not differences:
        return None, None
    return min(
        differences,
        key=lambda item: float(item[1]["error_rad"]),
    )


def _motion_state_unknown(result) -> bool:
    return bool(result.data.get("motion_state_unknown"))


def _confirmed_gripper_width(
    bridge: "RobotBridge",
    *,
    samples: int = 3,
    timeout: float = 1.5,
) -> tuple[float | None, str | None]:
    """Require fresh, stable feedback messages received after close."""
    snapshot = motion.read_joint_state(bridge)
    if not snapshot.ok:
        return None, snapshot.error or "joint feedback unavailable"
    sequence = snapshot.data.get("sequence")
    if sequence is None:
        return None, "joint feedback has no freshness sequence"

    deadline = time.monotonic() + timeout
    widths: list[float] = []
    for _ in range(samples):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return None, "fresh gripper feedback timed out"
        state = motion.read_joint_state(
            bridge,
            newer_than=int(sequence),
            timeout=remaining,
        )
        if not state.ok:
            return None, state.error or "fresh gripper feedback unavailable"
        sequence = state.data.get("sequence")
        width = state.data.get("gripper_width")
        if sequence is None or width is None:
            return None, "fresh gripper feedback is incomplete"
        widths.append(float(width))
    if max(widths) - min(widths) > 0.004:
        return None, (
            "gripper feedback is not stable "
            f"(spread={max(widths) - min(widths):.4f}m)"
        )
    return float(median(widths)), None


def _retreat_with_gripper_state(
    bridge: "RobotBridge",
    path: PlannedGraspPath,
) -> bool:
    """Keep the current gripper command while recovering to carry."""
    retreat = motion.execute_planned(bridge, path.retreat)
    if not retreat.ok and _motion_state_unknown(retreat):
        return False
    planned_retreat_succeeded = retreat.ok
    if not planned_retreat_succeeded:
        candidate = path.candidate
        retreat = motion.move_to_pose(
            bridge,
            *candidate.retreat_xyz,
            list(candidate.pose_quat_xyzw),
        )
        if not retreat.ok and _motion_state_unknown(retreat):
            return False
    # The precomputed carry trajectory is valid only when its exact planned
    # retreat trajectory reached the corresponding end_state.  A fallback
    # pose move can end in another IK branch, so continuing with the stale
    # trajectory would violate its start-state assumption.
    carry = (
        motion.execute_planned(bridge, path.to_carry)
        if planned_retreat_succeeded
        else motion.go_carry(bridge)
    )
    return bool(retreat.ok and carry.ok)


def _grasp_candidate_shape(
    bridge: "RobotBridge",
    x: float,
    y: float,
    z: float,
    geometry: dict,
    geo: GraspGeometry,
    *,
    planner,
    object_kind: str,
) -> SkillResult:
    _key, excluded = _candidate_session(bridge, x, y, z)
    planning = planner(
        bridge,
        geometry,
        excluded_candidate_ids=excluded,
    )
    if not planning.ok:
        return _fail(
            planning.error or f"No complete {object_kind} grasp path",
            "candidate_planning",
            retryable=False,
            holding=False,
            candidate_count=len(planning.candidates),
        )
    path = planning.selected_path
    candidate = path.candidate
    endpoint_detail = ""
    if (
        object_kind == "transparent_bottle"
        and geometry.get("orientation_class") == "upright"
    ):
        # x/y/z are the already validated perception output passed by the
        # orchestrator.  Do not reach into ObjectGeometry's serialized dict
        # here (its public field is ``center``, not the internal center_xyz).
        visual_axis = (float(x), float(y), float(z))
        overtravel = math.dist(visual_axis, candidate.pose_xyz)
        endpoint_detail = (
            f", visual_axis=({visual_axis[0]:.4f},"
            f"{visual_axis[1]:.4f},{visual_axis[2]:.4f}), "
            f"axis_overtravel={overtravel:.3f}m"
        )
    print(
        f"[grasp] kind={object_kind}, selected={candidate.candidate_id}, "
        f"angle_from_table={90.0 - candidate.tilt_deg:.0f}deg "
        f"(tilt_from_vertical={candidate.tilt_deg:.0f}deg), "
        f"pose=({candidate.pose_xyz[0]:.4f},"
        f"{candidate.pose_xyz[1]:.4f},{candidate.pose_xyz[2]:.4f}), "
        f"width={candidate.gripper_width:.3f}, "
        f"evaluated={planning.batch.cheap_checks}, "
        f"full_plans={planning.batch.full_plans}, "
        f"time={planning.batch.elapsed_s:.2f}s{endpoint_detail}",
        flush=True,
    )

    opened = motion.control_gripper(bridge, geo.gripper_open, duration=2.0)
    if not opened.ok:
        return _fail(
            opened.error or "open gripper failed",
            "open_gripper",
            retryable=True,
            holding=False,
            selected_candidate=candidate.to_dict(),
        )

    pregrasp = motion.execute_planned(bridge, path.to_pregrasp)
    if not pregrasp.ok:
        _reject_candidate(bridge, candidate.candidate_id)
        unknown = _motion_state_unknown(pregrasp)
        recovered = False if unknown else motion.go_home(bridge).ok
        return _fail(
            pregrasp.error or "pregrasp execution failed",
            "pregrasp",
            recovered=recovered,
            retryable=recovered and not unknown,
            holding=False,
            motion_state_unknown=unknown,
            selected_candidate=candidate.to_dict(),
        )

    approach = motion.execute_planned(bridge, path.approach)
    if not approach.ok:
        _reject_candidate(bridge, candidate.candidate_id)
        unknown = _motion_state_unknown(approach)
        recovered = False
        if not unknown:
            recovery = motion.move_cartesian(
                bridge,
                *candidate.pregrasp_xyz,
                list(candidate.pose_quat_xyzw),
                velocity=geo.descent_vel,
                accel=geo.descent_accel,
            )
            recovered = recovery.ok
            if not recovered and not _motion_state_unknown(recovery):
                recovered = motion.go_home(bridge).ok
        return _fail(
            approach.error or "approach execution failed",
            "approach",
            recovered=recovered,
            retryable=recovered and not unknown,
            holding=False,
            motion_state_unknown=unknown,
            selected_candidate=candidate.to_dict(),
        )

    if (
        object_kind == "transparent_bottle"
        and geometry.get("orientation_class") == "upright"
    ):
        # Execution success means the controller accepted the trajectory.
        # Record the measured TCP before closing so a geometric miss can be
        # distinguished from tracking or TCP-frame errors on the next run.
        try:
            reached = bridge.get_current_tcp_pose(timeout=1.0)["position"]
            endpoint_error = math.dist(
                tuple(float(value) for value in reached),
                candidate.pose_xyz,
            )
            print(
                f"[grasp] reached_tcp=({reached[0]:.4f},{reached[1]:.4f},"
                f"{reached[2]:.4f}), "
                f"endpoint_error={endpoint_error * 1000.0:.1f}mm",
                flush=True,
            )
        except Exception as error:
            print(
                f"[grasp] WARNING: cannot verify reached TCP before close: "
                f"{error}",
                flush=True,
            )

    closed = motion.control_gripper(bridge, geo.gripper_close, duration=2.0)
    if not closed.ok:
        if closed.data.get("stop_requested"):
            bridge.set_holding(False)
            return _fail(
                closed.error or "task stopped before gripper close",
                "task_stop",
                retryable=False,
                holding=False,
                stop_requested=True,
                selected_candidate=candidate.to_dict(),
            )
        # The actuator may have moved despite a missing result. Preserve its
        # state, leave the object vicinity, and do not switch candidates.
        recovered = _retreat_with_gripper_state(bridge, path)
        bridge.set_holding(None)
        return _fail(
            closed.error or "close gripper result is uncertain",
            "close_gripper",
            recovered=recovered,
            retryable=False,
            holding=None,
            selected_candidate=candidate.to_dict(),
        )

    width, feedback_error = _confirmed_gripper_width(bridge)
    if width is None:
        recovered = _retreat_with_gripper_state(bridge, path)
        bridge.set_holding(None)
        return _fail(
            f"{feedback_error}; holding state is uncertain",
            "holding_check",
            recovered=recovered,
            retryable=False,
            holding=None,
            gripper_width=None,
            selected_candidate=candidate.to_dict(),
        )

    holding = geo.is_holding(width)
    if not holding:
        motion.control_gripper(bridge, geo.gripper_open, duration=1.5)
        recovered = _retreat_with_gripper_state(bridge, path)
        bridge.set_holding(False)
        _reject_candidate(bridge, candidate.candidate_id)
        return _fail(
            "gripper closed without an object",
            "holding_check",
            recovered=recovered,
            retryable=recovered,
            holding=False,
            gripper_width=width,
            selected_candidate=candidate.to_dict(),
        )

    recovered = _retreat_with_gripper_state(bridge, path)
    bridge.set_holding(True)
    if not recovered:
        return _fail(
            "object is held but recovery to carry failed",
            "carry",
            recovered=False,
            retryable=False,
            holding=True,
            gripper_width=width,
            selected_candidate=candidate.to_dict(),
        )

    bridge.update_task_context(rejected_candidate_ids=[])
    selected_candidate = _candidate_with_reverse_context(path)
    return SkillResult.success(
        holding=True,
        state="holding",
        gripper_width=width,
        pick_x=x,
        pick_y=y,
        pick_z=z,
        geometry=geometry,
        selected_candidate=selected_candidate,
        candidate_evaluations=[
            {
                "candidate_id": item.candidate_id,
                "status": item.status.value,
                "reason": item.reason,
                "joint7_margin_deg": item.joint7_margin_deg,
            }
            for item in planning.batch.evaluations
        ],
    )


def _grasp_block(
    bridge: "RobotBridge",
    x: float,
    y: float,
    z: float,
    geometry: dict,
    geo: GraspGeometry,
) -> SkillResult:
    return _grasp_candidate_shape(
        bridge,
        x,
        y,
        z,
        geometry,
        geo,
        planner=plan_block_grasp,
        object_kind="block",
    )


def _grasp_cylinder(
    bridge: "RobotBridge",
    x: float,
    y: float,
    z: float,
    geometry: dict,
    geo: GraspGeometry,
) -> SkillResult:
    return _grasp_candidate_shape(
        bridge,
        x,
        y,
        z,
        geometry,
        geo,
        planner=plan_cylinder_grasp,
        object_kind="cylinder",
    )


def _grasp_transparent_bottle(
    bridge: "RobotBridge",
    x: float,
    y: float,
    z: float,
    geometry: dict,
    geo: GraspGeometry,
) -> SkillResult:
    return _grasp_candidate_shape(
        bridge,
        x,
        y,
        z,
        geometry,
        geo,
        planner=plan_transparent_bottle_grasp,
        object_kind="transparent_bottle",
    )


def _grasp_legacy(
    bridge: "RobotBridge", x: float, y: float, z: float,
    quat: list[float] | None, geometry: dict | None,
) -> SkillResult:
    geo = GraspGeometry.from_bridge(bridge)
    quat = quat or bridge.get_grasp_quat()

    check = motion.workspace_check(bridge, x, y)
    if not check.ok:
        return _fail(check.error or "Target outside workspace", "workspace_check", retryable=False)

    print(f"[grasp] target: x={x:.4f}, y={y:.4f}, z={z:.4f}, "
          f"approach_z={z + geo.approach_height:.4f}, "
          f"grasp_tcp_z={geo.grasp_tcp_z(z):.4f} ("
          f"fingertip_depth={geo.fingertip_depth:.3f})", flush=True)

    # Safety: the calibrated TCP is at the fingertip center; keep it above desk.
    desk_z = (
        float(geometry["local_desk_z"])
        if geometry and geometry.get("local_desk_z") is not None
        else bridge.get_desk_surface_z()
    )
    grasp_tcp_z = geo.grasp_tcp_z(z)
    if grasp_tcp_z < desk_z:
        clamped_depth = z - desk_z
        print(f"[grasp] WARNING: grasp TCP {grasp_tcp_z:.4f} below desk {desk_z:.4f}, "
              f"clamping fingertip_depth {geo.fingertip_depth:.3f}→{clamped_depth:.3f}",
              flush=True)
        geo = geo.with_fingertip_depth(clamped_depth)

    result = motion.control_gripper(bridge, geo.gripper_open, duration=2.0)
    if not result.ok:
        unknown = _motion_state_unknown(result)
        return _fail(
            result.error or "open gripper failed",
            "open_gripper",
            retryable=not unknown,
            holding=False,
            motion_state_unknown=unknown,
        )

    result = motion.move_to_pose(bridge, x, y, z + geo.approach_height, quat)
    if not result.ok:
        unknown = _motion_state_unknown(result)
        return _fail(
            result.error or "approach failed",
            "approach",
            retryable=not unknown,
            holding=False,
            motion_state_unknown=unknown,
        )

    result = motion.move_cartesian(
        bridge, x, y, geo.grasp_tcp_z(z), quat,
        velocity=geo.descent_vel, accel=geo.descent_accel)
    if not result.ok:
        unknown = _motion_state_unknown(result)
        recovered = (
            False if unknown
            else recover_to_safe(bridge, x, y, geo, quat)
        )
        return _fail(
            result.error or "descent failed",
            "descent",
            recovered=recovered,
            retryable=recovered and not unknown,
            holding=False,
            motion_state_unknown=unknown,
        )

    result = motion.control_gripper(bridge, geo.gripper_close, duration=2.0)
    if not result.ok:
        if result.data.get("stop_requested"):
            bridge.set_holding(False)
            return _fail(
                result.error or "task stopped before gripper close",
                "task_stop",
                retryable=False,
                holding=False,
                stop_requested=True,
            )
        bridge.set_holding(None)
        unknown = _motion_state_unknown(result)
        recovered = (
            False if unknown
            else recover_to_safe(bridge, x, y, geo, quat)
        )
        return _fail(
            result.error or "close gripper failed",
            "close_gripper",
            recovered=recovered,
            retryable=False,
            holding=None,
            motion_state_unknown=unknown,
        )

    width, feedback_error = _confirmed_gripper_width(bridge)
    if width is None:
        bridge.set_holding(None)
        recovered = recover_to_safe(bridge, x, y, geo, quat)
        return _fail(
            f"{feedback_error}; holding state is uncertain",
            "holding_check",
            recovered=recovered,
            retryable=False,
            holding=None,
        )
    holding = geo.is_holding(width)
    bridge.set_holding(holding)

    result = motion.move_cartesian(bridge, x, y, geo.safe_height, quat)
    if not result.ok:
        if _motion_state_unknown(result):
            return _fail(
                result.error or "lift state is unknown",
                "lift",
                retryable=False,
                holding=holding,
                gripper_width=width,
                motion_state_unknown=True,
            )
        result = motion.move_to_pose(bridge, x, y, geo.safe_height, quat)
    if not result.ok:
        unknown = _motion_state_unknown(result)
        recovered = (
            False if unknown
            else recover_to_safe(bridge, x, y, geo, quat)
        )
        return SkillResult.failure(
            result.error or "lift failed",
            failed_step="lift",
            recovered=recovered,
            retryable=False,
            holding=holding,
            gripper_width=width,
            motion_state_unknown=unknown,
        )

    return SkillResult.success(
        holding=holding,
        state="holding" if holding else "empty",
        gripper_width=width,
        pick_x=x,
        pick_y=y,
        pick_z=z,
        geometry=geometry,
    )


def grasp_object(bridge: "RobotBridge", x: float, y: float, z: float,
                 quat: list[float] | None = None,
                 geometry: dict | None = None,
                 target: str | None = None) -> SkillResult:
    """Use shape-specific candidates for blocks/cylinders."""
    geo = GraspGeometry.from_bridge(bridge)
    if _is_block_target(target):
        if not geometry:
            return _fail(
                "block grasp requires reliable oriented geometry",
                "candidate_geometry",
                retryable=False,
                holding=False,
            )
        return _grasp_block(bridge, x, y, z, geometry, geo)
    if is_transparent_bottle_target(target):
        if not geometry:
            return _fail(
                "transparent bottle grasp requires stable measured-label "
                "geometry",
                "candidate_geometry",
                retryable=False,
                holding=False,
            )
        return _grasp_transparent_bottle(
            bridge, x, y, z, geometry, geo,
        )
    if _is_cylinder_target(target):
        if not geometry:
            return _fail(
                "cylinder grasp requires reliable fitted geometry",
                "candidate_geometry",
                retryable=False,
                holding=False,
            )
        return _grasp_cylinder(bridge, x, y, z, geometry, geo)
    return _grasp_legacy(bridge, x, y, z, quat, geometry)


def _plan_reverse_place(
    bridge: "RobotBridge",
    candidate: GraspCandidate,
    timeout: float,
    reverse_joint_targets_deg: dict | None = None,
):
    started = time.monotonic()
    raw_targets = reverse_joint_targets_deg or {}
    targets: dict[str, list[float] | None] = {}
    for name in ("pregrasp", "grasp", "retreat"):
        value = raw_targets.get(name)
        if value is None:
            targets[name] = None
            continue
        if (
            not isinstance(value, (list, tuple))
            or len(value) != len(_ARM_JOINT_NAMES)
            or any(not math.isfinite(float(item)) for item in value)
        ):
            return None, f"invalid saved reverse joint target: {name}"
        targets[name] = [float(item) for item in value]

    def remaining(stages: int):
        available = timeout - (time.monotonic() - started)
        return available / max(1, stages)

    quat = list(candidate.pose_quat_xyzw)
    stage_timeout = remaining(3)
    if stage_timeout <= 0.0:
        return None, "reverse place planning deadline exceeded"
    if targets["retreat"] is not None:
        # Re-enter the exact IK branch used at the end of the original retreat.
        # Reversing a time-parameterized trajectory in place is unsafe, but
        # locking this joint endpoint preserves the original branch before the
        # two collision-aware Cartesian reverse segments are rebuilt.
        to_retreat = motion.plan_joints(
            bridge,
            targets["retreat"],
            timeout=stage_timeout,
        )
    else:
        to_retreat = motion.plan_to_pose(
            bridge,
            *candidate.retreat_xyz,
            quat,
            timeout=stage_timeout,
        )
    if not to_retreat.ok:
        return None, f"reverse preplace: {to_retreat.error}"
    retreat_plan = to_retreat.data["plan"]
    stage_timeout = remaining(2)
    if stage_timeout <= 0.0:
        return None, "reverse place planning deadline exceeded"
    to_grasp = motion.plan_cartesian(
        bridge,
        *candidate.pose_xyz,
        quat,
        timeout=stage_timeout,
        start_state=retreat_plan.end_state,
    )
    if not to_grasp.ok:
        return None, f"reverse approach: {to_grasp.error}"
    grasp_plan = to_grasp.data["plan"]
    branch_tolerance = bridge.get_reverse_branch_tolerance_rad()
    grasp_difference = _terminal_target_difference(
        grasp_plan, targets["grasp"],
    )
    branch_error = _branch_check_error(
        "reverse approach", grasp_difference, branch_tolerance,
    )
    if branch_error:
        return None, branch_error
    if grasp_difference is not None:
        print(
            f"[place] reverse approach branch: max_delta="
            f"{math.degrees(grasp_difference['error_rad']):.1f}deg "
            f"at {grasp_difference['joint']} "
            f"(limit={math.degrees(branch_tolerance):.1f}deg)",
            flush=True,
        )
    stage_timeout = remaining(1)
    if stage_timeout <= 0.0:
        return None, "reverse place planning deadline exceeded"
    open_retreat_start = bridge.node.robot_state_with_gripper_width(
        bridge.get_gripper_open_width(),
        base_state=grasp_plan.end_state,
    )
    to_pregrasp = motion.plan_cartesian(
        bridge,
        *candidate.pregrasp_xyz,
        quat,
        timeout=stage_timeout,
        start_state=open_retreat_start,
    )
    if not to_pregrasp.ok:
        return None, f"reverse retreat: {to_pregrasp.error}"
    pregrasp_plan = to_pregrasp.data["plan"]
    retreat_branch_targets = {
        "pregrasp": targets["pregrasp"],
    }
    same_free_space_pose = math.dist(
        candidate.pregrasp_xyz,
        candidate.retreat_xyz,
    ) <= 1e-4
    if same_free_space_pose:
        # With equal approach/retreat distances these are the same TCP pose.
        # A redundant 7-DOF arm may have reached that already validated pose
        # through two nearby joint endpoints during the forward plan.
        retreat_branch_targets["retreat"] = targets["retreat"]
    matched_label, pregrasp_difference = _closest_saved_branch(
        pregrasp_plan,
        retreat_branch_targets,
    )
    branch_error = _branch_check_error(
        (
            f"reverse retreat (closest saved {matched_label})"
            if matched_label else "reverse retreat"
        ),
        pregrasp_difference,
        branch_tolerance,
    )
    if branch_error:
        return None, branch_error
    if pregrasp_difference is not None:
        print(
            f"[place] reverse retreat branch: max_delta="
            f"{math.degrees(pregrasp_difference['error_rad']):.1f}deg "
            f"at {pregrasp_difference['joint']} "
            f"(matched={matched_label}, "
            f"limit={math.degrees(branch_tolerance):.1f}deg)",
            flush=True,
        )
    if time.monotonic() - started > timeout:
        return None, "reverse place planning deadline exceeded"
    plans = (retreat_plan, grasp_plan, pregrasp_plan)
    joint7_limit = bridge.get_joint7_soft_limit_deg()
    path_margin = _minimum_joint7_margin_deg(
        plans,
        joint7_limit,
    )
    path_error = _joint7_path_limit_error(
        path_margin,
        joint7_limit,
        path_label="reverse place path",
    )
    if path_error:
        return None, path_error
    return (
        retreat_plan,
        grasp_plan,
        pregrasp_plan,
    ), None


def _place_reverse(
    bridge: "RobotBridge",
    candidate_value: dict,
    x: float,
    y: float,
    z: float,
) -> SkillResult:
    candidate = GraspCandidate.from_dict(candidate_value)
    xy_error = (
        (candidate.pose_xyz[0] - float(x)) ** 2
        + (candidate.pose_xyz[1] - float(y)) ** 2
    ) ** 0.5
    if candidate.object_kind == "cylinder":
        position_spread = bridge.get_cylinder_position_max_spread()
    else:
        position_spread = bridge.get_block_xy_max_spread()
    allowed_xy_error = max(0.02, 2.0 * position_spread)
    if xy_error > allowed_xy_error:
        return _fail(
            (
                f"reverse candidate differs from stored pick position by "
                f"{xy_error:.3f}m (limit {allowed_xy_error:.3f}m)"
            ),
            "reverse_candidate_validation",
            retryable=False,
            holding=True,
        )
    plans, error = _plan_reverse_place(
        bridge,
        candidate,
        bridge.get_grasp_candidate_timeout(),
        candidate_value.get("reverse_joint_targets_deg"),
    )
    if plans is None:
        return _fail(
            error or "reverse place planning failed",
            "reverse_place_planning",
            retryable=False,
            holding=True,
        )

    to_retreat, to_grasp, to_pregrasp = plans
    for step, plan in (
        ("move_to_preplace", to_retreat),
        ("reverse_approach", to_grasp),
    ):
        result = motion.execute_planned(bridge, plan)
        if not result.ok:
            unknown = _motion_state_unknown(result)
            recovered = False if unknown else motion.go_carry(bridge).ok
            return _fail(
                result.error or f"{step} failed",
                step,
                recovered=recovered,
                retryable=False,
                holding=True,
                motion_state_unknown=unknown,
            )

    opened = motion.control_gripper(
        bridge,
        GraspGeometry.from_bridge(bridge).gripper_open,
    )
    if not opened.ok:
        if opened.data.get("stop_requested"):
            bridge.set_holding(True)
            return _fail(
                opened.error or "task stopped before object release",
                "task_stop",
                retryable=False,
                holding=True,
                stop_requested=True,
            )
        retreat_result = motion.execute_planned(bridge, to_pregrasp)
        recovered = retreat_result.ok
        if recovered:
            recovered = motion.go_carry(bridge).ok
        bridge.set_holding(None)
        return _fail(
            opened.error or "release result is uncertain",
            "open_gripper",
            recovered=recovered,
            retryable=False,
            holding=None,
        )

    bridge.set_holding(False)
    retreat = motion.execute_planned(bridge, to_pregrasp)
    if not retreat.ok:
        unknown = _motion_state_unknown(retreat)
        recovered = False if unknown else motion.go_home(bridge).ok
        return _fail(
            retreat.error or "reverse retreat failed after release",
            "reverse_retreat",
            recovered=recovered,
            retryable=False,
            holding=False,
            motion_state_unknown=unknown,
        )
    return SkillResult.success(
        holding=False,
        state="empty",
        place_x=x,
        place_y=y,
        place_z=z,
        reverse_candidate=candidate.to_dict(),
    )


def _place_legacy(bridge: "RobotBridge", x: float, y: float, z: float,
                  quat: list[float] | None = None) -> SkillResult:
    geo = GraspGeometry.from_bridge(bridge)
    quat = quat or bridge.get_grasp_quat()

    if z < 0.0:
        return _fail(f"place z={z:.3f} below desk surface", "safety_check", retryable=False)

    result = motion.workspace_check(bridge, x, y)
    if not result.ok:
        return _fail(result.error or "Target outside workspace", "workspace_check", retryable=False)

    result = motion.move_to_pose(bridge, x, y, geo.safe_height, quat)
    if not result.ok:
        unknown = _motion_state_unknown(result)
        return _fail(
            result.error or "move above failed",
            "move_above",
            retryable=not unknown,
            holding=True,
            motion_state_unknown=unknown,
        )

    result = motion.move_cartesian(bridge, x, y, geo.grasp_tcp_z(z), quat)
    if not result.ok:
        unknown = _motion_state_unknown(result)
        recovered = (
            False if unknown
            else recover_to_safe(bridge, x, y, geo, quat)
        )
        return _fail(
            result.error or "descent failed",
            "descent",
            recovered=recovered,
            retryable=recovered and not unknown,
            holding=True,
            motion_state_unknown=unknown,
        )

    result = motion.control_gripper(bridge, geo.gripper_open)
    if not result.ok:
        if result.data.get("stop_requested"):
            bridge.set_holding(True)
            return _fail(
                result.error or "task stopped before object release",
                "task_stop",
                retryable=False,
                holding=True,
                stop_requested=True,
            )
        unknown = _motion_state_unknown(result)
        recovered = (
            False if unknown
            else recover_to_safe(bridge, x, y, geo, quat)
        )
        if unknown:
            bridge.set_holding(None)
        return _fail(
            result.error or "open gripper failed",
            "open_gripper",
            recovered=recovered,
            retryable=False,
            holding=None if unknown else True,
            motion_state_unknown=unknown,
        )

    bridge.set_holding(False)
    return SkillResult.success(
        holding=False,
        state="empty",
        place_x=x,
        place_y=y,
        place_z=z,
    )


def _plan_translated_candidate_place(
    bridge: "RobotBridge",
    candidate: GraspCandidate,
    timeout: float,
):
    """Plan preplace, approach, and post-release retreat before execution."""
    started = time.monotonic()

    def remaining(stages: int) -> float:
        available = timeout - (time.monotonic() - started)
        return max(0.01, available / max(1, stages))

    quaternion = list(candidate.pose_quat_xyzw)
    preplace = motion.plan_to_pose(
        bridge,
        *candidate.pregrasp_xyz,
        quaternion,
        timeout=remaining(3),
    )
    if not preplace.ok:
        return None, f"preplace: {preplace.error}"
    preplace_plan = preplace.data["plan"]
    approach = motion.plan_cartesian(
        bridge,
        *candidate.pose_xyz,
        quaternion,
        timeout=remaining(2),
        start_state=preplace_plan.end_state,
    )
    if not approach.ok:
        return None, f"place approach: {approach.error}"
    approach_plan = approach.data["plan"]
    retreat = motion.plan_cartesian(
        bridge,
        *candidate.pregrasp_xyz,
        quaternion,
        timeout=remaining(1),
        start_state=approach_plan.end_state,
    )
    if not retreat.ok:
        return None, f"post-release retreat: {retreat.error}"
    retreat_plan = retreat.data["plan"]
    joint7_limit = bridge.get_joint7_soft_limit_deg()
    margin = _minimum_joint7_margin_deg(
        (preplace_plan, approach_plan, retreat_plan),
        joint7_limit,
    )
    limit_error = _joint7_path_limit_error(
        margin,
        joint7_limit,
        path_label="translated place path",
    )
    if limit_error:
        return None, limit_error
    return (preplace_plan, approach_plan, retreat_plan), None


def _place_translated_candidate(
    bridge: "RobotBridge",
    candidate_value: dict,
    x: float,
    y: float,
    z: float,
) -> SkillResult:
    try:
        candidate = GraspCandidate.from_dict(candidate_value)
    except (KeyError, TypeError, ValueError) as exc:
        return _fail(
            f"invalid placement candidate: {exc}",
            "placement_candidate_validation",
            retryable=False,
            holding=True,
        )
    if candidate.pose_xyz[2] <= bridge.get_desk_surface_z():
        return _fail(
            "placement candidate TCP is at or below the desk",
            "placement_candidate_validation",
            retryable=False,
            holding=True,
        )
    for point_name, point in (
        ("placement", candidate.pose_xyz),
        ("preplacement", candidate.pregrasp_xyz),
    ):
        workspace = motion.workspace_check(bridge, point[0], point[1])
        if not workspace.ok:
            return _fail(
                workspace.error or f"{point_name} leaves workspace",
                "workspace_check",
                retryable=False,
                holding=True,
            )

    plans, error = _plan_translated_candidate_place(
        bridge,
        candidate,
        bridge.get_grasp_candidate_timeout(),
    )
    if plans is None:
        return _fail(
            error or "translated place planning failed",
            "translated_place_planning",
            retryable=False,
            holding=True,
        )
    preplace_plan, approach_plan, retreat_plan = plans
    for step, plan in (
        ("move_to_preplace", preplace_plan),
        ("place_approach", approach_plan),
    ):
        result = motion.execute_planned(bridge, plan)
        if not result.ok:
            unknown = _motion_state_unknown(result)
            recovered = False if unknown else motion.go_carry(bridge).ok
            return _fail(
                result.error or f"{step} failed",
                step,
                recovered=recovered,
                retryable=False,
                holding=True,
                motion_state_unknown=unknown,
            )

    opened = motion.control_gripper(
        bridge,
        GraspGeometry.from_bridge(bridge).gripper_open,
    )
    if not opened.ok:
        if opened.data.get("stop_requested"):
            bridge.set_holding(True)
            return _fail(
                opened.error or "task stopped before object release",
                "task_stop",
                retryable=False,
                holding=True,
                stop_requested=True,
            )
        retreat = motion.execute_planned(bridge, retreat_plan)
        bridge.set_holding(None)
        return _fail(
            opened.error or "release result is uncertain",
            "open_gripper",
            recovered=retreat.ok,
            retryable=False,
            holding=None,
        )

    bridge.set_holding(False)
    retreat = motion.execute_planned(bridge, retreat_plan)
    if not retreat.ok:
        unknown = _motion_state_unknown(retreat)
        recovered = False if unknown else motion.go_home(bridge).ok
        return _fail(
            retreat.error or "post-release retreat failed",
            "place_retreat",
            recovered=recovered,
            retryable=False,
            holding=False,
            motion_state_unknown=unknown,
        )
    return SkillResult.success(
        holding=False,
        state="empty",
        place_x=x,
        place_y=y,
        place_z=z,
        placement_candidate=candidate.to_dict(),
    )


def place_object(bridge: "RobotBridge", x: float, y: float, z: float,
                 quat: list[float] | None = None,
                 reverse_candidate: dict | None = None,
                 placement_candidate: dict | None = None) -> SkillResult:
    """Reverse a selected shape-specific grasp path at the original pose."""
    holding = bridge.get_holding()
    if holding is not True:
        state = "unknown" if holding is None else "empty"
        return _fail(
            f"place requires confirmed holding=true (current state: {state})",
            "holding_guard",
            retryable=False,
            holding=holding,
        )
    if reverse_candidate is not None:
        return _place_reverse(bridge, reverse_candidate, x, y, z)
    if placement_candidate is not None:
        return _place_translated_candidate(
            bridge,
            placement_candidate,
            x,
            y,
            z,
        )
    return _place_legacy(bridge, x, y, z, quat)
