"""Prepare/bringup skill."""
from typing import TYPE_CHECKING

from .base import SkillResult
from ..components import infra, perception

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge


def _critical_node_counts(bridge: "RobotBridge") -> dict[str, int]:
    """Return ROS-graph counts without inspecting or mutating OS processes."""
    try:
        return bridge.get_node_counts({
            "move_group", "agx_arm_ctrl_single_node",
            "handeye_publisher", "dummy_publisher",
        })
    except Exception:
        return {}


def prepare(bridge: "RobotBridge", can_port: str = "can0",
            calib_name: str = "my_eih_calib_park",
            octomap_enabled: bool | None = None) -> SkillResult:
    if octomap_enabled is None:
        octomap_enabled = bridge.get_octomap_enabled_on_prepare()
    else:
        octomap_enabled = bool(octomap_enabled)
    counts = _critical_node_counts(bridge)
    if counts.get("dummy_publisher", 0):
        return SkillResult.failure(
            "easy_handeye calibration dummy_publisher is still running. "
            "Stop handeye_calibrate.launch.py before executing robot tasks.",
            failed_step="handeye_tf_check",
            retryable=False,
            node_counts=counts,
        )
    duplicates = {name: count for name, count in counts.items() if count > 1}
    if duplicates:
        details = ", ".join(f"{name}={count}" for name, count in sorted(duplicates.items()))
        return SkillResult.failure(
            f"Duplicate critical ROS nodes detected: {details}. "
            "prepare will not choose or kill OS processes automatically. "
            "Stop the duplicate launch manually, then run prepare again.",
            failed_step="duplicate_node_check",
            retryable=False,
            node_counts=counts,
        )
    initial = infra.bringup_status(bridge)
    initial_endpoints = initial.data.get("endpoints", {}) if initial.ok else {}
    required_endpoints = [
        "move_action", "camera_color", "handeye_publisher",
        "planning_scene_apply", "planning_scene_get", "tf",
    ]
    if octomap_enabled:
        required_endpoints.extend(["octomap_cloud", "octomap_control"])
    launch_required = not all(
        initial_endpoints.get(name) for name in required_endpoints
    )
    started = False
    if launch_required:
        result = infra.bringup_nodes(
            bridge,
            can_port=can_port,
            calib_name=calib_name,
            octomap_enabled=octomap_enabled,
        )
        started = True
        if not result.ok:
            return SkillResult.failure(
                result.error or "bringup failed",
                failed_step="bringup_nodes",
                retryable=True,
                **result.data,
            )

    # Enforce the configured state even when the cloud gate was already running.
    # Disabling also clears any voxels retained by MoveIt from an earlier run.
    octomap = bridge.set_octomap_enabled(octomap_enabled)
    if not octomap.get("success"):
        return SkillResult.failure(
            octomap.get("error") or "Failed to configure OctoMap",
            failed_step="octomap_configuration",
            retryable=True,
            octomap=octomap,
        )

    # The bridge may start before MoveIt's planning-scene service.  Add the
    # single permitted world object here as well, after bringup, and treat a
    # failure as a safety error.  No target collision box or OctoMap is needed.
    if bridge.get_desk_collision_enabled() and not bridge.node.add_desk_collision():
        return SkillResult.failure(
            "Failed to add the configured desk collision BOX",
            failed_step="desk_collision",
            retryable=True,
        )

    frame = perception.capture_image(bridge, timeout=5.0)
    health = bridge.health_status()
    if not health["ready"]:
        failures = ", ".join(health["failures"])
        camera_reason = health.get("camera", {}).get("last_rejection")
        detail = f"; camera_last_rejection={camera_reason}" if camera_reason else ""
        return SkillResult.failure(
            f"Robot health check failed: {failures}{detail}",
            failed_step="health_check",
            retryable=True,
            health=health,
            capture_error=None if frame.ok else frame.error,
        )

    return SkillResult.success(
        already_ready=not started,
        health=health,
        octomap=octomap,
    )
