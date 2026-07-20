"""Small service handlers used directly by the public MCP API.

Task execution belongs to the orchestrator and robot operations belong to
components/skills.  This module only contains the few administrative handlers
that do not form part of a task pipeline.
"""

from typing import TYPE_CHECKING

from .config import runtime_config

if TYPE_CHECKING:
    from .ros_bridge import RobotBridge
    from .vlm_client import VlmClient
    from .yolo_detector import YoloDetector


def get_status(bridge: "RobotBridge") -> dict:
    """Return the hardware and grasp-geometry status exposed by MCP."""
    js = bridge.get_joint_state()
    return {
        "success": True,
        **js,
        "holding": bridge.get_holding(),
        "workspace": bridge.get_workspace_bounds(),
        "safe_height": bridge.get_safe_height(),
        "desk_surface_z": bridge.get_desk_surface_z(),
        "grasp_geometry": {
            "configured_tcp_offset": bridge.get_tcp_offset(),
            "fingertip_depth": bridge.get_fingertip_depth(),
            "candidate_tilts_deg": bridge.get_grasp_tilt_angles_deg(),
            "cylinder_candidate_tilts_deg": (
                bridge.get_cylinder_tilt_angles_deg()
            ),
            "cylinder_side_grasp_height_ratio": (
                bridge.get_cylinder_side_grasp_height_ratio()
            ),
            "cylinder_diameter_range_m": [
                bridge.get_cylinder_min_diameter_m(),
                bridge.get_cylinder_max_diameter_m(),
            ],
            "cylinder_length_range_m": [
                bridge.get_cylinder_min_length_m(),
                bridge.get_cylinder_max_length_m(),
            ],
            "pregrasp_distance": bridge.get_grasp_pregrasp_distance(),
            "retreat_distance": bridge.get_grasp_retreat_distance(),
            "reverse_branch_tolerance_rad": (
                bridge.get_reverse_branch_tolerance_rad()
            ),
        },
        "observation_joints_deg": bridge.get_observation_joints_deg(),
        "carry_joints_deg": bridge.get_carry_joints_deg(),
        "health": bridge.health_status(),
    }


def stop(bridge: "RobotBridge") -> dict:
    """Request a cooperative task stop and clear diagnostic goal tracking."""
    ok, message = bridge.emergency_stop()
    return {"success": ok, "message": message}


def configure_octomap(bridge: "RobotBridge", enabled: bool) -> dict:
    """Enable live OctoMap updates or stop updates and clear existing voxels."""
    return bridge.set_octomap_enabled(enabled)


def configure_runtime(
    vlm: "VlmClient",
    yolo: "YoloDetector | None" = None,
    api_key: str | None = None,
    api_url: str | None = None,
    model: str | None = None,
    yolo_confidence: float | None = None,
    vlm_fallback: bool | None = None,
) -> dict:
    """Update detector settings that are safe to change at runtime."""
    changes = []
    if api_key is not None:
        vlm.api_key = api_key
        changes.append("api_key")
    if api_url is not None:
        vlm.api_url = api_url
        changes.append("api_url")
    if model is not None:
        vlm.model_name = model
        changes.append("model")
    if yolo_confidence is not None and yolo is not None:
        yolo.confidence = float(yolo_confidence)
        changes.append(f"yolo_confidence={yolo_confidence}")
    if vlm_fallback is not None:
        runtime_config.set_vlm_fallback(vlm_fallback)
        changes.append(f"vlm_fallback={'on' if vlm_fallback else 'off'}")

    current = {
        "api_url": vlm.api_url,
        "model": vlm.model_name,
        "api_key_set": bool(vlm.api_key),
        "yolo_confidence": yolo.confidence if yolo else None,
        "vlm_fallback": runtime_config.vlm_fallback,
    }
    message = f"Updated: {', '.join(changes)}" if changes else "No changes"
    return {"success": True, "message": message, "current": current}
