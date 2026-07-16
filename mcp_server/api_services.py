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
            "flange_to_tip": bridge.get_flange_to_tip(),
            "fingertip_depth": bridge.get_fingertip_depth(),
        },
        "health": bridge.health_status(),
    }


def stop(bridge: "RobotBridge") -> dict:
    """Cancel tracked robot motion goals."""
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
