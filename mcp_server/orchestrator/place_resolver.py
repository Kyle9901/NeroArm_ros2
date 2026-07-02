"""Rule-based place target resolution."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge


_PLACE_ALIASES = {
    "right": "right",
    "右": "right",
    "右边": "right",
    "left": "left",
    "左": "left",
    "左边": "left",
    "center": "center",
    "middle": "center",
    "中间": "center",
    "中央": "center",
    "front": "front",
    "前": "front",
    "前面": "front",
    "back": "back",
    "后": "back",
    "后面": "back",
}


# Conservative defaults inside the workspace in base_link frame.
_PLACE_TABLE = {
    "right": {"x": -0.20, "y": -0.35, "z": 0.0},
    "left": {"x": -0.20, "y": 0.10, "z": 0.0},
    "center": {"x": -0.30, "y": -0.12, "z": 0.0},
    "front": {"x": -0.15, "y": -0.12, "z": 0.0},
    "back": {"x": -0.45, "y": -0.12, "z": 0.0},
}


def resolve_place(bridge: "RobotBridge", place: str | dict) -> dict:
    if isinstance(place, dict):
        return {
            "x": float(place["x"]),
            "y": float(place["y"]),
            "z": float(place.get("z", 0.0)),
        }

    normalized = str(place).strip().lower()
    key = _PLACE_ALIASES.get(normalized)
    if key is None:
        for alias, canonical in _PLACE_ALIASES.items():
            if alias in normalized:
                key = canonical
                break
    if key is None:
        return bridge.get_place_pose()
    return dict(_PLACE_TABLE[key])
