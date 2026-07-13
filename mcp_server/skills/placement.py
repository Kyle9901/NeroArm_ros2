"""Resolve symbolic placement targets and height offsets."""

from .base import SkillResult


_PLACE_ALIASES = {
    "right": "right", "右": "right", "右边": "right",
    "left": "left", "左": "left", "左边": "left",
    "center": "center", "middle": "center", "中间": "center", "中央": "center",
    "front": "front", "前": "front", "前面": "front",
    "back": "back", "后": "back", "后面": "back",
}

_PLACE_TABLE = {
    "right": {"x": -0.20, "y": -0.35, "z": 0.0},
    "left": {"x": -0.20, "y": 0.10, "z": 0.0},
    "center": {"x": -0.30, "y": -0.12, "z": 0.0},
    "front": {"x": -0.15, "y": -0.12, "z": 0.0},
    "back": {"x": -0.45, "y": -0.12, "z": 0.0},
}


def _resolve_coordinates(bridge, place: str | dict) -> dict:
    if isinstance(place, dict):
        return {
            "x": float(place["x"]),
            "y": float(place["y"]),
            "z": float(place.get("z", 0.0)),
        }
    normalized = str(place).strip().lower()
    key = _PLACE_ALIASES.get(normalized)
    if key is None:
        key = next(
            (canonical for alias, canonical in _PLACE_ALIASES.items() if alias in normalized),
            None,
        )
    return dict(_PLACE_TABLE[key]) if key else bridge.get_place_pose()


def resolve_place(bridge, place: str | dict, **kwargs) -> SkillResult:
    try:
        return SkillResult.success(**_resolve_coordinates(bridge, place))
    except Exception as exc:
        return SkillResult.failure(
            str(exc),
            failed_step="resolve_place",
            retryable=False,
        )


def stack_on(
    bridge,
    x: float,
    y: float,
    z: float,
    height: float = 0.05,
    **kwargs,
) -> SkillResult:
    return SkillResult.success(x=x, y=y, z=z + height)
