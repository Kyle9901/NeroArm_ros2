"""Small, dependency-free target-shape routing helpers."""

from __future__ import annotations

import re


_BLOCK_KEYWORDS = ("方块", "物块", "cube", "block")
_TRANSPARENT_BOTTLE_KEYWORDS = (
    "水瓶",
    "矿泉水瓶",
    "塑料瓶",
    "饮料瓶",
    "bottle",
)
_CYLINDER_KEYWORDS = (
    "水瓶",
    "矿泉水瓶",
    "塑料瓶",
    "饮料瓶",
    "瓶子",
    "瓶",
    "bottle",
    "cylinder",
    "圆柱",
    "易拉罐",
    "罐子",
    "can",
)


def is_block_target(target: str | None) -> bool:
    if not target:
        return False
    lowered = target.lower()
    return any(keyword in lowered for keyword in _BLOCK_KEYWORDS)


def is_cylinder_target(target: str | None) -> bool:
    if not target:
        return False
    lowered = target.lower()
    for keyword in _CYLINDER_KEYWORDS:
        if keyword == "can":
            if re.search(r"\bcan\b", lowered):
                return True
        elif keyword in lowered:
            return True
    return False


def is_transparent_bottle_target(target: str | None) -> bool:
    """Return whether the current single transparent-bottle profile applies."""
    if not target:
        return False
    lowered = target.lower()
    return any(
        keyword in lowered
        for keyword in _TRANSPARENT_BOTTLE_KEYWORDS
    )


def uses_candidate_grasp(target: str | None) -> bool:
    return is_block_target(target) or is_cylinder_target(target)
