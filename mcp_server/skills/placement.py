"""Resolve symbolic and geometry-aware object placement targets."""

import math

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
    if key is None:
        raise ValueError(f"Unknown named placement zone: {place}")
    return dict(_PLACE_TABLE[key])


def resolve_place(bridge, place: str | dict, **kwargs) -> SkillResult:
    try:
        return SkillResult.success(**_resolve_coordinates(bridge, place))
    except Exception as exc:
        return SkillResult.failure(
            str(exc),
            failed_step="resolve_place",
            retryable=False,
        )


_RELATIVE_OFFSETS = {
    "right_of": (0.0, -1.0),
    "left_of": (0.0, 1.0),
    "in_front_of": (1.0, 0.0),
    "behind": (-1.0, 0.0),
}


def _point(value: dict, name: str, owner: str) -> tuple[float, float, float]:
    point = value.get(name)
    if not isinstance(point, dict):
        raise ValueError(f"{owner} geometry is missing {name}")
    try:
        result = tuple(float(point[axis]) for axis in ("x", "y", "z"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{owner} geometry {name} must contain numeric x/y/z") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{owner} geometry {name} must be finite")
    return result


def _validated_geometry(value: dict, owner: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{owner} geometry is missing")
    quality = value.get("quality")
    if not isinstance(quality, dict) or quality.get("reliable") is not True:
        raise ValueError(f"{owner} geometry is not marked reliable")
    if str(value.get("shape_kind", "unknown")) == "cylinder":
        raise ValueError(
            f"{owner} cylinder relation placement is not supported yet"
        )
    center = _point(value, "center", owner)
    surface = _point(value, "surface", owner)
    size = _point(value, "size", owner)
    if any(item <= 0.0 for item in size):
        raise ValueError(f"{owner} geometry size must be positive")
    try:
        height = float(value["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{owner} geometry height must be numeric") from exc
    if not math.isfinite(height) or height <= 0.0:
        raise ValueError(f"{owner} geometry height must be positive and finite")
    yaw = float(value.get("yaw_rad") or 0.0)
    if not math.isfinite(yaw):
        raise ValueError(f"{owner} geometry yaw must be finite")
    desk = value.get("local_desk_z")
    if desk is not None:
        desk = float(desk)
        if not math.isfinite(desk):
            raise ValueError(f"{owner} local desk height must be finite")
    return {
        "center": center,
        "surface": surface,
        "size": size,
        "height": height,
        "yaw": yaw,
        "desk": desk,
    }


def _translated_candidate(
    selected_candidate: dict,
    source_center: tuple[float, float, float],
    desired_center: tuple[float, float, float],
) -> dict:
    if not isinstance(selected_candidate, dict):
        raise ValueError("Selected grasp candidate is missing")
    pose = selected_candidate.get("pose_xyz")
    quaternion = selected_candidate.get("pose_quat_xyzw")
    try:
        pose_xyz = tuple(float(item) for item in pose)
        quat = tuple(float(item) for item in quaternion)
    except (TypeError, ValueError) as exc:
        raise ValueError("Selected grasp candidate pose is invalid") from exc
    if len(pose_xyz) != 3 or len(quat) != 4:
        raise ValueError("Selected grasp candidate must contain TCP xyz and quaternion")
    if not all(math.isfinite(item) for item in (*pose_xyz, *quat)):
        raise ValueError("Selected grasp candidate pose must be finite")

    delta = tuple(
        desired - original
        for desired, original in zip(desired_center, source_center)
    )
    target = tuple(position + shift for position, shift in zip(pose_xyz, delta))
    translated = dict(selected_candidate)
    translated["pose_xyz"] = list(target)
    translated["placement_source_center_xyz"] = list(source_center)
    translated["placement_target_center_xyz"] = list(desired_center)
    translated["placement_translation_xyz"] = list(delta)
    return translated


def _projected_half_extent(
    size: tuple[float, float, float],
    yaw: float,
    direction: tuple[float, float],
) -> float:
    local_x = (math.cos(yaw), math.sin(yaw))
    local_y = (-math.sin(yaw), math.cos(yaw))
    return (
        0.5 * size[0] * abs(direction[0] * local_x[0] + direction[1] * local_x[1])
        + 0.5 * size[1] * abs(direction[0] * local_y[0] + direction[1] * local_y[1])
    )


def stack_on(
    bridge,
    source_geometry: dict,
    support_geometry: dict,
    selected_candidate: dict,
    clearance: float | None = None,
    **kwargs,
) -> SkillResult:
    """Compute a translated TCP pose that puts the source on the support."""
    try:
        source = _validated_geometry(source_geometry, "source")
        support = _validated_geometry(support_geometry, "support")
        gap = (
            bridge.get_stack_clearance_m()
            if clearance is None else float(clearance)
        )
        if not math.isfinite(gap) or not 0.0 < gap <= 0.02:
            raise ValueError("Stack clearance must be in (0, 0.02]m")
        max_overhang = float(bridge.get_stack_max_overhang_m())
        if not math.isfinite(max_overhang) or not 0.0 <= max_overhang <= 0.03:
            raise ValueError("Stack maximum overhang must be in [0, 0.03]m")

        support_x = (math.cos(support["yaw"]), math.sin(support["yaw"]))
        support_y = (-support_x[1], support_x[0])
        source_half_x = _projected_half_extent(
            source["size"], source["yaw"], support_x
        )
        source_half_y = _projected_half_extent(
            source["size"], source["yaw"], support_y
        )
        support_half_x = support["size"][0] / 2.0
        support_half_y = support["size"][1] / 2.0
        overhang_x = max(0.0, source_half_x - support_half_x)
        overhang_y = max(0.0, source_half_y - support_half_y)
        if overhang_x > max_overhang or overhang_y > max_overhang:
            raise ValueError(
                "Source footprint is not sufficiently supported: "
                f"overhang=({overhang_x:.3f}, {overhang_y:.3f})m exceeds "
                f"{max_overhang:.3f}m"
            )
        placement_surface_z = support["surface"][2] + gap
        desired_center = (
            support["surface"][0],
            support["surface"][1],
            placement_surface_z + source["height"] / 2.0,
        )
        candidate = _translated_candidate(
            selected_candidate,
            source["center"],
            desired_center,
        )
        candidate["placement_mode"] = "stack"
        return SkillResult.success(
            x=desired_center[0],
            y=desired_center[1],
            z=placement_surface_z,
            object_center={
                "x": desired_center[0],
                "y": desired_center[1],
                "z": desired_center[2],
            },
            placement_candidate=candidate,
            clearance=gap,
            # The release gap is a motion clearance.  After release the object
            # is expected to settle onto the measured support surface.
            expected_surface_z=support["surface"][2] + source["height"],
            support_margin={
                "x": support_half_x - source_half_x,
                "y": support_half_y - source_half_y,
            },
            overhang={"x": overhang_x, "y": overhang_y},
        )
    except (TypeError, ValueError) as exc:
        return SkillResult.failure(
            str(exc),
            failed_step="stack_on",
            retryable=False,
        )


def offset_from(
    bridge,
    source_geometry: dict,
    reference_geometry: dict,
    selected_candidate: dict,
    relation: str,
    clearance: float | None = None,
    **kwargs,
) -> SkillResult:
    """Place beside a reference using both measured planar footprints."""
    try:
        source = _validated_geometry(source_geometry, "source")
        reference = _validated_geometry(reference_geometry, "reference")
        direction = _RELATIVE_OFFSETS.get(str(relation))
        if direction is None:
            raise ValueError(f"Unsupported relative placement relation: {relation}")
        gap = (
            bridge.get_relative_placement_clearance_m()
            if clearance is None else float(clearance)
        )
        if not math.isfinite(gap) or not 0.0 < gap <= 0.10:
            raise ValueError("Relative placement clearance must be in (0, 0.10]m")
        if source["desk"] is None or reference["desk"] is None:
            raise ValueError("Relative placement requires local desk heights")
        if abs(source["desk"] - reference["desk"]) > 0.03:
            raise ValueError("Source and reference desk heights are inconsistent")
        separation = (
            _projected_half_extent(reference["size"], reference["yaw"], direction)
            + _projected_half_extent(source["size"], source["yaw"], direction)
            + gap
        )
        desired_center = (
            reference["center"][0] + direction[0] * separation,
            reference["center"][1] + direction[1] * separation,
            reference["desk"] + source["height"] / 2.0,
        )
        candidate = _translated_candidate(
            selected_candidate,
            source["center"],
            desired_center,
        )
        candidate["placement_mode"] = str(relation)
        return SkillResult.success(
            x=desired_center[0],
            y=desired_center[1],
            z=reference["desk"],
            object_center={
                "x": desired_center[0],
                "y": desired_center[1],
                "z": desired_center[2],
            },
            separation=separation,
            clearance=gap,
            placement_candidate=candidate,
            expected_surface_z=reference["desk"] + source["height"],
        )
    except (TypeError, ValueError) as exc:
        return SkillResult.failure(
            str(exc),
            failed_step="offset_from",
            retryable=False,
        )


def verify_placement(
    bridge,
    observed_geometry: dict,
    expected_x: float,
    expected_y: float,
    expected_surface_z: float,
    xy_tolerance: float | None = None,
    z_tolerance: float | None = None,
    **kwargs,
) -> SkillResult:
    """Verify the released object's measured pose against the task postcondition."""
    try:
        observed = _validated_geometry(observed_geometry, "observed object")
        target_x = float(expected_x)
        target_y = float(expected_y)
        target_surface_z = float(expected_surface_z)
        if not all(math.isfinite(value) for value in (
            target_x, target_y, target_surface_z
        )):
            raise ValueError("Expected placement coordinates must be finite")
        xy_limit = (
            float(bridge.get_placement_verify_xy_tolerance_m())
            if xy_tolerance is None else float(xy_tolerance)
        )
        z_limit = (
            float(bridge.get_placement_verify_z_tolerance_m())
            if z_tolerance is None else float(z_tolerance)
        )
        if not math.isfinite(xy_limit) or not 0.0 < xy_limit <= 0.10:
            raise ValueError("Placement XY tolerance must be in (0, 0.10]m")
        if not math.isfinite(z_limit) or not 0.0 < z_limit <= 0.10:
            raise ValueError("Placement Z tolerance must be in (0, 0.10]m")

        dx = observed["surface"][0] - target_x
        dy = observed["surface"][1] - target_y
        dz = observed["surface"][2] - target_surface_z
        xy_error = math.hypot(dx, dy)
        if xy_error > xy_limit or abs(dz) > z_limit:
            raise ValueError(
                "Released object does not satisfy the placement postcondition: "
                f"xy_error={xy_error:.3f}m (limit {xy_limit:.3f}m), "
                f"surface_z_error={dz:+.3f}m (limit {z_limit:.3f}m)"
            )
        return SkillResult.success(
            verified=True,
            observed_x=observed["surface"][0],
            observed_y=observed["surface"][1],
            observed_surface_z=observed["surface"][2],
            xy_error=xy_error,
            surface_z_error=dz,
        )
    except (TypeError, ValueError) as exc:
        return SkillResult.failure(
            str(exc),
            failed_step="post_place_verification",
            retryable=False,
            holding=False,
        )
