"""Explicit object geometry shared by perception and manipulation."""

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class GeometryQuality:
    """Cross-frame consistency evidence for a metric object estimate."""

    requested_frames: int
    valid_frames: int
    inlier_frames: int
    reliable: bool
    position_spread_m: float | None = None
    height_spread_m: float | None = None
    size_spread_m: float | None = None
    desk_spread_m: float | None = None
    yaw_spread_rad: float | None = None
    depth_spread_mm: float | None = None
    rejection_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "requested_frames": self.requested_frames,
            "valid_frames": self.valid_frames,
            "inlier_frames": self.inlier_frames,
            "reliable": self.reliable,
            "position_spread_m": self.position_spread_m,
            "height_spread_m": self.height_spread_m,
            "size_spread_m": self.size_spread_m,
            "desk_spread_m": self.desk_spread_m,
            "yaw_spread_rad": self.yaw_spread_rad,
            "depth_spread_mm": self.depth_spread_mm,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class ObjectGeometry:
    """Object pose semantics in the base frame.

    ``surface_xyz`` is the measured top-surface grasp reference. ``size_xyz``
    stores lengths along the two planar object axes followed by height; it is
    not an axis-aligned base-frame bounding box.
    """

    surface_xyz: tuple[float, float, float]
    center_xyz: tuple[float, float, float] | None = None
    size_xyz: tuple[float, float, float] | None = None
    local_desk_z: float | None = None
    height: float | None = None
    height_source: str = "unknown"
    yaw_rad: float | None = None
    yaw_period_rad: float = math.pi
    primary_axis_xy: tuple[float, float] | None = None
    secondary_axis_xy: tuple[float, float] | None = None
    surface_depth_mm: float | None = None
    shape_kind: str = "unknown"
    axis_xyz: tuple[float, float, float] | None = None
    diameter_m: float | None = None
    length_m: float | None = None
    orientation_class: str | None = None
    quality: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        def xyz(value):
            if value is None:
                return None
            return {"x": value[0], "y": value[1], "z": value[2]}

        return {
            "surface": xyz(self.surface_xyz),
            "center": xyz(self.center_xyz),
            "size": xyz(self.size_xyz),
            "local_desk_z": self.local_desk_z,
            "height": self.height,
            "height_source": self.height_source,
            "yaw_rad": self.yaw_rad,
            "yaw_period_rad": self.yaw_period_rad,
            "primary_axis_xy": (
                list(self.primary_axis_xy) if self.primary_axis_xy is not None else None
            ),
            "secondary_axis_xy": (
                list(self.secondary_axis_xy) if self.secondary_axis_xy is not None else None
            ),
            "surface_depth_mm": self.surface_depth_mm,
            "shape_kind": self.shape_kind,
            "axis_xyz": (
                list(self.axis_xyz) if self.axis_xyz is not None else None
            ),
            "diameter_m": self.diameter_m,
            "length_m": self.length_m,
            "orientation_class": self.orientation_class,
            "quality": dict(self.quality),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "ObjectGeometry":
        def xyz(name):
            point = value.get(name)
            if point is None:
                return None
            return (float(point["x"]), float(point["y"]), float(point["z"]))

        def xy(name):
            axis = value.get(name)
            return None if axis is None else (float(axis[0]), float(axis[1]))

        return cls(
            surface_xyz=xyz("surface"),
            center_xyz=xyz("center"),
            size_xyz=xyz("size"),
            local_desk_z=value.get("local_desk_z"),
            height=value.get("height"),
            height_source=value.get("height_source", "unknown"),
            yaw_rad=value.get("yaw_rad"),
            yaw_period_rad=float(value.get("yaw_period_rad", math.pi)),
            primary_axis_xy=xy("primary_axis_xy"),
            secondary_axis_xy=xy("secondary_axis_xy"),
            surface_depth_mm=value.get("surface_depth_mm"),
            shape_kind=str(value.get("shape_kind", "unknown")),
            axis_xyz=(
                None
                if value.get("axis_xyz") is None
                else tuple(float(item) for item in value["axis_xyz"])
            ),
            diameter_m=(
                None
                if value.get("diameter_m") is None
                else float(value["diameter_m"])
            ),
            length_m=(
                None
                if value.get("length_m") is None
                else float(value["length_m"])
            ),
            orientation_class=value.get("orientation_class"),
            quality=dict(value.get("quality") or {}),
        )


@dataclass(frozen=True)
class GeometryAggregation:
    """Pure aggregation result; unreliable data never contains a geometry."""

    geometry: ObjectGeometry | None
    quality: GeometryQuality


def _periodic_mean_and_spread(
    angles: np.ndarray,
    period: float,
) -> tuple[float, float]:
    """Circular mean and maximum wrapped deviation for an undirected axis."""
    phase = angles * (2.0 * math.pi / period)
    vector = np.mean(np.exp(1j * phase))
    if abs(vector) <= 1e-12:
        mean = float(np.median(np.mod(angles, period)))
    else:
        mean = (math.atan2(vector.imag, vector.real) % (2.0 * math.pi))
        mean *= period / (2.0 * math.pi)
    wrapped = np.abs((angles - mean + period / 2.0) % period - period / 2.0)
    return mean, float(np.max(wrapped))


def aggregate_object_geometries(
    observations: list[ObjectGeometry],
    *,
    requested_frames: int = 5,
    min_valid_frames: int = 3,
    max_position_deviation_m: float = 0.015,
    max_height_deviation_m: float = 0.015,
    max_size_deviation_m: float | None = None,
    max_desk_deviation_m: float | None = None,
    max_yaw_deviation_rad: float = math.radians(15.0),
    max_depth_deviation_mm: float = 15.0,
) -> GeometryAggregation:
    """Median-fuse real-time observations and reject inconsistent sequences.

    The function contains no ROS/OpenCV dependencies and is deliberately easy
    to unit test.  A five-frame request may tolerate up to two missing/bad
    frames, but at least ``min_valid_frames`` mutually consistent inliers are
    required before any motion layer receives geometry.
    """
    valid = [
        item for item in observations
        if item is not None
        and len(item.surface_xyz) == 3
        and all(math.isfinite(float(value)) for value in item.surface_xyz)
        and item.height is not None
        and math.isfinite(float(item.height))
        and float(item.height) > 0.0
        and item.yaw_rad is not None
        and math.isfinite(float(item.yaw_rad))
        and item.size_xyz is not None
        and all(math.isfinite(float(value)) and float(value) > 0.0
                for value in item.size_xyz)
    ]
    reasons: list[str] = []
    if len(valid) < min_valid_frames:
        reasons.append(
            f"only {len(valid)}/{requested_frames} valid geometry frames "
            f"(need {min_valid_frames})"
        )
        quality = GeometryQuality(
            requested_frames=requested_frames,
            valid_frames=len(valid),
            inlier_frames=0,
            reliable=False,
            rejection_reasons=tuple(reasons),
        )
        return GeometryAggregation(None, quality)

    surfaces = np.asarray([item.surface_xyz for item in valid], dtype=np.float64)
    heights = np.asarray([item.height for item in valid], dtype=np.float64)
    sizes = np.asarray([item.size_xyz for item in valid], dtype=np.float64)
    desks = np.asarray([
        item.local_desk_z
        if item.local_desk_z is not None else float("nan")
        for item in valid
    ], dtype=np.float64)
    depths = np.asarray([
        item.surface_depth_mm
        if item.surface_depth_mm is not None else float("nan")
        for item in valid
    ], dtype=np.float64)
    surface_median = np.median(surfaces, axis=0)
    height_median = float(np.median(heights))
    # XY stability and depth/height stability are separate gates. Combining
    # XYZ here made two individually acceptable deviations reject each other
    # through Euclidean norm despite the parameter being named ``xy``.
    position_deviation = np.linalg.norm(
        surfaces[:, :2] - surface_median[:2],
        axis=1,
    )
    height_deviation = np.abs(heights - height_median)
    planar_size_median = np.median(sizes[:, :2], axis=0)
    size_deviation = np.max(
        np.abs(sizes[:, :2] - planar_size_median),
        axis=1,
    )
    max_size_deviation_m = (
        max_height_deviation_m
        if max_size_deviation_m is None
        else float(max_size_deviation_m)
    )
    max_desk_deviation_m = (
        max_height_deviation_m
        if max_desk_deviation_m is None
        else float(max_desk_deviation_m)
    )
    if np.any(np.isfinite(desks)):
        desk_median = float(np.nanmedian(desks))
        desk_deviation = np.abs(desks - desk_median)
    else:
        desk_deviation = np.zeros(len(valid), dtype=np.float64)

    yaw_period = float(np.median([item.yaw_period_rad for item in valid]))
    yaw_values = np.asarray([item.yaw_rad for item in valid], dtype=np.float64)
    yaw_median, _ = _periodic_mean_and_spread(yaw_values, yaw_period)
    yaw_deviation = np.abs(
        (yaw_values - yaw_median + yaw_period / 2.0) % yaw_period
        - yaw_period / 2.0
    )
    if np.any(np.isfinite(depths)):
        depth_median = float(np.nanmedian(depths))
        depth_deviation = np.abs(depths - depth_median)
    else:
        depth_median = float("nan")
        depth_deviation = np.zeros(len(valid), dtype=np.float64)

    inliers = (
        (position_deviation <= max_position_deviation_m)
        & (height_deviation <= max_height_deviation_m)
        & (size_deviation <= max_size_deviation_m)
        & (
            ~np.isfinite(desks)
            | (desk_deviation <= max_desk_deviation_m)
        )
        & (yaw_deviation <= max_yaw_deviation_rad)
        & (
            ~np.isfinite(depths)
            | (depth_deviation <= max_depth_deviation_mm)
        )
    )
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < min_valid_frames:
        reasons.append(
            f"only {inlier_count}/{len(valid)} mutually consistent frames "
            f"(need {min_valid_frames})"
        )

    selected = [item for item, keep in zip(valid, inliers) if bool(keep)]
    if not selected:
        selected = valid

    selected_surfaces = np.asarray(
        [item.surface_xyz for item in selected], dtype=np.float64,
    )
    selected_heights = np.asarray(
        [item.height for item in selected], dtype=np.float64,
    )
    selected_sizes = np.asarray(
        [item.size_xyz for item in selected], dtype=np.float64,
    )
    selected_yaws = np.asarray(
        [item.yaw_rad for item in selected], dtype=np.float64,
    )
    median_surface = np.median(selected_surfaces, axis=0)
    height = float(np.median(selected_heights))
    median_size = np.median(selected_sizes, axis=0)
    yaw_rad, yaw_spread = _periodic_mean_and_spread(selected_yaws, yaw_period)
    position_spread = float(np.max(np.linalg.norm(
        selected_surfaces[:, :2]
        - np.median(selected_surfaces[:, :2], axis=0),
        axis=1,
    )))
    height_spread = float(np.max(np.abs(
        selected_heights - np.median(selected_heights),
    )))
    size_spread = float(np.max(np.abs(
        selected_sizes[:, :2] - np.median(selected_sizes[:, :2], axis=0),
    )))

    selected_depths = np.asarray([
        item.surface_depth_mm
        if item.surface_depth_mm is not None else float("nan")
        for item in selected
    ], dtype=np.float64)
    if np.any(np.isfinite(selected_depths)):
        surface_depth_mm = float(np.nanmedian(selected_depths))
        depth_spread = float(np.nanmax(np.abs(
            selected_depths - surface_depth_mm,
        )))
    else:
        surface_depth_mm = None
        depth_spread = None

    local_desks = [
        float(item.local_desk_z) for item in selected
        if item.local_desk_z is not None
    ]
    local_desk_z = float(np.median(local_desks)) if local_desks else None
    desk_spread = (
        float(np.max(np.abs(
            np.asarray(local_desks) - np.median(local_desks),
        )))
        if local_desks else None
    )
    surface_z = (
        local_desk_z + height
        if local_desk_z is not None
        else float(median_surface[2])
    )
    surface_xyz = (
        float(median_surface[0]),
        float(median_surface[1]),
        surface_z,
    )
    center_xyz = (
        surface_xyz[0],
        surface_xyz[1],
        (
            local_desk_z + height / 2.0
            if local_desk_z is not None
            else surface_z - height / 2.0
        ),
    )
    size_xyz = (
        float(median_size[0]),
        float(median_size[1]),
        height,
    )
    primary_axis_xy = (math.cos(yaw_rad), math.sin(yaw_rad))
    secondary_axis_xy = (-math.sin(yaw_rad), math.cos(yaw_rad))

    reliable = not reasons
    quality = GeometryQuality(
        requested_frames=requested_frames,
        valid_frames=len(valid),
        inlier_frames=inlier_count,
        reliable=reliable,
        position_spread_m=position_spread,
        height_spread_m=height_spread,
        size_spread_m=size_spread,
        desk_spread_m=desk_spread,
        yaw_spread_rad=yaw_spread,
        depth_spread_mm=depth_spread,
        rejection_reasons=tuple(reasons),
    )
    if not reliable:
        return GeometryAggregation(None, quality)

    geometry = ObjectGeometry(
        surface_xyz=surface_xyz,
        center_xyz=center_xyz,
        size_xyz=size_xyz,
        local_desk_z=local_desk_z,
        height=height,
        height_source=f"realtime_depth_{requested_frames}frame_median",
        yaw_rad=yaw_rad,
        yaw_period_rad=yaw_period,
        primary_axis_xy=primary_axis_xy,
        secondary_axis_xy=secondary_axis_xy,
        surface_depth_mm=surface_depth_mm,
        quality=quality.to_dict(),
    )
    return GeometryAggregation(geometry, quality)


def aggregate_cylinder_geometries(
    observations: list[ObjectGeometry],
    *,
    requested_frames: int = 5,
    min_valid_frames: int = 3,
    max_position_deviation_m: float = 0.015,
    max_dimension_deviation_m: float = 0.015,
    max_desk_deviation_m: float = 0.015,
    max_axis_deviation_rad: float = math.radians(12.0),
    max_depth_deviation_mm: float = 15.0,
) -> GeometryAggregation:
    """Fuse live upright/lying cylinder fits without guessing missing depth."""

    valid = [
        item
        for item in observations
        if item is not None
        and item.shape_kind == "cylinder"
        and item.center_xyz is not None
        and item.axis_xyz is not None
        and item.diameter_m is not None
        and item.length_m is not None
        and item.orientation_class in {"upright", "lying"}
        and all(math.isfinite(float(value)) for value in item.center_xyz)
        and all(math.isfinite(float(value)) for value in item.axis_xyz)
        and math.isfinite(float(item.diameter_m))
        and math.isfinite(float(item.length_m))
        and float(item.diameter_m) > 0.0
        and float(item.length_m) > 0.0
    ]
    reasons: list[str] = []
    if len(valid) < min_valid_frames:
        reasons.append(
            f"only {len(valid)}/{requested_frames} valid cylinder frames "
            f"(need {min_valid_frames})"
        )
        quality = GeometryQuality(
            requested_frames=requested_frames,
            valid_frames=len(valid),
            inlier_frames=0,
            reliable=False,
            rejection_reasons=tuple(reasons),
        )
        return GeometryAggregation(None, quality)

    orientation_counts = {
        orientation: sum(
            item.orientation_class == orientation for item in valid
        )
        for orientation in ("upright", "lying")
    }
    orientation = max(orientation_counts, key=orientation_counts.get)
    same_orientation = [
        item for item in valid if item.orientation_class == orientation
    ]
    if len(same_orientation) < min_valid_frames:
        reasons.append(
            "upright/lying classification is not stable across frames"
        )

    centers = np.asarray(
        [item.center_xyz for item in same_orientation], dtype=np.float64,
    )
    axes = np.asarray(
        [item.axis_xyz for item in same_orientation], dtype=np.float64,
    )
    diameters = np.asarray(
        [item.diameter_m for item in same_orientation], dtype=np.float64,
    )
    lengths = np.asarray(
        [item.length_m for item in same_orientation], dtype=np.float64,
    )
    desks = np.asarray([
        item.local_desk_z
        if item.local_desk_z is not None else float("nan")
        for item in same_orientation
    ], dtype=np.float64)
    depths = np.asarray([
        item.surface_depth_mm
        if item.surface_depth_mm is not None else float("nan")
        for item in same_orientation
    ], dtype=np.float64)

    # Cylinder axes are undirected. Align their signs before taking a robust
    # component-wise median.
    reference = axes[0] / np.linalg.norm(axes[0])
    axes = np.asarray([
        axis / np.linalg.norm(axis)
        * (-1.0 if float(axis @ reference) < 0.0 else 1.0)
        for axis in axes
    ])
    axis_median = np.median(axes, axis=0)
    axis_median /= np.linalg.norm(axis_median)
    center_median = np.median(centers, axis=0)
    diameter_median = float(np.median(diameters))
    length_median = float(np.median(lengths))
    desk_median = (
        float(np.nanmedian(desks))
        if np.any(np.isfinite(desks)) else None
    )
    depth_median = (
        float(np.nanmedian(depths))
        if np.any(np.isfinite(depths)) else None
    )

    position_deviation = np.linalg.norm(
        centers - center_median, axis=1,
    )
    diameter_deviation = np.abs(diameters - diameter_median)
    length_deviation = np.abs(lengths - length_median)
    axis_deviation = np.arccos(np.clip(
        np.abs(axes @ axis_median), 0.0, 1.0,
    ))
    desk_deviation = (
        np.abs(desks - desk_median)
        if desk_median is not None else np.zeros(len(centers))
    )
    depth_deviation = (
        np.abs(depths - depth_median)
        if depth_median is not None else np.zeros(len(centers))
    )
    inliers = (
        (position_deviation <= max_position_deviation_m)
        & (diameter_deviation <= max_dimension_deviation_m)
        & (length_deviation <= max_dimension_deviation_m)
        & (axis_deviation <= max_axis_deviation_rad)
        & (
            ~np.isfinite(desks)
            | (desk_deviation <= max_desk_deviation_m)
        )
        & (
            ~np.isfinite(depths)
            | (depth_deviation <= max_depth_deviation_mm)
        )
    )
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < min_valid_frames:
        reasons.append(
            f"only {inlier_count}/{len(same_orientation)} mutually "
            f"consistent cylinder frames (need {min_valid_frames})"
        )
    selected = [
        item for item, keep in zip(same_orientation, inliers) if bool(keep)
    ]
    if not selected:
        selected = same_orientation

    selected_centers = np.asarray(
        [item.center_xyz for item in selected], dtype=np.float64,
    )
    selected_axes = np.asarray(
        [item.axis_xyz for item in selected], dtype=np.float64,
    )
    reference = selected_axes[0] / np.linalg.norm(selected_axes[0])
    selected_axes = np.asarray([
        axis / np.linalg.norm(axis)
        * (-1.0 if float(axis @ reference) < 0.0 else 1.0)
        for axis in selected_axes
    ])
    axis = np.median(selected_axes, axis=0)
    axis /= np.linalg.norm(axis)
    if orientation == "upright":
        axis = np.asarray([0.0, 0.0, 1.0])
    else:
        axis[2] = 0.0
        axis /= np.linalg.norm(axis)
        if axis[0] < -1e-9 or (
            abs(float(axis[0])) <= 1e-9 and axis[1] < 0.0
        ):
            axis = -axis

    center = np.median(selected_centers, axis=0)
    diameter = float(np.median([
        item.diameter_m for item in selected
    ]))
    length = float(np.median([
        item.length_m for item in selected
    ]))
    selected_desks = [
        float(item.local_desk_z) for item in selected
        if item.local_desk_z is not None
    ]
    local_desk_z = (
        float(np.median(selected_desks)) if selected_desks else None
    )
    if local_desk_z is not None:
        center[2] = local_desk_z + (
            length / 2.0 if orientation == "upright" else diameter / 2.0
        )
    vertical_height = length if orientation == "upright" else diameter
    surface_xyz = (
        float(center[0]),
        float(center[1]),
        float(
            local_desk_z + vertical_height
            if local_desk_z is not None
            else center[2] + vertical_height / 2.0
        ),
    )
    selected_depths = [
        float(item.surface_depth_mm) for item in selected
        if item.surface_depth_mm is not None
    ]
    surface_depth_mm = (
        float(np.median(selected_depths)) if selected_depths else None
    )
    position_spread = float(np.max(np.linalg.norm(
        selected_centers - np.median(selected_centers, axis=0),
        axis=1,
    )))
    diameter_spread = float(np.max(np.abs(
        np.asarray([item.diameter_m for item in selected]) - diameter,
    )))
    length_spread = float(np.max(np.abs(
        np.asarray([item.length_m for item in selected]) - length,
    )))
    axis_spread = float(np.max(np.arccos(np.clip(
        np.abs(selected_axes @ axis), 0.0, 1.0,
    ))))
    desk_spread = (
        float(np.max(np.abs(
            np.asarray(selected_desks) - np.median(selected_desks),
        )))
        if selected_desks else None
    )
    depth_spread = (
        float(np.max(np.abs(
            np.asarray(selected_depths) - np.median(selected_depths),
        )))
        if selected_depths else None
    )
    reliable = not reasons
    quality = GeometryQuality(
        requested_frames=requested_frames,
        valid_frames=len(valid),
        inlier_frames=inlier_count,
        reliable=reliable,
        position_spread_m=position_spread,
        height_spread_m=length_spread,
        size_spread_m=diameter_spread,
        desk_spread_m=desk_spread,
        yaw_spread_rad=axis_spread,
        depth_spread_mm=depth_spread,
        rejection_reasons=tuple(reasons),
    )
    if not reliable:
        return GeometryAggregation(None, quality)

    yaw = (
        0.0
        if orientation == "upright"
        else math.atan2(float(axis[1]), float(axis[0]))
    )
    geometry = ObjectGeometry(
        surface_xyz=surface_xyz,
        center_xyz=tuple(float(value) for value in center),
        size_xyz=(diameter, diameter, length),
        local_desk_z=local_desk_z,
        height=vertical_height,
        height_source=f"realtime_cylinder_{requested_frames}frame_median",
        yaw_rad=yaw,
        yaw_period_rad=math.pi,
        primary_axis_xy=(float(axis[0]), float(axis[1])),
        secondary_axis_xy=(-float(axis[1]), float(axis[0])),
        surface_depth_mm=surface_depth_mm,
        shape_kind="cylinder",
        axis_xyz=tuple(float(value) for value in axis),
        diameter_m=diameter,
        length_m=length,
        orientation_class=orientation,
        quality=quality.to_dict(),
    )
    return GeometryAggregation(geometry, quality)
