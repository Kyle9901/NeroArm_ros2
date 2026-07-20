"""Pure metric cylinder fitting for live RGB-D point clouds."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..models import ObjectGeometry


def _quantile_span(values: np.ndarray) -> tuple[float, float, float]:
    low, high = np.quantile(values, (0.02, 0.98))
    return float(low), float(high), float(high - low)


def _canonical_axis(axis: np.ndarray, orientation: str) -> np.ndarray:
    result = np.asarray(axis, dtype=np.float64)
    result /= np.linalg.norm(result)
    if orientation == "upright":
        if result[2] < 0.0:
            result = -result
        return result
    result[2] = 0.0
    result /= np.linalg.norm(result)
    if result[0] < -1e-9 or (
        abs(float(result[0])) <= 1e-9 and result[1] < 0.0
    ):
        result = -result
    return result


def _fit_circle(points_2d: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Robust algebraic circle fit with residual trimming."""

    points = np.asarray(points_2d, dtype=np.float64)
    if len(points) < 30:
        raise ValueError("cylinder cross-section has fewer than 30 depth points")
    selected = points
    center = None
    radius = None
    for _ in range(4):
        design = np.column_stack((
            2.0 * selected[:, 0],
            2.0 * selected[:, 1],
            np.ones(len(selected)),
        ))
        target = np.sum(selected * selected, axis=1)
        solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank < 3:
            raise ValueError("cylinder circle fit is degenerate")
        center = solution[:2]
        radius_sq = float(solution[2] + center @ center)
        if radius_sq <= 0.0:
            raise ValueError("cylinder circle fit returned a non-positive radius")
        radius = math.sqrt(radius_sq)
        residuals = np.abs(
            np.linalg.norm(points - center, axis=1) - radius
        )
        keep_limit = float(np.quantile(residuals, 0.75))
        selected = points[residuals <= max(keep_limit, 0.001)]
        if len(selected) < 30:
            break
    residuals = np.abs(
        np.linalg.norm(points - center, axis=1) - radius
    )
    return center, float(radius), float(np.median(residuals))


def fit_cylinder_geometry(
    points_base: Sequence[Sequence[float]],
    *,
    local_desk_z: float,
    min_diameter_m: float = 0.035,
    max_diameter_m: float = 0.085,
    min_length_m: float = 0.08,
    max_length_m: float = 0.30,
    lying_axis_max_deviation_deg: float = 30.0,
    surface_depth_mm: float | None = None,
) -> ObjectGeometry:
    """Fit either a desk-supported upright or lying circular cylinder.

    Only measured depth points are accepted. Ambiguous diagonal axes, poor
    circle fits and dimensions outside the configured envelope are rejected.
    """

    points = np.asarray(points_base, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("cylinder points must have shape Nx3")
    points = points[np.all(np.isfinite(points), axis=1)]
    heights = points[:, 2] - float(local_desk_z)
    points = points[(heights >= 0.002) & (heights <= max_length_m + 0.03)]
    if len(points) < 80:
        raise ValueError(
            f"cylinder has only {len(points)} valid points above the desk "
            "(need 80)"
        )

    centroid = np.median(points, axis=0)
    covariance = np.cov(points - centroid, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    dominant = eigenvectors[:, order[0]]
    if eigenvalues[0] <= 1e-10 or eigenvalues[1] <= 1e-12:
        raise ValueError("cylinder point cloud is degenerate")
    axis_verticality = abs(float(dominant[2]))
    lying_verticality_limit = math.sin(math.radians(
        float(lying_axis_max_deviation_deg)
    ))
    if not 0.0 < float(lying_axis_max_deviation_deg) <= 45.0:
        raise ValueError(
            "lying_axis_max_deviation_deg must be in (0, 45]"
        )
    if axis_verticality >= 0.75:
        orientation = "upright"
    elif axis_verticality <= lying_verticality_limit:
        orientation = "lying"
    else:
        raise ValueError(
            f"cylinder axis is diagonal (|axis_z|={axis_verticality:.2f}); "
            f"lying limit is {lying_verticality_limit:.2f}"
        )
    axis = _canonical_axis(dominant, orientation)

    if orientation == "upright":
        z_low, z_high, observed_height = _quantile_span(points[:, 2])
        # A standing target is supported by the measured local desk. Using the
        # desk contact avoids treating an occluded lower rim as the bottom.
        length = z_high - float(local_desk_z)
        if observed_height < min_length_m * 0.55:
            raise ValueError(
                f"upright cylinder exposes only {observed_height:.3f}m of "
                "vertical depth support"
            )
        middle = points[
            (points[:, 2] >= float(local_desk_z) + length * 0.20)
            & (points[:, 2] <= float(local_desk_z) + length * 0.80)
        ]
        circle_center, radius, circle_residual = _fit_circle(middle[:, :2])
        center = np.asarray([
            circle_center[0],
            circle_center[1],
            float(local_desk_z) + length / 2.0,
        ])
        axis = np.asarray([0.0, 0.0, 1.0])
    else:
        horizontal_axis = axis
        cross_axis = np.asarray([
            -horizontal_axis[1], horizontal_axis[0], 0.0,
        ])
        along = points @ horizontal_axis
        across = points @ cross_axis
        along_low, along_high, length = _quantile_span(along)
        cross_section = np.column_stack((across, points[:, 2]))
        circle_center, radius, circle_residual = _fit_circle(cross_section)
        contact_error = abs(
            float(circle_center[1]) - radius - float(local_desk_z)
        )
        if contact_error > 0.015:
            raise ValueError(
                f"lying cylinder circle misses desk contact by "
                f"{contact_error:.3f}m"
            )
        along_center = (along_low + along_high) / 2.0
        center = (
            horizontal_axis * along_center
            + cross_axis * float(circle_center[0])
        )
        center[2] = float(local_desk_z) + radius

    diameter = 2.0 * radius
    if not min_diameter_m <= diameter <= max_diameter_m:
        raise ValueError(
            f"cylinder diameter {diameter:.3f}m is outside "
            f"[{min_diameter_m:.3f}, {max_diameter_m:.3f}]m"
        )
    if not min_length_m <= length <= max_length_m:
        raise ValueError(
            f"cylinder length {length:.3f}m is outside "
            f"[{min_length_m:.3f}, {max_length_m:.3f}]m"
        )
    maximum_circle_residual = max(0.004, diameter * 0.08)
    if circle_residual > maximum_circle_residual:
        raise ValueError(
            f"cylinder circle residual {circle_residual:.4f}m exceeds "
            f"{maximum_circle_residual:.4f}m"
        )
    aspect_ratio = length / diameter
    if aspect_ratio < 1.15:
        raise ValueError(
            f"cylinder aspect ratio {aspect_ratio:.2f} is too close to a sphere"
        )

    vertical_height = length if orientation == "upright" else diameter
    surface = (
        float(center[0]),
        float(center[1]),
        float(local_desk_z) + vertical_height,
    )
    yaw = (
        0.0
        if orientation == "upright"
        else math.atan2(float(axis[1]), float(axis[0]))
    )
    quality = {
        "reliable": True,
        "valid_depth_points": int(len(points)),
        "circle_residual_median_mm": circle_residual * 1000.0,
        "axis_verticality": axis_verticality,
        "pca_axis_ratio": float(eigenvalues[0] / eigenvalues[1]),
        "aspect_ratio": aspect_ratio,
    }
    return ObjectGeometry(
        surface_xyz=surface,
        center_xyz=tuple(float(value) for value in center),
        size_xyz=(diameter, diameter, length),
        local_desk_z=float(local_desk_z),
        height=vertical_height,
        height_source="realtime_depth_cylinder_fit",
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
        quality=quality,
    )
