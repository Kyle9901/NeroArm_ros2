"""Sparse-depth geometry for one calibrated transparent water bottle.

This module deliberately does not reconstruct a cylinder.  It uses only
measured points that stand above a locally fitted desk, then isolates the
central label support.  The pure NumPy implementation is shared by the live
diagnostic and its hardware-free unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TransparentBottleAnalysis:
    orientation: str
    height_p90_m: float
    height_p95_m: float
    reliable_count: int
    label_count: int
    label_surface_xyz: tuple[float, float, float]
    cap_or_axis_xy: tuple[float, float]
    horizontal_axis_xy: tuple[float, float]
    horizontal_span_m: float
    tcp_xyz: tuple[float, float, float]
    reliable_mask: np.ndarray
    label_mask: np.ndarray
    cap_mask: np.ndarray
    image_axis_uv: tuple[float, float]
    image_axis_center_uv: tuple[float, float]
    label_axis_half_span_px: float
    connected_ratio: float


@dataclass(frozen=True)
class TransparentBottleConsensus:
    """Quality-filtered multi-frame pose and TCP consensus."""

    ready: bool
    orientation: str
    inlier_indices: tuple[int, ...]
    height_p90_m: float | None
    height_p95_m: float | None
    label_gate: int
    tcp_xyz: tuple[float, float, float] | None
    tcp_spread_m: float | None
    reason: str


def classify_pose_from_height_percentiles(
    height_p90_m: float,
    height_p95_m: float,
    *,
    upright_min_p90_m: float,
    upright_min_p95_m: float,
    lying_min_p90_m: float,
    lying_min_p95_m: float,
    lying_max_p90_m: float,
    lying_max_p95_m: float,
) -> str:
    """Classify only from robust height percentiles, with a safety gap."""

    p90 = float(height_p90_m)
    p95 = float(height_p95_m)
    if p90 >= upright_min_p90_m and p95 >= upright_min_p95_m:
        return "upright"
    if (
        p90 >= lying_min_p90_m
        and p95 >= lying_min_p95_m
        and p90 <= lying_max_p90_m
        and p95 <= lying_max_p95_m
    ):
        return "lying"
    return "ambiguous"


def aggregate_transparent_bottle_measurements(
    measurements: list[dict],
    *,
    minimum_frames: int,
    minimum_label_points: int,
    maximum_tcp_spread_m: float,
    upright_min_p90_m: float,
    upright_min_p95_m: float,
    lying_min_p90_m: float,
    lying_min_p95_m: float,
    lying_max_p90_m: float,
    lying_max_p95_m: float,
) -> TransparentBottleConsensus:
    """Fuse frames with orientation-appropriate measured-depth support.

    Low-return transparent frames are excluded before pose percentiles are
    computed. This prevents a standing bottle's sparse bottom reflections from
    voting for the lying class.
    """

    required = max(1, int(minimum_frames))
    absolute_gate = max(3, int(minimum_label_points))
    if not measurements:
        return TransparentBottleConsensus(
            False, "unknown", (), None, None, absolute_gate, None, None,
            f"no measurable frames (need {required})",
        )

    upright_candidates = tuple(
        index for index, item in enumerate(measurements)
        if item.get("orientation") == "upright"
        and int(item.get("label_depth_points", 0)) >= absolute_gate
        and item.get("reliable_heights_m")
    )
    best_upright_count = max(
        (
            int(measurements[index].get("label_depth_points", 0))
            for index in upright_candidates
        ),
        default=0,
    )
    upright_gate = max(
        absolute_gate,
        math.ceil(best_upright_count * 0.70),
    )
    upright_eligible = tuple(
        index for index in upright_candidates
        if int(measurements[index].get("label_depth_points", 0))
        >= upright_gate
    )
    # A lying bottle is accepted only after entering the configured physical
    # height band. It therefore does not need to reproduce the best frame's
    # label return density: transparent plastic often alternates between a
    # dense frame and sparse measured patches while remaining stationary.
    lying_eligible = tuple(
        index for index, item in enumerate(measurements)
        if item.get("orientation") == "lying"
        and int(item.get("label_depth_points", 0)) >= absolute_gate
        and item.get("reliable_heights_m")
    )
    if len(upright_eligible) >= required:
        eligible = upright_eligible
        label_gate = upright_gate
    elif len(lying_eligible) >= required:
        eligible = lying_eligible
        label_gate = absolute_gate
    elif len(upright_eligible) >= len(lying_eligible):
        eligible = upright_eligible
        label_gate = upright_gate
    else:
        eligible = lying_eligible
        label_gate = absolute_gate
    if len(eligible) < required:
        return TransparentBottleConsensus(
            False, "unknown", eligible, None, None, label_gate, None, None,
            f"only {len(eligible)} quality frames (need {required}, "
            f"label gate {label_gate})",
        )

    pooled_heights = np.concatenate([
        np.asarray(
            measurements[index]["reliable_heights_m"], dtype=np.float64,
        )
        for index in eligible
    ])
    pooled_heights = pooled_heights[np.isfinite(pooled_heights)]
    if pooled_heights.size == 0:
        return TransparentBottleConsensus(
            False, "unknown", (), None, None, label_gate, None, None,
            "quality frames have no finite height samples",
        )
    p90, p95 = np.quantile(pooled_heights, (0.90, 0.95))
    orientation = classify_pose_from_height_percentiles(
        float(p90),
        float(p95),
        upright_min_p90_m=upright_min_p90_m,
        upright_min_p95_m=upright_min_p95_m,
        lying_min_p90_m=lying_min_p90_m,
        lying_min_p95_m=lying_min_p95_m,
        lying_max_p90_m=lying_max_p90_m,
        lying_max_p95_m=lying_max_p95_m,
    )
    if orientation == "ambiguous":
        return TransparentBottleConsensus(
            False, orientation, eligible, float(p90), float(p95),
            label_gate, None, None,
            f"pooled pose is ambiguous (p90={p90 * 1000.0:.1f}mm, "
            f"p95={p95 * 1000.0:.1f}mm)",
        )

    inliers = tuple(
        index for index in eligible
        if measurements[index].get("orientation") == orientation
    )
    if len(inliers) < required:
        return TransparentBottleConsensus(
            False, orientation, inliers, float(p90), float(p95),
            label_gate, None, None,
            f"only {len(inliers)} stable {orientation} frames "
            f"(need {required}, label gate {label_gate})",
        )

    tcp_points = np.asarray(
        [measurements[index]["tcp_xyz"] for index in inliers],
        dtype=np.float64,
    )
    if tcp_points.shape != (len(inliers), 3) or not np.all(
        np.isfinite(tcp_points)
    ):
        return TransparentBottleConsensus(
            False, orientation, inliers, float(p90), float(p95),
            label_gate, None, None, "inlier TCP values are invalid",
        )
    tcp_median = np.median(tcp_points, axis=0)
    tcp_spread = float(np.max(np.linalg.norm(
        tcp_points - tcp_median, axis=1,
    )))
    if tcp_spread > float(maximum_tcp_spread_m):
        return TransparentBottleConsensus(
            False, orientation, inliers, float(p90), float(p95),
            label_gate,
            tuple(float(value) for value in tcp_median),
            tcp_spread,
            f"TCP spread {tcp_spread * 1000.0:.1f}mm exceeds "
            f"{float(maximum_tcp_spread_m) * 1000.0:.1f}mm",
        )
    return TransparentBottleConsensus(
        True,
        orientation,
        inliers,
        float(p90),
        float(p95),
        label_gate,
        tuple(float(value) for value in tcp_median),
        tcp_spread,
        "stable",
    )


def _principal_axis(points_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_2d, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("principal-axis input must contain at least 3 XY points")
    center = np.median(points, axis=0)
    covariance = np.cov(points - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        raise ValueError("principal axis is degenerate")
    axis = axis / norm
    if axis[0] < -1e-9 or (
        abs(float(axis[0])) <= 1e-9 and axis[1] < 0.0
    ):
        axis = -axis
    return center, axis


def _largest_depth_component(
    pixels_uv: np.ndarray,
    depths_mm: np.ndarray,
    *,
    gap_px: int,
    depth_tolerance_mm: float,
) -> np.ndarray:
    """Return a mask for the largest spatially/depth-continuous component."""

    pixels = np.asarray(pixels_uv, dtype=np.int32)
    depths = np.asarray(depths_mm, dtype=np.float64)
    count = len(pixels)
    if count == 0:
        return np.zeros(0, dtype=bool)

    parent = np.arange(count, dtype=np.int32)
    sizes = np.ones(count, dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    # The label is a small ROI, so a direct neighborhood search remains cheap
    # and avoids allocating an image-sized graph. Missing pixels may be crossed
    # only when both measured endpoints remain close in depth.
    cell_size = max(1, int(gap_px))
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, (u, v) in enumerate(pixels):
        cell = (int(u) // cell_size, int(v) // cell_size)
        for cell_v in range(cell[1] - 1, cell[1] + 2):
            for cell_u in range(cell[0] - 1, cell[0] + 2):
                for other in buckets.get((cell_u, cell_v), ()):
                    delta = pixels[other] - pixels[index]
                    if int(delta @ delta) > gap_px * gap_px:
                        continue
                    if abs(float(depths[other] - depths[index])) > depth_tolerance_mm:
                        continue
                    union(index, other)
        buckets.setdefault(cell, []).append(index)

    roots = np.asarray([find(index) for index in range(count)], dtype=np.int32)
    unique, root_counts = np.unique(roots, return_counts=True)
    largest_root = int(unique[int(np.argmax(root_counts))])
    return roots == largest_root


def analyze_transparent_bottle_points(
    pixels_uv,
    depths_mm,
    points_base,
    heights_above_desk_m,
    *,
    local_desk_z: float,
    minimum_height_m: float,
    maximum_height_m: float,
    label_axis_fraction: float,
    minimum_label_points: int,
    upright_min_p90_m: float,
    upright_min_p95_m: float,
    lying_min_p90_m: float,
    lying_min_p95_m: float,
    lying_max_p90_m: float,
    lying_max_p95_m: float,
) -> TransparentBottleAnalysis:
    """Measure label support and a safe TCP center from sparse real depth."""

    pixels = np.asarray(pixels_uv, dtype=np.float64)
    depths = np.asarray(depths_mm, dtype=np.float64)
    points = np.asarray(points_base, dtype=np.float64)
    heights = np.asarray(heights_above_desk_m, dtype=np.float64)
    sample_count = len(heights)
    if (
        pixels.shape != (sample_count, 2)
        or points.shape != (sample_count, 3)
        or depths.shape != (sample_count,)
    ):
        raise ValueError("bottle point arrays have inconsistent shapes")
    if not 0.10 <= float(label_axis_fraction) <= 0.80:
        raise ValueError("label_axis_fraction must be in [0.10, 0.80]")

    finite = (
        np.all(np.isfinite(pixels), axis=1)
        & np.all(np.isfinite(points), axis=1)
        & np.isfinite(depths)
        & np.isfinite(heights)
    )
    reliable = (
        finite
        & (depths > 0.0)
        & (heights >= float(minimum_height_m))
        & (heights <= float(maximum_height_m))
    )
    reliable_indices = np.flatnonzero(reliable)
    if len(reliable_indices) < int(minimum_label_points):
        raise ValueError(
            f"only {len(reliable_indices)} reliable points above the desk "
            f"(need {int(minimum_label_points)})"
        )

    reliable_heights = heights[reliable]
    p90, p95 = np.quantile(reliable_heights, (0.90, 0.95))
    orientation = classify_pose_from_height_percentiles(
        float(p90),
        float(p95),
        upright_min_p90_m=upright_min_p90_m,
        upright_min_p95_m=upright_min_p95_m,
        lying_min_p90_m=lying_min_p90_m,
        lying_min_p95_m=lying_min_p95_m,
        lying_max_p90_m=lying_max_p90_m,
        lying_max_p95_m=lying_max_p95_m,
    )

    image_center, image_axis = _principal_axis(pixels[reliable])
    projected = (pixels[reliable] - image_center) @ image_axis
    low, high = np.quantile(projected, (0.02, 0.98))
    axial_center = (float(low) + float(high)) * 0.5
    axial_span = max(float(high - low), 1.0)
    label_half_span = axial_span * float(label_axis_fraction) * 0.5
    axial_distance = np.abs(projected - axial_center)
    central_local = axial_distance <= label_half_span
    central_indices = reliable_indices[central_local]
    if (
        orientation == "lying"
        and len(central_indices) < int(minimum_label_points)
    ):
        # The grasp XY for a lying bottle comes from its measured long-axis
        # midpoint, not from a semantic label centroid.  Preserve only actual
        # depth returns nearest that midpoint for visualization/quality counts;
        # no missing depth is synthesized.
        nearest = np.argsort(axial_distance, kind="stable")[
            : int(minimum_label_points)
        ]
        central_indices = reliable_indices[nearest]
    elif len(central_indices) < int(minimum_label_points):
        raise ValueError(
            f"central label band has only {len(central_indices)} points "
            f"(need {int(minimum_label_points)})"
        )

    bbox_scale = max(
        float(np.ptp(pixels[reliable, 0])),
        float(np.ptp(pixels[reliable, 1])),
        1.0,
    )
    gap_px = int(np.clip(round(bbox_scale * 0.025), 3, 7))
    depth_tolerance_mm = float(np.clip(
        np.median(depths[central_indices]) * 0.035,
        12.0,
        30.0,
    ))
    keep_central = _largest_depth_component(
        pixels[central_indices],
        depths[central_indices],
        gap_px=gap_px,
        depth_tolerance_mm=depth_tolerance_mm,
    )
    connected_indices = central_indices[keep_central]
    if orientation == "lying":
        # Height and long-axis span provide the physical constraints for the
        # lying case. Sparse midpoint returns may be split by transparent gaps,
        # so retain the measured midpoint support instead of requiring one
        # image-connected label component.
        label_indices = central_indices
    else:
        label_indices = connected_indices
        if len(label_indices) < int(minimum_label_points):
            raise ValueError(
                f"largest label depth component has only {len(label_indices)} "
                f"points (need {int(minimum_label_points)})"
            )
    label_mask = np.zeros(sample_count, dtype=bool)
    label_mask[label_indices] = True
    label_surface = np.median(points[label_mask], axis=0)

    # The highest reliable support is the cap/neck for an upright bottle. Its
    # XY median is much closer to the vertical bottle axis than the visible
    # label surface. For a lying bottle this value is diagnostic only.
    cap_threshold = float(np.quantile(reliable_heights, 0.95))
    cap_mask = reliable & (heights >= cap_threshold)
    if int(np.count_nonzero(cap_mask)) < 3:
        cap_mask = reliable & (heights >= float(np.max(reliable_heights)) - 0.005)
    cap_xy = np.median(points[cap_mask, :2], axis=0)

    horizontal_center, horizontal_axis = _principal_axis(
        points[reliable, :2]
    )
    horizontal_projection = (
        points[reliable, :2] - horizontal_center
    ) @ horizontal_axis
    horizontal_low, horizontal_high = np.quantile(
        horizontal_projection, (0.02, 0.98),
    )
    horizontal_span = float(horizontal_high - horizontal_low)
    horizontal_axis_center = horizontal_center + horizontal_axis * (
        (float(horizontal_low) + float(horizontal_high)) * 0.5
    )
    if orientation == "upright":
        tcp = np.asarray([
            float(cap_xy[0]),
            float(cap_xy[1]),
            float(label_surface[2]),
        ])
    else:
        # XY is the midpoint of the measured bottle long axis. Z is centered
        # between the desk and the robust measured upper-surface p90. No
        # configured bottle radius and no synthesized depth are used.
        tcp = np.asarray([
            float(horizontal_axis_center[0]),
            float(horizontal_axis_center[1]),
            float(local_desk_z) + float(p90) * 0.5,
        ])

    return TransparentBottleAnalysis(
        orientation=orientation,
        height_p90_m=float(p90),
        height_p95_m=float(p95),
        reliable_count=int(np.count_nonzero(reliable)),
        label_count=int(np.count_nonzero(label_mask)),
        label_surface_xyz=tuple(float(value) for value in label_surface),
        cap_or_axis_xy=(float(cap_xy[0]), float(cap_xy[1])),
        horizontal_axis_xy=(
            float(horizontal_axis[0]),
            float(horizontal_axis[1]),
        ),
        horizontal_span_m=horizontal_span,
        tcp_xyz=tuple(float(value) for value in tcp),
        reliable_mask=reliable,
        label_mask=label_mask,
        cap_mask=cap_mask,
        image_axis_uv=(float(image_axis[0]), float(image_axis[1])),
        image_axis_center_uv=(
            float(image_center[0] + image_axis[0] * axial_center),
            float(image_center[1] + image_axis[1] * axial_center),
        ),
        label_axis_half_span_px=float(label_half_span),
        connected_ratio=float(len(connected_indices) / len(central_indices)),
    )
