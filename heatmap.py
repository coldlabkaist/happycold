from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from trajectory import infer_pixel_scale, resolve_bodypart_coordinate_columns


@dataclass(frozen=True)
class HeatmapResult:
    values: np.ndarray
    x_limit: float
    y_limit: float
    valid_frame_count: int


def _axis_extent(df: pd.DataFrame, columns: list[str], frame_extent: int) -> float:
    if not columns:
        return float(frame_extent)
    has_pixel_scale = any(infer_pixel_scale(df[column], frame_extent) == 1.0 for column in columns)
    return float(frame_extent if has_pixel_scale else 1.0)


def _grid_shape(x_limit: float, y_limit: float, max_bins: int) -> tuple[int, int]:
    safe_x = max(float(x_limit), 1e-6)
    safe_y = max(float(y_limit), 1e-6)
    if safe_x >= safe_y:
        return max_bins, max(64, int(round(max_bins * safe_y / safe_x)))
    return max(64, int(round(max_bins * safe_x / safe_y))), max_bins


def _clamped_display_bounds(
    display_bounds: tuple[float, float, float, float],
    frame_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    frame_width = max(1, int(frame_size[0]))
    frame_height = max(1, int(frame_size[1]))
    left, top, right, bottom = (float(value) for value in display_bounds)
    if not all(np.isfinite((left, top, right, bottom))):
        raise ValueError("Selected spatial range contains invalid coordinates.")
    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))
    left = max(0.0, min(float(frame_width), left))
    right = max(0.0, min(float(frame_width), right))
    top = max(0.0, min(float(frame_height), top))
    bottom = max(0.0, min(float(frame_height), bottom))
    if right - left < 1.0 or bottom - top < 1.0:
        raise ValueError("Selected spatial range is empty.")
    return left, top, right, bottom


def _coordinate_tensor(
    df: pd.DataFrame,
    bodyparts: list[str],
    normalized: bool,
    frame_size: tuple[int, int],
    rectified_size: tuple[int, int] | None,
    display_bounds: tuple[float, float, float, float] | None,
) -> tuple[np.ndarray, float, float]:
    available_coordinates = [
        (bodypart, columns)
        for bodypart in bodyparts
        if (columns := resolve_bodypart_coordinate_columns(df, bodypart, normalized=normalized)) is not None
    ]
    if not available_coordinates:
        raise ValueError("No bodyparts with matching x/y columns were found.")

    coordinate_columns = [
        column
        for _bodypart, columns in available_coordinates
        for column in columns
    ]
    numeric = df.loc[:, coordinate_columns].apply(pd.to_numeric, errors="coerce")
    coordinates = numeric.to_numpy(dtype=np.float32, copy=True).reshape(len(df), len(available_coordinates), 2)

    if normalized:
        if rectified_size is None:
            raise ValueError("Rectified display size is required for a normalized heatmap.")
        rect_width = max(1, int(rectified_size[0]))
        rect_height = max(1, int(rectified_size[1]))
        coordinates[:, :, 0] *= float(rect_width)
        coordinates[:, :, 1] *= float(rect_height)
        if display_bounds is not None:
            left, top, right, bottom = _clamped_display_bounds(
                display_bounds,
                (rect_width, rect_height),
            )
            coordinates[:, :, 0] -= left
            coordinates[:, :, 1] -= top
            return coordinates, right - left, bottom - top
        return coordinates, float(rect_width), float(rect_height)

    frame_width = max(1, int(frame_size[0]))
    frame_height = max(1, int(frame_size[1]))
    x_columns = [columns[0] for _bodypart, columns in available_coordinates]
    y_columns = [columns[1] for _bodypart, columns in available_coordinates]
    if display_bounds is not None:
        left, top, right, bottom = _clamped_display_bounds(
            display_bounds,
            (frame_width, frame_height),
        )
        for index, (x_col, y_col) in enumerate(zip(x_columns, y_columns)):
            coordinates[:, index, 0] *= infer_pixel_scale(df[x_col], frame_width)
            coordinates[:, index, 1] *= infer_pixel_scale(df[y_col], frame_height)
        coordinates[:, :, 0] -= left
        coordinates[:, :, 1] -= top
        return coordinates, right - left, bottom - top

    return (
        coordinates,
        _axis_extent(df, x_columns, frame_width),
        _axis_extent(df, y_columns, frame_height),
    )


def _scaled_hull_polygon(
    points: np.ndarray,
    x_limit: float,
    y_limit: float,
    grid_width: int,
    grid_height: int,
) -> np.ndarray | None:
    hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2)).reshape(-1, 2)
    if len(hull) < 3:
        return None

    polygon = np.empty_like(hull, dtype=np.float32)
    polygon[:, 0] = np.clip(
        hull[:, 0] * ((grid_width - 1) / max(x_limit, 1e-6)),
        0.0,
        grid_width - 1,
    )
    polygon[:, 1] = np.clip(
        hull[:, 1] * ((grid_height - 1) / max(y_limit, 1e-6)),
        0.0,
        grid_height - 1,
    )
    polygon = np.rint(polygon).astype(np.int32)
    if len(np.unique(polygon, axis=0)) < 3 or cv2.contourArea(polygon) <= 0.0:
        return None
    return polygon


def _add_polygon(counts: np.ndarray, polygon: np.ndarray) -> None:
    x_min = max(0, int(polygon[:, 0].min()))
    x_max = min(counts.shape[1] - 1, int(polygon[:, 0].max()))
    y_min = max(0, int(polygon[:, 1].min()))
    y_max = min(counts.shape[0] - 1, int(polygon[:, 1].max()))
    if x_max < x_min or y_max < y_min:
        return

    local_polygon = polygon - np.array([x_min, y_min], dtype=np.int32)
    local_mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    cv2.fillPoly(local_mask, [local_polygon], 1)
    counts[y_min : y_max + 1, x_min : x_max + 1] += local_mask


def calculate_body_occupancy_heatmap(
    df: pd.DataFrame,
    bodyparts: list[str],
    frame_size: tuple[int, int],
    normalized: bool = False,
    rectified_size: tuple[int, int] | None = None,
    max_bins: int = 180,
    blur_sigma: float = 1.2,
    display_bounds: tuple[float, float, float, float] | None = None,
) -> HeatmapResult:
    """Calculate body-volume occupancy without depending on any Qt/UI state."""
    coordinates, x_limit, y_limit = _coordinate_tensor(
        df,
        bodyparts,
        normalized,
        frame_size,
        rectified_size,
        display_bounds,
    )
    grid_width, grid_height = _grid_shape(x_limit, y_limit, max(64, int(max_bins)))
    counts = np.zeros((grid_height, grid_width), dtype=np.uint32)
    finite_points = np.isfinite(coordinates).all(axis=2)
    valid_frame_count = 0

    candidate_rows = np.flatnonzero(finite_points.sum(axis=1) >= 3)
    for row_index in candidate_rows:
        points = coordinates[row_index, finite_points[row_index]]
        polygon = _scaled_hull_polygon(points, x_limit, y_limit, grid_width, grid_height)
        if polygon is None:
            continue
        _add_polygon(counts, polygon)
        valid_frame_count += 1

    if valid_frame_count <= 0:
        raise ValueError("No valid body polygon could be built from the current CSV/time range.")

    heatmap = counts.astype(np.float32)
    if blur_sigma > 0:
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    heatmap /= float(valid_frame_count)
    if float(np.nanmax(heatmap)) <= 0.0:
        raise ValueError("No valid body polygon occupancy was found for the current CSV/time range.")

    return HeatmapResult(
        values=heatmap,
        x_limit=x_limit,
        y_limit=y_limit,
        valid_frame_count=valid_frame_count,
    )
