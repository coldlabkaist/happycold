from __future__ import annotations

import numpy as np
import pandas as pd

from trajectory import infer_pixel_scale


FRAME_COLUMN_CANDIDATES = ("frame idx", "frame_idx", "frame index", "frame_index", "frame")
TRACK_COLUMN_CANDIDATES = ("track", "track_id", "track id")
INSTANCE_COLUMN_CANDIDATES = (
    "instance",
    "instance_id",
    "instance id",
    "track",
    "track_id",
    "track id",
)


def _canonical_column_name(name: str) -> str:
    return "".join(character.lower() for character in str(name) if character.isalnum())


def _find_matching_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lookup = {_canonical_column_name(column): column for column in df.columns}
    for candidate in candidates:
        match = lookup.get(_canonical_column_name(candidate))
        if match is not None:
            return match
    return None


def _ordered_groups(df: pd.DataFrame) -> list[np.ndarray]:
    positions = np.arange(len(df), dtype=np.int64)
    instance_column = _find_matching_column(df, INSTANCE_COLUMN_CANDIDATES)
    frame_column = _find_matching_column(df, FRAME_COLUMN_CANDIDATES)

    if instance_column is None:
        groups = [positions]
    else:
        grouping_frame = pd.DataFrame(
            {
                "_position": positions,
                "_instance": df[instance_column].to_numpy(copy=False),
            }
        )
        groups = [
            group["_position"].to_numpy(dtype=np.int64)
            for _instance, group in grouping_frame.groupby("_instance", sort=False, dropna=False)
        ]

    if frame_column is None:
        return groups

    frame_values = pd.to_numeric(df[frame_column], errors="coerce").to_numpy(dtype=np.float64)
    ordered_groups: list[np.ndarray] = []
    for group_positions in groups:
        group_frames = frame_values[group_positions]
        sort_keys = np.where(np.isfinite(group_frames), group_frames, np.inf)
        order = np.argsort(sort_keys, kind="stable")
        ordered_groups.append(group_positions[order])
    return ordered_groups


def _interpolate_internal(
    values: np.ndarray,
    groups: list[np.ndarray],
    extrapolate: bool = False,
) -> np.ndarray:
    interpolated = values.copy()
    for positions in groups:
        if len(positions) < 2:
            continue
        group_values = pd.Series(values[positions], dtype="float64")
        filled = group_values.interpolate(method="linear", limit_area="inside")
        if extrapolate:
            # Boundary extrapolation intentionally uses the nearest valid coordinate.
            # This avoids unstable long-range slopes at the start or end of a track.
            filled = filled.bfill().ffill()
        interpolated[positions] = filled.to_numpy(dtype=np.float64)
    return interpolated


def build_automatic_removal_dataframe(
    df: pd.DataFrame,
    bodyparts: list[str],
    region_mask: np.ndarray,
    width: int,
    height: int,
    anchor_bodypart: str,
    removal_mode: str,
) -> pd.DataFrame:
    """Remove complete skeleton coordinates selected by an anchor node and region."""
    if removal_mode not in {"inside", "outside"}:
        raise ValueError("Automatic removal mode must be 'inside' or 'outside'.")
    if anchor_bodypart not in bodyparts:
        raise ValueError(f"Anchor node is not available in this CSV: {anchor_bodypart}")
    if region_mask.ndim != 2 or region_mask.shape != (height, width) or not np.any(region_mask):
        raise ValueError("Draw an automatic-removal region first.")

    x_column = f"{anchor_bodypart}.x"
    y_column = f"{anchor_bodypart}.y"
    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError(f"Anchor node coordinates are missing: {anchor_bodypart}")

    anchor_x = pd.to_numeric(df[x_column], errors="coerce").to_numpy(dtype=np.float64)
    anchor_y = pd.to_numeric(df[y_column], errors="coerce").to_numpy(dtype=np.float64)
    finite_anchor = np.isfinite(anchor_x) & np.isfinite(anchor_y)
    pixel_x = np.zeros(len(df), dtype=np.int64)
    pixel_y = np.zeros(len(df), dtype=np.int64)
    pixel_x[finite_anchor] = np.rint(
        anchor_x[finite_anchor] * infer_pixel_scale(df[x_column], width)
    ).astype(np.int64)
    pixel_y[finite_anchor] = np.rint(
        anchor_y[finite_anchor] * infer_pixel_scale(df[y_column], height)
    ).astype(np.int64)
    in_bounds = (
        finite_anchor
        & (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
    )
    inside_region = np.zeros(len(df), dtype=bool)
    valid_positions = np.flatnonzero(in_bounds)
    inside_region[valid_positions] = region_mask[
        pixel_y[valid_positions],
        pixel_x[valid_positions],
    ].astype(bool)
    if removal_mode == "inside":
        remove_rows = finite_anchor & in_bounds & inside_region
    else:
        remove_rows = finite_anchor & (~in_bounds | ~inside_region)

    result = df.copy()
    skeleton_columns: list[str] = []
    for bodypart in bodyparts:
        for suffix in (".x", ".y", ".score"):
            column = f"{bodypart}{suffix}"
            if column in result.columns:
                skeleton_columns.append(column)
    instance_score_column = _find_matching_column(
        result,
        ("instance.score", "instance_score", "instance score"),
    )
    if instance_score_column is not None:
        skeleton_columns.append(instance_score_column)
    if skeleton_columns and np.any(remove_rows):
        result.loc[remove_rows, skeleton_columns] = np.nan
    return result


def build_interpolation_pipeline_dataframe(
    df: pd.DataFrame,
    bodyparts: list[str],
    region_mask: np.ndarray | None,
    width: int,
    height: int,
    removal_mode: str,
    anchor_bodypart: str | None,
    interpolate: bool,
    extrapolate: bool = False,
) -> pd.DataFrame:
    """Run automatic skeleton removal, followed by optional temporal interpolation."""
    if removal_mode not in {"none", "inside", "outside"}:
        raise ValueError("Removal mode must be 'none', 'inside', or 'outside'.")
    result = df.copy()
    if removal_mode != "none":
        if region_mask is None or anchor_bodypart is None:
            raise ValueError("Automatic removal needs a region and anchor node.")
        result = build_automatic_removal_dataframe(
            result,
            bodyparts,
            region_mask,
            width,
            height,
            anchor_bodypart,
            removal_mode,
        )
    if interpolate:
        result = build_interpolated_dataframe(
            result,
            bodyparts,
            None,
            width,
            height,
            "all",
            extrapolate=extrapolate,
        )
    return result


def build_interpolated_dataframe(
    df: pd.DataFrame,
    bodyparts: list[str],
    region_mask: np.ndarray | None,
    width: int,
    height: int,
    region_mode: str,
    extrapolate: bool = False,
) -> pd.DataFrame:
    """Fill internal coordinate gaps, optionally restricted by a spatial region filter.

    Existing values are never overwritten. If a track column is present, each track is
    interpolated independently. Internal gaps use linear interpolation. When ``extrapolate``
    is enabled, leading and trailing gaps use the nearest valid coordinate in that track.
    Region membership is evaluated after interpolation when ``region_mode`` is ``inside``
    or ``outside``; ``all`` skips the spatial filter.
    """
    if region_mode not in {"all", "inside", "outside"}:
        raise ValueError("Interpolation region mode must be 'all', 'inside', or 'outside'.")
    if region_mode != "all":
        if region_mask is None or region_mask.ndim != 2 or region_mask.shape != (height, width):
            raise ValueError("Interpolation region mask does not match the video dimensions.")
        if not np.any(region_mask):
            raise ValueError("Draw an interpolation region first.")

    result = df.copy()
    groups = _ordered_groups(df)
    region = None if region_mode == "all" else region_mask.astype(bool)

    for bodypart in bodyparts:
        x_column = f"{bodypart}.x"
        y_column = f"{bodypart}.y"
        if x_column not in df.columns or y_column not in df.columns:
            continue

        original_x = pd.to_numeric(df[x_column], errors="coerce").to_numpy(dtype=np.float64)
        original_y = pd.to_numeric(df[y_column], errors="coerce").to_numpy(dtype=np.float64)
        missing_x = ~np.isfinite(original_x)
        missing_y = ~np.isfinite(original_y)
        candidate_missing = missing_x | missing_y
        if not candidate_missing.any():
            continue

        interpolated_x = _interpolate_internal(original_x, groups, extrapolate)
        interpolated_y = _interpolate_internal(original_y, groups, extrapolate)
        finite_pair = np.isfinite(interpolated_x) & np.isfinite(interpolated_y)
        candidates = candidate_missing & finite_pair
        if not candidates.any():
            continue

        if region_mode == "all":
            accepted = candidates
        else:
            pixel_x = np.zeros(len(df), dtype=np.int64)
            pixel_y = np.zeros(len(df), dtype=np.int64)
            finite_positions = np.flatnonzero(finite_pair)
            x_scale = infer_pixel_scale(df[x_column], width)
            y_scale = infer_pixel_scale(df[y_column], height)
            pixel_x[finite_positions] = np.rint(interpolated_x[finite_positions] * x_scale).astype(np.int64)
            pixel_y[finite_positions] = np.rint(interpolated_y[finite_positions] * y_scale).astype(np.int64)
            in_bounds = (
                (pixel_x >= 0)
                & (pixel_x < width)
                & (pixel_y >= 0)
                & (pixel_y < height)
            )
            inside_region = np.zeros(len(df), dtype=bool)
            valid_positions = np.flatnonzero(candidates & in_bounds)
            inside_region[valid_positions] = region[pixel_y[valid_positions], pixel_x[valid_positions]]
            allowed_region = inside_region if region_mode == "inside" else ~inside_region
            accepted = candidates & in_bounds & allowed_region

        if np.any(accepted & missing_x):
            output_x = original_x.copy()
            fill_x = accepted & missing_x
            output_x[fill_x] = interpolated_x[fill_x]
            result[x_column] = output_x
        if np.any(accepted & missing_y):
            output_y = original_y.copy()
            fill_y = accepted & missing_y
            output_y[fill_y] = interpolated_y[fill_y]
            result[y_column] = output_y

    return result
