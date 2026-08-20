from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


FRAME_COLUMN_CANDIDATES = ("frame idx", "frame_idx", "frame index", "frame_index", "frame")
INSTANCE_COLUMN_CANDIDATES = (
    "instance",
    "instance_id",
    "instance id",
    "track",
    "track_id",
    "track id",
)


@dataclass(frozen=True)
class SmoothingMethodSpec:
    key: str
    label: str
    apply: Callable[..., pd.DataFrame]


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
        ordered_groups.append(group_positions[np.argsort(sort_keys, kind="stable")])
    return ordered_groups


def _median_prefer_original(values: np.ndarray, original_value: float) -> float:
    finite = np.sort(values[np.isfinite(values)].astype(np.float64))
    if finite.size == 0:
        return float("nan")
    midpoint = finite.size // 2
    if finite.size % 2 == 1:
        return float(finite[midpoint])
    candidates = finite[midpoint - 1 : midpoint + 1]
    if np.isfinite(original_value) and np.any(np.isclose(candidates, original_value, rtol=1e-9, atol=1e-9)):
        return float(original_value)
    return float(np.mean(candidates))


def _window_positions(
    group_positions: np.ndarray,
    group_order_index: int,
    frame_values: np.ndarray | None,
    half_window: int,
) -> np.ndarray:
    if frame_values is None:
        start = max(0, group_order_index - half_window)
        stop = min(len(group_positions), group_order_index + half_window + 1)
        return group_positions[start:stop]

    current_frame = frame_values[group_positions[group_order_index]]
    if not np.isfinite(current_frame):
        start = max(0, group_order_index - half_window)
        stop = min(len(group_positions), group_order_index + half_window + 1)
        return group_positions[start:stop]

    group_frames = frame_values[group_positions]
    in_window = (
        np.isfinite(group_frames)
        & (group_frames >= current_frame - half_window)
        & (group_frames <= current_frame + half_window)
    )
    return group_positions[in_window]


def anchor_median_smoothing(
    df: pd.DataFrame,
    bodyparts: list[str],
    anchor_bodypart: str,
    window_size: int,
) -> pd.DataFrame:
    if window_size not in {3, 5}:
        raise ValueError("Smoothing window size must be 3 or 5.")
    if anchor_bodypart not in bodyparts:
        raise ValueError(f"Anchor node is not available in this CSV: {anchor_bodypart}")

    anchor_x_col = f"{anchor_bodypart}.x"
    anchor_y_col = f"{anchor_bodypart}.y"
    if anchor_x_col not in df.columns or anchor_y_col not in df.columns:
        raise ValueError(f"Anchor node coordinates are missing: {anchor_bodypart}")

    result = df.copy()
    half_window = window_size // 2
    frame_column = _find_matching_column(df, FRAME_COLUMN_CANDIDATES)
    frame_values = (
        None
        if frame_column is None
        else pd.to_numeric(df[frame_column], errors="coerce").to_numpy(dtype=np.float64)
    )
    anchor_x = pd.to_numeric(df[anchor_x_col], errors="coerce").to_numpy(dtype=np.float64)
    anchor_y = pd.to_numeric(df[anchor_y_col], errors="coerce").to_numpy(dtype=np.float64)

    coordinate_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for bodypart in bodyparts:
        x_col = f"{bodypart}.x"
        y_col = f"{bodypart}.y"
        if x_col in df.columns and y_col in df.columns:
            coordinate_arrays[bodypart] = (
                pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=np.float64),
                pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float64),
            )

    output_arrays = {
        bodypart: (x_values.copy(), y_values.copy())
        for bodypart, (x_values, y_values) in coordinate_arrays.items()
    }

    for group_positions in _ordered_groups(df):
        for group_order_index, row_position in enumerate(group_positions):
            current_anchor_x = anchor_x[row_position]
            current_anchor_y = anchor_y[row_position]
            if not (np.isfinite(current_anchor_x) and np.isfinite(current_anchor_y)):
                continue

            window = _window_positions(group_positions, group_order_index, frame_values, half_window)
            valid_anchor = np.isfinite(anchor_x[window]) & np.isfinite(anchor_y[window])
            anchor_window = window[valid_anchor]
            if anchor_window.size == 0:
                continue

            median_x = _median_prefer_original(anchor_x[anchor_window], current_anchor_x)
            median_y = _median_prefer_original(anchor_y[anchor_window], current_anchor_y)
            if not (np.isfinite(median_x) and np.isfinite(median_y)):
                continue

            dx = median_x - current_anchor_x
            dy = median_y - current_anchor_y
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                continue

            for bodypart, (source_x, source_y) in coordinate_arrays.items():
                output_x, output_y = output_arrays[bodypart]
                if np.isfinite(source_x[row_position]):
                    output_x[row_position] = source_x[row_position] + dx
                if np.isfinite(source_y[row_position]):
                    output_y[row_position] = source_y[row_position] + dy

    for bodypart, (output_x, output_y) in output_arrays.items():
        result[f"{bodypart}.x"] = output_x
        result[f"{bodypart}.y"] = output_y
    return result


SMOOTHING_METHODS: dict[str, SmoothingMethodSpec] = {
    "anchor_median_smoothing": SmoothingMethodSpec(
        key="anchor_median_smoothing",
        label="Anchor Median Smoothing",
        apply=anchor_median_smoothing,
    )
}


def build_smoothed_dataframe(
    df: pd.DataFrame,
    bodyparts: list[str],
    method: str,
    anchor_bodypart: str | None,
    window_size: int,
) -> pd.DataFrame:
    method_spec = SMOOTHING_METHODS.get(str(method))
    if method_spec is None:
        raise ValueError(f"Unknown smoothing method: {method}")
    if anchor_bodypart is None:
        raise ValueError("Select an anchor node for smoothing.")
    return method_spec.apply(
        df,
        bodyparts=bodyparts,
        anchor_bodypart=anchor_bodypart,
        window_size=int(window_size),
    )
