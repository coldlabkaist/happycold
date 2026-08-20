from __future__ import annotations

import math

import pandas as pd
from matplotlib.figure import Figure

from trajectory import infer_pixel_scale, resolve_bodypart_coordinate_columns


TRACK_COLUMN_CANDIDATES = ("track", "track_id", "track id")
FRAME_COLUMN_CANDIDATES = ("frame idx", "frame_idx", "frame index", "frame_index", "frame")
TRAJECTORY_COLORS = [
    "#2563eb",
    "#ea580c",
    "#16a34a",
    "#dc2626",
    "#7c3aed",
    "#0891b2",
    "#4f46e5",
    "#a16207",
]


def canonical_column_name(name: str) -> str:
    return "".join(character.lower() for character in str(name) if character.isalnum())


def find_matching_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lookup = {canonical_column_name(column): column for column in df.columns}
    for candidate in candidates:
        match = lookup.get(canonical_column_name(candidate))
        if match is not None:
            return match
    return None


def axis_extent(df: pd.DataFrame, columns: list[str], frame_extent: int) -> float:
    if not columns:
        return float(frame_extent)
    has_pixel_scale = any(infer_pixel_scale(df[column], frame_extent) == 1.0 for column in columns)
    return float(frame_extent if has_pixel_scale else 1.0)


def plot_limits(
    df: pd.DataFrame,
    bodyparts: list[str],
    normalized: bool,
    frame_width: int | None,
    frame_height: int | None,
    normalized_display_size: tuple[int, int] | None = None,
) -> tuple[float, float]:
    if normalized:
        if normalized_display_size is not None:
            return float(normalized_display_size[0]), float(normalized_display_size[1])
        return 1.0, 1.0
    if frame_width is None or frame_height is None:
        return 1.0, 1.0
    x_columns = [f"{bodypart}.x" for bodypart in bodyparts if f"{bodypart}.x" in df.columns]
    y_columns = [f"{bodypart}.y" for bodypart in bodyparts if f"{bodypart}.y" in df.columns]
    return (
        axis_extent(df, x_columns, frame_width),
        axis_extent(df, y_columns, frame_height),
    )


def track_groups(
    df: pd.DataFrame,
    track_col: str | None,
    frame_col: str | None,
) -> list[tuple[str | None, pd.DataFrame]]:
    if track_col is None:
        return [(None, df)]

    groups: list[tuple[str | None, pd.DataFrame]] = []
    for track_value, group_df in df.groupby(track_col, dropna=False, sort=False):
        if frame_col is not None:
            group_df = (
                group_df.assign(_trajectory_order=pd.to_numeric(group_df[frame_col], errors="coerce"))
                .sort_values("_trajectory_order", kind="stable")
                .drop(columns="_trajectory_order")
            )
        groups.append((None if pd.isna(track_value) else str(track_value), group_df))
    return groups


def scaled_bodypart_xy(
    df: pd.DataFrame,
    bodypart: str,
    normalized: bool,
    normalized_display_size: tuple[int, int] | None,
) -> tuple[pd.Series, pd.Series]:
    coordinate_columns = resolve_bodypart_coordinate_columns(df, bodypart, normalized=normalized)
    if coordinate_columns is None:
        return pd.Series(float("nan"), index=df.index), pd.Series(float("nan"), index=df.index)
    x_col, y_col = coordinate_columns
    x_values = pd.to_numeric(df[x_col], errors="coerce")
    y_values = pd.to_numeric(df[y_col], errors="coerce")
    if normalized_display_size is None:
        return x_values, y_values
    return (
        x_values * float(normalized_display_size[0]),
        y_values * float(normalized_display_size[1]),
    )


def plot_bodypart(
    ax,
    df: pd.DataFrame,
    bodypart: str,
    track_col: str | None,
    frame_col: str | None,
    normalized: bool,
    normalized_display_size: tuple[int, int] | None,
) -> None:
    groups = track_groups(df, track_col, frame_col)
    for index, (track_label, group_df) in enumerate(groups):
        x_values, y_values = scaled_bodypart_xy(
            group_df,
            bodypart,
            normalized,
            normalized_display_size,
        )
        ax.plot(
            x_values,
            y_values,
            color=TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)],
            linewidth=1.1,
            label=track_label,
        )
    if track_col is not None and len(groups) > 1:
        ax.legend(fontsize=7, loc="best")


def build_trajectory_figure(
    df: pd.DataFrame,
    bodyparts: list[str],
    normalized: bool,
    frame_width: int | None,
    frame_height: int | None,
    normalized_display_size: tuple[int, int] | None = None,
) -> Figure:
    figure = Figure(figsize=(10, 7), tight_layout=True)
    if not bodyparts:
        return figure

    columns = min(3, max(1, len(bodyparts)))
    rows = math.ceil(len(bodyparts) / columns)
    track_col = find_matching_column(df, TRACK_COLUMN_CANDIDATES)
    frame_col = find_matching_column(df, FRAME_COLUMN_CANDIDATES)
    x_limit, y_limit = plot_limits(
        df,
        bodyparts,
        normalized,
        frame_width,
        frame_height,
        normalized_display_size,
    )

    for index, bodypart in enumerate(bodyparts, start=1):
        ax = figure.add_subplot(rows, columns, index)
        plot_bodypart(
            ax,
            df,
            bodypart,
            track_col,
            frame_col,
            normalized,
            normalized_display_size,
        )
        ax.set_title(bodypart, fontsize=10)
        ax.set_xlim(0, x_limit)
        ax.set_ylim(y_limit, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    return figure
