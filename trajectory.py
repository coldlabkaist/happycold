from __future__ import annotations

import pandas as pd


def bodypart_coordinate_columns(bodypart: str, normalized: bool = False) -> tuple[str, str]:
    if normalized:
        return f"{bodypart}.x_normalized", f"{bodypart}.y_normalized"
    return f"{bodypart}.x", f"{bodypart}.y"


def resolve_bodypart_coordinate_columns(
    df: pd.DataFrame,
    bodypart: str,
    normalized: bool = False,
) -> tuple[str, str] | None:
    preferred = bodypart_coordinate_columns(bodypart, normalized=normalized)
    if all(column in df.columns for column in preferred):
        return preferred

    # Older normalized dataframes stored normalized values in the original x/y
    # columns. Keep accepting them in previews and heatmaps.
    legacy = bodypart_coordinate_columns(bodypart, normalized=False)
    if normalized and all(column in df.columns for column in legacy):
        return legacy
    return None


def bodyparts_from_dataframe(df: pd.DataFrame) -> list[str]:
    columns = set(df.columns)
    return [
        column[:-2]
        for column in df.columns
        if column.endswith(".x") and f"{column[:-2]}.y" in columns
    ]


def infer_pixel_scale(series: pd.Series, frame_extent: float) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return frame_extent
    return frame_extent if numeric.max() <= 1.5 else 1.0
