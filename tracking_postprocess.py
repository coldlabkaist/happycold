from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trajectory import bodyparts_from_dataframe


FRAME_COLUMN_CANDIDATES = (
    "frame_idx",
    "frame",
    "frame_index",
    "frame idx",
    "frame index",
)
TRACK_COLUMN_CANDIDATES = (
    "_coop_track_id",
    "instance.id",
    "track",
    "track_id",
    "test_track",
    "target_track",
)
PRIMARY_CONFIDENCE_COLUMNS = (
    "instance.score",
    "instance.visibility",
    "score",
    "confidence",
)
CANONICAL_POINT_ALIASES = {
    "nose": ("nose",),
    "body_c": ("body_c", "bodycenter", "body_centre", "center"),
    "body_l": ("body_l", "left_body"),
    "body_r": ("body_r", "right_body"),
    "tail": ("tail", "tail_base", "tailbase", "ano"),
}
LENGTH_OUTLIER_EDGES = (
    ("body_c", "body_l", "body_c-body_l"),
    ("body_c", "body_r", "body_c-body_r"),
    ("body_c", "tail", "body_c-tail"),
    ("body_c", "nose", "body_c-nose"),
)


@dataclass(frozen=True)
class DuplicateSkeletonCandidate:
    frame_value: int
    row_index_a: object
    row_index_b: object
    track_id_a: str | None
    track_id_b: str | None
    confidence_a: float
    confidence_b: float
    distance_sum: float


@dataclass(frozen=True)
class LengthOutlierCandidate:
    frame_value: int
    row_index: object
    track_id: str | None
    segments: tuple[str, ...]
    max_deviation: float


@dataclass(frozen=True)
class TrackingRepairStageResult:
    dataframe: pd.DataFrame
    candidate_count: int
    affected_count: int
    warning: str | None = None

    @property
    def removed_count(self) -> int:
        """Compatibility alias used by the duplicate-removal stage."""
        return self.affected_count


@dataclass(frozen=True)
class TrackingPostprocessConfig:
    remove_duplicates: bool = True
    duplicate_distance_threshold: float = 120.0
    remove_length_outliers: bool = True
    deviation_mode: str = "both"
    high_z_threshold: float = 3.5
    low_z_threshold: float = 3.5


@dataclass(frozen=True)
class TrackingPostprocessResult:
    dataframe: pd.DataFrame
    duplicate_candidates: int
    duplicate_removed: int
    length_outlier_candidates: int
    length_outliers_invalidated: int
    warnings: tuple[str, ...]

    @property
    def length_outliers_removed(self) -> int:
        """Compatibility alias for callers written before rows were preserved."""
        return self.length_outliers_invalidated


def _canonical_name(value: object) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {_canonical_name(column): column for column in columns}
    for candidate in candidates:
        match = lookup.get(_canonical_name(candidate))
        if match is not None:
            return match
    return None


def detect_frame_column(columns: list[str]) -> str | None:
    return _find_column(columns, FRAME_COLUMN_CANDIDATES)


def detect_track_column(columns: list[str]) -> str | None:
    return _find_column(columns, TRACK_COLUMN_CANDIDATES)


def robust_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 1e-6:
        return 1.4826 * mad
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    if iqr > 1e-6:
        return iqr / 1.349
    standard_deviation = float(np.std(values))
    return standard_deviation if standard_deviation > 1e-6 else 0.0


def _real_detection_mask(dataframe: pd.DataFrame) -> np.ndarray:
    if "_coop_dense_placeholder" not in dataframe.columns:
        return np.ones(len(dataframe), dtype=bool)
    values = pd.to_numeric(dataframe["_coop_dense_placeholder"], errors="coerce").fillna(0)
    return values.to_numpy(dtype=float) == 0.0


def _coordinate_pairs(dataframe: pd.DataFrame, bodyparts: list[str] | None) -> list[tuple[str, str, str]]:
    available_bodyparts = list(bodyparts or bodyparts_from_dataframe(dataframe))
    return [
        (bodypart, f"{bodypart}.x", f"{bodypart}.y")
        for bodypart in available_bodyparts
        if f"{bodypart}.x" in dataframe.columns and f"{bodypart}.y" in dataframe.columns
    ]


def _coordinate_tensor_pixels(
    dataframe: pd.DataFrame,
    coordinate_pairs: list[tuple[str, str, str]],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = np.full((len(dataframe), len(coordinate_pairs), 2), np.nan, dtype=float)
    numeric_columns: list[np.ndarray] = []
    for _bodypart, x_column, y_column in coordinate_pairs:
        numeric_columns.append(pd.to_numeric(dataframe[x_column], errors="coerce").to_numpy(dtype=float))
        numeric_columns.append(pd.to_numeric(dataframe[y_column], errors="coerce").to_numpy(dtype=float))
    finite_parts = [values[np.isfinite(values)] for values in numeric_columns if np.any(np.isfinite(values))]
    merged = np.concatenate(finite_parts) if finite_parts else np.array([], dtype=float)
    normalized = bool(
        merged.size == 0
        or (float(np.min(merged)) >= 0.0 and float(np.max(merged)) <= 1.5)
    )
    scale_x = float(max(1, width)) if normalized else 1.0
    scale_y = float(max(1, height)) if normalized else 1.0
    for point_index, (_bodypart, x_column, y_column) in enumerate(coordinate_pairs):
        x_values = pd.to_numeric(dataframe[x_column], errors="coerce").to_numpy(dtype=float)
        y_values = pd.to_numeric(dataframe[y_column], errors="coerce").to_numpy(dtype=float)
        tensor[:, point_index, 0] = x_values * scale_x
        tensor[:, point_index, 1] = y_values * scale_y
    return tensor, np.isfinite(tensor).all(axis=2)


def _track_ids(dataframe: pd.DataFrame, track_column: str | None) -> np.ndarray:
    if track_column is None:
        return np.full(len(dataframe), None, dtype=object)
    values: list[str | None] = []
    for value in dataframe[track_column]:
        if pd.isna(value) or not str(value).strip():
            values.append(None)
        else:
            values.append(str(value).strip())
    return np.asarray(values, dtype=object)


def _confidence_values(dataframe: pd.DataFrame) -> np.ndarray:
    columns = list(dataframe.columns)
    primary = _find_column(columns, PRIMARY_CONFIDENCE_COLUMNS)
    if primary is not None:
        return pd.to_numeric(dataframe[primary], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    score_columns = [
        column
        for column in columns
        if column.lower().endswith(".score") or column.lower().endswith(".visibility")
    ]
    if not score_columns:
        return np.zeros(len(dataframe), dtype=float)
    return (
        dataframe[score_columns]
        .apply(pd.to_numeric, errors="coerce")
        .mean(axis=1)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def find_duplicate_skeleton_candidates(
    dataframe: pd.DataFrame,
    *,
    bodyparts: list[str] | None = None,
    criteria: float = 120.0,
    width: int = 1,
    height: int = 1,
) -> tuple[tuple[DuplicateSkeletonCandidate, ...], str | None]:
    if dataframe.empty:
        return (), "No tracking rows were available."
    working = dataframe.loc[_real_detection_mask(dataframe)].copy()
    frame_column = detect_frame_column(list(working.columns))
    if frame_column is None:
        return (), "Tracking CSV does not contain a frame column."
    coordinate_pairs = _coordinate_pairs(working, bodyparts)
    if not coordinate_pairs:
        return (), "Tracking CSV does not contain coordinate pairs."

    frame_values = pd.to_numeric(working[frame_column], errors="coerce").to_numpy(dtype=float)
    coordinate_tensor, valid_coordinate_mask = _coordinate_tensor_pixels(
        working,
        coordinate_pairs,
        width,
        height,
    )
    row_indices = working.index.to_numpy(copy=True)
    track_ids = _track_ids(working, detect_track_column(list(working.columns)))
    confidences = _confidence_values(working)
    candidates: list[DuplicateSkeletonCandidate] = []

    finite_frames = sorted({int(value) for value in frame_values[np.isfinite(frame_values)]})
    threshold = max(0.0, float(criteria))
    for frame_value in finite_frames:
        positions = np.flatnonzero(frame_values == frame_value)
        frame_coordinates = coordinate_tensor[positions]
        frame_valid = valid_coordinate_mask[positions]
        for left_index in range(max(0, len(positions) - 1)):
            valid_pairs = frame_valid[left_index + 1 :] & frame_valid[left_index]
            used_counts = valid_pairs.sum(axis=1)
            if not np.any(used_counts > 0):
                continue
            differences = frame_coordinates[left_index + 1 :] - frame_coordinates[left_index]
            distances = np.hypot(differences[:, :, 0], differences[:, :, 1])
            distances[~valid_pairs] = 0.0
            distance_sums = distances.sum(axis=1)
            for relative_index in np.flatnonzero((used_counts > 0) & (distance_sums <= threshold)):
                right_index = left_index + 1 + int(relative_index)
                left_position = int(positions[left_index])
                right_position = int(positions[right_index])
                candidates.append(
                    DuplicateSkeletonCandidate(
                        frame_value=frame_value,
                        row_index_a=row_indices[left_position],
                        row_index_b=row_indices[right_position],
                        track_id_a=track_ids[left_position],
                        track_id_b=track_ids[right_position],
                        confidence_a=float(confidences[left_position]),
                        confidence_b=float(confidences[right_position]),
                        distance_sum=float(distance_sums[relative_index]),
                    )
                )
    candidates.sort(key=lambda item: (item.frame_value, item.distance_sum))
    return tuple(candidates), None


def remove_duplicate_skeletons(
    dataframe: pd.DataFrame,
    *,
    bodyparts: list[str] | None = None,
    criteria: float = 120.0,
    width: int = 1,
    height: int = 1,
) -> TrackingRepairStageResult:
    candidates, warning = find_duplicate_skeleton_candidates(
        dataframe,
        bodyparts=bodyparts,
        criteria=criteria,
        width=width,
        height=height,
    )
    if warning is not None or not candidates:
        return TrackingRepairStageResult(dataframe.copy(), len(candidates), 0, warning)

    adjacency: dict[object, set[object]] = {}
    confidence_by_row: dict[object, float] = {}
    for candidate in candidates:
        adjacency.setdefault(candidate.row_index_a, set()).add(candidate.row_index_b)
        adjacency.setdefault(candidate.row_index_b, set()).add(candidate.row_index_a)
        confidence_by_row[candidate.row_index_a] = max(
            confidence_by_row.get(candidate.row_index_a, float("-inf")),
            candidate.confidence_a,
        )
        confidence_by_row[candidate.row_index_b] = max(
            confidence_by_row.get(candidate.row_index_b, float("-inf")),
            candidate.confidence_b,
        )

    visited: set[object] = set()
    remove_rows: set[object] = set()
    position_by_index = {index: position for position, index in enumerate(dataframe.index)}
    for start_row in adjacency:
        if start_row in visited:
            continue
        component: list[object] = []
        stack = [start_row]
        visited.add(start_row)
        while stack:
            row_index = stack.pop()
            component.append(row_index)
            for neighbor in adjacency.get(row_index, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        keep_row = min(
            component,
            key=lambda row_index: (
                -confidence_by_row.get(row_index, 0.0),
                position_by_index.get(row_index, 0),
            ),
        )
        remove_rows.update(row_index for row_index in component if row_index != keep_row)

    result = dataframe.drop(index=list(remove_rows)).copy()
    return TrackingRepairStageResult(result, len(candidates), len(remove_rows), None)


def _canonical_bodypart_lookup(bodyparts: list[str]) -> dict[str, str]:
    normalized = {_canonical_name(bodypart): bodypart for bodypart in bodyparts}
    resolved: dict[str, str] = {}
    for canonical, aliases in CANONICAL_POINT_ALIASES.items():
        for alias in aliases:
            match = normalized.get(_canonical_name(alias))
            if match is not None:
                resolved[canonical] = match
                break
    return resolved


def find_length_outlier_candidates(
    dataframe: pd.DataFrame,
    *,
    bodyparts: list[str] | None = None,
    high_z_threshold: float = 3.5,
    low_z_threshold: float = 3.5,
    deviation_mode: str = "both",
    width: int = 1,
    height: int = 1,
) -> tuple[tuple[LengthOutlierCandidate, ...], str | None]:
    if dataframe.empty:
        return (), "No tracking rows were available."
    working = dataframe.loc[_real_detection_mask(dataframe)].copy()
    frame_column = detect_frame_column(list(working.columns))
    if frame_column is None:
        return (), "Tracking CSV does not contain a frame column."
    available_bodyparts = list(bodyparts or bodyparts_from_dataframe(working))
    canonical = _canonical_bodypart_lookup(available_bodyparts)
    available_edges = [
        (canonical[start], canonical[end], label)
        for start, end, label in LENGTH_OUTLIER_EDGES
        if start in canonical and end in canonical
    ]
    if not available_edges:
        return (), "Tracking CSV does not contain the body/nose/tail keypoints needed for length outlier detection."

    row_count = len(working)
    distance_matrix = np.full((row_count, len(available_edges)), np.nan, dtype=float)
    edge_bodyparts = list(dict.fromkeys(
        bodypart
        for start, end, _label in available_edges
        for bodypart in (start, end)
    ))
    edge_points, _ = _coordinate_tensor_pixels(
        working,
        [(bodypart, f"{bodypart}.x", f"{bodypart}.y") for bodypart in edge_bodyparts],
        width,
        height,
    )
    point_index = {bodypart: index for index, bodypart in enumerate(edge_bodyparts)}
    for edge_index, (start, end, _label) in enumerate(available_edges):
        start_xy = edge_points[:, point_index[start], :]
        end_xy = edge_points[:, point_index[end], :]
        valid = np.isfinite(start_xy).all(axis=1) & np.isfinite(end_xy).all(axis=1)
        distance_matrix[valid, edge_index] = np.hypot(
            start_xy[valid, 0] - end_xy[valid, 0],
            start_xy[valid, 1] - end_xy[valid, 1],
        )

    upward = np.full_like(distance_matrix, np.nan)
    downward = np.full_like(distance_matrix, np.nan)
    for edge_index in range(len(available_edges)):
        values = distance_matrix[:, edge_index]
        finite_values = values[np.isfinite(values)]
        if finite_values.size < 4:
            continue
        scale = robust_scale(finite_values)
        if scale <= 0.0:
            continue
        median = float(np.median(finite_values))
        finite = np.isfinite(values)
        upward[finite, edge_index] = (values[finite] - median) / scale
        downward[finite, edge_index] = (median - values[finite]) / scale

    normalized_mode = str(deviation_mode or "both").strip().lower()
    if normalized_mode == "high_only":
        deviations = upward
        outlier_mask = np.isfinite(upward) & (upward >= float(high_z_threshold))
    elif normalized_mode == "low_only":
        deviations = downward
        outlier_mask = np.isfinite(downward) & (downward >= float(low_z_threshold))
    else:
        deviations = np.maximum(upward, downward)
        outlier_mask = (
            (np.isfinite(upward) & (upward >= float(high_z_threshold)))
            | (np.isfinite(downward) & (downward >= float(low_z_threshold)))
        )

    frame_values = pd.to_numeric(working[frame_column], errors="coerce").to_numpy(dtype=float)
    row_indices = working.index.to_numpy(copy=True)
    track_ids = _track_ids(working, detect_track_column(list(working.columns)))
    segment_labels = tuple(edge[2] for edge in available_edges)
    candidates: list[LengthOutlierCandidate] = []
    for position in np.flatnonzero(np.any(outlier_mask, axis=1) & np.isfinite(frame_values)):
        segment_mask = outlier_mask[position]
        selected_deviations = deviations[position, segment_mask]
        finite_deviations = selected_deviations[np.isfinite(selected_deviations)]
        if finite_deviations.size == 0:
            continue
        candidates.append(
            LengthOutlierCandidate(
                frame_value=int(frame_values[position]),
                row_index=row_indices[position],
                track_id=track_ids[position],
                segments=tuple(
                    segment_labels[index]
                    for index in np.flatnonzero(segment_mask)
                ),
                max_deviation=float(np.max(finite_deviations)),
            )
        )
    candidates.sort(key=lambda item: (item.frame_value, -item.max_deviation))
    return tuple(candidates), None


def _skeleton_payload_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return coordinates and confidence values without frame/track identifiers."""
    columns: list[str] = []
    for bodypart in bodyparts_from_dataframe(dataframe):
        for suffix in ("x", "y", "score", "confidence", "visibility", "likelihood"):
            column = f"{bodypart}.{suffix}"
            if column in dataframe.columns and column not in columns:
                columns.append(column)
    for candidate in PRIMARY_CONFIDENCE_COLUMNS:
        column = _find_column(list(dataframe.columns), (candidate,))
        if column is not None and column not in columns:
            columns.append(column)
    return columns


def invalidate_length_outlier_skeletons(
    dataframe: pd.DataFrame,
    *,
    bodyparts: list[str] | None = None,
    high_z_threshold: float = 3.5,
    low_z_threshold: float = 3.5,
    deviation_mode: str = "both",
    width: int = 1,
    height: int = 1,
) -> TrackingRepairStageResult:
    """Keep outlier rows and mark their full skeleton payload missing for interpolation."""
    candidates, warning = find_length_outlier_candidates(
        dataframe,
        bodyparts=bodyparts,
        high_z_threshold=high_z_threshold,
        low_z_threshold=low_z_threshold,
        deviation_mode=deviation_mode,
        width=width,
        height=height,
    )
    if warning is not None or not candidates:
        return TrackingRepairStageResult(dataframe.copy(), len(candidates), 0, warning)
    affected_rows = {candidate.row_index for candidate in candidates}
    payload_columns = _skeleton_payload_columns(dataframe)
    result = dataframe.copy()
    if payload_columns:
        result.loc[list(affected_rows), payload_columns] = np.nan
    return TrackingRepairStageResult(
        result,
        len(candidates),
        len(affected_rows),
        None,
    )


def remove_length_outlier_skeletons(
    dataframe: pd.DataFrame,
    **kwargs,
) -> TrackingRepairStageResult:
    """Compatibility wrapper; Z-score outliers are now invalidated instead of removed."""
    return invalidate_length_outlier_skeletons(dataframe, **kwargs)


def run_tracking_postprocess(
    dataframe: pd.DataFrame,
    *,
    config: TrackingPostprocessConfig,
    bodyparts: list[str] | None = None,
    width: int = 1,
    height: int = 1,
) -> TrackingPostprocessResult:
    active = dataframe.copy()
    warnings: list[str] = []
    duplicate_candidates = 0
    duplicate_removed = 0
    length_candidates = 0
    length_invalidated = 0

    if config.remove_duplicates:
        duplicate_result = remove_duplicate_skeletons(
            active,
            bodyparts=bodyparts,
            criteria=config.duplicate_distance_threshold,
            width=width,
            height=height,
        )
        active = duplicate_result.dataframe
        duplicate_candidates = duplicate_result.candidate_count
        duplicate_removed = duplicate_result.removed_count
        if duplicate_result.warning:
            warnings.append(duplicate_result.warning)

    if config.remove_length_outliers:
        length_result = invalidate_length_outlier_skeletons(
            active,
            bodyparts=bodyparts,
            high_z_threshold=config.high_z_threshold,
            low_z_threshold=config.low_z_threshold,
            deviation_mode=config.deviation_mode,
            width=width,
            height=height,
        )
        active = length_result.dataframe
        length_candidates = length_result.candidate_count
        length_invalidated = length_result.affected_count
        if length_result.warning:
            warnings.append(length_result.warning)

    return TrackingPostprocessResult(
        dataframe=active,
        duplicate_candidates=duplicate_candidates,
        duplicate_removed=duplicate_removed,
        length_outlier_candidates=length_candidates,
        length_outliers_invalidated=length_invalidated,
        warnings=tuple(dict.fromkeys(warnings)),
    )


__all__ = [
    "DuplicateSkeletonCandidate",
    "LengthOutlierCandidate",
    "TrackingPostprocessConfig",
    "TrackingPostprocessResult",
    "TrackingRepairStageResult",
    "detect_frame_column",
    "detect_track_column",
    "find_duplicate_skeleton_candidates",
    "find_length_outlier_candidates",
    "invalidate_length_outlier_skeletons",
    "remove_duplicate_skeletons",
    "remove_length_outlier_skeletons",
    "robust_scale",
    "run_tracking_postprocess",
]
