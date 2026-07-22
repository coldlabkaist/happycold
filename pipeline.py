from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


INVALID_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class DataFramePipelineStage:
    key: str
    label: str
    transform: Callable[[pd.DataFrame], pd.DataFrame]
    output_mode: str = "append"
    allow_row_removal: bool = False

    def __post_init__(self) -> None:
        if self.output_mode not in {"append", "replace"}:
            raise ValueError("Pipeline stage output_mode must be 'append' or 'replace'.")
        if self.allow_row_removal and self.output_mode != "replace":
            raise ValueError("Only replace stages may remove rows.")


@dataclass(frozen=True)
class PipelineStageSummary:
    key: str
    label: str
    added_columns: tuple[str, ...]
    derived_columns: tuple[str, ...]
    replaced_columns: tuple[str, ...]
    changed_cells: int
    elapsed_seconds: float
    input_rows: int
    output_rows: int

    @property
    def removed_rows(self) -> int:
        return max(0, self.input_rows - self.output_rows)


@dataclass(frozen=True)
class PipelineRunResult:
    dataframe: pd.DataFrame
    stages: tuple[PipelineStageSummary, ...]

    @property
    def added_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            column
            for stage in self.stages
            for column in (*stage.added_columns, *stage.derived_columns)
        ))

    @property
    def replaced_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            column
            for stage in self.stages
            for column in stage.replaced_columns
        ))


def normalize_filename_affix(value: str) -> str:
    cleaned = INVALID_FILENAME_CHARACTERS.sub("_", str(value).strip())
    return cleaned.strip(" ._")


def build_pipeline_output_filename(
    source_path: str | Path,
    prefix: str = "",
    suffix: str = "processed",
) -> str:
    source_stem = Path(source_path).stem
    parts = [
        part
        for part in (
            normalize_filename_affix(prefix),
            source_stem,
            normalize_filename_affix(suffix),
        )
        if part
    ]
    return f"{'_'.join(parts)}.csv"


def _unique_derived_column_name(
    source_column: str,
    suffix: str,
    occupied_columns: set[str],
) -> str:
    base = f"{source_column}_{suffix}"
    candidate = base
    counter = 2
    while candidate in occupied_columns:
        candidate = f"{base}_{counter}"
        counter += 1
    return candidate


def _changed_cell_count(before: pd.Series, after: pd.Series) -> int:
    equal = before.eq(after) | (before.isna() & after.isna())
    return int((~equal.fillna(False)).sum())


def run_dataframe_pipeline(
    source_df: pd.DataFrame,
    stages: list[DataFramePipelineStage] | tuple[DataFramePipelineStage, ...],
) -> PipelineRunResult:
    """Run stages in memory without mutating the source dataframe.

    ``replace`` stages update the working data (and may remove existing rows when
    explicitly allowed). ``append`` stages keep the working data intact and only
    accumulate new result columns. If an append stage changes an existing column,
    that value is stored under a stage-suffixed derived column instead.
    """
    active_df = source_df.copy()
    summaries: list[PipelineStageSummary] = []

    for stage in stages:
        before_df = active_df
        started_at = time.perf_counter()
        next_df = stage.transform(before_df)
        elapsed_seconds = time.perf_counter() - started_at

        if not isinstance(next_df, pd.DataFrame):
            raise TypeError(f"Pipeline stage '{stage.label}' did not return a DataFrame.")
        removed_columns = [column for column in before_df.columns if column not in next_df.columns]
        if removed_columns:
            raise ValueError(
                f"Pipeline stage '{stage.label}' removed columns: {', '.join(removed_columns)}."
            )
        rows_changed = len(next_df) != len(before_df) or not next_df.index.equals(before_df.index)
        if rows_changed:
            if not stage.allow_row_removal:
                raise ValueError(
                    f"Pipeline stage '{stage.label}' changed the row count or index. "
                    "Pipeline stages must preserve rows unless row removal is explicitly enabled."
                )
            if (
                len(next_df) > len(before_df)
                or not next_df.index.is_unique
                or not next_df.index.isin(before_df.index).all()
            ):
                raise ValueError(
                    f"Pipeline stage '{stage.label}' added or re-keyed rows. "
                    "Repair stages may only remove existing rows."
                )

        added_columns: list[str] = []
        derived_columns: list[str] = []
        replaced_columns: list[str] = []
        changed_cells = 0
        before_aligned = before_df.loc[next_df.index]

        if stage.output_mode == "replace":
            active_df = next_df.copy()
            for column in next_df.columns:
                if column not in before_df.columns:
                    added_columns.append(column)
                    continue
                count = _changed_cell_count(before_aligned[column], next_df[column])
                if count:
                    replaced_columns.append(column)
                    changed_cells += count
        else:
            active_df = before_df.copy()
            occupied_columns = set(active_df.columns)
            for column in next_df.columns:
                if column not in before_df.columns:
                    active_df[column] = next_df[column]
                    occupied_columns.add(column)
                    added_columns.append(column)
                    continue
                count = _changed_cell_count(before_df[column], next_df[column])
                if not count:
                    continue
                suffix = normalize_filename_affix(stage.key) or "result"
                derived_column = _unique_derived_column_name(column, suffix, occupied_columns)
                active_df[derived_column] = next_df[column]
                occupied_columns.add(derived_column)
                derived_columns.append(derived_column)
                changed_cells += count

        summaries.append(
            PipelineStageSummary(
                key=stage.key,
                label=stage.label,
                added_columns=tuple(added_columns),
                derived_columns=tuple(derived_columns),
                replaced_columns=tuple(replaced_columns),
                changed_cells=changed_cells,
                elapsed_seconds=elapsed_seconds,
                input_rows=len(before_df),
                output_rows=len(next_df),
            )
        )

    return PipelineRunResult(dataframe=active_df, stages=tuple(summaries))
