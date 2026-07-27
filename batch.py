from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from trajectory import bodyparts_from_dataframe


class ExportCancelled(Exception):
    """Raised by export tasks when a user-requested cancel can be honored safely."""


def raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise ExportCancelled()


def _callable_accepts_cancel(callback: Callable[..., object]) -> bool:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return False
    return (
        "should_cancel" in signature.parameters
        or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    )


def call_export_item(
    export_item: Callable[..., Path],
    item: "BatchItem",
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    if _callable_accepts_cancel(export_item):
        return Path(export_item(item, should_cancel=should_cancel))
    return Path(export_item(item))


def call_single_export(
    export_item: Callable[..., Path],
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    if _callable_accepts_cancel(export_item):
        return Path(export_item(should_cancel=should_cancel))
    return Path(export_item())


@dataclass(frozen=True)
class BatchItem:
    video_path: Path
    csv_path: Path
    source_df: pd.DataFrame
    bodyparts: list[str]
    width: int
    height: int
    scale_x: float
    scale_y: float


@dataclass
class BatchRunResult:
    saved_paths: list[Path] = field(default_factory=list)
    skipped_auto_missing: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
    cancelled: bool = False

    @property
    def saved_count(self) -> int:
        return len(self.saved_paths)


def run_batch_exports(
    selected_videos: list[Path],
    source_width: int,
    source_height: int,
    csv_candidates_for: Callable[[Path], list[Path]],
    video_size_for: Callable[[Path], tuple[int, int] | None],
    export_item: Callable[[BatchItem], Path],
    progress: Callable[[int, int, Path], None] | None = None,
    yield_events: Callable[[], None] | None = None,
    require_bodyparts: bool = True,
    item_finished: Callable[[int, int, Path, str, int, int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> BatchRunResult:
    """Run the shared video/CSV batch pipeline and collect per-item outcomes."""
    safe_source_width = max(1, int(source_width))
    safe_source_height = max(1, int(source_height))
    result = BatchRunResult()
    total = len(selected_videos)

    for index, video_path in enumerate(selected_videos, start=1):
        if should_cancel is not None and should_cancel():
            result.cancelled = True
            break
        if progress is not None:
            progress(index, total, video_path)
        if yield_events is not None:
            yield_events()

        outcome = "failed"
        try:
            csv_candidates = csv_candidates_for(video_path)
            if not csv_candidates:
                result.skipped_auto_missing.append(video_path)
                outcome = "skipped"
                continue
            csv_path = csv_candidates[0]

            video_size = video_size_for(video_path)
            if video_size is None:
                raise ValueError("Could not read video size.")
            width, height = video_size
            width = max(1, int(width))
            height = max(1, int(height))

            source_df = pd.read_csv(csv_path)
            bodyparts = bodyparts_from_dataframe(source_df)
            if require_bodyparts and not bodyparts:
                raise ValueError("No bodyparts found in the CSV.")

            item = BatchItem(
                video_path=video_path,
                csv_path=csv_path,
                source_df=source_df,
                bodyparts=bodyparts,
                width=width,
                height=height,
                scale_x=width / safe_source_width,
                scale_y=height / safe_source_height,
            )
            raise_if_cancelled(should_cancel)
            result.saved_paths.append(call_export_item(export_item, item, should_cancel))
            outcome = "saved"
        except ExportCancelled:
            result.cancelled = True
            outcome = "cancelled"
            break
        except Exception as exc:
            result.failed.append((video_path, str(exc)))
            outcome = "failed"
        finally:
            if item_finished is not None:
                item_finished(
                    index,
                    total,
                    video_path,
                    outcome,
                    result.saved_count,
                    len(result.skipped_auto_missing),
                    len(result.failed),
                )

    return result
