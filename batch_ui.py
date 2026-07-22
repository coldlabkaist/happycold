from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, QSize, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from batch import BatchItem, BatchRunResult, run_batch_exports


@dataclass(frozen=True)
class BatchVideoChoice:
    path: Path
    label: str
    csv_path: Path | None
    selected: bool = False

    @property
    def csv_ready(self) -> bool:
        return self.csv_path is not None


class BatchVideoListRow(QWidget):
    """Compact one-line video row with CSV readiness aligned at the right."""

    def __init__(self, choice: BatchVideoChoice, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("batchVideoListRow")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(8)

        name_label = QLabel(choice.label)
        name_label.setObjectName("batchVideoName")
        name_label.setMinimumWidth(0)
        name_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        status_label = QLabel("Ready" if choice.csv_ready else "No CSV")
        status_label.setProperty("csvState", "ready" if choice.csv_ready else "missing")
        status_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        status_label.setFixedWidth(58)
        if choice.csv_path is not None:
            details = f"Video: {choice.path}\nMatching CSV: {choice.csv_path}"
            status_label.setToolTip(f"Matching CSV: {choice.csv_path}")
        else:
            details = (
                f"Video: {choice.path}\n"
                "No automatically matched CSV; this video will be skipped."
            )
            status_label.setToolTip(
                "No automatically matched CSV; this video will be skipped."
            )
        name_label.setToolTip(details)
        layout.addWidget(name_label, stretch=1)
        layout.addWidget(status_label)


class BatchVideoSelectionDialog(QDialog):
    """Shared video selection UI for batch CSV and multi-pipeline exports."""

    PATH_ROLE = int(Qt.ItemDataRole.UserRole)
    CSV_READY_ROLE = PATH_ROLE + 1

    def __init__(
        self,
        *,
        title: str,
        info_text: str,
        warning_text: str,
        start_button_text: str,
        choices: list[BatchVideoChoice],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("batchVideoSelectionDialog")
        self.setWindowTitle(title)
        self.setModal(True)
        self.setSizeGripEnabled(True)
        self.setMinimumSize(520, 540)
        self.resize(720, 640)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("batchSelectionHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 13, 16, 13)
        header_layout.setSpacing(3)
        eyebrow = QLabel(
            "MULTI PIPELINE" if "pipeline" in title.lower() else "BATCH CSV SAVE"
        )
        eyebrow.setObjectName("batchSelectionEyebrow")
        heading = QLabel(title)
        heading.setObjectName("batchSelectionTitle")
        heading.setWordWrap(True)
        description = QLabel(info_text)
        description.setWordWrap(True)
        description.setProperty("muted", True)
        header_layout.addWidget(eyebrow)
        header_layout.addWidget(heading)
        header_layout.addWidget(description)
        root_layout.addWidget(header)

        notice = QFrame()
        notice.setObjectName("batchSelectionNotice")
        notice_layout = QVBoxLayout(notice)
        notice_layout.setContentsMargins(12, 9, 12, 10)
        notice_layout.setSpacing(3)
        notice_title = QLabel("Before you start")
        notice_title.setObjectName("batchSelectionNoticeTitle")
        notice_text = warning_text.strip()
        if notice_text.lower().startswith("warning:"):
            notice_text = notice_text[len("warning:"):].lstrip()
        self.warning_label = QLabel(notice_text)
        self.warning_label.setObjectName("batchSelectionWarning")
        self.warning_label.setWordWrap(True)
        notice_layout.addWidget(notice_title)
        notice_layout.addWidget(self.warning_label)
        root_layout.addWidget(notice)

        selection_group = QGroupBox("1. Select Videos")
        selection_group.setObjectName("batchSelectionGroup")
        selection_layout = QVBoxLayout(selection_group)
        selection_layout.setContentsMargins(10, 12, 10, 10)
        selection_layout.setSpacing(7)

        quick_actions = QHBoxLayout()
        quick_actions.setSpacing(6)
        self.select_all_button = QPushButton("Select All")
        self.select_ready_button = QPushButton("CSV Ready Only")
        self.select_ready_button.setToolTip(
            "Select only videos that have an automatically matched CSV."
        )
        self.clear_button = QPushButton("Clear")
        quick_actions.addWidget(self.select_all_button)
        quick_actions.addWidget(self.select_ready_button)
        quick_actions.addWidget(self.clear_button)
        quick_actions.addStretch(1)
        selection_layout.addLayout(quick_actions)

        self.selection_status_label = QLabel()
        self.selection_status_label.setObjectName("batchSelectionStatus")
        self.selection_status_label.setWordWrap(False)
        selection_layout.addWidget(self.selection_status_label)

        self.list_widget = QListWidget()
        self.list_widget.setObjectName("batchVideoList")
        self.list_widget.setAlternatingRowColors(False)
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list_widget.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.list_widget.setSelectionRectVisible(True)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.setSpacing(0)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_widget.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        selection_layout.addWidget(self.list_widget, stretch=1)
        root_layout.addWidget(selection_group, stretch=1)

        footer = QFrame()
        footer.setObjectName("batchSelectionFooter")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(12, 8, 12, 8)
        footer_layout.setSpacing(0)

        action_row = QHBoxLayout()
        action_row.setSpacing(7)
        self.cancel_button = QPushButton("Cancel")
        self.start_button = QPushButton(start_button_text)
        self.start_button.setProperty("primary", True)
        self.start_button.setDefault(True)
        action_row.addStretch(1)
        action_row.addWidget(self.cancel_button)
        action_row.addWidget(self.start_button)
        footer_layout.addLayout(action_row)
        root_layout.addWidget(footer)

        for choice in choices:
            item = QListWidgetItem()
            item.setData(self.PATH_ROLE, str(choice.path))
            item.setData(self.CSV_READY_ROLE, choice.csv_ready)
            item.setSizeHint(QSize(0, 30))
            item.setToolTip(
                f"Video: {choice.path}\n"
                + (
                    f"Matching CSV: {choice.csv_path}"
                    if choice.csv_path is not None
                    else "No automatically matched CSV; this video will be skipped."
                )
            )
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(
                item,
                BatchVideoListRow(choice, self.list_widget),
            )
            item.setSelected(choice.selected)

        self.select_all_button.clicked.connect(self._toggle_select_all)
        self.select_ready_button.clicked.connect(self._select_ready_only)
        self.clear_button.clicked.connect(self._clear_selection)
        self.cancel_button.clicked.connect(self.reject)
        self.start_button.clicked.connect(self._accept_if_ready)
        self.list_widget.itemSelectionChanged.connect(self._refresh_selection_controls)
        self._refresh_selection_controls()

    @staticmethod
    def _item_path(item: QListWidgetItem) -> Path | None:
        value = item.data(BatchVideoSelectionDialog.PATH_ROLE)
        return Path(str(value)) if value else None

    @staticmethod
    def _item_is_ready(item: QListWidgetItem) -> bool:
        return bool(item.data(BatchVideoSelectionDialog.CSV_READY_ROLE))

    def _selection_counts(self) -> tuple[int, int, int, int]:
        selected_items = self.list_widget.selectedItems()
        selected_count = len(selected_items)
        selected_ready_count = sum(
            1 for item in selected_items if self._item_is_ready(item)
        )
        ready_count = sum(
            1
            for index in range(self.list_widget.count())
            if self._item_is_ready(self.list_widget.item(index))
        )
        return selected_count, self.list_widget.count(), selected_ready_count, ready_count

    def _set_selection(self, predicate) -> None:
        self.list_widget.blockSignals(True)
        try:
            for index in range(self.list_widget.count()):
                item = self.list_widget.item(index)
                item.setSelected(bool(predicate(item)))
        finally:
            self.list_widget.blockSignals(False)
        self._refresh_selection_controls()

    def _toggle_select_all(self) -> None:
        selected_count, total_count, _selected_ready, _ready = self._selection_counts()
        select_everything = not (total_count > 0 and selected_count == total_count)
        self._set_selection(lambda _item: select_everything)

    def _select_ready_only(self) -> None:
        self._set_selection(self._item_is_ready)

    def _clear_selection(self) -> None:
        self._set_selection(lambda _item: False)

    def _refresh_selection_controls(self) -> None:
        selected_count, total_count, selected_ready_count, ready_count = (
            self._selection_counts()
        )
        selected_missing_count = selected_count - selected_ready_count
        all_selected = total_count > 0 and selected_count == total_count
        self.select_all_button.setText("Deselect All" if all_selected else "Select All")
        self.select_all_button.setEnabled(total_count > 0)
        self.select_ready_button.setEnabled(
            ready_count > 0
            and (selected_count != ready_count or selected_ready_count != ready_count)
        )
        self.clear_button.setEnabled(selected_count > 0)
        self.start_button.setEnabled(selected_ready_count > 0)
        self.start_button.setDefault(selected_ready_count > 0)

        parts = [
            f"Selected: {selected_count} / {total_count}",
            f"Ready to process: {selected_ready_count} / {ready_count}",
        ]
        if selected_missing_count:
            parts.append(f"Will skip: {selected_missing_count}")
        elif ready_count == 0:
            parts.append("No matching CSVs found")
        self.selection_status_label.setText("   |   ".join(parts))

        if selected_ready_count == 0:
            self.start_button.setToolTip(
                "Select at least one video with a matching CSV."
            )
        elif selected_missing_count:
            self.start_button.setToolTip(
                "Selected videos without matching CSVs will be skipped."
            )
        else:
            self.start_button.setToolTip("")

    def _accept_if_ready(self) -> None:
        _selected, _total, selected_ready, _ready = self._selection_counts()
        if selected_ready <= 0:
            return
        self.accept()

    def selected_paths(self) -> list[Path]:
        return [
            path
            for item in self.list_widget.selectedItems()
            if (path := self._item_path(item)) is not None
        ]


class BatchExportWorker(QObject):
    """Run one shared batch export without touching UI objects."""

    item_started = pyqtSignal(int, int, str)
    item_finished = pyqtSignal(int, int, str, str, int, int, int)
    completed = pyqtSignal(object)
    crashed = pyqtSignal(str)

    def __init__(
        self,
        *,
        selected_videos: list[Path],
        source_width: int,
        source_height: int,
        csv_candidates_for: Callable[[Path], list[Path]],
        video_size_for: Callable[[Path], tuple[int, int] | None],
        export_item: Callable[[BatchItem], Path],
        require_bodyparts: bool = True,
    ) -> None:
        super().__init__()
        self._selected_videos = list(selected_videos)
        self._source_width = int(source_width)
        self._source_height = int(source_height)
        self._csv_candidates_for = csv_candidates_for
        self._video_size_for = video_size_for
        self._export_item = export_item
        self._require_bodyparts = bool(require_bodyparts)
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = run_batch_exports(
                selected_videos=self._selected_videos,
                source_width=self._source_width,
                source_height=self._source_height,
                csv_candidates_for=self._csv_candidates_for,
                video_size_for=self._video_size_for,
                export_item=self._export_item,
                progress=lambda index, total, path: self.item_started.emit(
                    index, total, str(path)
                ),
                item_finished=lambda index, total, path, outcome, saved, skipped, failed: (
                    self.item_finished.emit(
                        index,
                        total,
                        str(path),
                        outcome,
                        saved,
                        skipped,
                        failed,
                    )
                ),
                should_cancel=self._cancel_event.is_set,
                require_bodyparts=self._require_bodyparts,
            )
        except Exception as exc:
            self.crashed.emit(str(exc))
            return
        self.completed.emit(result)


class BatchProgressDialog(QDialog):
    cancel_requested = pyqtSignal()

    def __init__(
        self,
        title: str,
        activity_text: str,
        total: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._running = True
        self._activity_text = activity_text
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setMinimumWidth(540)
        self.resize(620, 390)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        self.status_label = QLabel(activity_text)
        self.status_label.setStyleSheet("font-size: 15px; font-weight: 700;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.current_file_label = QLabel("Preparing the background worker...")
        self.current_file_label.setWordWrap(True)
        self.current_file_label.setProperty("muted", True)
        layout.addWidget(self.current_file_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, max(1, int(total)))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m files")
        layout.addWidget(self.progress_bar)

        self.counter_label = QLabel("Saved 0  ·  Skipped 0  ·  Failed 0")
        self.counter_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.counter_label)

        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("Per-file results will appear here.")
        self.details.setMaximumBlockCount(500)
        layout.addWidget(self.details, stretch=1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.cancel_button = QPushButton("Cancel after current file")
        self.close_button = QPushButton("Close")
        self.close_button.setEnabled(False)
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

        self.cancel_button.clicked.connect(self._request_cancel)
        self.close_button.clicked.connect(self.accept)

    @property
    def is_running(self) -> bool:
        return self._running

    def _request_cancel(self) -> None:
        if not self._running:
            return
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancelling after the current file finishes...")
        self.cancel_requested.emit()

    def update_item_started(self, index: int, total: int, path_text: str) -> None:
        path = Path(path_text)
        self.status_label.setText(self._activity_text)
        self.current_file_label.setText(f"Processing {index}/{total}: {path.name}")
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(max(0, index - 1))

    def update_item_finished(
        self,
        index: int,
        total: int,
        path_text: str,
        outcome: str,
        saved: int,
        skipped: int,
        failed: int,
    ) -> None:
        path = Path(path_text)
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(index)
        self.counter_label.setText(
            f"Saved {saved}  ·  Skipped {skipped}  ·  Failed {failed}"
        )
        outcome_labels = {
            "saved": "Saved",
            "skipped": "Skipped — no matching CSV",
            "failed": "Failed",
        }
        self.details.appendPlainText(
            f"{index:>3}/{total}  {outcome_labels.get(outcome, outcome)}  ·  {path.name}"
        )

    def mark_completed(self, result: BatchRunResult) -> None:
        self._running = False
        if result.cancelled:
            self.status_label.setText("Batch cancelled")
            self.current_file_label.setText(
                "The current file was allowed to finish safely; remaining files were not started."
            )
        elif result.failed:
            self.status_label.setText("Batch completed with warnings")
            self.current_file_label.setText("Review failed items in the details below.")
            self.progress_bar.setValue(self.progress_bar.maximum())
        else:
            self.status_label.setText("Batch completed")
            self.current_file_label.setText("All selected files have been processed.")
            self.progress_bar.setValue(self.progress_bar.maximum())
        self.counter_label.setText(
            f"Saved {result.saved_count}  ·  "
            f"Skipped {len(result.skipped_auto_missing)}  ·  Failed {len(result.failed)}"
        )
        for path, reason in result.failed:
            self.details.appendPlainText(f"      Error · {path.name}: {reason}")
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)
        self.close_button.setProperty("primary", True)
        self.close_button.setFocus()

    def mark_crashed(self, message: str) -> None:
        self._running = False
        self.status_label.setText("Batch stopped unexpectedly")
        self.current_file_label.setText(message or "Unknown worker error")
        self.details.appendPlainText(f"Worker error · {message}")
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)

    def closeEvent(self, event) -> None:
        if self._running:
            self.status_label.setText(
                "Batch is still running. Use Cancel to stop after the current file."
            )
            event.ignore()
            return
        super().closeEvent(event)
