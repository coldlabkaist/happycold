from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from batch import BatchItem, BatchRunResult
from interpolation import build_interpolation_pipeline_dataframe
from pipeline import (
    DataFramePipelineStage,
    PipelineRunResult,
    build_pipeline_output_filename,
    normalize_filename_affix,
    run_dataframe_pipeline,
)
from shared import (
    MaskRecord,
    RoomRecord,
    build_chamber_mark_dataframe,
    build_circle_detection_dataframe,
    build_normalized_dataframe,
    build_occlusion_dataframe,
)
from tracking_postprocess import invalidate_length_outlier_skeletons, remove_duplicate_skeletons


PIPELINE_STAGE_LABELS = {
    "duplicate_removal": "Duplicate Skeleton Removal",
    "zscore_removal": "Robust Z-score Outlier Invalidation",
    "clean_repair": "Region Removal / Interpolation",
    "square": "Square Normalization",
    "chamber": "Chamber Detection",
    "circle": "Circle Detection",
    "occlusion": "Occlusion Detection",
}


def _duplicate_stage_dataframe(
    dataframe,
    *,
    bodyparts: list[str],
    criteria: float,
    width: int,
    height: int,
):
    return remove_duplicate_skeletons(
        dataframe,
        bodyparts=bodyparts,
        criteria=criteria,
        width=width,
        height=height,
    ).dataframe


def _zscore_stage_dataframe(
    dataframe,
    *,
    bodyparts: list[str],
    high_z_threshold: float,
    low_z_threshold: float,
    deviation_mode: str,
    width: int,
    height: int,
):
    return invalidate_length_outlier_skeletons(
        dataframe,
        bodyparts=bodyparts,
        high_z_threshold=high_z_threshold,
        low_z_threshold=low_z_threshold,
        deviation_mode=deviation_mode,
        width=width,
        height=height,
    ).dataframe


@dataclass(frozen=True)
class PipelineSnapshot:
    enabled_stages: frozenset[str]
    duplicate_distance_threshold: float
    zscore_high_threshold: float
    zscore_low_threshold: float
    zscore_deviation_mode: str
    interpolation_mask: np.ndarray | None
    interpolation_removal_mode: str
    interpolation_anchor: str | None
    interpolation_enabled: bool
    interpolation_extrapolate: bool
    chamber_mask: np.ndarray | None
    chamber_boundary_mode: str
    rooms: tuple[RoomRecord, ...]
    circle_geometry: tuple[tuple[float, float], float, float] | None
    masks: tuple[MaskRecord, ...]
    occlusion_quad_points: tuple[tuple[float, float], ...]
    square_points: tuple[tuple[float, float], ...]


class PipelinePanelMixin:
    @staticmethod
    def _build_output_affix_layout(
        prefix_edit: QLineEdit,
        suffix_edit: QLineEdit,
    ) -> QGridLayout:
        """Build the compact two-column Prefix/Suffix editor shared by Output panels."""
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(3)
        prefix_label = QLabel("Prefix")
        suffix_label = QLabel("Suffix")
        prefix_label.setProperty("muted", True)
        suffix_label.setProperty("muted", True)
        grid.addWidget(prefix_label, 0, 0)
        grid.addWidget(suffix_label, 0, 1)
        grid.addWidget(prefix_edit, 1, 0)
        grid.addWidget(suffix_edit, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        return grid

    def _build_output_workspace(self) -> QWidget:
        workspace = QFrame()
        workspace.setObjectName("outputWorkspace")
        workspace.setMinimumHeight(190)
        workspace.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.output_workspace_title_label = QLabel("CURRENT OUTPUT")
        self.output_workspace_title_label.setObjectName("outputWorkspaceTitle")
        layout.addWidget(self.output_workspace_title_label)

        self.output_workspace_stack = QStackedWidget()
        self.output_workspace_stack.addWidget(self._build_current_output_panel())
        self.output_workspace_stack.addWidget(self._build_trajectory_output_panel())
        self.output_workspace_stack.addWidget(self._build_pipeline_output_panel())
        layout.addWidget(self.output_workspace_stack, stretch=1)
        self._set_output_workspace_mode(self.mode_tabs.currentIndex())
        return workspace

    def _build_current_output_panel(self) -> QWidget:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setObjectName("currentOutputScroll")
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(5)

        def add_caption(text: str) -> None:
            caption = QLabel(text)
            caption.setProperty("muted", True)
            layout.addWidget(caption)

        self.save_folder_label = QLabel("-")
        self.save_folder_label.setWordWrap(True)
        self.save_folder_label.setMinimumWidth(0)
        self.save_folder_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.save_folder_label.setProperty("muted", True)
        self.choose_save_folder_button = QPushButton("Choose Folder")

        self.current_output_prefix_edit = QLineEdit(
            str(self.settings.get("current_output_prefix", ""))
        )
        self.current_output_prefix_edit.setPlaceholderText("Optional")
        self.current_output_prefix_edit.setMaxLength(80)
        self.current_output_prefix_edit.setMinimumWidth(0)
        self.current_output_suffix_edit = QLineEdit(
            str(self.settings.get("current_output_suffix", ""))
        )
        self.current_output_suffix_edit.setPlaceholderText("Optional")
        self.current_output_suffix_edit.setMaxLength(80)
        self.current_output_suffix_edit.setMinimumWidth(0)
        saved_current_suffix = self.current_output_suffix_edit.text()
        if "current_output_auto_suffix" in self.settings:
            saved_auto_suffix = self.settings.get("current_output_auto_suffix")
            self._current_output_auto_suffix: str | None = (
                str(saved_auto_suffix)
                if saved_auto_suffix in {"", "annotated"}
                and saved_current_suffix == str(saved_auto_suffix)
                else None
            )
        else:
            # Migrate older settings: blank and "annotated" were the only automatic values.
            self._current_output_auto_suffix = (
                saved_current_suffix if saved_current_suffix in {"", "annotated"} else None
            )

        self.current_output_label = QLabel("-")
        self.current_output_label.setWordWrap(True)
        self.current_output_label.setMinimumWidth(0)
        self.current_output_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.current_output_label.setProperty("muted", True)
        self.save_current_button = QPushButton("Save CSV")
        self.save_current_button.setProperty("primary", True)
        self.save_multiple_button = QPushButton("Save Multiple CSVs")
        self.save_multiple_button.setProperty("batchAction", True)
        self.save_multiple_button.setToolTip(
            "Apply current mode settings to multiple selected videos at once."
        )

        add_caption("Folder")
        layout.addWidget(self.save_folder_label)
        layout.addWidget(self.choose_save_folder_button)
        self.current_output_affix_layout = self._build_output_affix_layout(
            self.current_output_prefix_edit,
            self.current_output_suffix_edit,
        )
        layout.addLayout(self.current_output_affix_layout)
        add_caption("Output")
        layout.addWidget(self.current_output_label)

        button_row = QHBoxLayout()
        button_row.setSpacing(6)
        button_row.addWidget(self.save_current_button, stretch=1)
        button_row.addWidget(self.save_multiple_button, stretch=1)
        layout.addLayout(button_row)
        layout.addStretch(1)

        scroll.setWidget(content)
        panel_layout.addWidget(scroll)

        self.current_output_prefix_edit.textChanged.connect(
            self._on_current_output_naming_changed
        )
        self.current_output_suffix_edit.textChanged.connect(
            self._on_current_output_naming_changed
        )
        self.current_output_suffix_edit.textEdited.connect(
            self._on_current_output_suffix_edited
        )
        self.current_output_prefix_edit.editingFinished.connect(self._save_settings)
        self.current_output_suffix_edit.editingFinished.connect(self._save_settings)
        return panel

    def _build_trajectory_output_panel(self) -> QWidget:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setObjectName("trajectoryOutputScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)

        info = QLabel(
            "Batch exports use the coordinate mode and optional time range configured in Inspect > Trajectory."
        )
        info.setWordWrap(True)
        info.setProperty("muted", True)
        layout.addWidget(info)

        self.trajectory_output_folder_label = QLabel(str(self.save_folder))
        self.trajectory_output_folder_label.setWordWrap(True)
        self.trajectory_output_folder_label.setMinimumWidth(0)
        self.trajectory_output_folder_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.trajectory_output_folder_label.setProperty("muted", True)
        self.trajectory_choose_save_folder_button = QPushButton("Choose Folder")
        folder_form = QFormLayout()
        folder_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        folder_form.addRow("Folder", self.trajectory_output_folder_label)
        folder_form.addRow("", self.trajectory_choose_save_folder_button)
        layout.addLayout(folder_form)

        self.square_batch_heatmap_overlay_button = QPushButton(
            "Save Multiple Heatmaps with Background"
        )
        self.square_batch_heatmap_overlay_button.setProperty("batchAction", True)
        self.square_batch_heatmap_overlay_button.setEnabled(False)
        self.square_batch_heatmap_overlay_button.setToolTip(
            "Save background-composited heatmap PNG files for multiple selected videos."
        )
        self.square_batch_heatmap_plain_button = QPushButton(
            "Save Multiple Heatmaps Only"
        )
        self.square_batch_heatmap_plain_button.setProperty("batchAction", True)
        self.square_batch_heatmap_plain_button.setEnabled(False)
        self.square_batch_heatmap_plain_button.setToolTip(
            "Save background-free heatmap PNG files for multiple selected videos."
        )
        export_group = QGroupBox("Batch Heatmap Images")
        export_layout = QVBoxLayout(export_group)
        export_layout.addWidget(self.square_batch_heatmap_overlay_button)
        export_layout.addWidget(self.square_batch_heatmap_plain_button)
        layout.addWidget(export_group)

        save_note = QLabel(
            "For one image, open a preview and right-click it to use Save As."
        )
        save_note.setWordWrap(True)
        save_note.setProperty("muted", True)
        layout.addWidget(save_note)
        layout.addStretch(1)

        scroll.setWidget(content)
        panel_layout.addWidget(scroll)

        self.trajectory_choose_save_folder_button.clicked.connect(self.choose_save_folder)
        self.square_batch_heatmap_overlay_button.clicked.connect(
            self.save_multiple_square_heatmaps_with_background
        )
        self.square_batch_heatmap_plain_button.clicked.connect(
            self.save_multiple_square_heatmaps_without_background
        )
        return panel

    def _build_pipeline_tab(self) -> QWidget:
        tab = QWidget()
        tab.setMinimumWidth(0)
        tab.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(8, 8, 8, 8)
        tab_layout.setSpacing(7)

        info = QLabel(
            "Repair updates the new CSV; normalization and analysis append new columns."
        )
        info.setWordWrap(True)
        info.setMinimumWidth(0)
        info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        info.setProperty("muted", True)
        info.setToolTip(
            "All selected stages share one in-memory working dataset. Open Settings to configure a stage."
        )
        tab_layout.addWidget(info)

        scroll = QScrollArea()
        scroll.setObjectName("pipelineScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(0)
        scroll_content = QWidget()
        scroll_content.setMinimumWidth(0)
        scroll_content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(4, 4, 4, 8)
        scroll_layout.setSpacing(8)

        self.pipeline_stage_controls: dict[str, dict[str, QWidget]] = {}
        saved_stages = self.settings.get("pipeline_stages", {})
        if not isinstance(saved_stages, dict):
            saved_stages = {}
        phase_specs = (
            (
                "1-1. Tracking Repair",
                (
                    ("duplicate_removal", "Duplicates", self.TAB_TRACKING_REPAIR),
                    ("zscore_removal", "Z-score outliers", self.TAB_TRACKING_REPAIR),
                    ("clean_repair", "Region repair", self.TAB_INTERPOLATION),
                ),
            ),
            (
                "1-2. Square Normalize",
                (("square", "Square coordinates", self.TAB_SQUARE),),
            ),
            (
                "2. Analysis Columns",
                (
                    ("chamber", "Chamber membership", self.TAB_CHAMBER),
                    ("circle", "Circle in/out", self.TAB_CIRCLE),
                    ("occlusion", "Occlusion", self.TAB_OCCLUSION),
                ),
            ),
        )
        for phase_title, stage_specs in phase_specs:
            phase_group = QGroupBox(phase_title)
            phase_group.setMinimumWidth(0)
            phase_group.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            phase_layout = QVBoxLayout(phase_group)
            phase_layout.setContentsMargins(8, 10, 8, 8)
            phase_layout.setSpacing(7)
            for key, checkbox_text, tab_index in stage_specs:
                phase_layout.addWidget(
                    self._build_pipeline_stage_card(
                        key,
                        checkbox_text,
                        tab_index,
                        checked=bool(saved_stages.get(key, False)),
                    )
                )
            scroll_layout.addWidget(phase_group)
        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_content)
        tab_layout.addWidget(scroll, stretch=1)

        for controls in self.pipeline_stage_controls.values():
            checkbox = controls["checkbox"]
            assert isinstance(checkbox, QCheckBox)
            checkbox.toggled.connect(self._on_pipeline_option_changed)
        return tab

    def _build_pipeline_output_panel(self) -> QWidget:
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setObjectName("pipelineOutputScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(0)
        content = QWidget()
        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)

        folder_form = QFormLayout()
        folder_form.setHorizontalSpacing(8)
        folder_form.setVerticalSpacing(6)
        folder_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)

        self.pipeline_save_folder_label = QLabel("-")
        self.pipeline_save_folder_label.setWordWrap(True)
        self.pipeline_save_folder_label.setMinimumWidth(0)
        self.pipeline_save_folder_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.pipeline_save_folder_label.setProperty("muted", True)
        self.pipeline_choose_save_folder_button = QPushButton("Choose Folder")

        self.pipeline_prefix_edit = QLineEdit(
            str(self.settings.get("pipeline_prefix", ""))
        )
        self.pipeline_prefix_edit.setPlaceholderText("Optional")
        self.pipeline_prefix_edit.setMaxLength(80)
        self.pipeline_prefix_edit.setMinimumWidth(0)
        self.pipeline_suffix_edit = QLineEdit(
            str(self.settings.get("pipeline_suffix", "processed"))
        )
        self.pipeline_suffix_edit.setPlaceholderText("Optional")
        self.pipeline_suffix_edit.setMaxLength(80)
        self.pipeline_suffix_edit.setMinimumWidth(0)
        self.pipeline_output_preview_label = QLabel("-")
        self.pipeline_output_preview_label.setWordWrap(True)
        self.pipeline_output_preview_label.setMinimumWidth(0)
        self.pipeline_output_preview_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.pipeline_output_preview_label.setProperty("muted", True)

        folder_form.addRow("Folder", self.pipeline_save_folder_label)
        folder_form.addRow("", self.pipeline_choose_save_folder_button)
        layout.addLayout(folder_form)
        self.pipeline_output_affix_layout = self._build_output_affix_layout(
            self.pipeline_prefix_edit,
            self.pipeline_suffix_edit,
        )
        layout.addLayout(self.pipeline_output_affix_layout)
        preview_form = QFormLayout()
        preview_form.setHorizontalSpacing(8)
        preview_form.setVerticalSpacing(6)
        preview_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        preview_form.addRow("Preview", self.pipeline_output_preview_label)
        layout.addLayout(preview_form)

        output_note = QLabel(
            "The source CSV is never changed. Duplicate rows may be removed in the new CSV; repair stages "
            "update working coordinates, while normalization and analysis append columns."
        )
        output_note.setWordWrap(True)
        output_note.setProperty("muted", True)
        layout.addWidget(output_note)

        self.pipeline_status_label = QLabel("Select at least one stage.")
        self.pipeline_status_label.setWordWrap(True)
        self.pipeline_status_label.setProperty("muted", True)
        layout.addWidget(self.pipeline_status_label)

        self.pipeline_preview_button = QPushButton("Preview Result")
        self.pipeline_preview_button.setMinimumWidth(0)
        layout.addWidget(self.pipeline_preview_button)

        run_row = QVBoxLayout()
        run_row.setSpacing(5)
        self.pipeline_run_current_button = QPushButton("Save CSV")
        self.pipeline_run_current_button.setMinimumWidth(0)
        self.pipeline_run_current_button.setProperty("primary", True)
        self.pipeline_run_batch_button = QPushButton("Save Multiple CSVs")
        self.pipeline_run_batch_button.setMinimumWidth(0)
        self.pipeline_run_batch_button.setProperty("batchAction", True)
        run_row.addWidget(self.pipeline_run_current_button)
        run_row.addWidget(self.pipeline_run_batch_button)
        layout.addLayout(run_row)
        layout.addStretch(1)

        scroll.setWidget(content)
        panel_layout.addWidget(scroll)

        self.pipeline_prefix_edit.textChanged.connect(self._refresh_pipeline_ui)
        self.pipeline_suffix_edit.textChanged.connect(self._refresh_pipeline_ui)
        self.pipeline_prefix_edit.editingFinished.connect(self._save_settings)
        self.pipeline_suffix_edit.editingFinished.connect(self._save_settings)
        self.pipeline_choose_save_folder_button.clicked.connect(self.choose_save_folder)
        self.pipeline_preview_button.clicked.connect(self.preview_pipeline_result)
        self.pipeline_run_current_button.clicked.connect(self.run_current_pipeline)
        self.pipeline_run_batch_button.clicked.connect(self.run_batch_pipeline)
        return panel

    def _set_output_workspace_mode(self, tab_index: int) -> None:
        if not hasattr(self, "output_workspace_stack"):
            return

        workspace = getattr(self, "output_workspace", None)
        if workspace is not None:
            workspace.setVisible(tab_index != self.TAB_PIN)

        if tab_index == self.TAB_PIPELINE:
            page_index = 2
            title = "PIPELINE OUTPUT"
        elif tab_index == self.TAB_TRAJECTORY:
            page_index = 1
            title = "TRAJECTORY OUTPUT"
        else:
            page_index = 0
            title = "CURRENT OUTPUT"
        self.output_workspace_stack.setCurrentIndex(page_index)
        self.output_workspace_title_label.setText(title)

    def _on_current_output_naming_changed(self, *_args) -> None:
        self._refresh_output_ui()

    def _on_current_output_suffix_edited(self, _text: str) -> None:
        # Once the user types, tab changes must never replace that explicit choice.
        self._current_output_auto_suffix = None

    def _default_current_output_suffix_for_tab(self, tab_index: int) -> str:
        if tab_index in {self.TAB_CHAMBER, self.TAB_CIRCLE, self.TAB_OCCLUSION}:
            return "annotated"
        return ""

    def _sync_current_output_default_suffix(self, tab_index: int) -> None:
        suffix_edit = getattr(self, "current_output_suffix_edit", None)
        if suffix_edit is None or getattr(self, "_current_output_auto_suffix", None) is None:
            return
        current = suffix_edit.text()
        previous_auto = self._current_output_auto_suffix
        if current != previous_auto:
            self._current_output_auto_suffix = None
            return
        desired = self._default_current_output_suffix_for_tab(tab_index)
        if current == desired:
            return
        suffix_edit.blockSignals(True)
        suffix_edit.setText(desired)
        suffix_edit.blockSignals(False)
        self._current_output_auto_suffix = desired

    def _current_output_affixes(self) -> tuple[str, str]:
        prefix_edit = getattr(self, "current_output_prefix_edit", None)
        suffix_edit = getattr(self, "current_output_suffix_edit", None)
        raw_prefix = (
            prefix_edit.text()
            if prefix_edit is not None
            else str(self.settings.get("current_output_prefix", ""))
        )
        raw_suffix = (
            suffix_edit.text()
            if suffix_edit is not None
            else str(self.settings.get("current_output_suffix", ""))
        )
        return normalize_filename_affix(raw_prefix), normalize_filename_affix(raw_suffix)

    @staticmethod
    def _apply_output_affixes(output_path: Path, prefix: str, suffix: str) -> Path:
        parts = [part for part in (prefix, output_path.stem, suffix) if part]
        return output_path.with_name(f"{'_'.join(parts)}{output_path.suffix}")

    def _apply_current_output_affixes(self, output_path: Path) -> Path:
        prefix, suffix = self._current_output_affixes()
        return self._apply_output_affixes(output_path, prefix, suffix)

    def _build_pipeline_stage_card(
        self,
        key: str,
        checkbox_text: str,
        tab_index: int,
        checked: bool,
    ) -> QFrame:
        card = QFrame()
        card.setObjectName("pipelineStageCard")
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setMinimumWidth(0)
        card.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 7, 8, 7)
        layout.setSpacing(4)

        checkbox = QCheckBox(checkbox_text)
        checkbox.setMinimumWidth(0)
        checkbox.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        checkbox.setToolTip(PIPELINE_STAGE_LABELS.get(key, checkbox_text))
        checkbox.setChecked(checked)
        status = QLabel("Disabled")
        status.setMinimumWidth(0)
        status.setWordWrap(False)
        status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        status.setProperty("muted", True)
        status.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        summary = QLabel("-")
        summary.setMinimumWidth(0)
        summary.setWordWrap(True)
        summary.setProperty("muted", True)
        summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        open_button = QPushButton("Settings")
        open_button.setMinimumWidth(0)
        open_button.setToolTip("Open the tab that configures this pipeline stage.")
        open_button.clicked.connect(
            lambda _checked=False, index=tab_index: self._open_pipeline_settings(index)
        )

        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        header_row.addWidget(checkbox, stretch=1)
        header_row.addWidget(status, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(header_row)
        layout.addWidget(summary)
        layout.addWidget(open_button)
        self.pipeline_stage_controls[key] = {
            "checkbox": checkbox,
            "status": status,
            "summary": summary,
        }
        return card

    def _open_pipeline_settings(self, tab_index: int) -> None:
        actions = getattr(self, "tab_visibility_actions", {})
        action = actions.get(tab_index)
        if action is not None and not action.isChecked():
            action.setChecked(True)
        if hasattr(self, "workflow_category_bar"):
            self.workflow_category_bar.setCurrentIndex(self._mode_tab_category(tab_index))
            self._apply_workflow_category_visibility(select_first=False)
        self.mode_tabs.setCurrentIndex(tab_index)

    def _on_pipeline_option_changed(self, _checked: bool) -> None:
        self._refresh_pipeline_ui()
        self._save_settings()

    def _pipeline_settings_payload(self) -> dict:
        controls = getattr(self, "pipeline_stage_controls", {})
        stages = {
            key: bool(stage_controls["checkbox"].isChecked())
            for key, stage_controls in controls.items()
        }

        def _text_or_setting(widget_name: str, setting_name: str, default: str) -> str:
            widget = getattr(self, widget_name, None)
            return widget.text() if widget is not None else str(self.settings.get(setting_name, default))

        return {
            "pipeline_stages": stages,
            "pipeline_prefix": _text_or_setting("pipeline_prefix_edit", "pipeline_prefix", ""),
            "pipeline_suffix": _text_or_setting("pipeline_suffix_edit", "pipeline_suffix", "processed"),
            "current_output_prefix": _text_or_setting(
                "current_output_prefix_edit", "current_output_prefix", ""
            ),
            "current_output_suffix": _text_or_setting(
                "current_output_suffix_edit", "current_output_suffix", ""
            ),
            "current_output_auto_suffix": getattr(
                self, "_current_output_auto_suffix", None
            ),
        }

    def _selected_pipeline_stages(self) -> frozenset[str]:
        if not hasattr(self, "pipeline_stage_controls"):
            return frozenset()
        return frozenset(
            key
            for key, controls in self.pipeline_stage_controls.items()
            if controls["checkbox"].isChecked()
        )

    def _capture_pipeline_snapshot(self) -> PipelineSnapshot:
        interpolation_mask = (
            None
            if self.interpolation_mask is None
            else self.interpolation_mask.copy().astype(np.uint8)
        )
        chamber_mask = (
            None
            if self.chamber_mask is None
            else self.chamber_mask.copy().astype(np.uint8)
        )
        rooms = tuple(
            RoomRecord(
                name=room.name,
                color=QColor(room.color),
                mask=room.mask.copy().astype(np.uint8),
            )
            for room in self._effective_room_records()
        )
        masks = tuple(
            MaskRecord(
                name=record.name,
                color=QColor(record.color),
                mask=record.mask.copy().astype(np.uint8),
                margin=int(record.margin),
                margin_mode=str(record.margin_mode),
            )
            for record in self.mask_records.values()
        )
        return PipelineSnapshot(
            enabled_stages=self._selected_pipeline_stages(),
            duplicate_distance_threshold=self.tracking_duplicate_criteria_spinbox.value(),
            zscore_high_threshold=self.tracking_zscore_high_spinbox.value(),
            zscore_low_threshold=self.tracking_zscore_low_spinbox.value(),
            zscore_deviation_mode=str(self.tracking_zscore_mode_combo.currentData()),
            interpolation_mask=interpolation_mask,
            interpolation_removal_mode=self._selected_interpolation_removal_mode(),
            interpolation_anchor=self._selected_interpolation_anchor(),
            interpolation_enabled=self._interpolation_enabled(),
            interpolation_extrapolate=self._interpolation_extrapolation_enabled(),
            chamber_mask=chamber_mask,
            chamber_boundary_mode=getattr(self, "chamber_boundary_mode", "custom"),
            rooms=rooms,
            circle_geometry=self.frame_viewer.circle_geometry(),
            masks=masks,
            occlusion_quad_points=tuple(self.frame_viewer.occ_margin_points),
            square_points=tuple(self.frame_viewer.square_points),
        )

    @staticmethod
    def _pipeline_stage_errors(snapshot: PipelineSnapshot, stage_key: str) -> list[str]:
        errors: list[str] = []
        if stage_key == "clean_repair":
            if snapshot.interpolation_removal_mode == "none" and not snapshot.interpolation_enabled:
                errors.append("Enable automatic removal or interpolation in Interpolation.")
            if snapshot.interpolation_removal_mode != "none":
                if snapshot.interpolation_mask is None or not np.any(snapshot.interpolation_mask):
                    errors.append("Draw an automatic-removal region in Interpolation.")
                if snapshot.interpolation_anchor is None:
                    errors.append("Select an anchor node in Interpolation.")
        elif stage_key == "chamber":
            if snapshot.chamber_mask is None or not np.any(snapshot.chamber_mask):
                errors.append("Define a chamber area in Chamber Mark.")
            if not snapshot.rooms or not any(np.any(room.mask) for room in snapshot.rooms):
                errors.append("Add at least one non-empty room in Chamber Mark.")
        elif stage_key == "circle":
            if snapshot.circle_geometry is None:
                errors.append("Draw a circle in Circle Detection.")
        elif stage_key == "occlusion":
            if not snapshot.masks or not any(np.any(record.mask) for record in snapshot.masks):
                errors.append("Create at least one non-empty occlusion mask.")
            if (
                any(record.margin_mode == "geometric" for record in snapshot.masks)
                and len(snapshot.occlusion_quad_points) != 4
            ):
                errors.append("Set four geometric margin points in Occlusion Detect.")
        elif stage_key == "square" and len(snapshot.square_points) != 4:
            errors.append("Select four points in Prepare > Normalize.")
        return errors

    def _pipeline_validation_errors(self, snapshot: PipelineSnapshot) -> list[str]:
        errors: list[str] = []
        if self.video_state is None:
            errors.append("Load a video first.")
        if self.csv_df is None or self.csv_path is None:
            errors.append("Load a CSV first.")
        if not self.bodyparts:
            errors.append("The CSV does not contain bodypart x/y columns.")
        if not snapshot.enabled_stages:
            errors.append("Select at least one pipeline stage.")
        for stage_key in snapshot.enabled_stages:
            errors.extend(
                f"{PIPELINE_STAGE_LABELS[stage_key]}: {message}"
                for message in self._pipeline_stage_errors(snapshot, stage_key)
            )

        output_path = self._pipeline_output_path()
        if output_path is not None and self.csv_path is not None:
            try:
                same_path = output_path.resolve() == self.csv_path.resolve()
            except OSError:
                same_path = output_path.absolute() == self.csv_path.absolute()
            if same_path:
                errors.append("Set a prefix or suffix so the pipeline does not overwrite the source CSV.")
        return errors

    def _pipeline_stage_summary(self, snapshot: PipelineSnapshot, stage_key: str) -> str:
        if stage_key == "duplicate_removal":
            return (
                "Deletes only lower-confidence duplicate rows · "
                f"Distance threshold: {snapshot.duplicate_distance_threshold:.2f}px sum"
            )
        if stage_key == "zscore_removal":
            mode = snapshot.zscore_deviation_mode.replace("_", " ")
            return (
                f"Keeps rows; skeleton x/y/score → NaN · Side: {mode} · "
                f"High: {snapshot.zscore_high_threshold:.2f} · Low: {snapshot.zscore_low_threshold:.2f}"
            )
        if stage_key == "clean_repair":
            removal = snapshot.interpolation_removal_mode.replace("none", "off")
            anchor = snapshot.interpolation_anchor or "-"
            if snapshot.interpolation_enabled and snapshot.interpolation_extrapolate:
                interpolation = "on (internal gaps + start/end nearest-value extrapolation)"
            elif snapshot.interpolation_enabled:
                interpolation = "on (original + newly invalidated internal gaps)"
            else:
                interpolation = "off"
            return f"Removal: {removal} · Anchor: {anchor} · Interpolation: {interpolation}"
        if stage_key == "chamber":
            chamber_ready = snapshot.chamber_mask is not None and bool(np.any(snapshot.chamber_mask))
            boundary = "full frame" if snapshot.chamber_boundary_mode == "full_frame" else "custom"
            return f"Chamber: {boundary if chamber_ready else 'not set'} · Rooms: {len(snapshot.rooms)}"
        if stage_key == "circle":
            if snapshot.circle_geometry is None:
                return "Circle: not set"
            center, _base_radius, adjusted_radius = snapshot.circle_geometry
            return f"Center: ({center[0]:.1f}, {center[1]:.1f}) · Radius: {adjusted_radius:.1f}px"
        if stage_key == "occlusion":
            geometric = sum(record.margin_mode == "geometric" for record in snapshot.masks)
            return (
                f"Masks: {len(snapshot.masks)} · Geometric masks: {geometric} · "
                f"Reference points: {len(snapshot.occlusion_quad_points)}/4"
            )
        return f"Square points: {len(snapshot.square_points)}/4"

    def _refresh_pipeline_ui(self, *_args) -> None:
        if not hasattr(self, "pipeline_stage_controls"):
            return
        snapshot = self._capture_pipeline_snapshot()
        for key, controls in self.pipeline_stage_controls.items():
            selected = key in snapshot.enabled_stages
            errors = self._pipeline_stage_errors(snapshot, key)
            status = controls["status"]
            summary = controls["summary"]
            assert isinstance(status, QLabel)
            assert isinstance(summary, QLabel)
            summary.setText(self._pipeline_stage_summary(snapshot, key))
            if not selected:
                status.setText("Disabled")
                status.setStyleSheet("")
                status.setToolTip("")
            elif errors:
                status.setText("Needs setup")
                status.setStyleSheet("color: #b45309; font-weight: 600;")
                status.setToolTip("\n".join(errors))
            else:
                status.setText("Ready")
                status.setStyleSheet("color: #15803d; font-weight: 600;")
                status.setToolTip("")

        self.pipeline_save_folder_label.setText(str(self.save_folder))
        self.pipeline_save_folder_label.setToolTip(str(self.save_folder))
        output_path = self._pipeline_output_path()
        preview_text = str(output_path) if output_path is not None else "Load a CSV to preview the output name."
        self.pipeline_output_preview_label.setText(preview_text)
        self.pipeline_output_preview_label.setToolTip(preview_text)

        errors = self._pipeline_validation_errors(snapshot)
        if errors:
            self.pipeline_status_label.setText(errors[0])
            self.pipeline_status_label.setToolTip("\n".join(errors))
        else:
            self.pipeline_status_label.setText(
                f"Ready · {len(snapshot.enabled_stages)} stage(s) · new combined CSV; source unchanged"
            )
            self.pipeline_status_label.setToolTip("")
        ready = not errors
        self.pipeline_preview_button.setEnabled(ready)
        self.pipeline_run_current_button.setEnabled(ready)
        batch_ready = ready and self.video_list.count() > 0
        self.pipeline_run_batch_button.setEnabled(batch_ready)
        if errors:
            unavailable_tooltip = (
                "Saving is available when the pipeline is ready.\n\n"
                + "\n".join(errors)
            )
            self.pipeline_preview_button.setToolTip(unavailable_tooltip)
            self.pipeline_run_current_button.setToolTip(unavailable_tooltip)
            self.pipeline_run_batch_button.setToolTip(unavailable_tooltip)
        else:
            self.pipeline_preview_button.setToolTip(
                "Run the configured stages without writing a CSV and summarize the result."
            )
            self.pipeline_run_current_button.setToolTip(
                "Run the validated pipeline and save one combined CSV."
            )
            self.pipeline_run_batch_button.setToolTip(
                "Run the validated pipeline for multiple videos."
                if batch_ready
                else "Load a video folder containing at least one video to enable batch saving."
            )

    def _pipeline_output_path(self) -> Path | None:
        if self.csv_path is None:
            return None
        return self._pipeline_output_path_for(self.csv_path)

    def _pipeline_output_path_for(
        self,
        csv_path: Path,
        prefix: str | None = None,
        suffix: str | None = None,
    ) -> Path:
        resolved_prefix = self.pipeline_prefix_edit.text() if prefix is None else prefix
        resolved_suffix = self.pipeline_suffix_edit.text() if suffix is None else suffix
        filename = build_pipeline_output_filename(csv_path, resolved_prefix, resolved_suffix)
        return self.save_folder / filename

    @staticmethod
    def _ordered_pipeline_stage_keys(snapshot: PipelineSnapshot) -> tuple[str, ...]:
        order = (
            "duplicate_removal",
            "zscore_removal",
            "clean_repair",
            "square",
            "chamber",
            "circle",
            "occlusion",
        )
        return tuple(key for key in order if key in snapshot.enabled_stages)

    @staticmethod
    def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
        resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        return (resized > 0).astype(np.uint8)

    def _pipeline_stages_for_item(
        self,
        item: BatchItem,
        snapshot: PipelineSnapshot,
    ) -> list[DataFramePipelineStage]:
        stages: list[DataFramePipelineStage] = []
        if "duplicate_removal" in snapshot.enabled_stages:
            stages.append(
                DataFramePipelineStage(
                    key="duplicate_removal",
                    label=PIPELINE_STAGE_LABELS["duplicate_removal"],
                    output_mode="replace",
                    allow_row_removal=True,
                    transform=partial(
                        _duplicate_stage_dataframe,
                        bodyparts=item.bodyparts,
                        criteria=snapshot.duplicate_distance_threshold,
                        width=item.width,
                        height=item.height,
                    ),
                )
            )
        if "zscore_removal" in snapshot.enabled_stages:
            stages.append(
                DataFramePipelineStage(
                    key="zscore_removal",
                    label=PIPELINE_STAGE_LABELS["zscore_removal"],
                    output_mode="replace",
                    transform=partial(
                        _zscore_stage_dataframe,
                        bodyparts=item.bodyparts,
                        high_z_threshold=snapshot.zscore_high_threshold,
                        low_z_threshold=snapshot.zscore_low_threshold,
                        deviation_mode=snapshot.zscore_deviation_mode,
                        width=item.width,
                        height=item.height,
                    ),
                )
            )
        if "clean_repair" in snapshot.enabled_stages:
            region_mask = None
            if snapshot.interpolation_removal_mode != "none":
                assert snapshot.interpolation_mask is not None
                region_mask = self._resize_mask(snapshot.interpolation_mask, item.width, item.height)
            stages.append(
                DataFramePipelineStage(
                    key="clean_repair",
                    label=PIPELINE_STAGE_LABELS["clean_repair"],
                    output_mode="replace",
                    transform=partial(
                        build_interpolation_pipeline_dataframe,
                        bodyparts=item.bodyparts,
                        region_mask=region_mask,
                        width=item.width,
                        height=item.height,
                        removal_mode=snapshot.interpolation_removal_mode,
                        anchor_bodypart=snapshot.interpolation_anchor,
                        interpolate=snapshot.interpolation_enabled,
                        extrapolate=snapshot.interpolation_extrapolate,
                    ),
                )
            )

        if "square" in snapshot.enabled_stages:
            square_points = tuple(
                (x * item.scale_x, y * item.scale_y)
                for x, y in snapshot.square_points
            )
            stages.append(
                DataFramePipelineStage(
                    key="square",
                    label=PIPELINE_STAGE_LABELS["square"],
                    transform=partial(
                        build_normalized_dataframe,
                        bodyparts=item.bodyparts,
                        quad_points=list(square_points),
                        width=item.width,
                        height=item.height,
                    ),
                )
            )

        if "chamber" in snapshot.enabled_stages:
            assert snapshot.chamber_mask is not None
            if snapshot.chamber_boundary_mode == "full_frame":
                chamber_mask = np.ones((item.height, item.width), dtype=np.uint8)
            else:
                chamber_mask = self._resize_mask(snapshot.chamber_mask, item.width, item.height)
            rooms: list[RoomRecord] = []
            for room in snapshot.rooms:
                resized_room = self._resize_mask(room.mask, item.width, item.height)
                rooms.append(
                    RoomRecord(
                        name=room.name,
                        color=room.color,
                        mask=np.logical_and(resized_room > 0, chamber_mask > 0).astype(np.uint8),
                    )
                )
            stages.append(
                DataFramePipelineStage(
                    key="chamber",
                    label=PIPELINE_STAGE_LABELS["chamber"],
                    transform=partial(
                        build_chamber_mark_dataframe,
                        bodyparts=item.bodyparts,
                        rooms=rooms,
                        width=item.width,
                        height=item.height,
                    ),
                )
            )

        if "circle" in snapshot.enabled_stages:
            assert snapshot.circle_geometry is not None
            center, _base_radius, adjusted_radius = snapshot.circle_geometry
            scaled_center = (center[0] * item.scale_x, center[1] * item.scale_y)
            scaled_radius = max(1.0, adjusted_radius * ((item.scale_x + item.scale_y) / 2.0))
            stages.append(
                DataFramePipelineStage(
                    key="circle",
                    label=PIPELINE_STAGE_LABELS["circle"],
                    transform=partial(
                        build_circle_detection_dataframe,
                        bodyparts=item.bodyparts,
                        center=scaled_center,
                        radius=scaled_radius,
                        width=item.width,
                        height=item.height,
                    ),
                )
            )

        if "occlusion" in snapshot.enabled_stages:
            masks = tuple(
                MaskRecord(
                    name=record.name,
                    color=record.color,
                    mask=self._resize_mask(record.mask, item.width, item.height),
                    margin=record.margin,
                    margin_mode=record.margin_mode,
                )
                for record in snapshot.masks
            )
            quad_points = tuple(
                (x * item.scale_x, y * item.scale_y)
                for x, y in snapshot.occlusion_quad_points
            )
            stages.append(
                DataFramePipelineStage(
                    key="occlusion",
                    label=PIPELINE_STAGE_LABELS["occlusion"],
                    transform=partial(
                        build_occlusion_dataframe,
                        bodyparts=item.bodyparts,
                        masks=list(masks),
                        width=item.width,
                        height=item.height,
                        quad_points=list(quad_points),
                    ),
                )
            )

        return stages

    def _current_pipeline_item(self) -> BatchItem:
        assert self.video_state is not None
        assert self.csv_path is not None
        assert self.csv_df is not None
        return BatchItem(
            video_path=self.video_state.path,
            csv_path=self.csv_path,
            source_df=self.csv_df,
            bodyparts=list(self.bodyparts),
            width=self.video_state.width,
            height=self.video_state.height,
            scale_x=1.0,
            scale_y=1.0,
        )

    def _execute_pipeline_item(
        self,
        item: BatchItem,
        snapshot: PipelineSnapshot,
    ) -> PipelineRunResult:
        stages = self._pipeline_stages_for_item(item, snapshot)
        return run_dataframe_pipeline(item.source_df, stages)

    def preview_pipeline_result(self) -> None:
        snapshot = self._capture_pipeline_snapshot()
        errors = self._pipeline_validation_errors(snapshot)
        if errors:
            self._refresh_pipeline_ui()
            return
        try:
            result = self._execute_pipeline_item(self._current_pipeline_item(), snapshot)
        except Exception as exc:
            QMessageBox.warning(self, "Pipeline Preview", f"Could not run the pipeline preview.\n\n{exc}")
            return

        columns = list(result.added_columns)
        column_lines = columns[:24]
        if len(columns) > len(column_lines):
            column_lines.append(f"... and {len(columns) - len(column_lines)} more")
        stage_lines: list[str] = []
        for stage in result.stages:
            changes: list[str] = []
            if stage.removed_rows:
                changes.append(f"rows {stage.input_rows:,} → {stage.output_rows:,}")
            if stage.replaced_columns:
                changes.append(
                    f"updated {len(stage.replaced_columns)} working column(s), {stage.changed_cells:,} cell(s)"
                )
            appended_count = len(stage.added_columns) + len(stage.derived_columns)
            if appended_count:
                changes.append(f"appended {appended_count} column(s)")
            detail = " · " + " · ".join(changes) if changes else " · no data changes"
            stage_lines.append(f"{stage.label}: {stage.elapsed_seconds * 1000.0:.1f} ms{detail}")
        message = (
            "Source CSV: unchanged\n"
            f"Rows: {len(self.csv_df):,} → {len(result.dataframe):,}\n"
            f"Columns: {len(self.csv_df.columns):,} → {len(result.dataframe.columns):,}\n\n"
            "Stages:\n- "
            + "\n- ".join(stage_lines)
        )
        if result.replaced_columns:
            message += f"\n\nWorking columns updated: {len(result.replaced_columns)}"
        if column_lines:
            message += "\n\nAppended columns:\n- " + "\n- ".join(column_lines)
        QMessageBox.information(self, "Pipeline Preview", message)

    def run_current_pipeline(self) -> None:
        snapshot = self._capture_pipeline_snapshot()
        errors = self._pipeline_validation_errors(snapshot)
        if errors:
            self._refresh_pipeline_ui()
            return
        output_path = self._pipeline_output_path()
        if output_path is None:
            return
        if output_path.exists():
            answer = QMessageBox.question(
                self,
                "Overwrite Pipeline CSV",
                f"The output already exists. Overwrite it?\n\n{output_path}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            result = self._execute_pipeline_item(self._current_pipeline_item(), snapshot)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.dataframe.to_csv(output_path, index=False)
        except Exception as exc:
            QMessageBox.warning(self, "Run Pipeline", f"Could not save the combined pipeline CSV.\n\n{exc}")
            return
        self.statusBar().showMessage(f"Pipeline CSV saved: {output_path}")
        QMessageBox.information(
            self,
            "Run Pipeline",
            f"Saved a new combined CSV with {len(result.dataframe.columns)} columns. "
            f"The source CSV was not changed.\n\n{output_path}",
        )

    def run_batch_pipeline(self) -> None:
        if self._focus_active_batch_progress():
            return
        snapshot = self._capture_pipeline_snapshot()
        errors = self._pipeline_validation_errors(snapshot)
        if errors:
            self._refresh_pipeline_ui()
            return
        assert self.video_state is not None
        selected_videos = self._select_videos_for_batch(
            dialog_title="Select Videos For Multi Pipeline",
            info_text="Select videos that will use the enabled pipeline stages and current tab settings.",
            warning_text=(
                "Warning:\n"
                "- The top auto-detected CSV candidate is used for each video.\n"
                "- Regions, masks, circles, and square points are scaled from the current reference video.\n"
                "- One combined CSV is saved per video. Existing files with the same name are overwritten."
            ),
            start_button_text="Run Batch Pipeline",
        )
        if not selected_videos:
            return

        prefix = self.pipeline_prefix_edit.text()
        suffix = self.pipeline_suffix_edit.text()
        save_folder = Path(self.save_folder)
        source_width = int(self.video_state.width)
        source_height = int(self.video_state.height)

        def _export_item(item: BatchItem) -> Path:
            result = self._execute_pipeline_item(item, snapshot)
            output_path = save_folder / build_pipeline_output_filename(
                item.csv_path,
                prefix,
                suffix,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.dataframe.to_csv(output_path, index=False)
            return output_path

        def _on_completed(result: BatchRunResult) -> None:
            state = "cancelled" if result.cancelled else "finished"
            self.statusBar().showMessage(
                f"Batch pipeline {state}: saved={result.saved_count}, "
                f"skipped={len(result.skipped_auto_missing)}, failed={len(result.failed)}"
            )

        self._start_batch_export(
            title="Multi Pipeline Progress",
            activity_text="Running the pipeline in the background...",
            selected_videos=selected_videos,
            source_width=source_width,
            source_height=source_height,
            export_item=_export_item,
            on_completed=_on_completed,
        )
