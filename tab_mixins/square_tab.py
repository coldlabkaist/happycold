from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from batch import BatchItem, BatchRunResult
from heatmap import calculate_body_occupancy_heatmap
from heatmap_plot import build_heatmap_figure
from shared import (
    build_normalized_dataframe,
    build_rectified_geometry,
    dataframe_points_to_pixels,
)
from ui_controls import NoWheelComboBox, NoWheelDoubleSpinBox


class HeatmapPreviewDialog(QDialog):
    def __init__(
        self,
        figure: Figure,
        title: str,
        default_export_name: str,
        transparent_export: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.figure = figure
        self._default_export_name = default_export_name
        self._transparent_export = transparent_export

        self.setWindowTitle(title)
        self.resize(1180, 820)

        layout = QVBoxLayout(self)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.canvas)
        self.canvas.draw_idle()

    def _show_context_menu(self, position: QPoint) -> None:
        menu = QMenu(self)
        save_action = menu.addAction("Save As...")
        selected_action = menu.exec(self.canvas.mapToGlobal(position))
        if selected_action == save_action:
            self._save_figure_as()

    def _save_figure_as(self) -> None:
        start_dir = str(Path.cwd() / f"{self._default_export_name}.png")
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Heatmap Image",
            start_dir,
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;SVG Image (*.svg);;PDF Document (*.pdf)",
        )
        if not file_path:
            return

        output_path = Path(file_path)
        if output_path.suffix == "" and selected_filter:
            if "PNG" in selected_filter:
                output_path = output_path.with_suffix(".png")
            elif "JPEG" in selected_filter:
                output_path = output_path.with_suffix(".jpg")
            elif "SVG" in selected_filter:
                output_path = output_path.with_suffix(".svg")
            elif "PDF" in selected_filter:
                output_path = output_path.with_suffix(".pdf")
        self.figure.savefig(output_path, dpi=220, transparent=self._transparent_export)


class SquareTabMixin:
    FRAME_COLUMN_CANDIDATES = ("frame idx", "frame_idx", "frame index", "frame_index", "frame")
    CUT_MODE_START_DURATION = "start_duration"
    CUT_MODE_DURATION_END = "duration_end"
    CUT_MODE_START_END = "start_end"

    def _build_square_tab(self) -> QWidget:
        tab = QWidget()
        tooltip = "Choose four corners for perspective normalization. Original coordinates are preserved."
        tab.setToolTip(tooltip)
        layout = QVBoxLayout(tab)

        info = QLabel(
            "Click four corners on the frame to define the normalization area. "
            "Saving appends x_normalized/y_normalized columns and keeps the original coordinates."
        )
        info.setWordWrap(True)
        info.setToolTip(tooltip)

        self.square_points_label = QLabel("Selected points: 0 / 4")
        self.square_points_label.setWordWrap(True)
        self.square_points_label.setToolTip("The current four-point selection in image coordinates.")

        self.square_reset_button = QPushButton("Reset Points")
        self.square_reset_button.setToolTip("Clear the current four-point selection and start over.")

        layout.addWidget(info)
        layout.addWidget(self.square_points_label)
        layout.addWidget(self.square_reset_button)
        layout.addStretch(1)
        return tab

    def _build_trajectory_tab(self) -> QWidget:
        tab = QWidget()
        tooltip = "Inspect raw or normalized trajectories and heatmaps without modifying the CSV."
        tab.setToolTip(tooltip)
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setObjectName("trajectoryControlsScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        info = QLabel(
            "Choose an optional time or spatial range, then open a trajectory or heatmap "
            "preview. This tab never changes the source CSV."
        )
        info.setWordWrap(True)
        info.setToolTip(tooltip)
        layout.addWidget(info)

        self.trajectory_coordinate_mode_label = QLabel("Coordinate space: waiting for CSV")
        self.trajectory_coordinate_mode_label.setWordWrap(True)
        self.trajectory_coordinate_mode_label.setProperty("muted", True)
        layout.addWidget(self.trajectory_coordinate_mode_label)

        self.square_limit_trajectory_checkbox = QCheckBox("Limit preview to a time range")
        self.square_limit_trajectory_checkbox.setToolTip(
            "When enabled, only frames in the resolved range are used by previews and batch heatmaps."
        )

        self.square_cut_group = QGroupBox("Time Range Settings")
        cut_layout = QGridLayout(self.square_cut_group)
        cut_layout.setContentsMargins(8, 8, 8, 8)
        cut_layout.setHorizontalSpacing(6)
        cut_layout.setVerticalSpacing(6)
        cut_layout.setColumnStretch(1, 1)

        self.square_cut_mode_label = QLabel("Range Type")
        self.square_cut_mode_combo = NoWheelComboBox()
        self.square_cut_mode_combo.addItem("Start + Duration", self.CUT_MODE_START_DURATION)
        self.square_cut_mode_combo.addItem("Duration + End", self.CUT_MODE_DURATION_END)
        self.square_cut_mode_combo.addItem("Start + End", self.CUT_MODE_START_END)
        self.square_cut_mode_combo.setMinimumWidth(0)

        self.square_start_label = QLabel("Start")
        self.square_start_spinbox = NoWheelDoubleSpinBox()
        self.square_start_spinbox.setRange(0.0, 10_000_000.0)
        self.square_start_spinbox.setDecimals(0)
        self.square_start_spinbox.setSingleStep(1.0)
        self.square_start_spinbox.setValue(0.0)
        self.square_start_use_current_button = QPushButton("Use Current")

        self.square_duration_label = QLabel("Duration")
        self.square_duration_spinbox = NoWheelDoubleSpinBox()
        self.square_duration_spinbox.setRange(0.0, 10_000_000.0)
        self.square_duration_spinbox.setValue(300.0)
        self.square_duration_unit_combo = NoWheelComboBox()
        self.square_duration_unit_combo.addItem("seconds", "sec")
        self.square_duration_unit_combo.addItem("frames", "frame")
        self.square_duration_unit_combo.addItem("minutes", "min")
        self.square_duration_unit_combo.setCurrentIndex(1)
        self.square_duration_unit_combo.setEditable(True)
        self.square_duration_unit_combo.lineEdit().setReadOnly(True)
        self.square_duration_unit_combo.lineEdit().setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.square_duration_unit_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for index in range(self.square_duration_unit_combo.count()):
            self.square_duration_unit_combo.setItemData(
                index,
                int(Qt.AlignmentFlag.AlignCenter),
                Qt.ItemDataRole.TextAlignmentRole,
            )

        self.square_end_label = QLabel("End")
        self.square_end_spinbox = NoWheelDoubleSpinBox()
        self.square_end_spinbox.setRange(0.0, 10_000_000.0)
        self.square_end_spinbox.setDecimals(0)
        self.square_end_spinbox.setSingleStep(1.0)
        self.square_end_spinbox.setValue(0.0)
        self.square_end_use_current_button = QPushButton("Use Current")

        self.square_cut_range_label = QLabel("Resolved frames: -")
        self.square_cut_range_label.setWordWrap(True)

        for spinbox in (
            self.square_start_spinbox,
            self.square_duration_spinbox,
            self.square_end_spinbox,
        ):
            spinbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            spinbox.setMinimumWidth(0)
        for side_control in (
            self.square_start_use_current_button,
            self.square_end_use_current_button,
            self.square_duration_unit_combo,
        ):
            side_control.setMinimumWidth(0)
            side_control.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        cut_layout.addWidget(self.square_cut_mode_label, 0, 0)
        cut_layout.addWidget(self.square_cut_mode_combo, 0, 1, 1, 2)
        cut_layout.addWidget(self.square_start_label, 1, 0)
        cut_layout.addWidget(self.square_start_spinbox, 1, 1)
        cut_layout.addWidget(self.square_start_use_current_button, 1, 2)
        cut_layout.addWidget(self.square_duration_label, 2, 0)
        cut_layout.addWidget(self.square_duration_spinbox, 2, 1)
        cut_layout.addWidget(self.square_duration_unit_combo, 2, 2)
        cut_layout.addWidget(self.square_end_label, 3, 0)
        cut_layout.addWidget(self.square_end_spinbox, 3, 1)
        cut_layout.addWidget(self.square_end_use_current_button, 3, 2)
        cut_layout.addWidget(self.square_cut_range_label, 4, 0, 1, 3)

        range_group = QGroupBox("1. Choose Data Range")
        range_layout = QVBoxLayout(range_group)
        range_layout.addWidget(self.square_limit_trajectory_checkbox)
        range_layout.addWidget(self.square_cut_group)

        self.trajectory_limit_region_checkbox = QCheckBox(
            "Limit preview and heatmap to a rectangle"
        )
        self.trajectory_limit_region_checkbox.setToolTip(
            "Coordinates outside the rectangle are ignored; rows and source data remain unchanged."
        )
        self.trajectory_region_status_label = QLabel("Spatial range: Full frame")
        self.trajectory_region_status_label.setWordWrap(True)
        self.trajectory_region_status_label.setProperty("muted", True)
        self.trajectory_draw_region_button = QPushButton("Draw / Redraw Rectangle")
        self.trajectory_clear_region_button = QPushButton("Clear Rectangle")
        region_button_row = QHBoxLayout()
        region_button_row.setSpacing(6)
        region_button_row.addWidget(self.trajectory_draw_region_button, stretch=1)
        region_button_row.addWidget(self.trajectory_clear_region_button)
        spatial_group = QGroupBox("Spatial Range")
        spatial_layout = QVBoxLayout(spatial_group)
        spatial_layout.addWidget(self.trajectory_limit_region_checkbox)
        spatial_layout.addWidget(self.trajectory_region_status_label)
        spatial_layout.addLayout(region_button_row)
        range_layout.addWidget(spatial_group)
        layout.addWidget(range_group)

        preview_help = QLabel(
            "Trajectory shows paths by bodypart. Heatmap shows spatial occupancy, with or without the video frame."
        )
        preview_help.setWordWrap(True)
        preview_help.setProperty("muted", True)
        self.square_preview_button = QPushButton("Preview Trajectory")
        self.square_preview_button.setProperty("primary", True)
        self.square_preview_button.setEnabled(False)
        self.square_preview_button.setToolTip(
            "Preview raw coordinates with zero normalization points or normalized coordinates with four points."
        )
        self.square_heatmap_overlay_button = QPushButton("Preview Heatmap with Background")
        self.square_heatmap_overlay_button.setEnabled(False)
        self.square_heatmap_overlay_button.setToolTip(
            "Show a body-volume probability heatmap over the current video frame."
        )
        self.square_heatmap_plain_button = QPushButton("Preview Heatmap Only")
        self.square_heatmap_plain_button.setEnabled(False)
        self.square_heatmap_plain_button.setToolTip(
            "Show a body-volume probability heatmap without the video frame."
        )

        preview_group = QGroupBox("2. Open a Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.addWidget(preview_help)
        preview_layout.addWidget(self.square_preview_button)
        preview_layout.addWidget(self.square_heatmap_overlay_button)
        preview_layout.addWidget(self.square_heatmap_plain_button)
        layout.addWidget(preview_group)

        output_hint = QLabel(
            "Need files for several videos? Use TRAJECTORY OUTPUT below for batch heatmap saving."
        )
        output_hint.setWordWrap(True)
        output_hint.setProperty("muted", True)
        layout.addWidget(output_hint)
        layout.addStretch(1)

        scroll.setWidget(content)
        outer_layout.addWidget(scroll)

        self.square_limit_trajectory_checkbox.toggled.connect(self._refresh_square_time_controls)
        self.trajectory_limit_region_checkbox.toggled.connect(
            self._on_trajectory_region_limit_toggled
        )
        self.trajectory_draw_region_button.clicked.connect(self._start_trajectory_region_draw)
        self.trajectory_clear_region_button.clicked.connect(self._clear_trajectory_region)
        self.frame_viewer.trajectory_region_changed.connect(
            self._on_trajectory_region_changed
        )
        self.square_cut_mode_combo.currentIndexChanged.connect(self._on_square_cut_mode_changed)
        self.square_duration_unit_combo.currentIndexChanged.connect(self._on_square_duration_unit_changed)
        self.square_start_spinbox.valueChanged.connect(self._on_square_cut_value_changed)
        self.square_duration_spinbox.valueChanged.connect(self._on_square_cut_value_changed)
        self.square_end_spinbox.valueChanged.connect(self._on_square_cut_value_changed)
        self.square_start_use_current_button.clicked.connect(self._set_square_start_from_current_frame)
        self.square_end_use_current_button.clicked.connect(self._set_square_end_from_current_frame)
        self.square_heatmap_overlay_button.clicked.connect(self.preview_square_heatmap_with_background)
        self.square_heatmap_plain_button.clicked.connect(self.preview_square_heatmap_without_background)

        self._refresh_square_duration_spinbox_format()
        self._refresh_square_current_buttons()
        self._refresh_square_time_controls()
        self._refresh_trajectory_region_controls()
        self._refresh_trajectory_coordinate_mode()
        return tab

    def _sync_trajectory_region_mode(self) -> None:
        if not hasattr(self, "trajectory_limit_region_checkbox"):
            self.frame_viewer.set_mode("inspect")
            return
        active = (
            hasattr(self, "mode_tabs")
            and self.mode_tabs.currentIndex() == self.TAB_TRAJECTORY
            and self.trajectory_limit_region_checkbox.isChecked()
        )
        self.frame_viewer.set_mode("trajectory_region" if active else "inspect")

    def _on_trajectory_region_limit_toggled(self, _checked: bool) -> None:
        self._sync_trajectory_region_mode()
        self._refresh_trajectory_region_controls()
        self._refresh_square_ui()

    def _start_trajectory_region_draw(self) -> None:
        self.trajectory_limit_region_checkbox.setChecked(True)
        self.frame_viewer.clear_trajectory_region()
        self._sync_trajectory_region_mode()
        self.frame_viewer.setFocus()

    def _clear_trajectory_region(self) -> None:
        self.frame_viewer.clear_trajectory_region()
        self._sync_trajectory_region_mode()

    def _on_trajectory_region_changed(self) -> None:
        self._refresh_trajectory_region_controls()
        # Keep drag feedback light; update dependent output only when the drag finishes.
        if not self.frame_viewer._trajectory_region_dragging:
            self._refresh_square_ui()

    def _refresh_trajectory_region_controls(self) -> None:
        if not hasattr(self, "trajectory_limit_region_checkbox"):
            return
        enabled = self.trajectory_limit_region_checkbox.isChecked()
        self.trajectory_draw_region_button.setEnabled(enabled)
        self.trajectory_clear_region_button.setEnabled(
            enabled and self.frame_viewer.trajectory_region_bounds() is not None
        )
        if not enabled:
            self.trajectory_region_status_label.setText("Spatial range: Full frame")
            return
        bounds = self.frame_viewer.trajectory_region_bounds()
        if bounds is None:
            self.trajectory_region_status_label.setText("Spatial range: Drag a rectangle in the viewer.")
            return
        left, top, right, bottom = bounds
        self.trajectory_region_status_label.setText(
            f"Spatial range: x {left:.0f}–{right:.0f}, y {top:.0f}–{bottom:.0f} px"
        )

    def _refresh_trajectory_coordinate_mode(self) -> None:
        label = getattr(self, "trajectory_coordinate_mode_label", None)
        if label is None:
            return
        if self.csv_df is None:
            label.setText("Coordinate space: load a CSV to begin.")
            return
        point_count = len(self.frame_viewer.square_points)
        if point_count == 0:
            label.setText("Coordinate space: Raw pixel coordinates (0 normalization points).")
        elif point_count == 4:
            label.setText("Coordinate space: Square-normalized coordinates (4 normalization points).")
        else:
            label.setText(
                f"Coordinate space: normalization is incomplete ({point_count} / 4 points). "
                "Finish or reset Prepare > Normalize to enable previews."
            )

    def _refresh_square_ui(self) -> None:
        detail = " | ".join(f"{index + 1}: ({point[0]:.1f}, {point[1]:.1f})" for index, point in enumerate(self.frame_viewer.square_points))
        self.square_points_label.setText(f"Selected points: {len(self.frame_viewer.square_points)} / 4\n{detail if detail else '-'}")
        point_count = len(self.frame_viewer.square_points)
        region_ready = (
            not hasattr(self, "trajectory_limit_region_checkbox")
            or not self.trajectory_limit_region_checkbox.isChecked()
            or self.frame_viewer.trajectory_region_bounds() is not None
        )
        self.square_preview_button.setEnabled(
            self.csv_df is not None and point_count in {0, 4} and region_ready
        )
        heatmap_ready = self._square_heatmap_ready()
        self.square_heatmap_overlay_button.setEnabled(heatmap_ready)
        self.square_heatmap_plain_button.setEnabled(heatmap_ready)
        if hasattr(self, "square_batch_heatmap_overlay_button"):
            self.square_batch_heatmap_overlay_button.setEnabled(heatmap_ready)
            self.square_batch_heatmap_plain_button.setEnabled(heatmap_ready)
        self._refresh_trajectory_coordinate_mode()
        if (
            hasattr(self, "mode_tabs")
            and self.mode_tabs.currentIndex() == self.TAB_TRAJECTORY
            and self.square_preview_button.isEnabled()
        ):
            self.square_preview_button.setFocus()
        self._refresh_square_current_buttons()
        self._refresh_square_cut_summary()
        self._refresh_trajectory_region_controls()
        self._refresh_output_ui()

    def _on_square_points_changed(self) -> None:
        self._refresh_square_ui()

    def preview_square_normalization(self) -> None:
        if self.csv_df is None or self.video_state is None:
            QMessageBox.warning(self, "Preview", "Load a video and CSV first.")
            return
        source_df = self._prepare_trajectory_source_dataframe(
            self.csv_df,
            self.bodyparts,
            self.video_state.width,
            self.video_state.height,
            message_title="Preview",
            show_message=True,
        )
        if source_df is None:
            return
        point_count = len(self.frame_viewer.square_points)
        if point_count == 0:
            self._show_normalized_preview(source_df, normalized=False)
            return
        if point_count != 4:
            QMessageBox.warning(self, "Preview", "Choose either zero points for a raw preview or four points for a normalized preview.")
            return
        normalized_df = build_normalized_dataframe(
            source_df,
            self.bodyparts,
            self.frame_viewer.square_points,
            self.video_state.width,
            self.video_state.height,
        )
        self._show_normalized_preview(normalized_df, normalized=True)

    @staticmethod
    def _canonical_column_name(name: str) -> str:
        return "".join(character.lower() for character in str(name) if character.isalnum())

    @classmethod
    def _find_matching_column(cls, df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
        lookup = {cls._canonical_column_name(column): column for column in df.columns}
        for candidate in candidates:
            match = lookup.get(cls._canonical_column_name(candidate))
            if match is not None:
                return match
        return None

    def _selected_square_cut_mode(self) -> str:
        return str(self.square_cut_mode_combo.currentData())

    def _selected_square_duration_unit(self) -> str:
        return str(self.square_duration_unit_combo.currentData())

    def _refresh_square_duration_spinbox_format(self) -> None:
        unit = self._selected_square_duration_unit()
        if unit == "frame":
            self.square_duration_spinbox.setDecimals(0)
            self.square_duration_spinbox.setSingleStep(1.0)
        else:
            self.square_duration_spinbox.setDecimals(2)
            self.square_duration_spinbox.setSingleStep(0.1)

    def _refresh_square_current_buttons(self) -> None:
        text = "Use current"
        self.square_start_use_current_button.setText(text)
        self.square_end_use_current_button.setText(text)

    def _square_current_frame_number(self) -> int:
        return int(getattr(self, "current_frame_number", 1))

    def _set_square_start_from_current_frame(self) -> None:
        self.square_start_spinbox.setValue(float(self._square_current_frame_number()))

    def _set_square_end_from_current_frame(self) -> None:
        self.square_end_spinbox.setValue(float(self._square_current_frame_number()))

    def _on_square_cut_mode_changed(self, _index: int) -> None:
        self._refresh_square_time_controls()
        self._refresh_square_cut_summary()

    def _on_square_duration_unit_changed(self, _index: int) -> None:
        self._refresh_square_duration_spinbox_format()
        self._refresh_square_time_controls()
        self._refresh_square_cut_summary()

    def _on_square_cut_value_changed(self, _value: float) -> None:
        self._refresh_square_cut_summary()

    def _refresh_square_time_controls(self) -> None:
        enabled = self.square_limit_trajectory_checkbox.isChecked()
        self.square_cut_mode_label.setEnabled(enabled)
        self.square_cut_mode_combo.setEnabled(enabled)
        if not enabled:
            self.square_start_label.setEnabled(False)
            self.square_start_spinbox.setEnabled(False)
            self.square_start_use_current_button.setEnabled(False)
            self.square_duration_label.setEnabled(False)
            self.square_duration_spinbox.setEnabled(False)
            self.square_duration_unit_combo.setEnabled(False)
            self.square_end_label.setEnabled(False)
            self.square_end_spinbox.setEnabled(False)
            self.square_end_use_current_button.setEnabled(False)
            self.square_cut_range_label.setEnabled(False)
            self.square_cut_range_label.setText("Range (frame): -")
            return

        mode = self._selected_square_cut_mode()
        start_editable = mode in {self.CUT_MODE_START_DURATION, self.CUT_MODE_START_END}
        duration_editable = mode in {self.CUT_MODE_START_DURATION, self.CUT_MODE_DURATION_END}
        end_editable = mode in {self.CUT_MODE_DURATION_END, self.CUT_MODE_START_END}

        self.square_start_label.setText("Start")
        self.square_duration_label.setText("Duration")
        self.square_end_label.setText("End")
        self.square_start_label.setEnabled(True)
        self.square_duration_label.setEnabled(True)
        self.square_end_label.setEnabled(True)
        self.square_start_spinbox.setEnabled(start_editable)
        self.square_duration_spinbox.setEnabled(duration_editable)
        self.square_end_spinbox.setEnabled(end_editable)
        self.square_start_use_current_button.setEnabled(start_editable)
        self.square_end_use_current_button.setEnabled(end_editable)
        self.square_duration_unit_combo.setEnabled(duration_editable)
        self.square_cut_range_label.setEnabled(True)

    def _to_frame_units(self, value: float, unit: str, fps: float) -> int | None:
        if unit == "frame":
            return int(round(value))
        if fps <= 0:
            return None
        if unit == "sec":
            return int(round(value * fps))
        if unit == "min":
            return int(round(value * 60.0 * fps))
        return None

    def _duration_frames_to_value(self, frames: int, unit: str, fps: float) -> float | None:
        if unit == "frame":
            return float(frames)
        if fps <= 0:
            return None
        if unit == "sec":
            return float(frames) / fps
        if unit == "min":
            return float(frames) / (fps * 60.0)
        return None

    def _resolve_square_cut_range_frames(self, show_message: bool) -> tuple[int, int] | None:
        mode = self._selected_square_cut_mode()
        fps = float(self.video_state.fps) if self.video_state is not None else 0.0
        start_frame = int(round(float(self.square_start_spinbox.value())))
        end_frame = int(round(float(self.square_end_spinbox.value())))
        duration_frame = self._to_frame_units(float(self.square_duration_spinbox.value()), self._selected_square_duration_unit(), fps)

        if duration_frame is None and mode in {self.CUT_MODE_START_DURATION, self.CUT_MODE_DURATION_END}:
            if show_message:
                QMessageBox.warning(self, "Preview", "Video FPS is not available, so seconds/minutes duration cannot be converted.")
            return None

        if mode == self.CUT_MODE_START_DURATION:
            if duration_frame is None or duration_frame <= 0:
                if show_message:
                    QMessageBox.warning(self, "Preview", "Duration must be greater than zero.")
                return None
            end_frame = start_frame + duration_frame - 1
        elif mode == self.CUT_MODE_DURATION_END:
            if duration_frame is None or duration_frame <= 0:
                if show_message:
                    QMessageBox.warning(self, "Preview", "Duration must be greater than zero.")
                return None
            start_frame = end_frame - duration_frame + 1

        if end_frame < start_frame:
            if show_message:
                QMessageBox.warning(self, "Preview", "End must be greater than or equal to Start.")
            return None
        if start_frame < 0:
            if show_message:
                QMessageBox.warning(self, "Preview", "Start frame must be 0 or greater.")
            return None
        return start_frame, end_frame

    def _refresh_square_cut_summary(self) -> None:
        if not self.square_limit_trajectory_checkbox.isChecked():
            self.square_cut_range_label.setText("Range (frame): -")
            return
        resolved = self._resolve_square_cut_range_frames(show_message=False)
        if resolved is None:
            self.square_cut_range_label.setText("Range (frame): invalid")
            return
        start_frame, end_frame = resolved
        frame_len = end_frame - start_frame + 1
        self.square_cut_range_label.setText(f"Range (frame): {start_frame} ~ {end_frame} ({frame_len} frames)")

        mode = self._selected_square_cut_mode()
        if mode == self.CUT_MODE_START_DURATION:
            self.square_end_spinbox.blockSignals(True)
            self.square_end_spinbox.setValue(float(end_frame))
            self.square_end_spinbox.blockSignals(False)
        elif mode == self.CUT_MODE_DURATION_END:
            self.square_start_spinbox.blockSignals(True)
            self.square_start_spinbox.setValue(float(start_frame))
            self.square_start_spinbox.blockSignals(False)
        elif mode == self.CUT_MODE_START_END:
            duration_frames = max(0, end_frame - start_frame + 1)
            duration_value = self._duration_frames_to_value(
                duration_frames,
                self._selected_square_duration_unit(),
                float(self.video_state.fps) if self.video_state is not None else 0.0,
            )
            if duration_value is not None:
                self.square_duration_spinbox.blockSignals(True)
                self.square_duration_spinbox.setValue(float(duration_value))
                self.square_duration_spinbox.blockSignals(False)

    def _filter_limited_trajectory(
        self,
        df: pd.DataFrame,
        message_title: str = "Preview",
        show_message: bool = True,
    ) -> pd.DataFrame | None:
        frame_col = self._find_matching_column(df, self.FRAME_COLUMN_CANDIDATES)
        if frame_col is None:
            if show_message:
                QMessageBox.warning(
                    self,
                    message_title,
                    "Could not find a frame column.\nLimited trajectory preview needs a frame/frame_idx style column.",
                )
            return None

        resolved = self._resolve_square_cut_range_frames(show_message=show_message)
        if resolved is None:
            return None
        start_frame, end_frame = resolved

        frame_values = pd.to_numeric(df[frame_col], errors="coerce")
        limited_df = df.loc[frame_values.between(start_frame, end_frame, inclusive="both")].copy()
        if limited_df.empty:
            if show_message:
                QMessageBox.information(self, message_title, "No trajectory data found in the selected time range.")
            return None
        return limited_df

    def _filter_trajectory_spatial_region(
        self,
        df: pd.DataFrame,
        bodyparts: list[str],
        width: int,
        height: int,
        bounds: tuple[float, float, float, float] | None,
        *,
        message_title: str,
        show_message: bool,
    ) -> pd.DataFrame | None:
        if bounds is None:
            if show_message:
                QMessageBox.information(
                    self,
                    message_title,
                    "Draw a rectangular spatial range in the viewer first.",
                )
            return None
        left, top, right, bottom = bounds
        filtered = df.copy()
        found_columns = False
        kept_any = False
        for bodypart in bodyparts:
            x_col = f"{bodypart}.x"
            y_col = f"{bodypart}.y"
            if x_col not in df.columns or y_col not in df.columns:
                continue
            found_columns = True
            points = dataframe_points_to_pixels(df, x_col, y_col, width, height)
            valid = ~np.isnan(points).any(axis=1)
            inside = (
                valid
                & (points[:, 0] >= left)
                & (points[:, 0] <= right)
                & (points[:, 1] >= top)
                & (points[:, 1] <= bottom)
            )
            kept_any = kept_any or bool(np.any(inside))
            filtered.loc[~inside, [x_col, y_col]] = np.nan
        if not found_columns or not kept_any:
            if show_message:
                QMessageBox.information(
                    self,
                    message_title,
                    "No trajectory coordinates were found inside the selected rectangle.",
                )
            return None
        return filtered

    def _prepare_trajectory_source_dataframe(
        self,
        df: pd.DataFrame,
        bodyparts: list[str],
        width: int,
        height: int,
        *,
        message_title: str,
        show_message: bool,
    ) -> pd.DataFrame | None:
        prepared = df
        if self.square_limit_trajectory_checkbox.isChecked():
            prepared = self._filter_limited_trajectory(
                prepared,
                message_title=message_title,
                show_message=show_message,
            )
            if prepared is None:
                return None
        if self.trajectory_limit_region_checkbox.isChecked():
            prepared = self._filter_trajectory_spatial_region(
                prepared,
                bodyparts,
                width,
                height,
                self.frame_viewer.trajectory_region_bounds(),
                message_title=message_title,
                show_message=show_message,
            )
        return prepared

    def save_normalized_csv(self) -> None:
        if self.csv_df is None or self.video_state is None:
            QMessageBox.warning(self, "Save", "Load a video and CSV first.")
            return
        output = self._normalized_output_path()
        if output is None:
            return
        try:
            df = build_normalized_dataframe(self.csv_df, self.bodyparts, self.frame_viewer.square_points, self.video_state.width, self.video_state.height)
            output.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output, index=False)
        except Exception as exc:
            QMessageBox.warning(self, "Save Warning", f"Could not overwrite the CSV.\nIt may be open in another program.\n\n{output}\n\n{exc}")
            return
        self.statusBar().showMessage(f"Normalized CSV saved: {output}")

    def _square_heatmap_ready(self) -> bool:
        point_count = len(self.frame_viewer.square_points)
        return (
            self.csv_df is not None
            and self.video_state is not None
            and bool(self.bodyparts)
            and point_count in {0, 4}
            and (
                not hasattr(self, "trajectory_limit_region_checkbox")
                or not self.trajectory_limit_region_checkbox.isChecked()
                or self.frame_viewer.trajectory_region_bounds() is not None
            )
        )

    def _square_heatmap_mode(self) -> str | None:
        point_count = len(self.frame_viewer.square_points)
        if point_count == 0:
            return "raw"
        if point_count == 4:
            return "normalized"
        return None

    def _square_heatmap_export_name(self, with_background: bool) -> str:
        mode = self._square_heatmap_mode()
        suffix = "with_frame" if with_background else "transparent"
        stem = self.csv_path.stem if self.csv_path is not None else "heatmap"
        return f"{stem}_{mode}_location_heatmap_{suffix}"

    def _square_heatmap_export_name_for(self, csv_path: Path, with_background: bool) -> str:
        mode = self._square_heatmap_mode()
        suffix = "with_frame" if with_background else "transparent"
        return f"{csv_path.stem}_{mode}_location_heatmap_{suffix}"

    def _square_heatmap_output_path_for(self, csv_path: Path, with_background: bool) -> Path:
        return self.save_folder / f"{self._square_heatmap_export_name_for(csv_path, with_background)}.png"

    def _square_heatmap_background_image(self, normalized: bool) -> np.ndarray | None:
        if self.current_frame_rgb is None:
            return None
        if not normalized:
            return self.current_frame_rgb
        matrix, _inverse, (rect_width, rect_height) = build_rectified_geometry(self.frame_viewer.square_points)
        return cv2.warpPerspective(
            self.current_frame_rgb,
            matrix,
            (rect_width, rect_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    @staticmethod
    def _square_heatmap_background_image_from_frame(
        frame_rgb: np.ndarray | None,
        normalized: bool,
        quad_points: list[tuple[float, float]],
    ) -> np.ndarray | None:
        if frame_rgb is None:
            return None
        if not normalized:
            return frame_rgb
        matrix, _inverse, (rect_width, rect_height) = build_rectified_geometry(quad_points)
        return cv2.warpPerspective(
            frame_rgb,
            matrix,
            (rect_width, rect_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    @staticmethod
    def _read_video_frame_rgb(video_path: Path, frame_number: int) -> np.ndarray | None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return None
        try:
            total_frames = max(1, int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
            target_frame = max(1, min(int(frame_number), total_frames))
            capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
            ok, frame = capture.read()
            if not ok or frame is None:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()

    def _build_square_heatmap_figure(self, df: pd.DataFrame, normalized: bool, with_background: bool) -> Figure:
        if self.video_state is None:
            raise ValueError("Load a video before building a heatmap.")
        background = self._square_heatmap_background_image(normalized) if with_background else None
        rectified_size = None
        if normalized:
            _matrix, _inverse, rectified_size = build_rectified_geometry(self.frame_viewer.square_points)
        return self._build_square_heatmap_figure_with_background(
            df=df,
            bodyparts=self.bodyparts,
            normalized=normalized,
            background=background,
            with_background=with_background,
            frame_size=(self.video_state.width, self.video_state.height),
            rectified_size=rectified_size,
        )

    def _build_square_heatmap_figure_with_background(
        self,
        df: pd.DataFrame,
        bodyparts: list[str],
        normalized: bool,
        background: np.ndarray | None,
        with_background: bool,
        frame_size: tuple[int, int],
        rectified_size: tuple[int, int] | None,
    ) -> Figure:
        heatmap = calculate_body_occupancy_heatmap(
            df=df,
            bodyparts=bodyparts,
            frame_size=frame_size,
            normalized=normalized,
            rectified_size=rectified_size,
        )
        return build_heatmap_figure(
            heatmap=heatmap,
            normalized=normalized,
            background=background,
            with_background=with_background,
        )

    def _square_heatmap_export_dataframe(self) -> tuple[pd.DataFrame, bool] | None:
        if self.csv_df is None or self.video_state is None:
            QMessageBox.warning(self, "Heatmap Preview", "Load a video and CSV first.")
            return None
        return self._square_heatmap_export_dataframe_for(
            self.csv_df,
            self.bodyparts,
            self.video_state.width,
            self.video_state.height,
            [tuple(point) for point in self.frame_viewer.square_points],
            "Heatmap Preview",
            show_message=True,
        )

    def _square_heatmap_export_dataframe_for(
        self,
        source_df: pd.DataFrame,
        bodyparts: list[str],
        width: int,
        height: int,
        square_points: list[tuple[float, float]],
        message_title: str,
        show_message: bool,
    ) -> tuple[pd.DataFrame, bool] | None:
        prepared_df = self._prepare_trajectory_source_dataframe(
            source_df,
            bodyparts,
            width,
            height,
            message_title=message_title,
            show_message=show_message,
        )
        if prepared_df is None:
            return None
        point_count = len(square_points)
        if point_count == 0:
            export_df = prepared_df
            normalized = False
        elif point_count == 4:
            export_df = build_normalized_dataframe(
                prepared_df,
                bodyparts,
                square_points,
                width,
                height,
            )
            normalized = True
        else:
            if show_message:
                QMessageBox.warning(
                    self,
                    message_title,
                    "Choose either zero points for a raw heatmap or four points for a normalized heatmap.",
                )
            return None

        return export_df, normalized

    def _show_square_heatmap_preview(self, with_background: bool) -> None:
        export_data = self._square_heatmap_export_dataframe()
        if export_data is None:
            return
        if with_background and self.current_frame_rgb is None:
            QMessageBox.warning(self, "Heatmap Preview", "Load a video frame first to preview a background-composited heatmap.")
            return

        export_df, normalized = export_data
        try:
            figure = self._build_square_heatmap_figure(export_df, normalized, with_background)
        except Exception as exc:
            QMessageBox.warning(self, "Heatmap Preview", f"Could not build the location heatmap preview.\n\n{exc}")
            return

        mode_label = "Normalized" if normalized else "Raw"
        bg_label = "With Background" if with_background else "Transparent"
        title = f"Happy CoLD - {mode_label} Location Heatmap ({bg_label})"
        preview_dialog = HeatmapPreviewDialog(
            figure,
            title,
            self._square_heatmap_export_name(with_background),
            transparent_export=not with_background,
            parent=self,
        )
        preview_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        preview_dialog.finished.connect(lambda _result, dialog=preview_dialog: self._on_square_heatmap_preview_closed(dialog))
        if not hasattr(self, "_square_heatmap_preview_windows"):
            self._square_heatmap_preview_windows = []
        self._square_heatmap_preview_windows.append(preview_dialog)
        preview_dialog.show()
        preview_dialog.raise_()
        preview_dialog.activateWindow()
        self.statusBar().showMessage("Location heatmap preview opened. Right-click the preview to Save As if needed.")

    def _on_square_heatmap_preview_closed(self, dialog: QDialog) -> None:
        try:
            self._square_heatmap_preview_windows.remove(dialog)
        except (AttributeError, ValueError):
            pass

    def preview_square_heatmap_with_background(self) -> None:
        self._show_square_heatmap_preview(with_background=True)

    def preview_square_heatmap_without_background(self) -> None:
        self._show_square_heatmap_preview(with_background=False)

    def _save_square_heatmap_figure_to_path(self, figure: Figure, output_path: Path, transparent: bool) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=220, transparent=transparent)
        figure.clear()

    def _save_multiple_square_heatmaps(self, with_background: bool) -> None:
        if self._focus_active_batch_progress():
            return
        if self.video_state is None or self.csv_df is None:
            QMessageBox.warning(self, "Batch Heatmap Save", "Load a reference video and CSV first.")
            return
        if not self._square_heatmap_ready():
            QMessageBox.warning(
                self,
                "Batch Heatmap Save",
                "Square heatmap export needs either zero points for raw mode or four points for normalized mode.",
            )
            return

        selected_videos = self._select_videos_for_batch(
            dialog_title="Select Videos For Batch Heatmap Save",
            info_text="Select videos to export body-volume probability heatmap images using the current Square settings.",
            warning_text=(
                "Warning:\n"
                "- Only videos with auto-detected CSV candidates are exported.\n"
                "- The top auto-detected CSV candidate is used for each video.\n"
                "- Current square points and optional time/spatial ranges are applied to all selected videos."
            ),
            start_button_text="Start Batch Heatmap Save",
        )
        if not selected_videos:
            return

        source_width = max(1, int(self.video_state.width))
        source_height = max(1, int(self.video_state.height))
        source_square_points = [tuple(point) for point in self.frame_viewer.square_points]
        frame_number = self._square_current_frame_number()
        limit_trajectory = self.square_limit_trajectory_checkbox.isChecked()
        limit_spatial_region = self.trajectory_limit_region_checkbox.isChecked()
        source_region_bounds = self.frame_viewer.trajectory_region_bounds()
        cut_range = (
            self._resolve_square_cut_range_frames(show_message=True)
            if limit_trajectory
            else None
        )
        if limit_trajectory and cut_range is None:
            return
        if limit_spatial_region and source_region_bounds is None:
            QMessageBox.information(
                self, "Batch Heatmap", "Draw a rectangular spatial range first."
            )
            return
        save_folder = Path(self.save_folder)
        heatmap_mode = "normalized" if len(source_square_points) == 4 else "raw"
        image_suffix = "with_frame" if with_background else "transparent"

        def _export_item(item: BatchItem) -> Path:
            source_df = item.source_df
            if limit_spatial_region and source_region_bounds is not None:
                left, top, right, bottom = source_region_bounds
                scaled_bounds = (
                    left * item.scale_x,
                    top * item.scale_y,
                    right * item.scale_x,
                    bottom * item.scale_y,
                )
                source_df = self._filter_trajectory_spatial_region(
                    source_df,
                    item.bodyparts,
                    item.width,
                    item.height,
                    scaled_bounds,
                    message_title="Batch Heatmap",
                    show_message=False,
                )
                if source_df is None:
                    raise ValueError(
                        "No trajectory coordinates were found inside the selected rectangle."
                    )
            scaled_square_points = [
                (x * item.scale_x, y * item.scale_y)
                for x, y in source_square_points
            ]
            if scaled_square_points:
                export_df = build_normalized_dataframe(
                    source_df,
                    item.bodyparts,
                    scaled_square_points,
                    item.width,
                    item.height,
                )
                normalized = True
            else:
                export_df = source_df
                normalized = False
            if limit_trajectory:
                frame_col = self._find_matching_column(
                    export_df,
                    self.FRAME_COLUMN_CANDIDATES,
                )
                if frame_col is None:
                    raise ValueError("Could not find a frame column for the selected time range.")
                start_frame, end_frame = cut_range
                frame_values = pd.to_numeric(export_df[frame_col], errors="coerce")
                export_df = export_df.loc[
                    frame_values.between(start_frame, end_frame, inclusive="both")
                ].copy()
                if export_df.empty:
                    raise ValueError("No trajectory data was found in the selected time range.")

            rectified_size = None
            if normalized:
                _matrix, _inverse, rectified_size = build_rectified_geometry(scaled_square_points)

            background = None
            if with_background:
                background_frame = self._read_video_frame_rgb(item.video_path, frame_number)
                if background_frame is None:
                    raise ValueError("Could not read the background frame from the video.")
                background = self._square_heatmap_background_image_from_frame(
                    background_frame,
                    normalized,
                    scaled_square_points,
                )

            figure = self._build_square_heatmap_figure_with_background(
                df=export_df,
                bodyparts=item.bodyparts,
                normalized=normalized,
                background=background,
                with_background=with_background,
                frame_size=(item.width, item.height),
                rectified_size=rectified_size,
            )
            output_path = save_folder / (
                f"{item.csv_path.stem}_{heatmap_mode}_location_heatmap_{image_suffix}.png"
            )
            self._save_square_heatmap_figure_to_path(
                figure,
                output_path,
                transparent=not with_background,
            )
            return output_path

        def _on_completed(result: BatchRunResult) -> None:
            state = "cancelled" if result.cancelled else "finished"
            self.statusBar().showMessage(
                f"Batch heatmap save {state}: saved={result.saved_count}, "
                f"skipped={len(result.skipped_auto_missing)}, failed={len(result.failed)}"
            )

        self._start_batch_export(
            title="Batch Heatmap Save Progress",
            activity_text="Saving heatmap images in the background...",
            selected_videos=selected_videos,
            source_width=source_width,
            source_height=source_height,
            export_item=_export_item,
            on_completed=_on_completed,
        )

    def save_multiple_square_heatmaps_with_background(self) -> None:
        self._save_multiple_square_heatmaps(with_background=True)

    def save_multiple_square_heatmaps_without_background(self) -> None:
        self._save_multiple_square_heatmaps(with_background=False)
