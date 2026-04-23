import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from happycold_shared import build_normalized_dataframe


class SquareTabMixin:
    FRAME_COLUMN_CANDIDATES = ("frame idx", "frame_idx", "frame index", "frame_index", "frame")
    CUT_MODE_START_DURATION = "start_duration"
    CUT_MODE_DURATION_END = "duration_end"
    CUT_MODE_START_END = "start_end"

    def _build_square_tab(self) -> QWidget:
        tab = QWidget()
        tooltip = "Choose four corners for normalization and trajectory preview."
        tab.setToolTip(tooltip)
        layout = QVBoxLayout(tab)

        info = QLabel(
            "Click four corners on the frame. "
            "With 4 points, Preview shows normalized trajectory. "
            "With 0 points, Preview shows raw trajectory."
        )
        info.setWordWrap(True)
        info.setToolTip(tooltip)

        self.square_points_label = QLabel("Selected points: 0 / 4")
        self.square_points_label.setWordWrap(True)
        self.square_points_label.setToolTip("The current four-point selection in image coordinates.")

        self.square_reset_button = QPushButton("Reset Points")
        self.square_reset_button.setToolTip("Clear the current four-point selection and start over.")
        self.square_preview_button = QPushButton("Preview")
        self.square_preview_button.setEnabled(False)
        self.square_preview_button.setToolTip("With 4 points, open a normalized trajectory preview. With 0 points, open a raw trajectory preview.")

        button_row = QHBoxLayout()
        button_row.addWidget(self.square_reset_button)
        button_row.addWidget(self.square_preview_button)

        self.square_limit_trajectory_checkbox = QCheckBox("Show trajectory only for a limited time range")
        self.square_limit_trajectory_checkbox.setToolTip("Limit previewed trajectory to the selected cut parameters.")

        self.square_cut_group = QGroupBox("Cut Parameters")
        cut_layout = QGridLayout(self.square_cut_group)
        cut_layout.setContentsMargins(8, 8, 8, 8)
        cut_layout.setHorizontalSpacing(8)
        cut_layout.setVerticalSpacing(6)
        cut_layout.setColumnStretch(1, 1)

        self.square_cut_mode_label = QLabel("Mode:")
        self.square_cut_mode_combo = QComboBox()
        self.square_cut_mode_combo.addItem("Start + Duration", self.CUT_MODE_START_DURATION)
        self.square_cut_mode_combo.addItem("Duration + End", self.CUT_MODE_DURATION_END)
        self.square_cut_mode_combo.addItem("Start + End", self.CUT_MODE_START_END)

        self.square_start_label = QLabel("Start")
        self.square_start_spinbox = QDoubleSpinBox()
        self.square_start_spinbox.setRange(0.0, 10_000_000.0)
        self.square_start_spinbox.setDecimals(0)
        self.square_start_spinbox.setSingleStep(1.0)
        self.square_start_spinbox.setValue(0.0)
        self.square_start_use_current_button = QPushButton("Use current")

        self.square_duration_label = QLabel("Duration")
        self.square_duration_spinbox = QDoubleSpinBox()
        self.square_duration_spinbox.setRange(0.0, 10_000_000.0)
        self.square_duration_spinbox.setValue(300.0)
        self.square_duration_unit_combo = QComboBox()
        self.square_duration_unit_combo.addItem("seconds", "sec")
        self.square_duration_unit_combo.addItem("frames", "frame")
        self.square_duration_unit_combo.addItem("minutes", "min")
        self.square_duration_unit_combo.setCurrentIndex(1)
        # Keep combobox text centered like the adjacent fixed-width controls.
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
        self.square_end_spinbox = QDoubleSpinBox()
        self.square_end_spinbox.setRange(0.0, 10_000_000.0)
        self.square_end_spinbox.setDecimals(0)
        self.square_end_spinbox.setSingleStep(1.0)
        self.square_end_spinbox.setValue(0.0)
        self.square_end_use_current_button = QPushButton("Use current")

        self.square_cut_range_label = QLabel("Range (frame): -")
        self.square_cut_range_label.setWordWrap(True)

        for spinbox in (self.square_start_spinbox, self.square_duration_spinbox, self.square_end_spinbox):
            spinbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            spinbox.setMinimumWidth(260)

        side_control_width = 140
        self.square_start_use_current_button.setFixedWidth(side_control_width)
        self.square_end_use_current_button.setFixedWidth(side_control_width)
        self.square_duration_unit_combo.setFixedWidth(side_control_width)

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

        layout.addWidget(info)
        layout.addWidget(self.square_points_label)
        layout.addLayout(button_row)
        layout.addStretch(1)
        layout.addWidget(self.square_limit_trajectory_checkbox)
        layout.addWidget(self.square_cut_group)

        self.square_limit_trajectory_checkbox.toggled.connect(self._refresh_square_time_controls)
        self.square_cut_mode_combo.currentIndexChanged.connect(self._on_square_cut_mode_changed)
        self.square_duration_unit_combo.currentIndexChanged.connect(self._on_square_duration_unit_changed)
        self.square_start_spinbox.valueChanged.connect(self._on_square_cut_value_changed)
        self.square_duration_spinbox.valueChanged.connect(self._on_square_cut_value_changed)
        self.square_end_spinbox.valueChanged.connect(self._on_square_cut_value_changed)
        self.square_start_use_current_button.clicked.connect(self._set_square_start_from_current_frame)
        self.square_end_use_current_button.clicked.connect(self._set_square_end_from_current_frame)

        self._refresh_square_duration_spinbox_format()
        self._refresh_square_current_buttons()
        self._refresh_square_time_controls()
        return tab

    def _refresh_square_ui(self) -> None:
        detail = " | ".join(f"{index + 1}: ({point[0]:.1f}, {point[1]:.1f})" for index, point in enumerate(self.frame_viewer.square_points))
        self.square_points_label.setText(f"Selected points: {len(self.frame_viewer.square_points)} / 4\n{detail if detail else '-'}")
        self.square_preview_button.setEnabled(self.csv_df is not None)
        if (
            hasattr(self, "mode_tabs")
            and self.mode_tabs.currentIndex() == 0
            and self.square_preview_button.isEnabled()
        ):
            self.square_preview_button.setFocus()
        self._refresh_square_current_buttons()
        self._refresh_square_cut_summary()
        self._refresh_output_ui()

    def _on_square_points_changed(self) -> None:
        self._refresh_square_ui()

    def preview_square_normalization(self) -> None:
        if self.csv_df is None or self.video_state is None:
            QMessageBox.warning(self, "Preview", "Load a video and CSV first.")
            return
        point_count = len(self.frame_viewer.square_points)
        if point_count == 0:
            preview_df = self.csv_df
            if self.square_limit_trajectory_checkbox.isChecked():
                preview_df = self._filter_limited_trajectory(preview_df)
                if preview_df is None:
                    return
            self._show_normalized_preview(preview_df, normalized=False)
            return
        if point_count != 4:
            QMessageBox.warning(self, "Preview", "Choose either zero points for a raw preview or four points for a normalized preview.")
            return
        normalized_df = build_normalized_dataframe(self.csv_df, self.bodyparts, self.frame_viewer.square_points, self.video_state.width, self.video_state.height)
        if self.square_limit_trajectory_checkbox.isChecked():
            normalized_df = self._filter_limited_trajectory(normalized_df)
            if normalized_df is None:
                return
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

    def _filter_limited_trajectory(self, df: pd.DataFrame) -> pd.DataFrame | None:
        frame_col = self._find_matching_column(df, self.FRAME_COLUMN_CANDIDATES)
        if frame_col is None:
            QMessageBox.warning(
                self,
                "Preview",
                "Could not find a frame column.\nLimited trajectory preview needs a frame/frame_idx style column.",
            )
            return None

        resolved = self._resolve_square_cut_range_frames(show_message=True)
        if resolved is None:
            return None
        start_frame, end_frame = resolved

        frame_values = pd.to_numeric(df[frame_col], errors="coerce")
        limited_df = df.loc[frame_values.between(start_frame, end_frame, inclusive="both")].copy()
        if limited_df.empty:
            QMessageBox.information(self, "Preview", "No trajectory data found in the selected time range.")
            return None
        return limited_df

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
