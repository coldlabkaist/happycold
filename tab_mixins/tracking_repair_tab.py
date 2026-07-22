from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from tracking_postprocess import (
    TrackingPostprocessConfig,
    find_duplicate_skeleton_candidates,
    find_length_outlier_candidates,
    remove_duplicate_skeletons,
    run_tracking_postprocess,
)
from ui_controls import NoWheelComboBox, NoWheelDoubleSpinBox


class TrackingRepairTabMixin:
    TRACKING_REPAIR_DISPLAY_LIMIT = 10

    def _build_tracking_repair_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 12)
        layout.setSpacing(10)

        flow = QLabel(
            "1. Remove duplicate detections  →  2. Invalidate robust Z-score length outliers  "
            "→  3. Continue to region repair and interpolation"
        )
        flow.setWordWrap(True)
        flow.setProperty("muted", True)
        layout.addWidget(flow)

        duplicate_group = QGroupBox("1. Duplicate Skeleton Removal")
        duplicate_form = QFormLayout(duplicate_group)
        duplicate_form.setHorizontalSpacing(8)
        duplicate_form.setVerticalSpacing(7)
        self.tracking_duplicate_enabled_checkbox = QCheckBox("Apply duplicate removal")
        self.tracking_duplicate_enabled_checkbox.setChecked(
            bool(self.settings.get("tracking_duplicate_enabled", True))
        )
        self.tracking_duplicate_criteria_spinbox = NoWheelDoubleSpinBox()
        self.tracking_duplicate_criteria_spinbox.setRange(0.0, 1_000_000.0)
        self.tracking_duplicate_criteria_spinbox.setDecimals(2)
        self.tracking_duplicate_criteria_spinbox.setValue(
            float(self.settings.get("tracking_duplicate_criteria", 120.0))
        )
        self.tracking_duplicate_criteria_spinbox.setSuffix(" px sum")
        duplicate_note = QLabel(
            "Skeleton pairs in the same frame are compared using the summed distance of all "
            "jointly valid bodyparts. The higher-confidence row is kept."
        )
        duplicate_note.setWordWrap(True)
        duplicate_note.setProperty("muted", True)
        duplicate_form.addRow(self.tracking_duplicate_enabled_checkbox)
        duplicate_form.addRow("Distance threshold", self.tracking_duplicate_criteria_spinbox)
        duplicate_form.addRow("", duplicate_note)
        layout.addWidget(duplicate_group)

        zscore_group = QGroupBox("2. Robust Z-score Length Outlier Invalidation")
        zscore_form = QFormLayout(zscore_group)
        zscore_form.setHorizontalSpacing(8)
        zscore_form.setVerticalSpacing(7)
        self.tracking_zscore_enabled_checkbox = QCheckBox("Invalidate length outliers")
        self.tracking_zscore_enabled_checkbox.setChecked(
            bool(self.settings.get("tracking_zscore_enabled", True))
        )
        self.tracking_zscore_mode_combo = NoWheelComboBox()
        self.tracking_zscore_mode_combo.addItem("Both", "both")
        self.tracking_zscore_mode_combo.addItem("High only", "high_only")
        self.tracking_zscore_mode_combo.addItem("Low only", "low_only")
        saved_mode = str(self.settings.get("tracking_zscore_mode", "both"))
        saved_mode_index = self.tracking_zscore_mode_combo.findData(saved_mode)
        self.tracking_zscore_mode_combo.setCurrentIndex(max(0, saved_mode_index))
        self.tracking_zscore_high_spinbox = NoWheelDoubleSpinBox()
        self.tracking_zscore_high_spinbox.setRange(0.1, 100.0)
        self.tracking_zscore_high_spinbox.setDecimals(2)
        self.tracking_zscore_high_spinbox.setValue(
            float(self.settings.get("tracking_zscore_high", 3.5))
        )
        self.tracking_zscore_low_spinbox = NoWheelDoubleSpinBox()
        self.tracking_zscore_low_spinbox.setRange(0.1, 100.0)
        self.tracking_zscore_low_spinbox.setDecimals(2)
        self.tracking_zscore_low_spinbox.setValue(
            float(self.settings.get("tracking_zscore_low", 3.5))
        )
        zscore_note = QLabel(
            "Uses the median and MAD for body-center to left/right body, nose, and tail "
            "segments. A flagged row is kept, while its skeleton coordinates and scores become NaN so interpolation can fill both original and newly invalidated gaps."
        )
        zscore_note.setWordWrap(True)
        zscore_note.setProperty("muted", True)
        zscore_form.addRow(self.tracking_zscore_enabled_checkbox)
        zscore_form.addRow("Deviation side", self.tracking_zscore_mode_combo)
        zscore_form.addRow("High threshold", self.tracking_zscore_high_spinbox)
        zscore_form.addRow("Low threshold", self.tracking_zscore_low_spinbox)
        zscore_form.addRow("", zscore_note)
        layout.addWidget(zscore_group)

        review_group = QGroupBox("Repair Examples")
        review_layout = QVBoxLayout(review_group)
        self.tracking_repair_summary_label = QLabel(
            "Load a video and CSV, then scan a few rows that would be removed or invalidated."
        )
        self.tracking_repair_summary_label.setWordWrap(True)
        self.tracking_repair_summary_label.setProperty("muted", True)
        self.tracking_repair_preview_button = QPushButton("Display Next")
        self.tracking_repair_preview_button.setProperty("primary", True)
        review_layout.addWidget(self.tracking_repair_summary_label)
        review_layout.addWidget(self.tracking_repair_preview_button)
        layout.addWidget(review_group)
        layout.addStretch(1)

        scroll.setWidget(content)
        tab_layout.addWidget(scroll)

        self._tracking_repair_display_cases: list[dict] = []
        self._tracking_repair_display_index = -1
        self._tracking_repair_display_signature = None
        self._tracking_repair_display_row_indices: frozenset[object] = frozenset()
        self._tracking_repair_deleted_row_indices: frozenset[object] = frozenset()
        self._tracking_repair_display_frame_number: int | None = None
        self.tracking_repair_preview_button.clicked.connect(
            self.display_next_tracking_repair_case
        )
        for checkbox in (
            self.tracking_duplicate_enabled_checkbox,
            self.tracking_zscore_enabled_checkbox,
        ):
            checkbox.toggled.connect(self._on_tracking_repair_setting_changed)
        for spinbox in (
            self.tracking_duplicate_criteria_spinbox,
            self.tracking_zscore_high_spinbox,
            self.tracking_zscore_low_spinbox,
        ):
            spinbox.valueChanged.connect(self._on_tracking_repair_setting_changed)
        self.tracking_zscore_mode_combo.currentIndexChanged.connect(
            self._on_tracking_repair_setting_changed
        )
        self._refresh_tracking_repair_ui()
        return tab

    def _tracking_repair_config(self) -> TrackingPostprocessConfig:
        return TrackingPostprocessConfig(
            remove_duplicates=self.tracking_duplicate_enabled_checkbox.isChecked(),
            duplicate_distance_threshold=self.tracking_duplicate_criteria_spinbox.value(),
            remove_length_outliers=self.tracking_zscore_enabled_checkbox.isChecked(),
            deviation_mode=str(self.tracking_zscore_mode_combo.currentData()),
            high_z_threshold=self.tracking_zscore_high_spinbox.value(),
            low_z_threshold=self.tracking_zscore_low_spinbox.value(),
        )

    def _tracking_repair_settings_payload(self) -> dict:
        if not hasattr(self, "tracking_duplicate_enabled_checkbox"):
            return {}
        config = self._tracking_repair_config()
        return {
            "tracking_duplicate_enabled": config.remove_duplicates,
            "tracking_duplicate_criteria": config.duplicate_distance_threshold,
            "tracking_zscore_enabled": config.remove_length_outliers,
            "tracking_zscore_mode": config.deviation_mode,
            "tracking_zscore_high": config.high_z_threshold,
            "tracking_zscore_low": config.low_z_threshold,
        }

    def _on_tracking_repair_setting_changed(self, *_args) -> None:
        self._invalidate_tracking_repair_display_cases()
        self._refresh_tracking_repair_ui()
        if hasattr(self, "output_workspace"):
            self._refresh_output_ui()
        self._save_settings()

    def _refresh_tracking_repair_ui(self) -> None:
        if not hasattr(self, "tracking_repair_preview_button"):
            return
        duplicate_enabled = self.tracking_duplicate_enabled_checkbox.isChecked()
        zscore_enabled = self.tracking_zscore_enabled_checkbox.isChecked()
        self.tracking_duplicate_criteria_spinbox.setEnabled(duplicate_enabled)
        self.tracking_zscore_mode_combo.setEnabled(zscore_enabled)
        mode = str(self.tracking_zscore_mode_combo.currentData())
        self.tracking_zscore_high_spinbox.setEnabled(zscore_enabled and mode != "low_only")
        self.tracking_zscore_low_spinbox.setEnabled(zscore_enabled and mode != "high_only")
        ready = (
            self.csv_df is not None
            and self.video_state is not None
            and bool(self.bodyparts)
            and (duplicate_enabled or zscore_enabled)
        )
        self.tracking_repair_preview_button.setEnabled(ready)
        if not duplicate_enabled and not zscore_enabled:
            self.tracking_repair_summary_label.setText("Enable at least one tracking repair step.")
        elif self.csv_df is None or self.video_state is None:
            self.tracking_repair_summary_label.setText(
                "Load a video and CSV, then scan a few rows that would be removed or invalidated."
            )
        elif (
            self._tracking_repair_display_cases
            and self._tracking_repair_display_signature
            == self._tracking_repair_review_signature()
            and 0 <= self._tracking_repair_display_index < len(self._tracking_repair_display_cases)
        ):
            self._set_tracking_repair_case_summary(
                self._tracking_repair_display_cases[self._tracking_repair_display_index]
            )
        else:
            enabled = []
            if duplicate_enabled:
                enabled.append("duplicate removal")
            if zscore_enabled:
                enabled.append("robust Z-score invalidation")
            self.tracking_repair_summary_label.setText(
                f"Ready · {len(self.csv_df):,} source row(s) · "
                + " → ".join(enabled)
                + f" · Display up to {self.TRACKING_REPAIR_DISPLAY_LIMIT} repair examples."
            )

    def _run_tracking_repair_for(
        self,
        dataframe,
        bodyparts,
        width: int,
        height: int,
        config=None,
    ):
        return run_tracking_postprocess(
            dataframe,
            config=self._tracking_repair_config() if config is None else config,
            bodyparts=bodyparts,
            width=width,
            height=height,
        )

    def _tracking_repair_review_signature(self) -> tuple | None:
        if self.csv_df is None or self.video_state is None:
            return None
        config = self._tracking_repair_config()
        return (
            id(self.csv_df),
            len(self.csv_df),
            self.video_state.width,
            self.video_state.height,
            tuple(self.bodyparts),
            config.remove_duplicates,
            config.duplicate_distance_threshold,
            config.remove_length_outliers,
            config.deviation_mode,
            config.high_z_threshold,
            config.low_z_threshold,
        )

    def _invalidate_tracking_repair_display_cases(self) -> None:
        self._tracking_repair_display_cases = []
        self._tracking_repair_display_index = -1
        self._tracking_repair_display_signature = None
        self._tracking_repair_display_row_indices = frozenset()
        self._tracking_repair_deleted_row_indices = frozenset()
        self._tracking_repair_display_frame_number = None
        if hasattr(self, "frame_viewer"):
            self._refresh_node_overlay()

    @staticmethod
    def _tracking_row_label(row_index: object) -> str:
        return str(row_index)

    def _build_duplicate_display_cases(self, dataframe, config) -> tuple[list[dict], object]:
        candidates, _warning = find_duplicate_skeleton_candidates(
            dataframe,
            bodyparts=self.bodyparts,
            criteria=config.duplicate_distance_threshold,
            width=self.video_state.width,
            height=self.video_state.height,
        )
        stage_result = remove_duplicate_skeletons(
            dataframe,
            bodyparts=self.bodyparts,
            criteria=config.duplicate_distance_threshold,
            width=self.video_state.width,
            height=self.video_state.height,
        )
        removed_rows = set(dataframe.index).difference(stage_result.dataframe.index)
        cases: list[dict] = []
        represented: set[object] = set()
        for candidate in candidates:
            pair = (candidate.row_index_a, candidate.row_index_b)
            deleted = tuple(row for row in pair if row in removed_rows and row not in represented)
            if not deleted:
                continue
            represented.update(deleted)
            tracks = f"{candidate.track_id_a or '-'} / {candidate.track_id_b or '-'}"
            cases.append(
                {
                    "kind": "Duplicate",
                    "frame_value": candidate.frame_value,
                    "row_indices": pair,
                    "deleted_row_indices": deleted,
                    "detail": (
                        f"delete row {', '.join(self._tracking_row_label(row) for row in deleted)} · "
                        f"tracks {tracks} · distance sum {candidate.distance_sum:.2f}px · "
                        f"confidence {candidate.confidence_a:.3f} / {candidate.confidence_b:.3f}"
                    ),
                }
            )
        return cases, stage_result

    def _build_length_display_cases(self, dataframe, config) -> tuple[list[dict], str | None]:
        candidates, warning = find_length_outlier_candidates(
            dataframe,
            bodyparts=self.bodyparts,
            high_z_threshold=config.high_z_threshold,
            low_z_threshold=config.low_z_threshold,
            deviation_mode=config.deviation_mode,
            width=self.video_state.width,
            height=self.video_state.height,
        )
        cases = [
            {
                "kind": "Length Z-score",
                "frame_value": candidate.frame_value,
                "row_indices": (candidate.row_index,),
                "deleted_row_indices": (candidate.row_index,),
                "detail": (
                    f"invalidate skeleton at row {self._tracking_row_label(candidate.row_index)} · "
                    f"track {candidate.track_id or '-'} · "
                    f"{', '.join(candidate.segments)} · max deviation {candidate.max_deviation:.2f}"
                ),
            }
            for candidate in candidates
        ]
        return cases, warning

    def _build_tracking_repair_display_cases(self) -> tuple[list[dict], tuple[str, ...]]:
        if self.csv_df is None or self.video_state is None:
            return [], ()
        config = self._tracking_repair_config()
        working = self.csv_df
        duplicate_cases: list[dict] = []
        length_cases: list[dict] = []
        warnings: list[str] = []
        if config.remove_duplicates:
            duplicate_cases, duplicate_stage = self._build_duplicate_display_cases(
                working, config
            )
            working = duplicate_stage.dataframe
            if duplicate_stage.warning:
                warnings.append(duplicate_stage.warning)
        if config.remove_length_outliers:
            length_cases, warning = self._build_length_display_cases(working, config)
            if warning:
                warnings.append(warning)

        limit = self.TRACKING_REPAIR_DISPLAY_LIMIT
        if duplicate_cases and length_cases:
            first_quota = max(1, limit // 2)
            selected = duplicate_cases[:first_quota] + length_cases[: limit - first_quota]
            if len(selected) < limit:
                selected_ids = {id(case) for case in selected}
                remaining = [
                    case
                    for case in duplicate_cases + length_cases
                    if id(case) not in selected_ids
                ]
                selected.extend(remaining[: limit - len(selected)])
        else:
            selected = (duplicate_cases or length_cases)[:limit]
        selected.sort(key=lambda case: (int(case["frame_value"]), str(case["kind"])))
        return selected, tuple(dict.fromkeys(warnings))

    def _tracking_case_frame_number(self, frame_value: object) -> int:
        frame_number = int(frame_value)
        if getattr(self, "_node_frame_zero_based", False):
            frame_number += 1
        return max(1, min(frame_number, self.video_state.frame_count))

    def _set_tracking_repair_case_summary(self, case: dict) -> None:
        current = self._tracking_repair_display_index + 1
        total = len(self._tracking_repair_display_cases)
        self.tracking_repair_summary_label.setText(
            f"Case {current}/{total} · {case['kind']} · frame {case['frame_value']} · "
            f"{case['detail']} · red = removed / invalidated, orange = comparison"
        )

    def _show_tracking_repair_display_case(self, case: dict) -> None:
        frame_number = self._tracking_case_frame_number(case["frame_value"])
        self._tracking_repair_display_row_indices = frozenset(case["row_indices"])
        self._tracking_repair_deleted_row_indices = frozenset(
            case["deleted_row_indices"]
        )
        self._tracking_repair_display_frame_number = frame_number
        if not self.show_nodes_checkbox.isChecked():
            self.show_nodes_checkbox.setChecked(True)
        if self.frame_slider.value() == frame_number:
            self._load_frame(frame_number)
        else:
            self.frame_slider.setValue(frame_number)
        self._set_tracking_repair_case_summary(case)
        self.statusBar().showMessage(
            f"Tracking repair example {self._tracking_repair_display_index + 1}/"
            f"{len(self._tracking_repair_display_cases)}"
        )

    def display_next_tracking_repair_case(self) -> None:
        if self.csv_df is None or self.video_state is None:
            QMessageBox.warning(self, "Tracking Repair", "Load a video and CSV first.")
            return
        config = self._tracking_repair_config()
        if not config.remove_duplicates and not config.remove_length_outliers:
            QMessageBox.information(self, "Tracking Repair", "Enable at least one repair step.")
            return
        signature = self._tracking_repair_review_signature()
        if self._tracking_repair_display_signature != signature:
            try:
                cases, warnings = self._build_tracking_repair_display_cases()
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Tracking Repair",
                    f"Could not find repair examples.\n\n{exc}",
                )
                return
            self._tracking_repair_display_cases = cases
            self._tracking_repair_display_index = -1
            self._tracking_repair_display_signature = signature
            if not cases:
                message = "No rows would be removed or invalidated with the current settings."
                if warnings:
                    message += " " + " ".join(warnings)
                self.tracking_repair_summary_label.setText(message)
                self._tracking_repair_display_row_indices = frozenset()
                self._tracking_repair_deleted_row_indices = frozenset()
                self._refresh_node_overlay()
                return
        self._tracking_repair_display_index = (
            self._tracking_repair_display_index + 1
        ) % len(self._tracking_repair_display_cases)
        self._show_tracking_repair_display_case(
            self._tracking_repair_display_cases[self._tracking_repair_display_index]
        )

    def preview_tracking_repair(self) -> None:
        """Compatibility wrapper for older callers."""
        self.display_next_tracking_repair_case()

    def _tracking_repair_output_path(self) -> Path | None:
        return None if self.csv_path is None else self._tracking_repair_output_path_for(self.csv_path)

    def _tracking_repair_output_path_for(self, csv_path: Path) -> Path:
        return self._apply_current_output_affixes(self.save_folder / f"{csv_path.stem}_tracking_postprocessed.csv")

    def save_tracking_repair_csv(self) -> None:
        if self.csv_df is None or self.video_state is None:
            QMessageBox.warning(self, "Tracking Repair", "Load a video and CSV first.")
            return
        output = self._tracking_repair_output_path()
        if output is None:
            return
        try:
            result = self._run_tracking_repair_for(
                self.csv_df,
                self.bodyparts,
                self.video_state.width,
                self.video_state.height,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            result.dataframe.to_csv(output, index=False)
        except Exception as exc:
            QMessageBox.warning(self, "Tracking Repair", f"Could not save tracking postprocess CSV.\n\n{exc}")
            return
        self.statusBar().showMessage(
            f"Tracking postprocess CSV saved: {output} "
            f"({len(self.csv_df):,} → {len(result.dataframe):,} rows; "
            f"duplicates removed={result.duplicate_removed:,}, "
            f"Z-score skeletons invalidated={result.length_outliers_invalidated:,})"
        )


__all__ = ["TrackingRepairTabMixin"]
