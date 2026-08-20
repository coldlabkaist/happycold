from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
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

from smoothing import SMOOTHING_METHODS, build_smoothed_dataframe
from ui_controls import NoWheelComboBox


class SmoothingTabMixin:
    def _build_smoothing_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setObjectName("smoothingScroll")
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 12)
        layout.setSpacing(10)

        info = QLabel(
            "Stabilize tracked coordinates by shifting each observed skeleton from an anchor-node temporal window."
        )
        info.setWordWrap(True)
        info.setProperty("muted", True)
        layout.addWidget(info)

        method_group = QGroupBox("Smoothing Method")
        method_form = QFormLayout(method_group)
        method_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.smoothing_method_combo = NoWheelComboBox()
        for spec in SMOOTHING_METHODS.values():
            self.smoothing_method_combo.addItem(spec.label, spec.key)
        saved_method = str(self.settings.get("smoothing_method", "anchor_median_smoothing"))
        method_index = self.smoothing_method_combo.findData(saved_method)
        self.smoothing_method_combo.setCurrentIndex(max(0, method_index))
        method_form.addRow("Method", self.smoothing_method_combo)
        layout.addWidget(method_group)

        self.smoothing_options_group = QGroupBox("Anchor Median Smoothing Options")
        options_form = QFormLayout(self.smoothing_options_group)
        options_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.smoothing_anchor_combo = NoWheelComboBox()
        self.smoothing_anchor_combo.addItem("Select anchor node...", None)
        self.smoothing_window_combo = NoWheelComboBox()
        self.smoothing_window_combo.addItem("3 frames", 3)
        self.smoothing_window_combo.addItem("5 frames", 5)
        saved_window = int(self.settings.get("smoothing_window_size", 3))
        window_index = self.smoothing_window_combo.findData(saved_window)
        self.smoothing_window_combo.setCurrentIndex(max(0, window_index))
        options_form.addRow("Anchor node", self.smoothing_anchor_combo)
        options_form.addRow("Window", self.smoothing_window_combo)
        layout.addWidget(self.smoothing_options_group)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.smoothing_preview_label = QLabel("No smoothing preview.")
        self.smoothing_preview_label.setWordWrap(True)
        self.smoothing_preview_label.setProperty("muted", True)
        preview_buttons = QHBoxLayout()
        self.smoothing_preview_button = QPushButton("Preview Current Frame")
        self.smoothing_preview_button.setProperty("primary", True)
        self.smoothing_clear_preview_button = QPushButton("Clear Preview")
        preview_buttons.addWidget(self.smoothing_preview_button)
        preview_buttons.addWidget(self.smoothing_clear_preview_button)
        preview_layout.addWidget(self.smoothing_preview_label)
        preview_layout.addLayout(preview_buttons)
        layout.addWidget(preview_group)

        layout.addStretch(1)
        scroll.setWidget(content)
        tab_layout.addWidget(scroll)

        self.smoothing_method_combo.currentIndexChanged.connect(self._on_smoothing_option_changed)
        self.smoothing_anchor_combo.currentIndexChanged.connect(self._on_smoothing_option_changed)
        self.smoothing_window_combo.currentIndexChanged.connect(self._on_smoothing_option_changed)
        self.smoothing_preview_button.clicked.connect(self.preview_smoothing_current_frame)
        self.smoothing_clear_preview_button.clicked.connect(self.clear_smoothing_preview)
        self._smoothing_preview_df = None
        self._smoothing_preview_active = False
        self._refresh_smoothing_ui()
        return tab

    def _selected_smoothing_method(self) -> str:
        if not hasattr(self, "smoothing_method_combo"):
            return "anchor_median_smoothing"
        return str(self.smoothing_method_combo.currentData())

    def _selected_smoothing_anchor(self) -> str | None:
        if not hasattr(self, "smoothing_anchor_combo"):
            return None
        value = self.smoothing_anchor_combo.currentData()
        return None if value is None else str(value)

    def _selected_smoothing_window_size(self) -> int:
        if not hasattr(self, "smoothing_window_combo"):
            return 3
        try:
            value = int(self.smoothing_window_combo.currentData())
        except (TypeError, ValueError):
            value = 3
        return value if value in {3, 5} else 3

    def _smoothing_settings_payload(self) -> dict:
        if not hasattr(self, "smoothing_method_combo"):
            return {}
        return {
            "smoothing_method": self._selected_smoothing_method(),
            "smoothing_anchor": self._selected_smoothing_anchor(),
            "smoothing_window_size": self._selected_smoothing_window_size(),
        }

    def _populate_smoothing_anchor_combo(self) -> None:
        if not hasattr(self, "smoothing_anchor_combo"):
            return
        expected_values: list[str | None] = [None, *self.bodyparts]
        current_values = [
            self.smoothing_anchor_combo.itemData(index)
            for index in range(self.smoothing_anchor_combo.count())
        ]
        if current_values == expected_values:
            return
        selected_anchor = self._selected_smoothing_anchor() or self.settings.get("smoothing_anchor")
        self.smoothing_anchor_combo.blockSignals(True)
        self.smoothing_anchor_combo.clear()
        self.smoothing_anchor_combo.addItem("Select anchor node...", None)
        for bodypart in self.bodyparts:
            self.smoothing_anchor_combo.addItem(bodypart, bodypart)
        selected_index = self.smoothing_anchor_combo.findData(selected_anchor)
        self.smoothing_anchor_combo.setCurrentIndex(max(0, selected_index))
        self.smoothing_anchor_combo.blockSignals(False)

    def _refresh_smoothing_ui(self) -> None:
        if not hasattr(self, "smoothing_preview_button"):
            return
        self._populate_smoothing_anchor_combo()
        ready = self.csv_df is not None and bool(self.bodyparts)
        has_anchor = self._selected_smoothing_anchor() is not None
        self.smoothing_anchor_combo.setEnabled(ready)
        self.smoothing_method_combo.setEnabled(ready)
        self.smoothing_window_combo.setEnabled(ready)
        self.smoothing_preview_button.setEnabled(ready and has_anchor)
        self.smoothing_clear_preview_button.setEnabled(bool(getattr(self, "_smoothing_preview_active", False)))
        if not ready:
            self.smoothing_preview_label.setText("Load a video and CSV first.")
        elif not has_anchor:
            self.smoothing_preview_label.setText("Select an anchor node.")
        elif getattr(self, "_smoothing_preview_active", False):
            self.smoothing_preview_label.setText("Smoothing preview is displayed on the current frame.")
        else:
            window = self._selected_smoothing_window_size()
            self.smoothing_preview_label.setText(f"Ready: anchor median smoothing with a {window}-frame window.")
        if hasattr(self, "save_current_button"):
            self._refresh_output_ui()

    def _on_smoothing_option_changed(self, *_args) -> None:
        self.clear_smoothing_preview(refresh_ui=False)
        self._refresh_smoothing_ui()
        if hasattr(self, "_refresh_pipeline_ui"):
            self._refresh_pipeline_ui()
        if hasattr(self, "_save_settings"):
            self._save_settings()

    def _smoothing_output_path(self) -> Path | None:
        return None if self.csv_path is None else self._smoothing_output_path_for(self.csv_path)

    def _smoothing_output_path_for(self, csv_path: Path) -> Path:
        return self._apply_current_output_affixes(self.save_folder / f"{csv_path.stem}_smoothed.csv")

    def _build_smoothing_dataframe(self, dataframe=None):
        source_df = self.csv_df if dataframe is None else dataframe
        if source_df is None:
            raise ValueError("Load a CSV first.")
        return build_smoothed_dataframe(
            source_df,
            list(self.bodyparts),
            self._selected_smoothing_method(),
            self._selected_smoothing_anchor(),
            self._selected_smoothing_window_size(),
        )

    def preview_smoothing_current_frame(self) -> None:
        if self.csv_df is None or self.video_state is None or not self.bodyparts:
            QMessageBox.warning(self, "Smoothing", "Load a video and CSV first.")
            return
        if self._selected_smoothing_anchor() is None:
            QMessageBox.warning(self, "Smoothing", "Select an anchor node.")
            return
        try:
            smoothed = self._build_smoothing_dataframe()
        except Exception as exc:
            QMessageBox.warning(self, "Smoothing", f"Could not build smoothing preview.\n\n{exc}")
            return
        self._smoothing_preview_df = smoothed
        self._smoothing_preview_active = True
        shifted = self._smoothing_preview_shift_count(self.current_frame_number)
        self.smoothing_preview_label.setText(
            f"Preview frame {self.current_frame_number}: {shifted:,} node(s) shifted."
        )
        self._refresh_node_overlay()
        self._refresh_smoothing_ui()

    def clear_smoothing_preview(self, refresh_ui: bool = True) -> None:
        self._smoothing_preview_df = None
        self._smoothing_preview_active = False
        if hasattr(self, "frame_viewer"):
            self._refresh_node_overlay()
        if refresh_ui:
            self._refresh_smoothing_ui()

    def _smoothing_preview_shift_count(self, frame_number: int) -> int:
        if self.csv_df is None or self._smoothing_preview_df is None:
            return 0
        source_rows = self._csv_rows_for_frame(frame_number)
        if source_rows.empty:
            return 0
        preview_rows = self._smoothing_preview_df.loc[
            self._smoothing_preview_df.index.intersection(source_rows.index)
        ]
        shifted = 0
        for row_index in preview_rows.index:
            for bodypart in self.bodyparts:
                x_col = f"{bodypart}.x"
                y_col = f"{bodypart}.y"
                if x_col not in self.csv_df.columns or y_col not in self.csv_df.columns:
                    continue
                before = self.csv_df.loc[row_index, [x_col, y_col]].to_numpy(dtype=np.float64)
                after = self._smoothing_preview_df.loc[row_index, [x_col, y_col]].to_numpy(dtype=np.float64)
                if np.all(np.isfinite(before)) and np.all(np.isfinite(after)) and not np.allclose(before, after):
                    shifted += 1
        return shifted

    def save_smoothed_csv(self) -> None:
        if self.csv_df is None or self.video_state is None or not self.bodyparts:
            QMessageBox.warning(self, "Smoothing", "Load a video and CSV first.")
            return
        output = self._smoothing_output_path()
        if output is None:
            return
        try:
            result_df = self._build_smoothing_dataframe()
            output.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output, index=False)
        except Exception as exc:
            QMessageBox.warning(self, "Smoothing", f"Could not save smoothed CSV.\n\n{exc}")
            return
        self.statusBar().showMessage(f"Smoothed CSV saved: {output}")
