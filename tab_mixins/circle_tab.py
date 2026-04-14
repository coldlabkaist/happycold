from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget

from happycold_shared import build_circle_detection_dataframe


class CircleTabMixin:
    def _build_circle_tab(self) -> QWidget:
        tab = QWidget()
        tooltip = "Draw a diameter on the frame. The adjusted radius is used to classify each bodypart as inside or outside the circle."
        tab.setToolTip(tooltip)
        layout = QVBoxLayout(tab)
        info = QLabel("Drag a diameter on the frame. All bodyparts are evaluated when saving.")
        info.setWordWrap(True)
        info.setToolTip(tooltip)
        self.circle_margin_slider = QSlider(Qt.Orientation.Horizontal)
        self.circle_margin_slider.setRange(-300, 300)
        self.circle_margin_slider.setValue(0)
        self.circle_margin_slider.setToolTip("Increase or decrease the detection radius in pixels.")
        self.circle_margin_spinbox = QSpinBox()
        self.circle_margin_spinbox.setRange(-300, 300)
        self.circle_margin_spinbox.setSuffix(" px")
        self.circle_margin_spinbox.setFixedWidth(96)
        self.circle_margin_spinbox.setToolTip("Precise pixel offset added to the drawn circle radius.")
        self.circle_margin_label = QLabel("Detection Margin")
        self.circle_margin_label.setToolTip("Positive values expand the circle. Negative values shrink it.")
        self.circle_summary_label = QLabel("Draw a circle first.")
        self.circle_summary_label.setWordWrap(True)
        self.circle_summary_label.setToolTip("Shows the center, base radius, and adjusted radius.")
        self.circle_reset_button = QPushButton("Reset Circle")
        self.circle_reset_button.setToolTip("Clear the current circle selection.")
        circle_margin_row = QHBoxLayout()
        circle_margin_row.addWidget(self.circle_margin_slider, stretch=1)
        circle_margin_row.addWidget(self.circle_margin_spinbox)
        layout.addWidget(info)
        layout.addWidget(self.circle_margin_label)
        layout.addLayout(circle_margin_row)
        layout.addWidget(self.circle_summary_label)
        layout.addWidget(self.circle_reset_button)
        layout.addStretch(1)
        return tab

    def _on_circle_margin_changed(self, value: int) -> None:
        self.default_circle_margin = int(value)
        if self.circle_margin_slider.value() != value:
            self.circle_margin_slider.blockSignals(True)
            self.circle_margin_slider.setValue(value)
            self.circle_margin_slider.blockSignals(False)
        if self.circle_margin_spinbox.value() != value:
            self.circle_margin_spinbox.blockSignals(True)
            self.circle_margin_spinbox.setValue(value)
            self.circle_margin_spinbox.blockSignals(False)
        if self.mode_tabs.currentIndex() == 2:
            self.frame_viewer.set_margin_value(float(value))
        self._refresh_circle_ui()
        self._save_settings()
        self._refresh_output_ui()

    def _refresh_circle_ui(self) -> None:
        geometry = self.frame_viewer.circle_geometry()
        if geometry is None:
            self.circle_summary_label.setText("Draw a circle first.")
        else:
            center, base_radius, adjusted_radius = geometry
            self.circle_summary_label.setText(f"center=({center[0]:.1f}, {center[1]:.1f}) | base radius={base_radius:.1f}px | adjusted radius={adjusted_radius:.1f}px | bodyparts={len(self.bodyparts)}")
        self._refresh_output_ui()

    def save_circle_detection_csv(self) -> None:
        if self.csv_df is None or self.video_state is None:
            QMessageBox.warning(self, "Save", "Load a video and CSV first.")
            return
        geometry = self.frame_viewer.circle_geometry()
        if geometry is None:
            QMessageBox.warning(self, "Save", "Draw a circle first.")
            return
        center, _, adjusted_radius = geometry
        output = self._circle_output_path()
        if output is None:
            return
        try:
            df = build_circle_detection_dataframe(self.csv_df, self.bodyparts, center, adjusted_radius, self.video_state.width, self.video_state.height)
            output.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output, index=False)
        except Exception as exc:
            QMessageBox.warning(self, "Save Warning", f"Could not overwrite the CSV.\nIt may be open in another program.\n\n{output}\n\n{exc}")
            return
        self.statusBar().showMessage(f"Detection CSV saved: {output}")
