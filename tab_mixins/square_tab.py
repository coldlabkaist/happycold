from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from happycold_shared import build_normalized_dataframe


class SquareTabMixin:
    def _build_square_tab(self) -> QWidget:
        tab = QWidget()
        tooltip = "Choose four corners of the arena or region. These points define the perspective correction used for normalization and geometric mask margin."
        tab.setToolTip(tooltip)
        layout = QVBoxLayout(tab)
        info = QLabel("Click four corners on the frame. Use Preview to inspect normalized trajectories.")
        info.setWordWrap(True)
        info.setToolTip(tooltip)
        self.square_points_label = QLabel("Selected points: 0 / 4")
        self.square_points_label.setWordWrap(True)
        self.square_points_label.setToolTip("The current four-point selection in image coordinates.")
        self.square_reset_button = QPushButton("Reset Points")
        self.square_reset_button.setToolTip("Clear the current four-point selection and start over.")
        self.square_preview_button = QPushButton("Preview")
        self.square_preview_button.setEnabled(False)
        self.square_preview_button.setToolTip("Open a plot preview of the normalized trajectories using the selected four points.")
        row = QHBoxLayout()
        row.addWidget(self.square_reset_button)
        row.addWidget(self.square_preview_button)
        layout.addWidget(info)
        layout.addWidget(self.square_points_label)
        layout.addLayout(row)
        layout.addStretch(1)
        return tab

    def _refresh_square_ui(self) -> None:
        detail = " | ".join(f"{index + 1}: ({point[0]:.1f}, {point[1]:.1f})" for index, point in enumerate(self.frame_viewer.square_points))
        self.square_points_label.setText(f"Selected points: {len(self.frame_viewer.square_points)} / 4\n{detail if detail else '-'}")
        self.square_preview_button.setEnabled(self.csv_df is not None and len(self.frame_viewer.square_points) == 4)
        self._refresh_output_ui()

    def _on_square_points_changed(self) -> None:
        self._refresh_square_ui()

    def preview_square_normalization(self) -> None:
        if self.csv_df is None or self.video_state is None or len(self.frame_viewer.square_points) != 4:
            QMessageBox.warning(self, "Preview", "Load CSV and choose four square points first.")
            return
        normalized_df = build_normalized_dataframe(self.csv_df, self.bodyparts, self.frame_viewer.square_points, self.video_state.width, self.video_state.height)
        self._show_normalized_preview(normalized_df)

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
