from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from shared import build_circle_detection_dataframe, circle_mask_geometry
from ui_controls import NoWheelSpinBox


def build_circle_mask(
    width: int,
    height: int,
    center: tuple[float, float],
    radius: float,
) -> np.ndarray:
    mask = np.zeros((max(1, int(height)), max(1, int(width))), dtype=np.uint8)
    cv2.circle(
        mask,
        (int(round(center[0])), int(round(center[1]))),
        max(1, int(round(radius))),
        1,
        -1,
    )
    return mask


def infer_circle_geometry_from_mask(
    mask: np.ndarray,
) -> tuple[tuple[float, float], float] | None:
    if mask.ndim != 2:
        raise ValueError("Circle mask must be a 2D image.")
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return None
    (center_x, center_y), radius = cv2.minEnclosingCircle(points)
    return (float(center_x), float(center_y)), max(1.0, float(radius))


def infer_circular_mask_geometry(
    mask: np.ndarray,
    *,
    min_circularity: float = 0.82,
    min_fill_ratio: float = 0.82,
    min_iou: float = 0.78,
) -> tuple[tuple[float, float], float] | None:
    """Infer circle geometry only when the binary mask is circle-like enough.

    This stricter helper is used for cross-tab paste into the Circle tool. The
    looser import path stays unchanged so existing PNG imports remain tolerant.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2 or not np.any(binary):
        return None
    component_count, _labels = cv2.connectedComponents(binary)
    if component_count != 2:
        return None
    points = cv2.findNonZero(binary)
    if points is None:
        return None
    (center_x, center_y), radius = cv2.minEnclosingCircle(points)
    radius = max(1.0, float(radius))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0:
        return None
    area = float(np.count_nonzero(binary))
    circularity = 4.0 * np.pi * area / (perimeter * perimeter)
    ideal = build_circle_mask(binary.shape[1], binary.shape[0], (center_x, center_y), radius)
    ideal_area = float(np.count_nonzero(ideal))
    if ideal_area <= 0:
        return None
    intersection = float(np.count_nonzero((binary > 0) & (ideal > 0)))
    union = float(np.count_nonzero((binary > 0) | (ideal > 0)))
    fill_ratio = area / ideal_area
    iou = intersection / max(1.0, union)
    if circularity < min_circularity or fill_ratio < min_fill_ratio or iou < min_iou:
        return None
    return (float(center_x), float(center_y)), radius


def write_circle_mask_bundle(
    mask_path: Path,
    manifest_path: Path,
    *,
    width: int,
    height: int,
    center: tuple[float, float],
    base_radius: float,
    margin: int,
    source: str = "exact",
) -> None:
    adjusted_radius = max(1.0, float(base_radius) + int(margin))
    mask = build_circle_mask(width, height, center, adjusted_radius) * 255
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"Could not write {mask_path}")
    metadata = {
        "format": "happycold_circle_mask_v1",
        "metadata_version": 2,
        "width": int(width),
        "height": int(height),
        "center": [float(center[0]), float(center[1])],
        "base_radius": float(base_radius),
        "margin": int(margin),
        "adjusted_radius": adjusted_radius,
        "geometry": circle_mask_geometry(
            center,
            base_radius,
            adjusted_radius,
            source=source,
        ),
    }
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_circle_mask_bundle(
    mask_path: Path,
    *,
    width: int,
    height: int,
    return_source: bool = False,
) -> tuple[tuple[float, float], float, int] | tuple[tuple[float, float], float, int, str]:
    image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read mask file: {mask_path}")
    if image.shape != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    inferred = infer_circle_geometry_from_mask(image)
    if inferred is None:
        raise ValueError("The selected circle mask is empty.")

    manifest_path = mask_path.with_suffix(".json")
    metadata: dict = {}
    geometry_source = "inferred"
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata = loaded
        except Exception:
            metadata = {}

    try:
        center_values = metadata["center"]
        source_width = max(1.0, float(metadata.get("width", width)))
        source_height = max(1.0, float(metadata.get("height", height)))
        scale_x = width / source_width
        scale_y = height / source_height
        radius_scale = (scale_x + scale_y) / 2.0
        center = (
            float(center_values[0]) * scale_x,
            float(center_values[1]) * scale_y,
        )
        base_radius = max(1.0, float(metadata["base_radius"]) * radius_scale)
        margin = int(round(float(metadata.get("margin", 0)) * radius_scale))
        geometry_metadata = metadata.get("geometry")
        geometry_source = (
            str(geometry_metadata.get("source", "exact")).lower()
            if isinstance(geometry_metadata, dict)
            else "exact"
        )
        if geometry_source not in {"exact", "inferred"}:
            geometry_source = "inferred"
    except (KeyError, TypeError, ValueError, IndexError):
        center, base_radius = inferred
        margin = 0
        geometry_source = "inferred"
    if return_source:
        return center, base_radius, margin, geometry_source
    return center, base_radius, margin


class CircleTabMixin:
    def _build_circle_tab(self) -> QWidget:
        tab = QWidget()
        tooltip = (
            "Draw or import a circle, then adjust it before saving inside/outside results."
        )
        tab.setToolTip(tooltip)
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setObjectName("circleControlsScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(0)
        content = QWidget()
        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        info = QLabel(
            "Draw a diameter in the viewer or import a mask below, "
            "then adjust the circle and confirm the result."
        )
        info.setWordWrap(True)
        info.setToolTip(tooltip)
        layout.addWidget(info)

        self.circle_draw_radio = QRadioButton("Draw")
        self.circle_transform_radio = QRadioButton("Transform")
        self.circle_draw_radio.setChecked(True)
        self.circle_draw_radio.setToolTip("Drag a diameter to create or replace the circle.")
        self.circle_transform_radio.setToolTip(
            "Drag to move the circle, or use the shared scale shortcuts."
        )
        self.circle_edit_group = QButtonGroup(self)
        self.circle_edit_group.addButton(self.circle_draw_radio)
        self.circle_edit_group.addButton(self.circle_transform_radio)
        edit_row = QHBoxLayout()
        edit_row.addWidget(self.circle_draw_radio)
        edit_row.addWidget(self.circle_transform_radio)
        edit_row.addStretch(1)
        self.circle_reset_button = QPushButton("Reset Circle")
        self.circle_reset_button.setToolTip("Clear the current circle.")

        edit_group = QGroupBox("1. Draw or Transform")
        edit_form = QFormLayout(edit_group)
        edit_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        edit_form.addRow("Edit Mode", edit_row)
        edit_form.addRow("Circle Tool", self.circle_reset_button)
        layout.addWidget(edit_group)

        self.circle_margin_slider = QSlider(Qt.Orientation.Horizontal)
        self.circle_margin_slider.setRange(-300, 300)
        self.circle_margin_slider.setValue(0)
        self.circle_margin_slider.setToolTip("Expand or shrink the detection radius.")
        self.circle_margin_spinbox = NoWheelSpinBox()
        self.circle_margin_spinbox.setRange(-300, 300)
        self.circle_margin_spinbox.setSuffix(" px")
        self.circle_margin_spinbox.setFixedWidth(88)
        self.circle_margin_spinbox.setToolTip("Exact radius offset in pixels.")
        margin_row = QHBoxLayout()
        margin_row.addWidget(self.circle_margin_slider, stretch=1)
        margin_row.addWidget(self.circle_margin_spinbox)

        adjust_group = QGroupBox("2. Adjust Detection Area")
        adjust_form = QFormLayout(adjust_group)
        adjust_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        adjust_form.addRow("Detection Margin", margin_row)
        layout.addWidget(adjust_group)

        self.circle_summary_label = QLabel("Draw or import a circle first.")
        self.circle_summary_label.setWordWrap(True)
        self.circle_summary_label.setToolTip(
            "Shows center, base radius, adjusted radius, and edit mode."
        )
        review_group = QGroupBox("3. Review")
        review_layout = QVBoxLayout(review_group)
        review_layout.addWidget(self.circle_summary_label)
        layout.addWidget(review_group)
        layout.addStretch(1)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll, stretch=1)

        self.import_circle_mask_button = QPushButton("Import Circle Mask")
        self.import_circle_mask_button.setToolTip(
            "Load a circle PNG and optional matching metadata JSON."
        )
        self.export_circle_mask_button = QPushButton("Export Circle Mask")
        self.export_circle_mask_button.setToolTip(
            "Save the adjusted circle PNG and editable geometry JSON."
        )
        file_actions = QHBoxLayout()
        file_actions.setSpacing(6)
        file_actions.addWidget(self.import_circle_mask_button, stretch=1)
        file_actions.addWidget(self.export_circle_mask_button, stretch=1)
        outer_layout.addLayout(file_actions)
        return tab

    def _sync_circle_mode(self, *_args) -> None:
        if self.mode_tabs.currentIndex() != self.TAB_CIRCLE:
            return
        self.frame_viewer.set_mode("circle")
        self.frame_viewer.set_margin_value(float(self.circle_margin_slider.value()))
        self.frame_viewer.set_circle_transform_mode(
            self.circle_transform_radio.isChecked()
            and self.frame_viewer.circle_geometry() is not None
        )
        self._refresh_circle_ui()

    def scale_circle(self, scale_factor: float) -> None:
        if scale_factor <= 0 or self.frame_viewer.circle_geometry() is None:
            return
        start = self.frame_viewer.circle_start
        end = self.frame_viewer.circle_end
        if start is None or end is None:
            return
        center_x = (start[0] + end[0]) / 2.0
        center_y = (start[1] + end[1]) / 2.0
        next_start = (
            center_x + (start[0] - center_x) * scale_factor,
            center_y + (start[1] - center_y) * scale_factor,
        )
        next_end = (
            center_x + (end[0] - center_x) * scale_factor,
            center_y + (end[1] - center_y) * scale_factor,
        )
        if self.video_state is not None and not all(
            0.0 <= point[0] <= self.video_state.width - 1
            and 0.0 <= point[1] <= self.video_state.height - 1
            for point in (next_start, next_end)
        ):
            return
        if hasattr(self, "_push_circle_undo"):
            self._push_circle_undo("scale circle")
        self.frame_viewer.circle_start = next_start
        self.frame_viewer.circle_end = next_end
        self.frame_viewer.update()
        self.frame_viewer.circle_changed.emit()

    def _set_circle_margin_value(self, value: int) -> None:
        self.default_circle_margin = int(value)
        if self.circle_margin_slider.value() != value:
            self.circle_margin_slider.blockSignals(True)
            self.circle_margin_slider.setValue(value)
            self.circle_margin_slider.blockSignals(False)
        if self.circle_margin_spinbox.value() != value:
            self.circle_margin_spinbox.blockSignals(True)
            self.circle_margin_spinbox.setValue(value)
            self.circle_margin_spinbox.blockSignals(False)
        if self.mode_tabs.currentIndex() == self.TAB_CIRCLE:
            self.frame_viewer.set_margin_value(float(value))

    def _on_circle_margin_changed(self, value: int) -> None:
        value = int(value)
        if (
            value != int(getattr(self, "default_circle_margin", 0))
            and self.frame_viewer.circle_geometry() is not None
            and hasattr(self, "_push_circle_undo")
        ):
            self._push_circle_undo("adjust circle margin")
        self._set_circle_margin_value(value)
        self._refresh_circle_ui()
        self._save_settings()
        self._refresh_output_ui()

    def _refresh_circle_ui(self) -> None:
        geometry = self.frame_viewer.circle_geometry()
        has_circle = geometry is not None
        self.circle_transform_radio.setEnabled(has_circle)
        self.import_circle_mask_button.setEnabled(self.video_state is not None)
        self.export_circle_mask_button.setEnabled(
            self.video_state is not None and has_circle
        )
        if not has_circle and self.circle_transform_radio.isChecked():
            self.circle_draw_radio.setChecked(True)
        if geometry is None:
            self.circle_summary_label.setText("Draw or import a circle first.")
        else:
            center, base_radius, adjusted_radius = geometry
            edit_mode = "transform" if self.circle_transform_radio.isChecked() else "draw"
            self.circle_summary_label.setText(
                f"mode={edit_mode} | center=({center[0]:.1f}, {center[1]:.1f}) | "
                f"base radius={base_radius:.1f}px | adjusted radius={adjusted_radius:.1f}px | "
                f"bodyparts={len(self.bodyparts)}"
            )
        self._refresh_output_ui()

    def export_circle_mask(self) -> None:
        if self.video_state is None:
            QMessageBox.information(self, "Circle Mask", "Load a video first.")
            return
        geometry = self.frame_viewer.circle_geometry()
        if geometry is None:
            QMessageBox.information(self, "Circle Mask", "Draw or import a circle first.")
            return
        mask_path = self._circle_mask_output_path()
        manifest_path = self._circle_manifest_output_path()
        if mask_path is None or manifest_path is None:
            return
        center, base_radius, _adjusted_radius = geometry
        try:
            write_circle_mask_bundle(
                mask_path,
                manifest_path,
                width=self.video_state.width,
                height=self.video_state.height,
                center=center,
                base_radius=base_radius,
                margin=self.circle_margin_slider.value(),
                source=getattr(self.frame_viewer, "circle_geometry_source", "exact"),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Circle Mask", f"Could not export circle mask.\n\n{exc}")
            return
        self.statusBar().showMessage(f"Circle mask exported: {mask_path}")
        QMessageBox.information(self, "Circle Mask", f"Circle mask exported to:\n{mask_path}")

    def import_circle_mask(self) -> None:
        if self.video_state is None:
            QMessageBox.information(self, "Import Circle Mask", "Load a video first.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Circle Mask PNG",
            "",
            "PNG Files (*.png)",
        )
        if path:
            self._import_circle_mask_file(Path(path))

    def _import_circle_mask_file(self, path: Path) -> None:
        if self.video_state is None:
            return
        try:
            center, base_radius, margin, geometry_source = load_circle_mask_bundle(
                path,
                width=self.video_state.width,
                height=self.video_state.height,
                return_source=True,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Import Circle Mask", str(exc))
            return
        margin = max(
            self.circle_margin_slider.minimum(),
            min(self.circle_margin_slider.maximum(), margin),
        )
        if hasattr(self, "_push_circle_undo"):
            self._push_circle_undo("import circle mask")
        self._set_circle_margin_value(margin)
        self.frame_viewer.circle_start = (center[0] - base_radius, center[1])
        self.frame_viewer.circle_end = (center[0] + base_radius, center[1])
        self.frame_viewer.circle_geometry_source = geometry_source
        self.frame_viewer.circle_changed.emit()
        self.statusBar().showMessage(f"Imported circle mask: {path.name}")

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
            df = build_circle_detection_dataframe(
                self.csv_df,
                self.bodyparts,
                center,
                adjusted_radius,
                self.video_state.width,
                self.video_state.height,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output, index=False)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Save Warning",
                f"Could not overwrite the CSV.\nIt may be open in another program."
                f"\n\n{output}\n\n{exc}",
            )
            return
        self.statusBar().showMessage(f"Detection CSV saved: {output}")
