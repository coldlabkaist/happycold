import json
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from happycold_shared import (
    MASK_PALETTE,
    MaskRecord,
    adjust_mask_by_mode,
    build_occlusion_dataframe,
    fill_circle_from_diameter,
    fill_polygon,
    order_quad_points,
    paint_brush,
)


class OcclusionTabMixin:
    def _build_occlusion_tab(self) -> QWidget:
        tab = QWidget()
        tooltip = "Create named masks for occlusion detection. You can draw, erase, transform masks, and choose simple or perspective-aware geometric margin."
        tab.setToolTip(tooltip)
        layout = QVBoxLayout(tab)
        header_row = QHBoxLayout()
        header_label = QLabel("Object Mask Controls")
        self.occlusion_help_button = QPushButton("?")
        self.occlusion_help_button.setFixedWidth(30)
        self.occlusion_help_button.setToolTip("Open occlusion controls help.")
        header_row.addWidget(header_label)
        header_row.addStretch(1)
        header_row.addWidget(self.occlusion_help_button)
        info = QLabel(
            "Create named masks for occlusion detection."
        )
        info.setWordWrap(True)
        info.setToolTip(tooltip)

        self.mask_combo = QComboBox()
        self.mask_combo.setToolTip("Choose which named mask is currently active.")
        self.mask_name_button = QPushButton("Add Mask")
        self.mask_name_button.setToolTip("Create a new empty occlusion mask.")
        self.mask_rename_button = QPushButton("Rename Mask")
        self.mask_rename_button.setToolTip("Rename the selected mask.")
        self.mask_delete_button = QPushButton("Delete Mask")
        self.mask_delete_button.setToolTip("Delete the selected mask.")
        self.mask_clear_button = QPushButton("Clear Mask")
        self.mask_clear_button.setToolTip("Erase all painted pixels in the selected mask.")

        name_row = QHBoxLayout()
        name_row.addWidget(self.mask_name_button)
        name_row.addWidget(self.mask_rename_button)
        name_row.addWidget(self.mask_delete_button)
        name_row.addWidget(self.mask_clear_button)

        self.mask_shape_combo = QComboBox()
        self.mask_shape_combo.addItems(["Rectangle 4 Points", "Circle Drag", "Free Drawing"])
        self.mask_shape_combo.setToolTip("Choose how new mask strokes are created on the frame.")
        self.mask_draw_radio = QRadioButton("Draw")
        self.mask_transform_radio = QRadioButton("Transform")
        self.mask_draw_radio.setChecked(True)
        self.mask_draw_radio.setToolTip("Paint or erase pixels into the selected mask.")
        self.mask_transform_radio.setToolTip("Move or scale the selected mask instead of painting.")
        self.mask_mode_group = QButtonGroup(self)
        self.mask_mode_group.addButton(self.mask_draw_radio)
        self.mask_mode_group.addButton(self.mask_transform_radio)
        self.mask_add_radio = QRadioButton("Add")
        self.mask_erase_radio = QRadioButton("Erase")
        self.mask_add_radio.setChecked(True)
        self.mask_add_radio.setToolTip("New strokes add pixels to the mask.")
        self.mask_erase_radio.setToolTip("New strokes remove pixels from the mask.")
        self.mask_action_group = QButtonGroup(self)
        self.mask_action_group.addButton(self.mask_add_radio)
        self.mask_action_group.addButton(self.mask_erase_radio)
        self.mask_brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.mask_brush_slider.setRange(1, 60)
        self.mask_brush_slider.setValue(12)
        self.mask_brush_slider.setToolTip("Brush radius for free drawing mode.")
        self.mask_brush_spinbox = QSpinBox()
        self.mask_brush_spinbox.setRange(1, 60)
        self.mask_brush_spinbox.setSuffix(" px")
        self.mask_brush_spinbox.setFixedWidth(96)
        self.mask_brush_spinbox.setToolTip("Precise brush radius for free drawing mode.")
        self.mask_brush_label = QLabel("Brush Size")
        self.mask_brush_label.setToolTip("Brush radius used while free drawing.")
        self.mask_margin_simple_radio = QRadioButton("Simple")
        self.mask_margin_geometric_radio = QRadioButton("Geometric")
        self.mask_margin_simple_radio.setChecked(True)
        self.mask_margin_mode_group = QButtonGroup(self)
        self.mask_margin_mode_group.addButton(self.mask_margin_simple_radio)
        self.mask_margin_mode_group.addButton(self.mask_margin_geometric_radio)
        self.mask_margin_slider = QSlider(Qt.Orientation.Horizontal)
        self.mask_margin_slider.setRange(-300, 300)
        self.mask_margin_slider.setValue(0)
        self.mask_margin_spinbox = QSpinBox()
        self.mask_margin_spinbox.setRange(-300, 300)
        self.mask_margin_spinbox.setSuffix(" px")
        self.mask_margin_spinbox.setFixedWidth(96)
        self.mask_margin_label = QLabel("Mask Margin")
        self.occ_margin_points_label = QLabel("Geometric Points: 0 / 4")
        self.occ_margin_points_label.setWordWrap(True)
        self.occ_margin_set_button = QPushButton("Set Square")
        self.mask_summary_label = QLabel("Select or create a mask.")
        self.mask_summary_label.setWordWrap(True)
        self.mask_summary_label.setToolTip("Shows base mask size, adjusted mask size, current margin mode, and edit mode.")
        self.import_mask_button = QPushButton("Import Mask PNG")
        self.import_mask_button.setToolTip("Load a single saved mask PNG.")
        self.import_mask_folder_button = QPushButton("Import Mask Folder")
        self.import_mask_folder_button.setToolTip("Load every mask PNG and saved metadata from a folder.")
        self.export_masks_button = QPushButton("Export Masks PNG")
        self.export_masks_button.setToolTip("Save the current masks and their margin settings to disk.")
        brush_row = QHBoxLayout()
        brush_row.addWidget(self.mask_brush_slider, stretch=1)
        brush_row.addWidget(self.mask_brush_spinbox)
        margin_mode_row = QHBoxLayout()
        margin_mode_row.addWidget(self.mask_margin_simple_radio)
        margin_mode_row.addWidget(self.mask_margin_geometric_radio)
        occ_margin_points_row = QHBoxLayout()
        occ_margin_points_row.addWidget(self.occ_margin_set_button)
        margin_row = QHBoxLayout()
        margin_row.addWidget(self.mask_margin_slider, stretch=1)
        margin_row.addWidget(self.mask_margin_spinbox)

        form = QFormLayout()
        form.addRow("Shape Tool", self.mask_shape_combo)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.mask_draw_radio)
        mode_row.addWidget(self.mask_transform_radio)
        form.addRow("Mode", mode_row)
        action_row = QHBoxLayout()
        action_row.addWidget(self.mask_add_radio)
        action_row.addWidget(self.mask_erase_radio)
        form.addRow("Action", action_row)
        form.addRow(self.mask_brush_label, brush_row)
        form.addRow("Margin Mode", margin_mode_row)
        form.addRow(self.occ_margin_points_label, occ_margin_points_row)
        form.addRow(self.mask_margin_label, margin_row)

        mask_margin_tooltip = (
            "Simple: expand or shrink the mask directly in the current frame.\n"
            "Geometric: apply the same margin after 4-point perspective rectification, then project it back.\n"
            "Geometric mode needs four reference points set inside Occlusion."
        )
        self.mask_margin_label.setToolTip(mask_margin_tooltip)
        self.mask_margin_slider.setToolTip(mask_margin_tooltip)
        self.mask_margin_spinbox.setToolTip(mask_margin_tooltip)
        self.mask_margin_simple_radio.setToolTip(mask_margin_tooltip)
        self.mask_margin_geometric_radio.setToolTip(mask_margin_tooltip)
        self.occ_margin_points_label.setToolTip("Four perspective reference points used only by occlusion geometric margin.")
        self.occ_margin_set_button.setToolTip("Start picking four geometric reference points, or reset an existing square.")

        import_row = QHBoxLayout()
        import_row.addWidget(self.import_mask_button)
        import_row.addWidget(self.import_mask_folder_button)
        import_row.addWidget(self.export_masks_button)

        layout.addLayout(header_row)
        layout.addWidget(info)
        layout.addWidget(self.mask_combo)
        layout.addLayout(name_row)
        layout.addLayout(form)
        layout.addLayout(import_row)
        layout.addWidget(self.mask_summary_label)
        return tab

    def show_occlusion_controls_help(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Occlusion Controls")
        dialog.setModal(True)
        dialog.resize(520, 360)
        layout = QVBoxLayout(dialog)

        title = QLabel("Object Mask Controls")
        title.setStyleSheet("font-size: 15px; font-weight: 700;")
        text = QLabel(
            "Transform mode:\n"
            "- Left drag: move mask\n"
            "- Ctrl + left click / drag: erase with current brush size\n"
            "- [: scale down\n"
            "- ]: scale up\n"
            "- E: scale down\n"
            "- R: scale up\n"
            "- Ctrl + [: rotate -4 degrees\n"
            "- Ctrl + ]: rotate +4 degrees\n\n"
            "- Ctrl + E: rotate -4 degrees\n"
            "- Ctrl + R: rotate +4 degrees\n\n"
            "Mode and selection:\n"
            "- Press D: switch to Draw mode\n"
            "- Press T: switch to Transform mode\n"
            "- Press 1..0: select mask #1..#10 in the list\n"
            "- Double-click near a mask object: select that mask\n"
            "- Mask move is clamped to frame bounds (cannot move outside)\n\n"
            "Draw mode:\n"
            "- Hold Ctrl while drawing to erase (Rectangle, Circle, Free Drawing)\n\n"
            "Viewer:\n"
            "- Mouse wheel: zoom viewer\n"
            "- Right drag: pan viewer\n\n"
            "Tips:\n"
            "- This help is available from the ? button in Occlusion tab.\n"
            "- Press F1 to open this dialog while using Occlusion."
        )
        text.setWordWrap(True)
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(title)
        layout.addWidget(text, stretch=1)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        dialog.exec()

    def _sync_occlusion_mode(self) -> None:
        if self.mode_tabs.currentIndex() != 4:
            self.frame_viewer.set_occ_margin_pick_mode(False)
            return
        mapping = {0: "occ_rect", 1: "occ_circle", 2: "occ_free"}
        self.frame_viewer.set_mode(mapping.get(self.mask_shape_combo.currentIndex(), "occ_rect"))
        self.frame_viewer.set_margin_value(float(self.mask_margin_slider.value()))
        self.frame_viewer.set_occ_transform_mode(self.mask_transform_radio.isChecked())
        if not self.mask_margin_geometric_radio.isChecked():
            self.frame_viewer.set_occ_margin_pick_mode(False)
        if any(record.margin_mode == "geometric" for record in self.mask_records.values()):
            self.frame_viewer.set_mask_records(self.mask_records, self.selected_mask_name, refresh=True)
        self._sync_draw_mode()
        self._refresh_occ_margin_points_ui()
        self._refresh_mask_ui()
        self._refresh_output_ui()

    def _sync_draw_mode(self) -> None:
        self.frame_viewer.free_draw_add = self.mask_add_radio.isChecked()

    def reset_masks(self) -> None:
        self.mask_records.clear()
        self.selected_mask_name = None
        self.mask_combo.blockSignals(True)
        self.mask_combo.clear()
        self.mask_combo.blockSignals(False)
        self.mask_margin_slider.blockSignals(True)
        self.mask_margin_slider.setValue(self.default_mask_margin)
        self.mask_margin_slider.blockSignals(False)
        self.mask_margin_spinbox.blockSignals(True)
        self.mask_margin_spinbox.setValue(self.default_mask_margin)
        self.mask_margin_spinbox.blockSignals(False)
        self._set_margin_mode_controls(self.default_mask_margin_mode)
        self.frame_viewer.set_mask_records(self.mask_records, self.selected_mask_name)
        self._refresh_mask_ui()

    def _selected_mask(self) -> MaskRecord | None:
        if self.selected_mask_name is None:
            return None
        return self.mask_records.get(self.selected_mask_name)

    def _selected_margin_mode(self) -> str:
        return "geometric" if self.mask_margin_geometric_radio.isChecked() else "simple"

    def _set_margin_mode_controls(self, mode: str) -> None:
        use_geometric = mode == "geometric"
        self.mask_margin_simple_radio.blockSignals(True)
        self.mask_margin_simple_radio.setChecked(not use_geometric)
        self.mask_margin_simple_radio.blockSignals(False)
        self.mask_margin_geometric_radio.blockSignals(True)
        self.mask_margin_geometric_radio.setChecked(use_geometric)
        self.mask_margin_geometric_radio.blockSignals(False)

    def _next_available_mask_name(self, base_name: str) -> str:
        candidate = base_name.strip() or "mask"
        if candidate not in self.mask_records:
            return candidate
        suffix = 2
        while f"{candidate}_{suffix}" in self.mask_records:
            suffix += 1
        return f"{candidate}_{suffix}"

    def add_mask(self) -> None:
        if self.video_state is None:
            QMessageBox.information(self, "Mask", "Load a video first.")
            return
        name, ok = QInputDialog.getText(self, "New Mask", "Mask name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self.mask_records:
            QMessageBox.warning(self, "Mask", "A mask with this name already exists.")
            return
        color = MASK_PALETTE[len(self.mask_records) % len(MASK_PALETTE)]
        self.mask_records[name] = MaskRecord(
            name=name,
            color=color,
            mask=np.zeros((self.video_state.height, self.video_state.width), dtype=np.uint8),
            margin=self.default_mask_margin,
            margin_mode=self.default_mask_margin_mode,
        )
        self.selected_mask_name = name
        self._rebuild_mask_list()

    def rename_mask(self) -> None:
        current = self._selected_mask()
        if current is None:
            return
        name, ok = QInputDialog.getText(self, "Rename Mask", "Mask name:", text=current.name)
        if not ok or not name.strip():
            return
        name = name.strip()
        if name == current.name:
            return
        if name in self.mask_records:
            QMessageBox.warning(self, "Mask", "A mask with this name already exists.")
            return
        record = self.mask_records.pop(current.name)
        record.name = name
        self.mask_records[name] = record
        self.selected_mask_name = name
        self._rebuild_mask_list()

    def delete_mask(self) -> None:
        current = self._selected_mask()
        if current is None:
            return
        del self.mask_records[current.name]
        self.selected_mask_name = sorted(self.mask_records)[0] if self.mask_records else None
        self._rebuild_mask_list()

    def clear_selected_mask(self) -> None:
        current = self._selected_mask()
        if current is not None:
            current.mask.fill(0)
            self.frame_viewer.set_mask_records(self.mask_records, self.selected_mask_name, refresh=True)
            self._refresh_mask_ui()

    def _rebuild_mask_list(self) -> None:
        self.mask_combo.blockSignals(True)
        self.mask_combo.clear()
        for name in sorted(self.mask_records):
            self.mask_combo.addItem(name, name)
        if self.selected_mask_name is not None:
            index = self.mask_combo.findData(self.selected_mask_name)
            if index >= 0:
                self.mask_combo.setCurrentIndex(index)
            elif self.mask_combo.count() > 0:
                self.selected_mask_name = self.mask_combo.itemData(0)
                self.mask_combo.setCurrentIndex(0)
        elif self.mask_combo.count() > 0:
            self.selected_mask_name = self.mask_combo.itemData(0)
            self.mask_combo.setCurrentIndex(0)
        self.mask_combo.blockSignals(False)
        self.frame_viewer.set_mask_records(self.mask_records, self.selected_mask_name)
        self._refresh_mask_ui()

    def _on_mask_selection_changed(self, index: int) -> None:
        self.selected_mask_name = self.mask_combo.itemData(index) if index >= 0 else None
        selected = self._selected_mask()
        if selected is not None:
            self.mask_margin_slider.blockSignals(True)
            self.mask_margin_slider.setValue(int(selected.margin))
            self.mask_margin_slider.blockSignals(False)
            self.mask_margin_spinbox.blockSignals(True)
            self.mask_margin_spinbox.setValue(int(selected.margin))
            self.mask_margin_spinbox.blockSignals(False)
            self._set_margin_mode_controls(selected.margin_mode)
        else:
            self.mask_margin_slider.blockSignals(True)
            self.mask_margin_slider.setValue(self.default_mask_margin)
            self.mask_margin_slider.blockSignals(False)
            self.mask_margin_spinbox.blockSignals(True)
            self.mask_margin_spinbox.setValue(self.default_mask_margin)
            self.mask_margin_spinbox.blockSignals(False)
            self._set_margin_mode_controls(self.default_mask_margin_mode)
        self.frame_viewer.set_mask_records(self.mask_records, self.selected_mask_name)
        self._refresh_mask_ui()

    def _on_mask_margin_changed(self, value: int) -> None:
        self.default_mask_margin = int(value)
        if self.mask_margin_slider.value() != value:
            self.mask_margin_slider.blockSignals(True)
            self.mask_margin_slider.setValue(value)
            self.mask_margin_slider.blockSignals(False)
        if self.mask_margin_spinbox.value() != value:
            self.mask_margin_spinbox.blockSignals(True)
            self.mask_margin_spinbox.setValue(value)
            self.mask_margin_spinbox.blockSignals(False)
        current = self._selected_mask()
        if current is not None:
            current.margin = value
        if self.mode_tabs.currentIndex() == 4:
            self.frame_viewer.set_margin_value(float(value))
        self.frame_viewer.set_mask_records(self.mask_records, self.selected_mask_name, refresh=True)
        self._save_settings()
        self._refresh_mask_ui()

    def _on_mask_margin_mode_changed(self, checked: bool) -> None:
        if not checked:
            return
        mode = self._selected_margin_mode()
        self.default_mask_margin_mode = mode
        current = self._selected_mask()
        if current is not None:
            current.margin_mode = mode
        if mode != "geometric":
            self.frame_viewer.set_occ_margin_pick_mode(False)
        self.frame_viewer.set_mask_records(self.mask_records, self.selected_mask_name, refresh=True)
        self._save_settings()
        self._refresh_occ_margin_points_ui()
        self._refresh_mask_ui()

    def _on_mask_brush_changed(self, value: int) -> None:
        self.default_mask_brush = int(value)
        if self.mask_brush_slider.value() != value:
            self.mask_brush_slider.blockSignals(True)
            self.mask_brush_slider.setValue(value)
            self.mask_brush_slider.blockSignals(False)
        if self.mask_brush_spinbox.value() != value:
            self.mask_brush_spinbox.blockSignals(True)
            self.mask_brush_spinbox.setValue(value)
            self.mask_brush_spinbox.blockSignals(False)
        self._save_settings()

    def _refresh_mask_ui(self) -> None:
        current = self._selected_mask()
        if current is None:
            self.mask_summary_label.setText("Select or create a mask.")
        else:
            pixels = int(current.mask.sum())
            geometric_pending = current.margin_mode == "geometric" and len(self.frame_viewer.occ_margin_points) != 4
            if pixels == 0:
                adjusted_pixels = 0
            else:
                try:
                    adjusted_pixels = int(
                        adjust_mask_by_mode(current.mask, current.margin, current.margin_mode, self.frame_viewer.occ_margin_points).sum()
                    )
                except ValueError:
                    adjusted_pixels = pixels
            mode_text = "Transform" if self.mask_transform_radio.isChecked() else ("Add" if self.mask_add_radio.isChecked() else "Erase")
            margin_text = f"{current.margin_mode} ({current.margin:+d}px)"
            if geometric_pending:
                margin_text += " | geometric needs 4 square points"
            self.mask_summary_label.setText(
                f"{current.name} | base pixels={pixels} | adjusted pixels={adjusted_pixels} | margin={margin_text} | mode={mode_text}"
            )
        self._refresh_output_ui()

    def _refresh_occ_margin_points_ui(self) -> None:
        is_expert = self.mask_margin_geometric_radio.isChecked()
        self.occ_margin_points_label.setVisible(is_expert)
        self.occ_margin_set_button.setVisible(is_expert)
        if not is_expert:
            return
        self.occ_margin_points_label.setText("Geometric Points")
        count = len(self.frame_viewer.occ_margin_points)
        if count == 4:
            self.occ_margin_set_button.setText("Reset Square")
        elif self.frame_viewer.occ_margin_pick_mode:
            self.occ_margin_set_button.setText("Cancel Set Square")
        else:
            self.occ_margin_set_button.setText("Set Square")

    def _on_occ_margin_set_clicked(self) -> None:
        if not self.mask_margin_geometric_radio.isChecked():
            return
        if len(self.frame_viewer.occ_margin_points) == 4:
            self.frame_viewer.clear_occ_margin_points()
            self.frame_viewer.set_occ_margin_pick_mode(False)
        elif self.frame_viewer.occ_margin_pick_mode:
            self.frame_viewer.set_occ_margin_pick_mode(False)
        else:
            if self.frame_viewer.occ_margin_points:
                self.frame_viewer.clear_occ_margin_points()
            self.frame_viewer.set_occ_margin_pick_mode(True)
        self._refresh_occ_margin_points_ui()

    def _on_occ_margin_points_changed(self) -> None:
        if len(self.frame_viewer.occ_margin_points) == 4 and self.frame_viewer.occ_margin_pick_mode:
            self.frame_viewer.set_occ_margin_pick_mode(False)
        if any(record.margin_mode == "geometric" for record in self.mask_records.values()):
            self.frame_viewer.set_mask_records(self.mask_records, self.selected_mask_name, refresh=True)
        self._refresh_occ_margin_points_ui()
        self._refresh_mask_ui()

    def _finalize_mask_draw(self) -> None:
        current = self._selected_mask()
        if current is not None:
            self._last_mask_preview_refresh = 0.0
            self.frame_viewer.refresh_mask_record(current.name, include_margin=True)
        self._refresh_mask_ui()

    def _finalize_occ_transform(self) -> None:
        current = self._selected_mask()
        if current is not None:
            # Rebuild margin only once after drag/erase transform interaction ends.
            self.frame_viewer.refresh_mask_record(current.name, include_margin=True)
        self._last_transform_preview_refresh = 0.0
        self._refresh_mask_ui()

    def erase_occ_transform_point(self, point: tuple[float, float]) -> None:
        current = self._selected_mask()
        if current is None or not self.mask_transform_radio.isChecked():
            return
        brush_radius = max(1, int(self.mask_brush_slider.value()))
        paint_brush(current.mask, point, point, brush_radius, 0)
        self.frame_viewer.refresh_mask_record(current.name, include_margin=False)

    def erase_occ_transform_segment(self, payload: tuple[tuple[float, float], tuple[float, float]]) -> None:
        current = self._selected_mask()
        if current is None or not self.mask_transform_radio.isChecked():
            return
        start, end = payload
        brush_radius = max(1, int(self.mask_brush_slider.value()))
        paint_brush(current.mask, start, end, brush_radius, 0)
        now = time.monotonic()
        if now - getattr(self, "_last_mask_preview_refresh", 0.0) >= 0.03:
            self._last_mask_preview_refresh = now
            self.frame_viewer.refresh_mask_record(current.name, include_margin=False)

    def apply_occ_rect_mask(self, payload: object) -> None:
        current = self._selected_mask()
        if current is None or self.mask_transform_radio.isChecked():
            return
        add = self.mask_add_radio.isChecked()
        points: list[tuple[float, float]]
        if (
            isinstance(payload, tuple)
            and len(payload) == 2
            and isinstance(payload[0], list)
            and isinstance(payload[1], bool)
        ):
            points = payload[0]
            add = payload[1]
        elif isinstance(payload, list):
            points = payload
        else:
            return
        fill_polygon(current.mask, order_quad_points(points).tolist(), 1 if add else 0)
        self.frame_viewer.refresh_mask_record(current.name, include_margin=True)
        self._refresh_mask_ui()

    def apply_occ_circle_mask(self, payload: object) -> None:
        current = self._selected_mask()
        if current is None or self.mask_transform_radio.isChecked():
            return
        if not isinstance(payload, tuple) or len(payload) < 4:
            return
        _, _, start, end = payload[:4]
        add = self.mask_add_radio.isChecked()
        if len(payload) >= 5 and isinstance(payload[4], bool):
            add = payload[4]
        if start is None or end is None:
            return
        fill_circle_from_diameter(current.mask, start, end, 1 if add else 0)
        self.frame_viewer.refresh_mask_record(current.name, include_margin=True)
        self.frame_viewer.clear_occ_circle()
        self._refresh_mask_ui()

    def apply_occ_free_segment(self, payload: tuple[tuple[float, float], tuple[float, float], bool]) -> None:
        current = self._selected_mask()
        if current is None or self.mask_transform_radio.isChecked():
            return
        start, end, add = payload
        paint_brush(current.mask, start, end, self.mask_brush_slider.value(), 1 if add else 0)
        now = time.monotonic()
        if now - getattr(self, "_last_mask_preview_refresh", 0.0) >= 0.03:
            self._last_mask_preview_refresh = now
            self.frame_viewer.refresh_mask_record(current.name, include_margin=False)

    def import_mask_png(self) -> None:
        if self.video_state is None:
            QMessageBox.information(self, "Import Mask", "Load a video first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open Mask PNG", "", "PNG Files (*.png)")
        if path:
            self._import_mask_file(Path(path))

    def import_mask_folder(self) -> None:
        if self.video_state is None:
            QMessageBox.information(self, "Import Masks", "Load a video first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Select mask folder", str(self.current_folder))
        if not folder:
            return
        folder_path = Path(folder)
        manifest = folder_path / "masks_manifest.json"
        manifest_data = {}
        if manifest.exists():
            try:
                manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                manifest_data = {}
        for path in sorted(folder_path.glob("*.png")):
            self._import_mask_file(path, manifest_data.get(path.stem, {}))

    def _import_mask_file(self, path: Path, metadata: dict | None = None) -> None:
        if self.video_state is None:
            return
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        if img.shape != (self.video_state.height, self.video_state.width):
            img = cv2.resize(img, (self.video_state.width, self.video_state.height), interpolation=cv2.INTER_NEAREST)
        name = self._next_available_mask_name(path.stem)
        margin = int((metadata or {}).get("margin", 0))
        margin_mode = str((metadata or {}).get("margin_mode", "simple"))
        if margin_mode not in {"simple", "geometric"}:
            margin_mode = "simple"
        color_hex = (metadata or {}).get("color")
        color = QColor(color_hex) if color_hex else MASK_PALETTE[len(self.mask_records) % len(MASK_PALETTE)]
        self.mask_records[name] = MaskRecord(name=name, color=color, mask=(img > 0).astype(np.uint8), margin=margin, margin_mode=margin_mode)
        self.selected_mask_name = name
        self._rebuild_mask_list()

    def export_masks(self) -> None:
        if not self.mask_records:
            QMessageBox.information(self, "Export Masks", "No masks to export.")
            return
        folder = self._mask_export_folder()
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for name, record in self.mask_records.items():
            img = record.mask.astype(np.uint8) * 255
            ok = cv2.imwrite(str(folder / f"{name}.png"), img)
            if not ok:
                QMessageBox.warning(self, "Export Masks", f"Could not write mask image:\n{folder / f'{name}.png'}")
                return
            manifest[name] = {"margin": int(record.margin), "margin_mode": record.margin_mode, "color": record.color.name()}
        try:
            (folder / "masks_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Export Masks", f"Could not write manifest:\n{folder / 'masks_manifest.json'}\n\n{exc}")
            return
        self.statusBar().showMessage(f"Masks exported to {folder}")
        QMessageBox.information(self, "Export Masks", f"Masks exported to:\n{folder}")

    def save_occlusion_csv(self) -> None:
        if self.csv_df is None or self.video_state is None or not self.mask_records:
            QMessageBox.warning(self, "Save", "Load video/CSV and prepare masks first.")
            return
        if any(record.margin_mode == "geometric" for record in self.mask_records.values()) and len(self.frame_viewer.occ_margin_points) != 4:
            QMessageBox.warning(self, "Save", "Geometric margin mode needs four occlusion geometric points.")
            return
        output = self._occlusion_output_path()
        if output is None:
            return
        try:
            df = build_occlusion_dataframe(
                self.csv_df,
                self.bodyparts,
                list(self.mask_records.values()),
                self.video_state.width,
                self.video_state.height,
                self.frame_viewer.occ_margin_points,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output, index=False)
        except Exception as exc:
            QMessageBox.warning(self, "Save Warning", f"Could not overwrite the CSV.\nIt may be open in another program.\n\n{output}\n\n{exc}")
            return
        self.statusBar().showMessage(f"Occlusion CSV saved: {output}")
