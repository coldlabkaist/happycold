import json
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
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

from shared import (
    MASK_PALETTE,
    MaskRecord,
    adjust_mask_by_mode,
    build_occlusion_dataframe,
    fill_circle_from_diameter,
    fill_polygon,
    order_quad_points,
    paint_brush,
    smooth_binary_mask_low,
)
from ui_controls import NoWheelComboBox, NoWheelSpinBox


class OcclusionTabMixin:
    def _build_occlusion_tab(self) -> QWidget:
        tab = QWidget()
        tooltip = (
            "Create named masks for occlusion detection, then draw, transform, and adjust them."
        )
        tab.setToolTip(tooltip)
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setObjectName("occlusionControlsScroll")
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
            "Add or import a mask, choose how to edit it, configure "
            "its margin, and check the summary before saving."
        )
        info.setWordWrap(True)
        info.setToolTip(tooltip)
        layout.addWidget(info)

        self.mask_combo = NoWheelComboBox()
        self.mask_combo.setMinimumWidth(0)
        self.mask_combo.setToolTip("Choose the mask that receives edits.")
        self.mask_name_button = QPushButton("Add Mask")
        self.mask_rename_button = QPushButton("Rename Mask")
        self.mask_delete_button = QPushButton("Delete Mask")
        self.mask_clear_button = QPushButton("Clear Mask")
        self.mask_name_button.setToolTip("Create a new empty mask.")
        self.mask_rename_button.setToolTip("Rename the selected mask.")
        self.mask_delete_button.setToolTip("Delete the selected mask.")
        self.mask_clear_button.setToolTip("Clear the selected mask only.")
        mask_actions = QGridLayout()
        mask_actions.setHorizontalSpacing(6)
        mask_actions.setVerticalSpacing(6)
        mask_actions.addWidget(self.mask_name_button, 0, 0)
        mask_actions.addWidget(self.mask_rename_button, 0, 1)
        mask_actions.addWidget(self.mask_delete_button, 1, 0)
        mask_actions.addWidget(self.mask_clear_button, 1, 1)

        target_group = QGroupBox("1. Create or Select Mask")
        target_layout = QVBoxLayout(target_group)
        target_form = QFormLayout()
        target_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        target_form.addRow("Current Mask", self.mask_combo)
        target_layout.addLayout(target_form)
        target_layout.addLayout(mask_actions)
        layout.addWidget(target_group)

        self.mask_shape_combo = NoWheelComboBox()
        self.mask_shape_combo.addItems(
            ["Rectangle 4 Points", "Circle Drag", "Free Drawing"]
        )
        self.mask_shape_combo.setMinimumWidth(0)
        self.mask_shape_combo.setToolTip("Choose how mask pixels are drawn in the viewer.")
        self.mask_draw_radio = QRadioButton("Draw")
        self.mask_transform_radio = QRadioButton("Transform")
        self.mask_draw_radio.setChecked(True)
        self.mask_draw_radio.setToolTip("Draw or erase mask pixels.")
        self.mask_transform_radio.setToolTip("Move, scale, or rotate the selected mask.")
        self.mask_mode_group = QButtonGroup(self)
        self.mask_mode_group.addButton(self.mask_draw_radio)
        self.mask_mode_group.addButton(self.mask_transform_radio)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.mask_draw_radio)
        mode_row.addWidget(self.mask_transform_radio)
        mode_row.addStretch(1)

        self.mask_add_radio = QRadioButton("Add")
        self.mask_erase_radio = QRadioButton("Erase")
        self.mask_add_radio.setChecked(True)
        self.mask_add_radio.setToolTip("New strokes add pixels.")
        self.mask_erase_radio.setToolTip("New strokes remove pixels.")
        self.mask_action_group = QButtonGroup(self)
        self.mask_action_group.addButton(self.mask_add_radio)
        self.mask_action_group.addButton(self.mask_erase_radio)
        action_row = QHBoxLayout()
        action_row.addWidget(self.mask_add_radio)
        action_row.addWidget(self.mask_erase_radio)
        action_row.addStretch(1)

        self.mask_brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.mask_brush_slider.setRange(1, 60)
        self.mask_brush_slider.setValue(12)
        self.mask_brush_slider.setToolTip("Brush radius for Free Drawing.")
        self.mask_brush_spinbox = NoWheelSpinBox()
        self.mask_brush_spinbox.setRange(1, 60)
        self.mask_brush_spinbox.setSuffix(" px")
        self.mask_brush_spinbox.setFixedWidth(88)
        self.mask_brush_spinbox.setToolTip("Exact brush radius.")
        self.mask_brush_label = QLabel("Brush Size")
        brush_row = QHBoxLayout()
        brush_row.addWidget(self.mask_brush_slider, stretch=1)
        brush_row.addWidget(self.mask_brush_spinbox)

        edit_group = QGroupBox("2. Draw or Transform")
        edit_form = QFormLayout(edit_group)
        edit_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        edit_form.addRow("Edit Mode", mode_row)
        edit_form.addRow("Mask Tool", self.mask_shape_combo)
        edit_form.addRow("Draw Action", action_row)
        edit_form.addRow(self.mask_brush_label, brush_row)
        layout.addWidget(edit_group)

        self.mask_margin_simple_radio = QRadioButton("Simple")
        self.mask_margin_geometric_radio = QRadioButton("Geometric")
        self.mask_margin_simple_radio.setChecked(True)
        self.mask_margin_mode_group = QButtonGroup(self)
        self.mask_margin_mode_group.addButton(self.mask_margin_simple_radio)
        self.mask_margin_mode_group.addButton(self.mask_margin_geometric_radio)
        margin_mode_row = QHBoxLayout()
        margin_mode_row.addWidget(self.mask_margin_simple_radio)
        margin_mode_row.addWidget(self.mask_margin_geometric_radio)
        margin_mode_row.addStretch(1)

        self.mask_margin_slider = QSlider(Qt.Orientation.Horizontal)
        self.mask_margin_slider.setRange(-300, 300)
        self.mask_margin_slider.setValue(0)
        self.mask_margin_spinbox = NoWheelSpinBox()
        self.mask_margin_spinbox.setRange(-300, 300)
        self.mask_margin_spinbox.setSuffix(" px")
        self.mask_margin_spinbox.setFixedWidth(88)
        self.mask_margin_label = QLabel("Mask Margin")
        margin_row = QHBoxLayout()
        margin_row.addWidget(self.mask_margin_slider, stretch=1)
        margin_row.addWidget(self.mask_margin_spinbox)

        self.occ_margin_points_label = QLabel("Geometric Points")
        self.occ_margin_points_label.setWordWrap(True)
        self.occ_margin_set_button = QPushButton("Set Square")
        points_row = QVBoxLayout()
        points_row.addWidget(self.occ_margin_set_button)

        margin_group = QGroupBox("3. Configure Margin")
        margin_form = QFormLayout(margin_group)
        margin_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        margin_form.addRow("Margin Mode", margin_mode_row)
        margin_form.addRow(self.occ_margin_points_label, points_row)
        margin_form.addRow(self.mask_margin_label, margin_row)
        layout.addWidget(margin_group)

        margin_tooltip = (
            "Simple changes the mask directly. Geometric applies the margin after "
            "four-point perspective rectification."
        )
        for control in (
            self.mask_margin_label,
            self.mask_margin_slider,
            self.mask_margin_spinbox,
            self.mask_margin_simple_radio,
            self.mask_margin_geometric_radio,
        ):
            control.setToolTip(margin_tooltip)
        self.occ_margin_points_label.setToolTip(
            "Four perspective points used by Geometric margin."
        )
        self.occ_margin_set_button.setToolTip(
            "Pick or reset four perspective reference points."
        )

        self.mask_summary_label = QLabel("Add or import a mask first.")
        self.mask_summary_label.setWordWrap(True)
        self.mask_summary_label.setToolTip(
            "Shows selected mask, pixel counts, margin, and edit mode."
        )
        review_group = QGroupBox("4. Review")
        review_layout = QVBoxLayout(review_group)
        review_layout.addWidget(self.mask_summary_label)
        layout.addWidget(review_group)
        layout.addStretch(1)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll, stretch=1)

        self.import_mask_button = QPushButton("Import Mask")
        self.import_mask_button.setToolTip("Load one mask PNG.")
        self.import_mask_folder_button = QPushButton("Import Mask Folder")
        self.import_mask_folder_button.setToolTip(
            "Load all mask PNGs and metadata from a folder."
        )
        self.export_masks_button = QPushButton("Export Masks")
        self.export_masks_button.setToolTip("Save masks and margin metadata.")
        file_actions = QGridLayout()
        file_actions.setHorizontalSpacing(6)
        file_actions.setVerticalSpacing(6)
        file_actions.addWidget(self.import_mask_button, 0, 0)
        file_actions.addWidget(self.import_mask_folder_button, 0, 1)
        file_actions.addWidget(self.export_masks_button, 1, 0, 1, 2)
        outer_layout.addLayout(file_actions)
        return tab

    def show_annotate_controls_help(self) -> None:
        dialog = QDialog(self)
        dialog.setObjectName("annotateControlsDialog")
        dialog.setWindowTitle("Annotate Shortcuts")
        dialog.setModal(True)
        dialog.resize(560, 440)
        dialog.setMinimumSize(440, 320)
        if self.styleSheet():
            dialog.setStyleSheet(self.styleSheet())

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QLabel("Annotate Shortcuts")
        title.setObjectName("annotateHelpTitle")
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 6, 0)
        content_layout.setSpacing(10)

        def add_shortcut_group(
            section_title: str,
            rows: list[tuple[str, str]],
        ) -> None:
            section = QGroupBox(section_title)
            section.setObjectName("annotateShortcutGroup")
            grid = QGridLayout(section)
            grid.setContentsMargins(14, 12, 14, 13)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(8)
            for row_index, (shortcut, description) in enumerate(rows):
                shortcut_label = QLabel(shortcut)
                shortcut_label.setProperty("shortcutKey", True)
                shortcut_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                shortcut_label.setMinimumWidth(120)
                description_label = QLabel(description)
                description_label.setWordWrap(True)
                description_label.setProperty("shortcutDescription", True)
                grid.addWidget(shortcut_label, row_index, 0)
                grid.addWidget(description_label, row_index, 1)
            grid.setColumnStretch(1, 1)
            content_layout.addWidget(section)

        add_shortcut_group(
            "Shared",
            [
                ("D", "Draw mode"),
                ("T", "Transform mode"),
                ("E  or  [", "Scale down"),
                ("R  or  ]", "Scale up"),
                ("Ctrl+E  or  Ctrl+[", "Rotate left (Chamber / Occlusion)"),
                ("Ctrl+R  or  Ctrl+]", "Rotate right (Chamber / Occlusion)"),
                ("Ctrl+Z", "Undo the latest mask content edit in the current tab"),
                ("F1", "Open this shortcut guide"),
            ],
        )
        add_shortcut_group(
            "Occlusion only",
            [
                ("1 - 9", "Select mask 1 - 9"),
                ("0", "Select mask 10"),
            ],
        )
        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)

        close_button = QPushButton("Close")
        close_button.setProperty("primary", True)
        close_button.setMinimumWidth(96)
        close_button.setDefault(True)
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        dialog.exec()

    def show_occlusion_controls_help(self) -> None:
        """Compatibility wrapper for older callers."""
        self.show_annotate_controls_help()

    def _sync_occlusion_mode(self) -> None:
        if self.mode_tabs.currentIndex() != self.TAB_OCCLUSION:
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
        self._invalidate_mask_transform_source()
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
            if hasattr(self, "_push_occlusion_undo"):
                self._push_occlusion_undo("clear occlusion mask")
            self._invalidate_mask_transform_source(current.name)
            current.mask.fill(0)
            self.frame_viewer.set_mask_records(self.mask_records, self.selected_mask_name, refresh=True)
            self._refresh_mask_ui()

    def _rebuild_mask_list(self) -> None:
        self._invalidate_mask_transform_source()
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
        self._invalidate_mask_transform_source()
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
        if self.mode_tabs.currentIndex() == self.TAB_OCCLUSION:
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
            current.mask = smooth_binary_mask_low(current.mask)
            self._invalidate_mask_transform_source(current.name)
            self._last_mask_preview_refresh = 0.0
            self.frame_viewer.refresh_mask_record(current.name, include_margin=True)
        self._occlusion_free_undo_open = False
        self._refresh_mask_ui()

    def _finalize_occ_transform(self) -> None:
        current = self._selected_mask()
        if current is not None:
            current.mask = smooth_binary_mask_low(current.mask)
            self._invalidate_mask_transform_source(current.name)
            # Rebuild margin only once after drag/erase transform interaction ends.
            self.frame_viewer.refresh_mask_record(current.name, include_margin=True)
        self._occlusion_transform_undo_open = False
        self._last_transform_preview_refresh = 0.0
        self._refresh_mask_ui()

    def erase_occ_transform_point(self, point: tuple[float, float]) -> None:
        current = self._selected_mask()
        if current is None or not self.mask_transform_radio.isChecked():
            return
        if not getattr(self, "_occlusion_transform_undo_open", False):
            if hasattr(self, "_push_occlusion_undo"):
                self._push_occlusion_undo("erase transformed occlusion mask")
            self._occlusion_transform_undo_open = True
        self._invalidate_mask_transform_source(current.name)
        brush_radius = max(1, int(self.mask_brush_slider.value()))
        paint_brush(current.mask, point, point, brush_radius, 0)
        self.frame_viewer.refresh_mask_record(current.name, include_margin=False)

    def erase_occ_transform_segment(self, payload: tuple[tuple[float, float], tuple[float, float]]) -> None:
        current = self._selected_mask()
        if current is None or not self.mask_transform_radio.isChecked():
            return
        if not getattr(self, "_occlusion_transform_undo_open", False):
            if hasattr(self, "_push_occlusion_undo"):
                self._push_occlusion_undo("erase transformed occlusion mask")
            self._occlusion_transform_undo_open = True
        self._invalidate_mask_transform_source(current.name)
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
        if hasattr(self, "_push_occlusion_undo"):
            self._push_occlusion_undo("draw occlusion rectangle")
        self._invalidate_mask_transform_source(current.name)
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
        if hasattr(self, "_push_occlusion_undo"):
            self._push_occlusion_undo("draw occlusion circle")
        self._invalidate_mask_transform_source(current.name)
        fill_circle_from_diameter(current.mask, start, end, 1 if add else 0)
        self.frame_viewer.refresh_mask_record(current.name, include_margin=True)
        self.frame_viewer.clear_occ_circle()
        self._refresh_mask_ui()

    def apply_occ_free_segment(self, payload: tuple[tuple[float, float], tuple[float, float], bool]) -> None:
        current = self._selected_mask()
        if current is None or self.mask_transform_radio.isChecked():
            return
        if not getattr(self, "_occlusion_free_undo_open", False):
            if hasattr(self, "_push_occlusion_undo"):
                self._push_occlusion_undo("brush occlusion mask")
            self._occlusion_free_undo_open = True
        start, end, add = payload
        self._invalidate_mask_transform_source(current.name)
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
