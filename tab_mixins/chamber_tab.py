import json
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
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
    QVBoxLayout,
    QWidget,
)

from shared import MASK_PALETTE, MaskTransformSource, RoomRecord, build_chamber_mark_dataframe, fill_circle_from_diameter, fill_polygon, order_quad_points
from ui_controls import NoWheelComboBox


class ChamberTabMixin:
    def _build_chamber_tab(self) -> QWidget:
        tab = QWidget()
        tooltip = (
            "Define a chamber first, then create named rooms inside it. Rooms cannot overlap "
            "and only pixels inside the chamber are accepted."
        )
        tab.setToolTip(tooltip)
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setObjectName("chamberControlsScroll")
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
            "Start with Edit Chamber. After drawing the chamber, "
            "add a room and switch the target to Edit Room."
        )
        info.setWordWrap(True)
        info.setToolTip(tooltip)
        layout.addWidget(info)

        self.chamber_boundary_status_label = QLabel("Current: Not set")
        self.chamber_boundary_status_label.setWordWrap(True)
        self.chamber_boundary_status_label.setProperty("muted", True)
        self.chamber_boundary_status_label.setToolTip(
            "Shows whether the chamber boundary is unset, full frame, or custom."
        )
        self.chamber_full_frame_button = QPushButton("Use Full Frame")
        self.chamber_full_frame_button.setToolTip(
            "Set the chamber boundary to the full current video frame."
        )
        self.chamber_reset_button = QPushButton("Clear Chamber && Rooms")
        self.chamber_reset_button.setToolTip("Clear the chamber and every room.")
        boundary_actions = QGridLayout()
        boundary_actions.setHorizontalSpacing(6)
        boundary_actions.setVerticalSpacing(6)
        boundary_actions.addWidget(self.chamber_full_frame_button, 0, 0)
        boundary_actions.addWidget(self.chamber_reset_button, 0, 1)

        boundary_group = QGroupBox("1. Set Chamber Boundary")
        boundary_layout = QVBoxLayout(boundary_group)
        boundary_layout.addWidget(self.chamber_boundary_status_label)
        boundary_layout.addLayout(boundary_actions)
        layout.addWidget(boundary_group)

        self.chamber_edit_chamber_radio = QRadioButton("Edit Chamber")
        self.chamber_edit_room_radio = QRadioButton("Edit Room")
        self.chamber_edit_chamber_radio.setChecked(True)
        self.chamber_edit_chamber_radio.setToolTip(
            "Add or transform shapes in the overall chamber area."
        )
        self.chamber_edit_room_radio.setToolTip(
            "Add or transform shapes in the selected room."
        )
        self.chamber_edit_mode_group = QButtonGroup(self)
        self.chamber_edit_mode_group.addButton(self.chamber_edit_chamber_radio)
        self.chamber_edit_mode_group.addButton(self.chamber_edit_room_radio)
        target_row = QHBoxLayout()
        target_row.addWidget(self.chamber_edit_chamber_radio)
        target_row.addWidget(self.chamber_edit_room_radio)
        target_row.addStretch(1)

        self.room_combo = NoWheelComboBox()
        self.room_combo.setMinimumWidth(0)
        self.room_combo.setToolTip("Choose the room that receives edits.")
        self.room_add_button = QPushButton("Add Room")
        self.room_rename_button = QPushButton("Rename Room")
        self.room_delete_button = QPushButton("Delete Room")
        self.room_clear_button = QPushButton("Clear Room")
        self.room_add_button.setToolTip("Create a new named room.")
        self.room_rename_button.setToolTip("Rename the selected room.")
        self.room_delete_button.setToolTip("Delete the selected room.")
        self.room_clear_button.setToolTip("Clear the selected room only.")
        room_actions = QGridLayout()
        room_actions.setHorizontalSpacing(6)
        room_actions.setVerticalSpacing(6)
        room_actions.addWidget(self.room_add_button, 0, 0)
        room_actions.addWidget(self.room_rename_button, 0, 1)
        room_actions.addWidget(self.room_delete_button, 1, 0)
        room_actions.addWidget(self.room_clear_button, 1, 1)

        target_group = QGroupBox("2. Choose Target")
        target_layout = QVBoxLayout(target_group)
        target_form = QFormLayout()
        target_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        target_form.addRow("Edit Target", target_row)
        target_form.addRow("Current Room", self.room_combo)
        target_layout.addLayout(target_form)
        target_layout.addLayout(room_actions)
        layout.addWidget(target_group)

        self.chamber_draw_radio = QRadioButton("Draw")
        self.chamber_transform_radio = QRadioButton("Transform")
        self.chamber_draw_radio.setChecked(True)
        self.chamber_draw_radio.setToolTip("Add shapes to the selected target.")
        self.chamber_transform_radio.setToolTip(
            "Move, scale, or rotate the selected target."
        )
        self.chamber_operation_group = QButtonGroup(self)
        self.chamber_operation_group.addButton(self.chamber_draw_radio)
        self.chamber_operation_group.addButton(self.chamber_transform_radio)
        operation_row = QHBoxLayout()
        operation_row.addWidget(self.chamber_draw_radio)
        operation_row.addWidget(self.chamber_transform_radio)
        operation_row.addStretch(1)

        self.chamber_shape_combo = NoWheelComboBox()
        self.chamber_shape_combo.addItems(["Rectangle 4 Points", "Circle Drag"])
        self.chamber_shape_combo.setMinimumWidth(0)
        self.chamber_shape_combo.setToolTip("Choose the next shape drawn in the viewer.")
        tool_layout = QVBoxLayout()
        tool_layout.addWidget(self.chamber_shape_combo)

        edit_group = QGroupBox("3. Draw or Transform")
        edit_form = QFormLayout(edit_group)
        edit_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        edit_form.addRow("Edit Mode", operation_row)
        edit_form.addRow("Shape Tool", tool_layout)
        layout.addWidget(edit_group)

        self.chamber_summary_label = QLabel("Define the chamber area first.")
        self.chamber_summary_label.setWordWrap(True)
        self.chamber_summary_label.setToolTip(
            "Shows chamber, room, and current target status."
        )
        review_group = QGroupBox("4. Review")
        review_layout = QVBoxLayout(review_group)
        review_layout.addWidget(self.chamber_summary_label)
        layout.addWidget(review_group)
        layout.addStretch(1)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll, stretch=1)

        self.import_chamber_mask_button = QPushButton("Import Chamber Mask")
        self.import_chamber_mask_button.setToolTip(
            "Load a chamber mask PNG and matching metadata JSON."
        )
        self.export_chamber_mask_button = QPushButton("Export Chamber Mask")
        self.export_chamber_mask_button.setToolTip(
            "Save a color-coded mask PNG and metadata JSON."
        )
        file_actions = QHBoxLayout()
        file_actions.setSpacing(6)
        file_actions.addWidget(self.import_chamber_mask_button, stretch=1)
        file_actions.addWidget(self.export_chamber_mask_button, stretch=1)
        outer_layout.addLayout(file_actions)

        self._chamber_transform_source: MaskTransformSource | None = None
        self._chamber_transform_source_key: str | None = None
        self._chamber_transform_angle = 0.0
        self._chamber_transform_scale = 1.0
        return tab

    def _full_frame_chamber_mask(self) -> np.ndarray | None:
        if self.video_state is None:
            return None
        return np.ones(
            (self.video_state.height, self.video_state.width),
            dtype=np.uint8,
        )

    def _is_full_frame_chamber_mask(self) -> bool:
        if self.video_state is None or self.chamber_mask is None:
            return False
        expected_shape = (self.video_state.height, self.video_state.width)
        return self.chamber_mask.shape == expected_shape and bool(np.all(self.chamber_mask > 0))

    def _resolved_chamber_boundary_mode(self) -> str:
        if self.chamber_mask is None or not np.any(self.chamber_mask):
            return "unset"
        if getattr(self, "chamber_boundary_mode", "custom") == "full_frame" and self._is_full_frame_chamber_mask():
            return "full_frame"
        return "custom"

    def _set_chamber_boundary_mode(self, mode: str) -> None:
        self.chamber_boundary_mode = mode if mode in {"unset", "full_frame", "custom"} else "custom"

    def _mark_chamber_boundary_custom(self) -> None:
        self._set_chamber_boundary_mode(
            "unset"
            if self.chamber_mask is None or not np.any(self.chamber_mask)
            else "custom"
        )

    def _commit_effective_room_masks(self) -> None:
        if self.video_state is None or not self.room_records:
            return
        for effective in self._effective_room_records():
            room = self.room_records.get(effective.name)
            if room is not None:
                room.mask = effective.mask.copy().astype(np.uint8)

    def set_chamber_to_full_frame(self, _checked: bool | None = None, *, show_message: bool = True) -> None:
        mask = self._full_frame_chamber_mask()
        if mask is None:
            if show_message:
                QMessageBox.information(self, "Chamber", "Load a video first.")
            return
        if self.chamber_mask is not None and np.any(self.chamber_mask):
            self._commit_effective_room_masks()
        self._invalidate_chamber_transform_source()
        self.chamber_mask = mask
        self._set_chamber_boundary_mode("full_frame")
        self.chamber_edit_chamber_radio.setChecked(True)
        self.frame_viewer.clear_chamber_rect_points()
        self.frame_viewer.clear_chamber_circle()
        self._refresh_chamber_viewer(refresh=True)
        self._sync_chamber_mode()
        if show_message:
            self.statusBar().showMessage("Chamber boundary set to full frame.", 4000)

    def reset_chamber_rooms_for_full_frame_video(self) -> None:
        mask = self._full_frame_chamber_mask()
        if mask is None:
            self.reset_chamber()
            return
        self._invalidate_chamber_transform_source()
        self.chamber_mask = mask
        self._set_chamber_boundary_mode("full_frame")
        self.room_records.clear()
        self.selected_room_name = None
        self.frame_viewer.clear_chamber_rect_points()
        self.frame_viewer.clear_chamber_circle()
        self.room_combo.blockSignals(True)
        self.room_combo.clear()
        self.room_combo.blockSignals(False)
        self._refresh_chamber_viewer(refresh=True)
        self._refresh_chamber_ui()

    def _effective_room_records(self) -> list[RoomRecord]:
        if self.video_state is None:
            return []
        shape = (self.video_state.height, self.video_state.width)
        chamber = (
            np.zeros(shape, dtype=bool)
            if self.chamber_mask is None
            else self.chamber_mask.astype(bool)
        )
        selected_name = (
            self.selected_room_name
            if self.chamber_edit_room_radio.isChecked()
            else None
        )
        ordered_names = [name for name in self.room_records if name != selected_name]
        if selected_name in self.room_records:
            ordered_names.append(selected_name)
        occupied = np.zeros(shape, dtype=bool)
        effective_masks: dict[str, np.ndarray] = {}
        for name in ordered_names:
            room = self.room_records[name]
            allowed = room.mask.astype(bool) & chamber & ~occupied
            effective_masks[name] = allowed.astype(np.uint8)
            occupied |= allowed
        return [
            RoomRecord(
                name=room.name,
                color=room.color,
                mask=effective_masks.get(room.name, np.zeros(shape, dtype=np.uint8)),
            )
            for room in self.room_records.values()
        ]

    def _effective_room_records_dict(self) -> dict[str, RoomRecord]:
        return {room.name: room for room in self._effective_room_records()}

    def _refresh_chamber_viewer(self, refresh: bool = True) -> None:
        self.frame_viewer.set_chamber_records(
            self.chamber_mask,
            self._effective_room_records_dict(),
            self.selected_room_name,
            refresh=refresh,
        )

    def _selected_chamber_layer(self) -> tuple[str, np.ndarray] | None:
        if self.chamber_edit_chamber_radio.isChecked():
            if self.chamber_mask is None or not np.any(self.chamber_mask):
                return None
            return "chamber", self.chamber_mask
        room = self._selected_room()
        if room is None or not np.any(room.mask):
            return None
        return f"room:{room.name}", room.mask

    def _set_selected_chamber_layer_mask(self, mask: np.ndarray) -> None:
        normalized = (mask > 0).astype(np.uint8)
        if self.chamber_edit_chamber_radio.isChecked():
            self.chamber_mask = normalized
            self._mark_chamber_boundary_custom()
        else:
            room = self._selected_room()
            if room is None:
                return
            room.mask = normalized
        self._refresh_chamber_viewer(refresh=True)
        if self.mode_tabs.currentIndex() == self.TAB_CHAMBER:
            self._sync_chamber_mode()
        else:
            self._refresh_chamber_ui()

    def _invalidate_chamber_transform_source(self) -> None:
        self._chamber_transform_source = None
        self._chamber_transform_source_key = None
        self._chamber_transform_angle = 0.0
        self._chamber_transform_scale = 1.0

    def _ensure_chamber_transform_source(self) -> bool:
        selected = self._selected_chamber_layer()
        if selected is None:
            return False
        key, mask = selected
        if (
            self._chamber_transform_source is not None
            and self._chamber_transform_source_key == key
            and self._chamber_transform_source.mask.shape == mask.shape
        ):
            return True
        try:
            source = MaskTransformSource.from_mask(mask)
        except ValueError:
            self._invalidate_chamber_transform_source()
            return False
        self._chamber_transform_source = source
        self._chamber_transform_source_key = key
        self._chamber_transform_angle = 0.0
        self._chamber_transform_scale = 1.0
        return True

    def _apply_chamber_affine_transform(
        self,
        *,
        angle_delta: float = 0.0,
        scale_multiplier: float = 1.0,
    ) -> None:
        if scale_multiplier <= 0 or not self._ensure_chamber_transform_source():
            return
        source = self._chamber_transform_source
        if source is None:
            return
        next_angle = (self._chamber_transform_angle + float(angle_delta)) % 360.0
        next_scale = self._chamber_transform_scale * float(scale_multiplier)
        if next_scale < 0.1 or next_scale > 10.0:
            return
        transformed = source.render(next_angle, next_scale)
        if not np.any(transformed):
            return
        self._chamber_transform_angle = next_angle
        self._chamber_transform_scale = next_scale
        self._set_selected_chamber_layer_mask(transformed)

    def translate_selected_chamber_layer(self, shift: tuple[int, int]) -> None:
        selected = self._selected_chamber_layer()
        if selected is None:
            return
        _key, mask = selected
        dx, dy = self._clamp_mask_shift(mask, *shift)
        if dx == 0 and dy == 0:
            return
        translated = np.zeros_like(mask)
        src_x0 = max(0, -dx)
        src_x1 = mask.shape[1] - max(0, dx)
        src_y0 = max(0, -dy)
        src_y1 = mask.shape[0] - max(0, dy)
        dst_x0 = max(0, dx)
        dst_y0 = max(0, dy)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        translated[dst_y0:dst_y1, dst_x0:dst_x1] = mask[
            src_y0:src_y1, src_x0:src_x1
        ]
        self._invalidate_chamber_transform_source()
        self._set_selected_chamber_layer_mask(translated)

    def scale_selected_chamber_layer(self, scale_factor: float) -> None:
        self._apply_chamber_affine_transform(scale_multiplier=scale_factor)

    def rotate_selected_chamber_layer(self, angle_degrees: float) -> None:
        self._apply_chamber_affine_transform(angle_delta=angle_degrees)

    def _on_chamber_target_or_mode_changed(self, *_args) -> None:
        self._invalidate_chamber_transform_source()
        self._refresh_chamber_viewer(refresh=True)
        self._sync_chamber_mode()

    def _selected_room(self) -> RoomRecord | None:
        if self.selected_room_name is None:
            return None
        return self.room_records.get(self.selected_room_name)

    def _next_available_room_name(self, base_name: str) -> str:
        candidate = base_name.strip() or "room"
        if candidate not in self.room_records:
            return candidate
        suffix = 2
        while f"{candidate}_{suffix}" in self.room_records:
            suffix += 1
        return f"{candidate}_{suffix}"

    def _selected_chamber_draw_mode(self) -> str:
        return "chamber_circle" if self.chamber_shape_combo.currentIndex() == 1 else "chamber_rect"

    def _sync_chamber_mode(self) -> None:
        if self.mode_tabs.currentIndex() != self.TAB_CHAMBER:
            return
        if self.chamber_shape_combo.currentIndex() == 0:
            self.frame_viewer.clear_chamber_circle()
        else:
            self.frame_viewer.clear_chamber_rect_points()
        self.frame_viewer.set_mode(self._selected_chamber_draw_mode())
        self.frame_viewer.set_margin_value(0.0)
        selected_layer = self._selected_chamber_layer()
        self.frame_viewer.set_chamber_transform_mode(
            self.chamber_transform_radio.isChecked() and selected_layer is not None,
            None if selected_layer is None else selected_layer[0],
            None if selected_layer is None else selected_layer[1],
        )
        self._refresh_chamber_ui()

    def _rebuild_room_list(self) -> None:
        self._invalidate_chamber_transform_source()
        self.room_combo.blockSignals(True)
        self.room_combo.clear()
        for name in sorted(self.room_records):
            self.room_combo.addItem(name, name)
        if self.selected_room_name is not None:
            index = self.room_combo.findData(self.selected_room_name)
            if index >= 0:
                self.room_combo.setCurrentIndex(index)
            elif self.room_combo.count() > 0:
                self.selected_room_name = self.room_combo.itemData(0)
                self.room_combo.setCurrentIndex(0)
        elif self.room_combo.count() > 0:
            self.selected_room_name = self.room_combo.itemData(0)
            self.room_combo.setCurrentIndex(0)
        self.room_combo.blockSignals(False)
        self._refresh_chamber_viewer(refresh=True)
        self._refresh_chamber_ui()

    def _on_room_selection_changed(self, index: int) -> None:
        self._invalidate_chamber_transform_source()
        self.selected_room_name = self.room_combo.itemData(index) if index >= 0 else None
        self._refresh_chamber_viewer(refresh=True)
        self._refresh_chamber_ui()

    def add_room(self) -> None:
        if self.video_state is None:
            QMessageBox.information(self, "Room", "Load a video first.")
            return
        name, ok = QInputDialog.getText(self, "Add Room", "Room name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self.room_records:
            QMessageBox.warning(self, "Room", "A room with this name already exists.")
            return
        color = MASK_PALETTE[len(self.room_records) % len(MASK_PALETTE)]
        self.room_records[name] = RoomRecord(name=name, color=color, mask=np.zeros((self.video_state.height, self.video_state.width), dtype=np.uint8))
        self.selected_room_name = name
        self.chamber_edit_room_radio.setChecked(True)
        self._rebuild_room_list()

    def rename_room(self) -> None:
        current = self._selected_room()
        if current is None:
            return
        name, ok = QInputDialog.getText(self, "Rename Room", "Room name:", text=current.name)
        if not ok or not name.strip():
            return
        name = name.strip()
        if name == current.name:
            return
        if name in self.room_records:
            QMessageBox.warning(self, "Room", "A room with this name already exists.")
            return
        record = self.room_records.pop(current.name)
        record.name = name
        self.room_records[name] = record
        self.selected_room_name = name
        self._rebuild_room_list()

    def delete_room(self) -> None:
        current = self._selected_room()
        if current is None:
            return
        del self.room_records[current.name]
        self.selected_room_name = sorted(self.room_records)[0] if self.room_records else None
        self._rebuild_room_list()

    def clear_selected_room(self) -> None:
        current = self._selected_room()
        if current is None:
            return
        self._invalidate_chamber_transform_source()
        current.mask.fill(0)
        self._refresh_chamber_viewer(refresh=True)
        self._refresh_chamber_ui()

    def reset_chamber(self) -> None:
        self._invalidate_chamber_transform_source()
        self.chamber_mask = None
        self._set_chamber_boundary_mode("unset")
        self.room_records.clear()
        self.selected_room_name = None
        self.frame_viewer.clear_chamber_rect_points()
        self.frame_viewer.clear_chamber_circle()
        self.room_combo.blockSignals(True)
        self.room_combo.clear()
        self.room_combo.blockSignals(False)
        self._refresh_chamber_viewer(refresh=True)
        self._refresh_chamber_ui()

    def _occupied_room_mask(self, exclude_name: str | None = None) -> np.ndarray | None:
        if self.video_state is None:
            return None
        occupied = np.zeros((self.video_state.height, self.video_state.width), dtype=bool)
        for room in self._effective_room_records():
            if exclude_name is not None and room.name == exclude_name:
                continue
            occupied |= room.mask.astype(bool)
        return occupied.astype(np.uint8)

    def _apply_chamber_shape_mask(self, shape_mask: np.ndarray) -> None:
        if self.video_state is None:
            return
        self._invalidate_chamber_transform_source()
        if self.chamber_edit_chamber_radio.isChecked():
            if self.chamber_mask is None:
                self.chamber_mask = np.zeros((self.video_state.height, self.video_state.width), dtype=np.uint8)
            self.chamber_mask = np.logical_or(self.chamber_mask.astype(bool), shape_mask.astype(bool)).astype(np.uint8)
            self._mark_chamber_boundary_custom()
        else:
            current = self._selected_room()
            if current is None:
                QMessageBox.information(self, "Room", "Add or select a room first.")
                return
            if self.chamber_mask is None or not np.any(self.chamber_mask):
                QMessageBox.information(self, "Room", "Define the chamber area before assigning rooms.")
                return
            blocked = self._occupied_room_mask(exclude_name=current.name)
            allowed = shape_mask.astype(bool) & self.chamber_mask.astype(bool)
            if blocked is not None:
                allowed &= ~blocked.astype(bool)
            current.mask = np.logical_or(current.mask.astype(bool), allowed).astype(np.uint8)
        self._refresh_chamber_viewer(refresh=True)
        self._refresh_chamber_ui()

    def apply_chamber_rect(self, points: list[tuple[float, float]]) -> None:
        if self.video_state is None:
            return
        shape_mask = np.zeros((self.video_state.height, self.video_state.width), dtype=np.uint8)
        fill_polygon(shape_mask, order_quad_points(points).tolist(), 1)
        self._apply_chamber_shape_mask(shape_mask)

    def apply_chamber_circle(self, payload: tuple[tuple[float, float], float, tuple[float, float], tuple[float, float]]) -> None:
        if self.video_state is None:
            return
        _, _, start, end = payload
        if start is None or end is None:
            return
        shape_mask = np.zeros((self.video_state.height, self.video_state.width), dtype=np.uint8)
        fill_circle_from_diameter(shape_mask, start, end, 1)
        self._apply_chamber_shape_mask(shape_mask)
        self.frame_viewer.clear_chamber_circle()

    def _refresh_chamber_ui(self) -> None:
        if not self.room_records and self.chamber_edit_room_radio.isChecked():
            self.chamber_edit_chamber_radio.setChecked(True)
            return
        chamber_pixels = int(self.chamber_mask.sum()) if self.chamber_mask is not None else 0
        effective_rooms = self._effective_room_records_dict()
        occupied_pixels = int(sum(int(room.mask.sum()) for room in effective_rooms.values()))
        free_pixels = max(0, chamber_pixels - occupied_pixels)
        current = self._selected_room()
        current_effective = effective_rooms.get(current.name) if current is not None else None
        current_name = current.name if current is not None else "-"
        current_pixels = int(current.mask.sum()) if current is not None else 0
        effective_pixels = int(current_effective.mask.sum()) if current_effective is not None else 0
        target_text = "chamber" if self.chamber_edit_chamber_radio.isChecked() else "room"
        operation_text = "transform" if self.chamber_transform_radio.isChecked() else "draw"
        self.chamber_summary_label.setText(
            f"target={target_text} | mode={operation_text} | chamber={chamber_pixels} px | "
            f"occupied={occupied_pixels} px | free={free_pixels} px | "
            f"current room={current_name} ({effective_pixels}/{current_pixels} effective/raw px) | "
            f"rooms={len(self.room_records)}"
        )
        has_video = self.video_state is not None
        has_chamber = self.chamber_mask is not None and bool(np.any(self.chamber_mask))
        boundary_mode = self._resolved_chamber_boundary_mode()
        if not has_video:
            boundary_text = "Current: load a video first"
        elif boundary_mode == "full_frame":
            boundary_text = f"Current: Full Frame ({self.video_state.width}x{self.video_state.height})"
        elif boundary_mode == "custom":
            boundary_text = f"Current: Custom ({chamber_pixels} px)"
        else:
            boundary_text = "Current: Not set"
        self.chamber_boundary_status_label.setText(boundary_text)
        self.chamber_full_frame_button.setEnabled(has_video and boundary_mode != "full_frame")
        self.chamber_full_frame_button.setText(
            "Full Frame Applied" if has_video and boundary_mode == "full_frame" else "Use Full Frame"
        )
        self.chamber_reset_button.setEnabled(has_chamber or bool(self.room_records))
        self.room_combo.setEnabled(bool(self.room_records))
        self.room_rename_button.setEnabled(current is not None)
        self.room_delete_button.setEnabled(current is not None)
        self.room_clear_button.setEnabled(current is not None)
        self.chamber_edit_room_radio.setEnabled(bool(self.room_records))
        has_transform_target = self._selected_chamber_layer() is not None
        self.chamber_transform_radio.setEnabled(has_transform_target)
        if not has_transform_target and self.chamber_transform_radio.isChecked():
            self.chamber_draw_radio.setChecked(True)
        self.chamber_shape_combo.setEnabled(self.chamber_draw_radio.isChecked())
        self.export_chamber_mask_button.setEnabled(has_chamber)
        self._refresh_output_ui()

    def _chamber_mask_rgb(self) -> np.ndarray | None:
        if self.video_state is None or self.chamber_mask is None or not np.any(self.chamber_mask):
            return None
        image = np.zeros((self.video_state.height, self.video_state.width, 3), dtype=np.uint8)
        chamber_bool = self.chamber_mask.astype(bool)
        occupied = np.zeros((self.video_state.height, self.video_state.width), dtype=bool)
        for room in self._effective_room_records():
            room_bool = room.mask.astype(bool)
            occupied |= room_bool
            image[room_bool, 0] = room.color.red()
            image[room_bool, 1] = room.color.green()
            image[room_bool, 2] = room.color.blue()
        chamber_only = chamber_bool & ~occupied
        image[chamber_only, 0] = 209
        image[chamber_only, 1] = 213
        image[chamber_only, 2] = 219
        return image

    def _chamber_overlay_rgb(self) -> np.ndarray | None:
        if self.current_frame_rgb is None:
            return None
        mask_rgb = self._chamber_mask_rgb()
        if mask_rgb is None:
            return None
        overlay = self.current_frame_rgb.copy()
        active = np.any(mask_rgb > 0, axis=2)
        blended = (overlay[active].astype(np.float32) * 0.58 + mask_rgb[active].astype(np.float32) * 0.42).clip(0, 255).astype(np.uint8)
        overlay[active] = blended
        return overlay

    def export_chamber_mask(self) -> None:
        if self.video_state is None:
            QMessageBox.information(self, "Chamber Mask", "Load a video first.")
            return
        mask_rgb = self._chamber_mask_rgb()
        if mask_rgb is None:
            QMessageBox.information(self, "Chamber Mask", "Define the chamber area first.")
            return
        mask_path = self._chamber_mask_output_path()
        manifest_path = self._chamber_manifest_output_path()
        if mask_path is None or manifest_path is None:
            return
        try:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mask_path), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Could not write {mask_path}")
            metadata = {
                "format": "happycold_chamber_mask_v1",
                "width": self.video_state.width,
                "height": self.video_state.height,
                "boundary_mode": self._resolved_chamber_boundary_mode(),
                "rooms": [{"name": room.name, "color": room.color.name()} for room in self.room_records.values()],
            }
            manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Chamber Mask", f"Could not export chamber mask.\n\n{exc}")
            return
        self.statusBar().showMessage(f"Chamber mask exported: {mask_path}")
        QMessageBox.information(self, "Chamber Mask", f"Chamber mask exported to:\n{mask_path}")

    def import_chamber_mask(self) -> None:
        if self.video_state is None:
            QMessageBox.information(self, "Import Chamber Mask", "Load a video first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open Chamber Mask PNG", "", "PNG Files (*.png)")
        if not path:
            return
        self._import_chamber_mask_file(Path(path))

    def _import_chamber_mask_file(self, path: Path) -> None:
        if self.video_state is None:
            return
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.warning(self, "Import Chamber Mask", f"Could not read mask file:\n{path}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (self.video_state.height, self.video_state.width):
            image = cv2.resize(image, (self.video_state.width, self.video_state.height), interpolation=cv2.INTER_NEAREST)

        manifest_path = path.with_suffix(".json")
        room_entries: list[dict] = []
        if manifest_path.exists():
            try:
                room_entries = json.loads(manifest_path.read_text(encoding="utf-8")).get("rooms", [])
            except Exception:
                room_entries = []

        chamber_mask = np.any(image > 0, axis=2).astype(np.uint8)
        imported_rooms: dict[str, RoomRecord] = {}
        if room_entries:
            for index, entry in enumerate(room_entries):
                name = str(entry.get("name", f"room_{index + 1}")).strip() or f"room_{index + 1}"
                color = QColor(str(entry.get("color", MASK_PALETTE[index % len(MASK_PALETTE)].name())))
                rgb = np.array([color.red(), color.green(), color.blue()], dtype=np.uint8)
                room_mask = np.all(image == rgb, axis=2).astype(np.uint8)
                if np.any(room_mask):
                    imported_rooms[name] = RoomRecord(name=name, color=color, mask=room_mask)
        else:
            unique_colors = np.unique(image.reshape(-1, 3), axis=0)
            inferred_index = 1
            for rgb in unique_colors:
                if np.all(rgb == 0) or np.all(rgb == np.array([209, 213, 219], dtype=np.uint8)):
                    continue
                color = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
                name = self._next_available_room_name(f"room_{inferred_index}")
                inferred_index += 1
                room_mask = np.all(image == rgb, axis=2).astype(np.uint8)
                if np.any(room_mask):
                    imported_rooms[name] = RoomRecord(name=name, color=color, mask=room_mask)

        self.chamber_mask = chamber_mask
        self._set_chamber_boundary_mode(
            "full_frame" if self._is_full_frame_chamber_mask() else "custom"
        )
        self.room_records = imported_rooms
        self.selected_room_name = sorted(imported_rooms)[0] if imported_rooms else None
        self.frame_viewer.clear_chamber_rect_points()
        self.frame_viewer.clear_chamber_circle()
        self._rebuild_room_list()
        self.statusBar().showMessage(f"Imported chamber mask: {path.name}")

    def save_chamber_outputs(self) -> None:
        if self.video_state is None or self.csv_df is None:
            QMessageBox.warning(self, "Save", "Load a video and CSV first.")
            return
        if self.chamber_mask is None or not np.any(self.chamber_mask):
            QMessageBox.warning(self, "Save", "Define the chamber area first.")
            return
        if not self.room_records:
            QMessageBox.warning(self, "Save", "Add at least one room first.")
            return

        csv_output = self._chamber_csv_output_path()
        mask_output = self._chamber_mask_output_path()
        overlay_output = self._chamber_overlay_output_path()
        manifest_output = self._chamber_manifest_output_path()
        if csv_output is None or mask_output is None or overlay_output is None or manifest_output is None:
            return

        mask_rgb = self._chamber_mask_rgb()
        overlay_rgb = self._chamber_overlay_rgb()
        if mask_rgb is None or overlay_rgb is None:
            QMessageBox.warning(self, "Save", "A frame and chamber mask are required.")
            return

        try:
            df = build_chamber_mark_dataframe(
                self.csv_df,
                self.bodyparts,
                self._effective_room_records(),
                self.video_state.width,
                self.video_state.height,
            )
            csv_output.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_output, index=False)
            if not cv2.imwrite(str(mask_output), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Could not write {mask_output}")
            if not cv2.imwrite(str(overlay_output), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Could not write {overlay_output}")
            metadata = {
                "format": "happycold_chamber_mask_v1",
                "width": self.video_state.width,
                "height": self.video_state.height,
                "boundary_mode": self._resolved_chamber_boundary_mode(),
                "rooms": [{"name": room.name, "color": room.color.name()} for room in self.room_records.values()],
            }
            manifest_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Save Warning", f"Could not save chamber outputs.\n\n{exc}")
            return

        self.statusBar().showMessage(f"Chamber outputs saved: {csv_output}")
