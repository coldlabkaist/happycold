import json
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from happycold_shared import MASK_PALETTE, RoomRecord, build_chamber_mark_dataframe, fill_circle_from_diameter, fill_polygon, order_quad_points


class ChamberTabMixin:
    def _build_chamber_tab(self) -> QWidget:
        tab = QWidget()
        tooltip = (
            "Define a chamber as the union of multiple rectangles or circles, then define named rooms inside it. "
            "Rooms cannot overlap each other and only the area inside the chamber is accepted."
        )
        tab.setToolTip(tooltip)
        layout = QVBoxLayout(tab)

        info = QLabel(
            "1) Build the chamber with repeated square or circle shapes. "
            "2) Add a room name, switch to room editing, and add shapes for that room."
        )
        info.setWordWrap(True)
        info.setToolTip(tooltip)

        self.chamber_shape_combo = QComboBox()
        self.chamber_shape_combo.addItems(["Rectangle 4 Points", "Circle Drag"])
        self.chamber_shape_combo.setToolTip("Choose whether the next chamber or room area is drawn by four points or by dragging a circle.")

        self.chamber_edit_chamber_radio = QRadioButton("Edit Chamber")
        self.chamber_edit_room_radio = QRadioButton("Edit Room")
        self.chamber_edit_chamber_radio.setChecked(True)
        self.chamber_edit_chamber_radio.setToolTip("Add new shapes to the overall chamber area.")
        self.chamber_edit_room_radio.setToolTip("Add new shapes to the selected room, clipped to the chamber and excluding other rooms.")
        self.chamber_edit_mode_group = QButtonGroup(self)
        self.chamber_edit_mode_group.addButton(self.chamber_edit_chamber_radio)
        self.chamber_edit_mode_group.addButton(self.chamber_edit_room_radio)

        self.room_combo = QComboBox()
        self.room_combo.setToolTip("Choose which room receives the next drawn shapes.")
        self.room_add_button = QPushButton("Add Room")
        self.room_add_button.setToolTip("Create a new named room.")
        self.room_rename_button = QPushButton("Rename Room")
        self.room_rename_button.setToolTip("Rename the selected room.")
        self.room_delete_button = QPushButton("Delete Room")
        self.room_delete_button.setToolTip("Delete the selected room and free its occupied area.")
        self.room_clear_button = QPushButton("Clear Room")
        self.room_clear_button.setToolTip("Remove all pixels from the selected room only.")
        self.chamber_reset_button = QPushButton("Reset Chamber")
        self.chamber_reset_button.setToolTip("Clear the entire chamber and every room assigned inside it.")

        mode_row = QHBoxLayout()
        mode_row.addWidget(self.chamber_edit_chamber_radio)
        mode_row.addWidget(self.chamber_edit_room_radio)

        chamber_row = QHBoxLayout()
        chamber_row.addWidget(self.chamber_shape_combo, stretch=1)
        chamber_row.addWidget(self.chamber_reset_button)

        room_row = QHBoxLayout()
        room_row.addWidget(self.room_add_button)
        room_row.addWidget(self.room_rename_button)
        room_row.addWidget(self.room_delete_button)
        room_row.addWidget(self.room_clear_button)

        self.chamber_summary_label = QLabel("Define the chamber area first.")
        self.chamber_summary_label.setWordWrap(True)
        self.chamber_summary_label.setToolTip("Shows chamber size, occupied room area, free chamber area, and the currently selected room.")

        self.import_chamber_mask_button = QPushButton("Import Chamber Mask")
        self.import_chamber_mask_button.setToolTip("Load a previously exported chamber mask PNG and its matching metadata JSON.")
        self.export_chamber_mask_button = QPushButton("Export Chamber Mask")
        self.export_chamber_mask_button.setToolTip("Save a color-coded room mask PNG and metadata JSON for later reloading.")
        import_export_row = QHBoxLayout()
        import_export_row.addWidget(self.import_chamber_mask_button)
        import_export_row.addWidget(self.export_chamber_mask_button)

        form = QFormLayout()
        form.addRow("Edit Target", mode_row)
        form.addRow("Chamber Tool", chamber_row)
        form.addRow("Current Room", self.room_combo)

        layout.addWidget(info)
        layout.addLayout(form)
        layout.addLayout(room_row)
        layout.addWidget(self.chamber_summary_label)
        layout.addStretch(1)
        layout.addLayout(import_export_row)
        return tab

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
        if self.mode_tabs.currentIndex() != 1:
            return
        if self.chamber_shape_combo.currentIndex() == 0:
            self.frame_viewer.clear_chamber_circle()
        else:
            self.frame_viewer.clear_chamber_rect_points()
        self.frame_viewer.set_mode(self._selected_chamber_draw_mode())
        self.frame_viewer.set_margin_value(0.0)
        self._refresh_chamber_ui()

    def _rebuild_room_list(self) -> None:
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
        self.frame_viewer.set_chamber_records(self.chamber_mask, self.room_records, self.selected_room_name, refresh=True)
        self._refresh_chamber_ui()

    def _on_room_selection_changed(self, index: int) -> None:
        self.selected_room_name = self.room_combo.itemData(index) if index >= 0 else None
        self.frame_viewer.set_chamber_records(self.chamber_mask, self.room_records, self.selected_room_name, refresh=True)
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
        current.mask.fill(0)
        self.frame_viewer.set_chamber_records(self.chamber_mask, self.room_records, self.selected_room_name, refresh=True)
        self._refresh_chamber_ui()

    def reset_chamber(self) -> None:
        self.chamber_mask = None
        self.room_records.clear()
        self.selected_room_name = None
        self.frame_viewer.clear_chamber_rect_points()
        self.frame_viewer.clear_chamber_circle()
        self.room_combo.blockSignals(True)
        self.room_combo.clear()
        self.room_combo.blockSignals(False)
        self.frame_viewer.set_chamber_records(self.chamber_mask, self.room_records, self.selected_room_name, refresh=True)
        self._refresh_chamber_ui()

    def _occupied_room_mask(self, exclude_name: str | None = None) -> np.ndarray | None:
        if self.video_state is None:
            return None
        occupied = np.zeros((self.video_state.height, self.video_state.width), dtype=bool)
        for name, room in self.room_records.items():
            if exclude_name is not None and name == exclude_name:
                continue
            occupied |= room.mask.astype(bool)
        return occupied.astype(np.uint8)

    def _apply_chamber_shape_mask(self, shape_mask: np.ndarray) -> None:
        if self.video_state is None:
            return
        if self.chamber_edit_chamber_radio.isChecked():
            if self.chamber_mask is None:
                self.chamber_mask = np.zeros((self.video_state.height, self.video_state.width), dtype=np.uint8)
            self.chamber_mask = np.logical_or(self.chamber_mask.astype(bool), shape_mask.astype(bool)).astype(np.uint8)
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
        self.frame_viewer.set_chamber_records(self.chamber_mask, self.room_records, self.selected_room_name, refresh=True)
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
        occupied_pixels = int(sum(int(room.mask.sum()) for room in self.room_records.values()))
        free_pixels = max(0, chamber_pixels - occupied_pixels)
        current = self._selected_room()
        current_name = current.name if current is not None else "-"
        current_pixels = int(current.mask.sum()) if current is not None else 0
        mode_text = "chamber" if self.chamber_edit_chamber_radio.isChecked() else "room"
        self.chamber_summary_label.setText(
            f"edit={mode_text} | chamber pixels={chamber_pixels} | occupied={occupied_pixels} | free={free_pixels} | current room={current_name} ({current_pixels} px) | rooms={len(self.room_records)}"
        )
        has_chamber = self.chamber_mask is not None and bool(np.any(self.chamber_mask))
        self.room_combo.setEnabled(bool(self.room_records))
        self.room_rename_button.setEnabled(current is not None)
        self.room_delete_button.setEnabled(current is not None)
        self.room_clear_button.setEnabled(current is not None)
        self.chamber_edit_room_radio.setEnabled(bool(self.room_records))
        self.export_chamber_mask_button.setEnabled(has_chamber)
        self._refresh_output_ui()

    def _chamber_mask_rgb(self) -> np.ndarray | None:
        if self.video_state is None or self.chamber_mask is None or not np.any(self.chamber_mask):
            return None
        image = np.zeros((self.video_state.height, self.video_state.width, 3), dtype=np.uint8)
        chamber_bool = self.chamber_mask.astype(bool)
        occupied = np.zeros((self.video_state.height, self.video_state.width), dtype=bool)
        for room in self.room_records.values():
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
                list(self.room_records.values()),
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
                "rooms": [{"name": room.name, "color": room.color.name()} for room in self.room_records.values()],
            }
            manifest_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception as exc:
            QMessageBox.warning(self, "Save Warning", f"Could not save chamber outputs.\n\n{exc}")
            return

        self.statusBar().showMessage(f"Chamber outputs saved: {csv_output}")
