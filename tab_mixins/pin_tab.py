import json
from string import ascii_lowercase, ascii_uppercase

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from shared import PinRecord
from ui_controls import NoWheelSpinBox


class PinTabMixin:
    def _build_pin_tab(self) -> QWidget:
        tab = QWidget()
        tooltip = "Place pins on the current frame to inspect absolute pixel coordinates and normalized coordinates."
        tab.setToolTip(tooltip)
        layout = QVBoxLayout(tab)
        info = QLabel("In Pin mode, left-click to place pins. Absolute and normalized coordinates are listed below.")
        info.setWordWrap(True)
        info.setToolTip(tooltip)
        self.pin_summary_label = QLabel("Pins: 0")
        self.pin_summary_label.setToolTip("Shows how many pins exist overall and on the current frame.")
        self.pin_list = QListWidget()
        self.pin_list.setToolTip("Right-click a pin to delete it. Each entry shows frame number, absolute coordinates, and normalized coordinates.")
        self.pin_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.pin_list.customContextMenuRequested.connect(self._show_pin_list_context_menu)
        self.pin_reset_button = QPushButton("Reset Pins")
        self.pin_reset_button.setToolTip("Remove every stored pin.")
        self.pin_remove_last_button = QPushButton("Remove Last Pin")
        self.pin_remove_last_button.setToolTip("Remove the most recently added pin.")
        row = QHBoxLayout()
        row.addWidget(self.pin_reset_button)
        row.addWidget(self.pin_remove_last_button)

        usage_group = QGroupBox("Use Pins While Drawing")
        usage_layout = QVBoxLayout(usage_group)
        usage_help = QLabel(
            "Pins can be shown while using Annotate / Inspect tools. If snap is enabled, a drawing point snaps to the nearest pin on the same frame only when it is inside the snap radius."
        )
        usage_help.setWordWrap(True)
        self.pin_show_outside_tab_checkbox = QCheckBox("Show pins in Annotate / Inspect tabs")
        self.pin_show_outside_tab_checkbox.setChecked(
            bool(self.settings.get("pin_show_outside_tab", True))
        )
        self.pin_snap_checkbox = QCheckBox("Snap drawing points to nearby pins")
        self.pin_snap_checkbox.setChecked(
            bool(self.settings.get("pin_snap_enabled", False))
        )
        self.pin_snap_radius_spinbox = NoWheelSpinBox()
        self.pin_snap_radius_spinbox.setRange(1, 500)
        self.pin_snap_radius_spinbox.setValue(20)
        self.pin_snap_radius_spinbox.setSuffix(" px")
        self.pin_snap_radius_spinbox.setMinimumWidth(0)
        radius_row = QHBoxLayout()
        radius_row.addWidget(QLabel("Snap radius"))
        radius_row.addWidget(self.pin_snap_radius_spinbox, stretch=1)
        self.pin_snap_status_label = QLabel("Snap: off")
        self.pin_snap_status_label.setWordWrap(True)
        usage_layout.addWidget(usage_help)
        usage_layout.addWidget(self.pin_show_outside_tab_checkbox)
        usage_layout.addWidget(self.pin_snap_checkbox)
        usage_layout.addLayout(radius_row)
        usage_layout.addWidget(self.pin_snap_status_label)

        metadata_row = QHBoxLayout()
        self.pin_import_button = QPushButton("Import Pins")
        self.pin_export_button = QPushButton("Export Pins")
        metadata_row.addWidget(self.pin_import_button)
        metadata_row.addWidget(self.pin_export_button)

        layout.addWidget(info)
        layout.addWidget(self.pin_summary_label)
        layout.addLayout(row)
        layout.addWidget(usage_group)
        layout.addLayout(metadata_row)
        layout.addWidget(self.pin_list, stretch=1)
        return tab

    def add_pin(self, image_point):
        if self.video_state is None:
            return
        pin_id = self._next_pin_id()
        pin = PinRecord(pin_id=pin_id, frame=self.current_frame_number, x=float(image_point[0]), y=float(image_point[1]))
        self.pins.append(pin)
        self.frame_viewer.set_pin_records(self.pins)
        self._refresh_pin_ui()

    def _next_pin_id(self) -> str:
        labels = list(ascii_uppercase) + list(ascii_lowercase)
        used = {pin.pin_id for pin in self.pins}
        counter = int(getattr(self, "pin_counter", len(self.pins)))
        while True:
            pin_id = labels[counter] if counter < len(labels) else f"P{counter + 1}"
            counter += 1
            if pin_id not in used:
                self.pin_counter = counter
                return pin_id

    def _sync_pin_counter_to_existing_pins(self) -> None:
        labels = list(ascii_uppercase) + list(ascii_lowercase)
        used = {pin.pin_id for pin in self.pins}
        counter = 0
        while True:
            pin_id = labels[counter] if counter < len(labels) else f"P{counter + 1}"
            if pin_id not in used:
                self.pin_counter = counter
                return
            counter += 1

    def _pin_list_label(self, pin: PinRecord) -> str:
        if self.video_state is not None:
            nx, ny = pin.normalized(self.video_state.width, self.video_state.height)
            return f"{pin.pin_id} | frame {pin.frame} | abs=({pin.x:.1f}, {pin.y:.1f}) | rel=({nx:.4f}, {ny:.4f})"
        return f"{pin.pin_id} | frame {pin.frame} | abs=({pin.x:.1f}, {pin.y:.1f})"

    def _refresh_pin_ui(self):
        self.pin_list.clear()
        if self.video_state is None:
            self.pin_summary_label.setText("Pins: 0")
        else:
            visible_pins = [pin for pin in self.pins if pin.frame == self.current_frame_number]
            self.pin_summary_label.setText(f"Pins: {len(self.pins)} total | {len(visible_pins)} on current frame")
        for index, pin in enumerate(self.pins):
            item = QListWidgetItem(self._pin_list_label(pin))
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.pin_list.addItem(item)

        if hasattr(self, "pin_export_button"):
            self.pin_export_button.setEnabled(bool(self.pins))
        self._sync_pin_viewer_options()

    def _pin_settings_payload(self) -> dict:
        if not hasattr(self, "pin_show_outside_tab_checkbox"):
            return {}
        return {
            "pin_show_outside_tab": bool(self.pin_show_outside_tab_checkbox.isChecked()),
            "pin_snap_enabled": bool(self.pin_snap_checkbox.isChecked()),
        }

    def _on_pin_option_changed(self, *args):
        self._sync_pin_viewer_options()
        if hasattr(self, "_save_settings"):
            self._save_settings()

    def _sync_pin_viewer_options(self):
        if not hasattr(self, "frame_viewer"):
            return
        show_outside = bool(
            hasattr(self, "pin_show_outside_tab_checkbox") and self.pin_show_outside_tab_checkbox.isChecked()
        )
        snap_enabled = bool(hasattr(self, "pin_snap_checkbox") and self.pin_snap_checkbox.isChecked())
        snap_radius = int(self.pin_snap_radius_spinbox.value()) if hasattr(self, "pin_snap_radius_spinbox") else 20
        if hasattr(self.frame_viewer, "set_pin_overlay_options"):
            self.frame_viewer.set_pin_overlay_options(
                show_outside_pin_mode=show_outside,
                snap_to_nearby_pins=snap_enabled,
                snap_radius=snap_radius,
            )
        if hasattr(self, "pin_snap_status_label"):
            if not snap_enabled:
                self.pin_snap_status_label.setText("Snap: off")
            elif not self.pins:
                self.pin_snap_status_label.setText("Snap: on, but no pins are available.")
            else:
                self.pin_snap_status_label.setText(
                    f"Snap: nearest pin within {snap_radius}px on the same frame."
                )

    def _show_pin_list_context_menu(self, position) -> None:
        item = self.pin_list.itemAt(position)
        if item is None:
            return
        index = item.data(Qt.ItemDataRole.UserRole)
        try:
            index = int(index)
        except (TypeError, ValueError):
            return
        if not (0 <= index < len(self.pins)):
            return
        pin = self.pins[index]
        menu = QMenu(self.pin_list)
        delete_action = menu.addAction(f"Delete pin {pin.pin_id}")
        chosen = menu.exec(self.pin_list.mapToGlobal(position))
        if chosen == delete_action:
            self.delete_pin_at_index(index)

    def delete_pin_at_index(self, index: int) -> None:
        if not (0 <= index < len(self.pins)):
            return
        pin = self.pins.pop(index)
        self.frame_viewer.set_pin_records(self.pins)
        self._refresh_pin_ui()
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage(f"Deleted pin {pin.pin_id}.", 2500)

    def remove_last_pin(self):
        if self.pins:
            self.pins.pop()
        self.frame_viewer.set_pin_records(self.pins)
        self._refresh_pin_ui()

    def reset_pins(self):
        self.pins.clear()
        self.pin_counter = 0
        self.frame_viewer.set_pin_records(self.pins)
        self._refresh_pin_ui()

    def _pin_export_default_path(self) -> str:
        if self.video_state is not None:
            return str(self.video_state.path.with_suffix(".pins.json"))
        return "pins_metadata.json"

    def _pin_metadata_payload(self) -> dict:
        video_metadata = None
        if self.video_state is not None:
            video_metadata = {
                "path": str(self.video_state.path),
                "width": int(self.video_state.width),
                "height": int(self.video_state.height),
                "frame_count": int(self.video_state.frame_count),
                "fps": float(self.video_state.fps),
            }
        pin_items = []
        for pin in self.pins:
            item = {
                "id": pin.pin_id,
                "pin_id": pin.pin_id,
                "frame": int(pin.frame),
                "x": float(pin.x),
                "y": float(pin.y),
            }
            if self.video_state is not None:
                nx, ny = pin.normalized(self.video_state.width, self.video_state.height)
                item["x_normalized"] = float(nx)
                item["y_normalized"] = float(ny)
            pin_items.append(item)
        return {
            "format": "happycold_pin_metadata_v1",
            "metadata_version": 1,
            "video": video_metadata,
            "pins": pin_items,
        }

    def export_pins_metadata(self) -> None:
        if not self.pins:
            QMessageBox.information(self, "Export Pins", "There are no pins to export.")
            return
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Pin Metadata",
            self._pin_export_default_path(),
            "Pin metadata (*.json);;All Files (*)",
        )
        if not output_path:
            return
        if not output_path.lower().endswith(".json"):
            output_path += ".json"
        try:
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(self._pin_metadata_payload(), handle, indent=2)
        except OSError as exc:
            QMessageBox.warning(self, "Export Pins", f"Could not export pins:\n{exc}")
            return
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage(f"Exported {len(self.pins)} pins.", 3000)

    def _pin_from_metadata_item(self, item: dict, fallback_index: int) -> PinRecord | None:
        if not isinstance(item, dict):
            return None
        pin_id = str(item.get("pin_id", item.get("id", f"P{fallback_index + 1}"))).strip() or f"P{fallback_index + 1}"
        try:
            frame = int(round(float(item.get("frame", getattr(self, "current_frame_number", 0)))))
        except (TypeError, ValueError):
            frame = int(getattr(self, "current_frame_number", 0))

        x_value = item.get("x")
        y_value = item.get("y")
        if (x_value is None or y_value is None) and self.video_state is not None:
            nx_value = item.get("x_normalized", item.get("x_norm"))
            ny_value = item.get("y_normalized", item.get("y_norm"))
            if nx_value is not None and ny_value is not None:
                try:
                    x_value = float(nx_value) * self.video_state.width
                    y_value = float(ny_value) * self.video_state.height
                except (TypeError, ValueError):
                    x_value = None
                    y_value = None
        try:
            x = float(x_value)
            y = float(y_value)
        except (TypeError, ValueError):
            return None
        if self.video_state is not None:
            x = max(0.0, min(float(self.video_state.width - 1), x))
            y = max(0.0, min(float(self.video_state.height - 1), y))
            frame = max(0, min(int(self.video_state.frame_count - 1), frame))
        return PinRecord(pin_id=pin_id, frame=frame, x=x, y=y)

    def import_pins_metadata(self) -> None:
        input_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Pin Metadata",
            "",
            "Pin metadata (*.json);;All Files (*)",
        )
        if not input_path:
            return
        try:
            with open(input_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Import Pins", f"Could not read pin metadata:\n{exc}")
            return

        raw_pins = payload.get("pins") if isinstance(payload, dict) else payload
        if not isinstance(raw_pins, list):
            QMessageBox.warning(self, "Import Pins", "This file does not contain a valid pins list.")
            return
        imported: list[PinRecord] = []
        for index, item in enumerate(raw_pins):
            pin = self._pin_from_metadata_item(item, index)
            if pin is not None:
                imported.append(pin)
        if not imported:
            QMessageBox.warning(self, "Import Pins", "No valid pins were found in this file.")
            return

        self.pins = imported
        self._sync_pin_counter_to_existing_pins()
        self.frame_viewer.set_pin_records(self.pins)
        self._refresh_pin_ui()
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage(f"Imported {len(self.pins)} pins.", 3000)
