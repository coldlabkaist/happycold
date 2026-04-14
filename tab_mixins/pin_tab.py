from string import ascii_lowercase, ascii_uppercase

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget

from happycold_shared import PinRecord


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
        self.pin_list.setToolTip("Each entry shows the pin label, frame number, absolute coordinates, and normalized coordinates.")
        self.pin_reset_button = QPushButton("Reset Pins")
        self.pin_reset_button.setToolTip("Remove every stored pin.")
        self.pin_remove_last_button = QPushButton("Remove Last Pin")
        self.pin_remove_last_button.setToolTip("Remove the most recently added pin.")
        row = QHBoxLayout()
        row.addWidget(self.pin_reset_button)
        row.addWidget(self.pin_remove_last_button)
        layout.addWidget(info)
        layout.addWidget(self.pin_summary_label)
        layout.addLayout(row)
        layout.addWidget(self.pin_list, stretch=1)
        return tab

    def add_pin(self, image_point: tuple[float, float]) -> None:
        if self.video_state is None:
            return
        pin_labels = ascii_uppercase + ascii_lowercase
        label = pin_labels[self.pin_counter % len(pin_labels)]
        self.pin_counter += 1
        self.pins.append(PinRecord(label, self.current_frame_number, image_point[0], image_point[1]))
        self.frame_viewer.set_pin_records(self.pins)
        self._refresh_pin_ui()

    def _refresh_pin_ui(self) -> None:
        self.pin_list.clear()
        if self.video_state is None:
            self.pin_summary_label.setText("Pins: 0")
            return
        visible_pins = [pin for pin in self.pins if pin.frame == self.current_frame_number]
        self.pin_summary_label.setText(f"Pins: {len(self.pins)} total | {len(visible_pins)} on current frame")
        for pin in self.pins:
            nx, ny = pin.normalized(self.video_state.width, self.video_state.height)
            self.pin_list.addItem(f"{pin.pin_id} | frame {pin.frame} | abs=({pin.x:.1f}, {pin.y:.1f}) | rel=({nx:.4f}, {ny:.4f})")

    def remove_last_pin(self) -> None:
        if self.pins:
            self.pins.pop()
            self.frame_viewer.set_pin_records(self.pins)
            self._refresh_pin_ui()

    def reset_pins(self) -> None:
        self.pins.clear()
        self.pin_counter = 0
        self.frame_viewer.set_pin_records(self.pins)
        self._refresh_pin_ui()
