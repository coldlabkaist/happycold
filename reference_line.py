"""Shared CAD-style reference line input for Direction and 3D Object."""
from __future__ import annotations

import math

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QSizePolicy, QWidget

from annotation_settings import preference_number
from occlusion_3d import reference_pixel_length
from ui_controls import NoWheelDoubleSpinBox


def reference_endpoint(anchor, pointer, length=None, orthogonal=False):
    """Return a continuous image-space endpoint, without clipping or rounding."""
    dx, dy = pointer[0] - anchor[0], pointer[1] - anchor[1]
    if orthogonal:
        if abs(dx) >= abs(dy):
            dy = 0.0
        else:
            dx = 0.0
    distance = math.hypot(dx, dy)
    if distance < 1e-10:
        return None
    scale = 1.0 if length is None else length / distance
    return anchor[0] + dx * scale, anchor[1] + dy * scale


class ReferenceLineControls(QWidget):
    options_changed = pyqtSignal(float, bool, bool)
    commit_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    shift_requested = pyqtSignal(bool)

    def __init__(self, saved=None):
        super().__init__()
        saved = saved if isinstance(saved, dict) else {}
        self._syncing = False
        self._length = preference_number(saved.get("length"), 100.0, 1, 1e9)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.length = NoWheelDoubleSpinBox()
        self.length.setRange(1, 1e9)
        self.length.setDecimals(2)
        self.length.setKeyboardTracking(False)
        self.length.setSuffix(" px")
        self.length.setValue(self._length)
        self.length.setMinimumWidth(110)
        self.length.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.length.setToolTip("Length in original video pixels. Press Enter to apply; Shift snaps to horizontal or vertical.")
        self.fixed = QCheckBox("Fixed")
        self.fixed.setChecked(saved.get("fixed") is True)
        self.fixed.setToolTip("Keep this pixel length while choosing the line direction or dragging an endpoint.")
        row.addWidget(self.length, 1)
        row.addWidget(self.fixed)
        self.length.valueChanged.connect(self._length_changed)
        self.fixed.toggled.connect(self._fixed_changed)
        self.length.installEventFilter(self)
        self.length.lineEdit().installEventFilter(self)
        self.fixed.installEventFilter(self)

    def preferences(self):
        return {"fixed": self.fixed.isChecked(), "length": self._length}

    def _length_changed(self, value):
        if self._syncing:
            return
        self._length = value
        self.fixed.blockSignals(True)
        self.fixed.setChecked(True)
        self.fixed.blockSignals(False)
        self.options_changed.emit(value, True, True)

    def _fixed_changed(self, checked):
        if not self._syncing:
            self.options_changed.emit(self._length, checked, False)

    def show_state(self, length, fixed):
        self._syncing = True
        try:
            self._length = length
            if not (self.length.hasFocus() and self.length.lineEdit().isModified()):
                self.length.setValue(length)
            self.fixed.setChecked(fixed)
        finally:
            self._syncing = False

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.length.interpretText()
                self.commit_requested.emit()
                return True
            if event.key() == Qt.Key.Key_Escape:
                self.cancel_requested.emit()
                return True
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and event.key() == Qt.Key.Key_Shift:
            self.shift_requested.emit(event.type() == QEvent.Type.KeyPress)
            return True
        return super().eventFilter(watched, event)


class ReferenceLineViewerMixin:
    def _init_reference_line_editor(self):
        self.reference_editor_key = None
        self.reference_fixed = False
        self.reference_pixel_length = 100.0
        self._reference_committed = []
        self._reference_pending = False
        self._reference_preview = None
        self._reference_cursor = None
        self._reference_shift = False
        self._reference_number = ""
        self._reference_typing = False
        self._reference_error = ""
        self._reference_adjust_index = None

    @property
    def reference_line_pending(self):
        return self.reference_editor_key is not None and self._reference_pending

    def configure_reference_line(self, key, points, fixed, length, *, redraw=False):
        points = [tuple(p) for p in points]
        reset = key != self.reference_editor_key or points != self._reference_committed
        if reset:
            self._reference_committed = points
            self.reference_pick_points = list(points)
            self._reference_pending = len(points) != 2
            self._reference_preview = self._reference_cursor = None
            self._reference_adjust_index = self._reference_drag_index = None
            self._reference_number = self._reference_error = ""
            self._reference_typing = False
            self._reference_shift = False
            self.reference_pixel_length = length
            if len(points) == 2:
                try:
                    self.reference_pixel_length = reference_pixel_length(points)
                except ValueError:
                    pass
        self.reference_editor_key = key
        self.reference_pick_active = True
        self.reference_fixed = fixed
        if redraw:
            self.reference_pick_points = []
            self._reference_pending = True
            self._reference_preview = self._reference_cursor = None
            self._reference_adjust_index = self._reference_drag_index = None
            self._reference_number = self._reference_error = ""
            self._reference_typing = False
        self._reference_notify()

    def deactivate_reference_line(self):
        if self.reference_editor_key is not None:
            self.reference_pick_points = list(self._reference_committed)
        self.reference_editor_key = None
        self.reference_pick_active = False
        self._reference_pending = False
        self._reference_preview = self._reference_cursor = None
        self._reference_adjust_index = self._reference_drag_index = None
        self._reference_number = self._reference_error = ""
        self._reference_typing = False
        self._reference_shift = False

    def reference_display_points(self):
        if len(self.reference_pick_points) == 1 and self._reference_preview is not None:
            return [self.reference_pick_points[0], self._reference_preview]
        return list(self.reference_pick_points)

    def reference_display_length(self):
        points = self.reference_display_points()
        if self.reference_fixed or len(points) != 2:
            return self.reference_pixel_length
        return max(1, math.dist(*points))

    def reference_instruction(self):
        if self._reference_error:
            return self._reference_error
        if not self._reference_pending:
            return ""
        if not self.reference_pick_points:
            return "Click a start point."
        return "Choose direction; type pixels. Enter/click: finish. Shift: H/V. Esc: cancel."

    def _reference_notify(self):
        self.reference_line_state_changed.emit()
        self.update()

    def _reference_valid(self, points):
        if self._reference_typing:
            try:
                number = float(self._reference_number)
            except ValueError:
                number = 0
            if not math.isfinite(number) or not 1 <= number <= 1e9:
                self._reference_error = "Enter a pixel length of at least 1 px."
                return False
        if len(points) != 2:
            self._reference_error = "Set a direction and a length of at least 1 px."
            return False
        if not all(math.isfinite(v) for p in points for v in p):
            self._reference_error = "Enter a finite pixel length."
            return False
        try:
            reference_pixel_length(points)
        except ValueError as exc:
            self._reference_error = str(exc)
            return False
        if not all(-1e-8 <= x <= self._frame_width + 1e-8 and -1e-8 <= y <= self._frame_height + 1e-8 for x, y in points):
            self._reference_error = "Endpoint is outside the video. Change direction or length."
            return False
        self._reference_error = ""
        return True

    def _reference_update_proposal(self):
        if not self.reference_pick_points or self._reference_cursor is None:
            return
        index = self._reference_adjust_index
        anchor = self.reference_pick_points[1 - index] if index is not None else self.reference_pick_points[0]
        endpoint = reference_endpoint(anchor, self._reference_cursor,
            self.reference_pixel_length if self.reference_fixed else None, self._reference_shift)
        if endpoint is None:
            self._reference_error = "Move the pointer to choose a direction."
            return
        if index is None:
            self._reference_preview = endpoint
        else:
            self.reference_pick_points[index] = endpoint
        self._reference_valid(self.reference_display_points())

    def set_reference_length(self, length, fixed, *, resize=False):
        self.reference_pixel_length = length
        self.reference_fixed = fixed
        self._reference_number = ""
        self._reference_typing = False
        if resize and len(self.reference_pick_points) == 2:
            self._reference_cursor = self.reference_pick_points[1]
            self._reference_adjust_index = 1
            self._reference_pending = True
        if self._reference_pending:
            self._reference_update_proposal()
        self._reference_notify()

    def set_reference_shift(self, pressed):
        self._reference_shift = pressed
        if self._reference_pending:
            self._reference_update_proposal()
            self._reference_notify()

    def cancel_reference_line(self):
        self.reference_pick_points = list(self._reference_committed)
        self._reference_pending = False
        self._reference_preview = self._reference_cursor = None
        self._reference_adjust_index = self._reference_drag_index = None
        self._reference_number = self._reference_error = ""
        self._reference_typing = False
        if len(self._reference_committed) == 2:
            try:
                self.reference_pixel_length = reference_pixel_length(self._reference_committed)
            except ValueError:
                pass
        self._reference_notify()

    def commit_reference_line(self):
        if not self.reference_line_pending:
            return False
        points = self.reference_display_points()
        if not self._reference_valid(points):
            self._reference_notify()
            return False
        changed = points != self._reference_committed
        self._reference_committed = list(points)
        self.reference_pick_points = list(points)
        self.reference_pixel_length = reference_pixel_length(points)
        self._reference_pending = False
        self._reference_preview = self._reference_cursor = None
        self._reference_adjust_index = self._reference_drag_index = None
        self._reference_number = self._reference_error = ""
        self._reference_typing = False
        if changed:
            self.reference_line_completed.emit(list(points))
        self._reference_notify()
        return True

    def _reference_mouse_press(self, event, point):
        self._reference_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if len(self.reference_pick_points) == 1:
            self._reference_cursor = point
            self._reference_update_proposal()
            self.commit_reference_line()
            return
        hit = self._point_hit_index(event.position(), self.reference_pick_points)
        if hit is not None:
            self._reference_drag_index = self._reference_adjust_index = hit
            self._reference_cursor = point
        elif not self.reference_pick_points:
            self.reference_pick_points = [point]
            self._reference_pending = True
            self._reference_cursor = point
            self._reference_error = ""
        elif self._reference_pending:
            self.commit_reference_line()
        self._reference_notify()

    def _reference_mouse_move(self, event, point):
        self._reference_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if len(self.reference_pick_points) == 1 or (self._reference_drag_index is not None and event.buttons() & Qt.MouseButton.LeftButton):
            self._reference_cursor = point
            self._reference_pending = True
            self._reference_update_proposal()
            self._reference_notify()

    def _reference_mouse_release(self, event, point):
        if self._reference_drag_index is not None:
            if point is not None and self._reference_pending:
                self._reference_cursor = point
                self._reference_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                self._reference_update_proposal()
                self.commit_reference_line()
            self._reference_drag_index = None
        self.update()

    def _reference_keyboard_active(self):
        return self.mode == "reference_line" and self.reference_pick_active and self.reference_editor_key is not None

    @staticmethod
    def _reference_digit(event):
        if Qt.Key.Key_0 <= event.key() <= Qt.Key.Key_9:
            return str(event.key() - Qt.Key.Key_0)
        if event.key() in (Qt.Key.Key_Period, Qt.Key.Key_Comma):
            return "."
        return "-" if event.key() == Qt.Key.Key_Minus else None

    def event(self, event):
        if event.type() == QEvent.Type.ShortcutOverride and self._reference_keyboard_active():
            modifiers = event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier)
            if not modifiers and (self.reference_pick_points or self._reference_pending):
                if self._reference_digit(event) is not None or event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape, Qt.Key.Key_Backspace):
                    event.accept()
                    return True
        return super().event(event)

    def _reference_key_press(self, event):
        if not self._reference_keyboard_active():
            return False
        if event.key() == Qt.Key.Key_Shift:
            self.set_reference_shift(True)
        elif event.key() == Qt.Key.Key_Escape:
            self.cancel_reference_line()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.commit_reference_line()
        elif (self.reference_pick_points or self._reference_pending) and not event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier):
            digit = self._reference_digit(event)
            if digit is None and event.key() != Qt.Key.Key_Backspace:
                return False
            if not self._reference_pending and len(self.reference_pick_points) == 2:
                self._reference_cursor = self.reference_pick_points[1]
                self._reference_adjust_index = 1
            self._reference_pending = True
            self._reference_typing = True
            if event.key() == Qt.Key.Key_Backspace:
                self._reference_number = self._reference_number[:-1]
            elif (digit != "." or "." not in self._reference_number) and (digit != "-" or not self._reference_number):
                if len(self._reference_number) < 15:
                    self._reference_number += digit
            try:
                value = float(self._reference_number)
            except ValueError:
                value = 0
            if 1 <= value <= 1e9:
                buffer = self._reference_number
                self.set_reference_length(value, True, resize=len(self.reference_pick_points) == 2 and not self._reference_pending)
                self._reference_number = buffer
                self._reference_typing = True
            else:
                self._reference_error = "Enter a pixel length of at least 1 px."
                self._reference_notify()
        else:
            return False
        event.accept()
        return True


class ReferenceLineTabMixin:
    def _build_reference_line_controls(self, tab, saved):
        controls = ReferenceLineControls(saved)
        controls.options_changed.connect(lambda length, fixed, resize: self._reference_options_changed(tab, controls, length, fixed, resize))
        controls.commit_requested.connect(lambda: self._reference_apply(tab, controls))
        controls.cancel_requested.connect(lambda: self._reference_cancel(tab))
        controls.shift_requested.connect(lambda pressed: self.frame_viewer.set_reference_shift(pressed) if self._reference_active_tab() == tab else None)
        return controls

    def _reference_active_tab(self):
        if self._direction_active():
            return "direction"
        if self._object3d_active():
            return "object3d"
        return None

    def _reference_options_changed(self, tab, controls, length, fixed, resize):
        if self._reference_active_tab() == tab and self.frame_viewer.reference_editor_key is not None:
            self.frame_viewer.set_reference_length(length, fixed, resize=resize)
        self._save_settings()

    def _reference_apply(self, tab, controls):
        if self._reference_active_tab() == tab and self.frame_viewer.reference_pick_active:
            self.frame_viewer.commit_reference_line()
            self.frame_viewer.setFocus()

    def _reference_cancel(self, tab):
        if self._reference_active_tab() == tab:
            self.frame_viewer.cancel_reference_line()
            self.frame_viewer.setFocus()

    def _reference_line_state_changed(self):
        viewer = self.frame_viewer
        key = viewer.reference_editor_key
        if key is None or key[0] != self._reference_active_tab():
            return
        if key[0] == "direction":
            controls = self.direction_line_input
            context = self._direction_calibration
            if context and context["kind"] == "line":
                context["pending"] = viewer.reference_line_pending
            label, ref = self.direction_range_status, self._direction_edit_reference()
        else:
            controls = self.object3d_line_input
            label, ref = self.object3d_reference_status, self._object3d_reference()
        controls.show_state(viewer.reference_display_length(), viewer.reference_fixed)
        text = viewer.reference_instruction()
        if not text and ref:
            try:
                ref.pixels_per_metre
                text = f"{math.dist(*ref.points):.2f} px = {ref.length:g} {ref.unit}."
            except ValueError:
                text = "Click a start point, then choose direction and length."
        label.setText(text)
        label.setVisible(bool(text))
        self._refresh_output_ui()
