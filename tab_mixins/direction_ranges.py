"""Compact range controls and video-local calibration for Direction."""
from __future__ import annotations

import copy
import math

import cv2

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QButtonGroup, QFormLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from direction import DirectionZones
from occlusion_3d import LengthReference, UNITS, convert_length
from ui_controls import NoWheelComboBox, NoWheelDoubleSpinBox


class DirectionRangesMixin:
    def _build_direction_ranges(self):
        self._direction_range_errors = {}
        self._direction_range_paths = {}
        self._direction_calibration = None
        self._direction_range_syncing = False
        self._direction_reference_target = "line"
        self._direction_reference_id = None
        group = QGroupBox("2. Ranges")
        layout = QVBoxLayout(group)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.direction_distance_node = NoWheelComboBox()
        form.addRow("Distance Node", self.direction_distance_node)
        self.direction_distance_label = form.labelForField(self.direction_distance_node)
        self.direction_distance_node.setToolTip("Used for Overlap, Near, and Far for every object, independently of the heading method.")
        self.direction_range_mode = NoWheelComboBox()
        for label, key in (("Pixels", "pixels"), ("Physical", "physical"), ("Perspective", "geometric")):
            self.direction_range_mode.addItem(label, key)
        form.addRow("Margins", self.direction_range_mode)
        self.direction_object_margin = NoWheelDoubleSpinBox()
        self.direction_near_margin = NoWheelDoubleSpinBox()
        self.direction_range_unit = NoWheelComboBox()
        self.direction_range_unit.addItems(list(UNITS))
        self.direction_range_unit.setCurrentText("cm")
        row = QHBoxLayout()
        row.addWidget(self.direction_object_margin, 1)
        row.addWidget(self.direction_range_unit)
        form.addRow("Object Margin", row)
        form.addRow("Near Margin", self.direction_near_margin)
        for spin in (self.direction_object_margin, self.direction_near_margin):
            spin.setKeyboardTracking(False)
            spin.setToolTip("Offset from the raw base shape. Near Margin must be at least Object Margin.")
            spin.valueChanged.connect(self._direction_range_changed)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.direction_calibration_toggle = QPushButton("Calibration")
        self.direction_calibration_toggle.setCheckable(True)
        self.direction_apply_ranges = QPushButton("Apply to All")
        self.direction_apply_ranges.setToolTip("Copy margins and units; keep each object's calibration.")
        row.addWidget(self.direction_calibration_toggle)
        row.addWidget(self.direction_apply_ranges)
        layout.addLayout(row)
        self.direction_calibration_panel = QWidget()
        self.direction_calibration_panel.setVisible(False)
        calibration = QVBoxLayout(self.direction_calibration_panel)
        calibration.setContentsMargins(0, 2, 0, 0)
        targets = QHBoxLayout()
        self.direction_reference_targets = QButtonGroup(self)
        self.direction_reference_target_buttons = {}
        for target, label in (("line", "Line"), ("object", "Object"), ("top", "Top"), ("bottom", "Bottom")):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setProperty("object3dTarget", target)
            button.setToolTip("Draw or adjust the selected reference line." if target == "line" else f"Choose the reference used for the {target} margin.")
            self.direction_reference_targets.addButton(button)
            self.direction_reference_target_buttons[target] = button
            button.clicked.connect(lambda _checked=False, key=target: self._direction_set_reference_target(key))
            targets.addWidget(button)
        self.direction_reference_target_row = QWidget()
        targets.setContentsMargins(0, 0, 0, 0)
        self.direction_reference_target_row.setLayout(targets)
        calibration.addWidget(self.direction_reference_target_row)
        self.direction_reference_controls = {}
        self.direction_reference_editor = QWidget()
        editor = QFormLayout(self.direction_reference_editor)
        editor.setContentsMargins(0, 0, 0, 0)
        editor.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.direction_reference_assignment = QWidget()
        assignments = QFormLayout(self.direction_reference_assignment)
        assignments.setContentsMargins(0, 0, 0, 0)
        assignments.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for key, label in (("edit", "Reference"), ("reference_id", "Reference"),
                           ("top_reference_id", "Top Reference"), ("bottom_reference_id", "Bottom Reference")):
            combo = NoWheelComboBox()
            target_form = editor if key == "edit" else assignments
            target_form.addRow(label, combo)
            self.direction_reference_controls[key] = (combo, target_form.labelForField(combo))
            combo.currentIndexChanged.connect(lambda _index, name=key: self._direction_reference_binding(name))
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        for label, handler in (("Add Reference", self._direction_new_reference), ("Rename", self._direction_rename_reference),
                               ("Delete", self._direction_delete_reference)):
            button = QPushButton(label)
            button.clicked.connect(handler)
            actions.addWidget(button)
        editor.addRow(actions)
        self.direction_reference_length = NoWheelDoubleSpinBox()
        self.direction_reference_length.setKeyboardTracking(False)
        self.direction_reference_length.setToolTip("Actual distance between the reference line's two endpoints.")
        self.direction_reference_unit = NoWheelComboBox()
        self.direction_reference_unit.addItems(list(UNITS))
        length = QHBoxLayout()
        length.setContentsMargins(0, 0, 0, 0)
        length.addWidget(self.direction_reference_length, 1)
        length.addWidget(self.direction_reference_unit)
        editor.addRow("Length", length)
        self.direction_redraw_line = QPushButton("Redraw Selected Line")
        self.direction_redraw_line.setToolTip("Click two new endpoints. Changing target cancels an unfinished redraw.")
        editor.addRow(self.direction_redraw_line)
        calibration.addWidget(self.direction_reference_editor)
        calibration.addWidget(self.direction_reference_assignment)
        self.direction_quad_button = QPushButton("Set Four Points")
        self.direction_quad_button.setCheckable(True)
        self.direction_quad_button.setProperty("object3dTarget", "line")
        self.direction_quad_button.setToolTip("Click the four corners of a reference square in the video.")
        calibration.addWidget(self.direction_quad_button)
        self.direction_range_status = QLabel()
        self.direction_range_status.setWordWrap(True)
        calibration.addWidget(self.direction_range_status)
        layout.addWidget(self.direction_calibration_panel)
        self.direction_range_mode.currentIndexChanged.connect(self._direction_range_changed)
        self.direction_range_unit.currentIndexChanged.connect(self._direction_range_changed)
        self.direction_apply_ranges.clicked.connect(self._direction_apply_ranges_to_all)
        self.direction_calibration_toggle.toggled.connect(self._direction_calibration_visibility)
        self.direction_reference_length.valueChanged.connect(self._direction_reference_setting)
        self.direction_reference_unit.currentIndexChanged.connect(self._direction_reference_setting)
        self.direction_redraw_line.clicked.connect(lambda: self._direction_edit_line(redraw=True))
        self.direction_quad_button.clicked.connect(self._direction_edit_quad)
        return group

    def _direction_range_object(self):
        index = self.direction_objects.currentRow()
        return self.direction_masks[index] if 0 <= index < len(self.direction_masks) else None

    def _direction_rebuild_ranges(self):
        self._direction_targets, self._direction_range_errors, self._direction_range_paths = {}, {}, {}
        for item in self.direction_masks:
            try:
                self._direction_targets[item.name] = DirectionZones(item)
            except (ValueError, TypeError, OverflowError, cv2.error) as exc:
                self._direction_range_errors[item.name] = str(exc)
        self._direction_viewer_dirty = True

    @staticmethod
    def _direction_reference_measured(ref):
        if ref is None:
            return False
        try:
            ref.pixels_per_metre
            return True
        except ValueError:
            return False

    def _direction_reuse_video_references(self):
        """Copy this video's available rulers without changing the source tab."""
        if not self.direction_masks:
            return False
        refs = {}
        for mask in self.direction_masks:
            for ref in mask.references:
                if ref.id not in refs or (not self._direction_reference_measured(refs[ref.id]) and self._direction_reference_measured(ref)):
                    refs[ref.id] = copy.deepcopy(ref)
        workspace = self.object3d_workspace
        shape = self.direction_masks[0].mask.shape
        if workspace.video_key == self._direction_video_key and (workspace.height, workspace.width) == shape:
            for source in workspace.references:
                if source.id not in refs:
                    refs[source.id] = copy.deepcopy(source)
                elif not self._direction_reference_measured(refs[source.id]) and self._direction_reference_measured(source):
                    refs[source.id].points = list(source.points)
        changed = False
        available = list(refs.values())
        default = next((r for r in available if self._direction_reference_measured(r)), available[0] if available else None)
        for mask in self.direction_masks:
            if mask.references != available:
                mask.references = copy.deepcopy(available)
                changed = True
            keys = ("reference_id",) if mask.top is None else ("top_reference_id", "bottom_reference_id")
            for key in keys:
                if getattr(mask.ranges, key) is None and default:
                    setattr(mask.ranges, key, default.id)
                    changed = True
            if any(self._direction_reference_measured(r) for r in mask.references):
                mask.ranges.calibration_video_key = self._direction_video_key
        return changed

    def _direction_sync_ranges(self):
        if self._direction_range_syncing:
            return
        self._direction_range_syncing = True
        try:
            item = self._direction_range_object()
            for control in (self.direction_range_mode, self.direction_range_unit, self.direction_object_margin,
                            self.direction_near_margin, self.direction_calibration_toggle):
                control.setEnabled(item is not None)
            self.direction_apply_ranges.setVisible(len(self.direction_masks) > 1)
            if item is None:
                self.direction_calibration_panel.setVisible(False)
                if self._direction_active():
                    self.frame_viewer.direction_show_calibration = False
                    self.frame_viewer.reference_lines = []
                return
            spec = item.ranges
            self.direction_range_mode.model().item(2).setEnabled(item.top is None)
            self.direction_range_mode.setCurrentIndex(self.direction_range_mode.findData(spec.mode))
            self.direction_range_unit.setCurrentText(spec.unit)
            physical = spec.mode == "physical"
            self.direction_range_unit.setVisible(physical)
            for spin, value in ((self.direction_object_margin, spec.object_margin), (self.direction_near_margin, spec.near_margin)):
                spin.setDecimals(5 if physical else 0)
                spin.setRange(-100000 / UNITS[spec.unit] if physical else -300, 100000 / UNITS[spec.unit] if physical else 300)
                spin.setSingleStep(.1 if physical else 1)
                spin.setSuffix(f" {spec.unit}" if physical else " px")
                spin.setValue(value)
            self.direction_calibration_toggle.setVisible(spec.mode != "pixels")
            self.direction_calibration_panel.setVisible(spec.mode != "pixels" and self.direction_calibration_toggle.isChecked())
            valid_targets = {"line", "object"} if item.top is None else {"line", "top", "bottom"}
            if self._direction_reference_target not in valid_targets:
                self._direction_reference_target = "object" if item.top is None else "top"
            target = self._direction_reference_target
            for key, button in self.direction_reference_target_buttons.items():
                button.setVisible(key in valid_targets)
                button.setChecked(key == target)
            self.direction_reference_target_row.setVisible(physical)
            self.direction_reference_editor.setVisible(physical and target == "line")
            self.direction_reference_assignment.setVisible(physical and target != "line")
            ids = {r.id for r in item.references}
            if self._direction_reference_id not in ids:
                preferred = spec.reference_id if item.top is None else spec.top_reference_id
                self._direction_reference_id = preferred if preferred in ids else next(iter(ids), None)
            for key, (combo, label) in self.direction_reference_controls.items():
                visible = key == "edit" or key == {"object": "reference_id", "top": "top_reference_id", "bottom": "bottom_reference_id"}.get(target)
                combo.setVisible(visible)
                label.setVisible(visible)
                combo.clear()
                combo.addItem("Select a reference...", None)
                for ref in item.references:
                    status = "Measured" if self._direction_reference_measured(ref) else "Needs line"
                    combo.addItem(f"{ref.name} - {status}", ref.id)
                selected = self._direction_reference_id if key == "edit" else getattr(spec, key)
                combo.setCurrentIndex(max(0, combo.findData(selected)))
            ref = self._direction_edit_reference()
            for widget in (self.direction_reference_length, self.direction_reference_unit, self.direction_redraw_line):
                widget.setEnabled(ref is not None)
            if ref:
                self.direction_reference_unit.setCurrentText(ref.unit)
                spin = self.direction_reference_length
                spin.setDecimals({"m": 8, "cm": 6, "mm": 5}[ref.unit])
                spin.setRange(1e-8 / UNITS[ref.unit], 100000 / UNITS[ref.unit])
                spin.setValue(ref.length)
            self.direction_quad_button.setVisible(spec.mode == "geometric")
            quad = bool(self._direction_calibration and self._direction_calibration["kind"] == "quad")
            self.direction_quad_button.setText("Cancel Four Points" if quad else "Set Four Points")
            self.direction_quad_button.setChecked(quad)
            text = self._direction_range_errors.get(item.name, "")
            if physical and target == "line":
                if self._direction_calibration and self._direction_calibration.get("pending", False):
                    text = "Click two endpoints or drag an endpoint."
                elif self._direction_reference_measured(ref):
                    text = f"{math.dist(*ref.points):.2f} px = {ref.length:g} {ref.unit}. Drag an endpoint to adjust."
                else:
                    text = "Select Line to draw a reference." if ref is None else "Click two endpoints in the video."
            elif quad:
                text = "Click the four corners of a reference square."
            elif not text and spec.mode == "geometric":
                text = f"{len(spec.quad_points)} / 4 perspective points"
            self.direction_range_status.setText(text)
            self.direction_range_status.setVisible(bool(text))
        finally:
            self._direction_range_syncing = False
        self._direction_sync_calibration_viewer()

    def _direction_range_changed(self, *_args):
        if self._direction_range_syncing:
            return
        item = self._direction_range_object()
        if item is None:
            return
        self._direction_cancel_calibration()
        spec = item.ranges
        mode, unit = self.direction_range_mode.currentData(), self.direction_range_unit.currentText()
        if mode != spec.mode:
            profiles = getattr(spec, "_mode_values", {})
            profiles[spec.mode] = (spec.object_margin, spec.near_margin)
            spec.object_margin, spec.near_margin = profiles.get(mode, (0, 2 if mode == "physical" else 20))
            spec._mode_values = profiles
            spec.mode = mode
        elif unit != spec.unit and mode == "physical":
            spec.object_margin = convert_length(spec.object_margin, spec.unit, unit)
            spec.near_margin = convert_length(spec.near_margin, spec.unit, unit)
        else:
            spec.object_margin, spec.near_margin = self.direction_object_margin.value(), self.direction_near_margin.value()
        spec.unit = unit
        self._direction_ranges_updated()

    def _direction_ranges_updated(self):
        self._direction_rebuild_ranges()
        self._direction_sync_ranges()
        self._direction_options_changed()

    def _direction_apply_ranges_to_all(self):
        item = self._direction_range_object()
        if item is None:
            return
        if item.ranges.mode == "geometric" and any(m.top is not None for m in self.direction_masks):
            QMessageBox.information(self, "Direction Ranges", "Use Pixels or Physical margins when applying ranges to 3D faces.")
            return
        for other in self.direction_masks:
            for key in ("mode", "unit", "object_margin", "near_margin"):
                setattr(other.ranges, key, getattr(item.ranges, key))
        self._direction_ranges_updated()

    def _direction_calibration_visibility(self, visible):
        if not visible:
            self._direction_cancel_calibration()
        item = self._direction_range_object()
        if visible and item and item.ranges.mode == "physical" and not item.references:
            self._direction_create_reference("Reference 1")
            self._direction_reference_target = "line"
            self._direction_rebuild_ranges()
        self._direction_sync_ranges()
        self._direction_options_changed()

    def _direction_set_reference_target(self, target):
        item = self._direction_range_object()
        if item is None:
            return
        self._direction_cancel_calibration()
        self._direction_reference_target = target
        if target == "line" and not item.references:
            self._direction_create_reference("Reference 1")
            self._direction_rebuild_ranges()
        elif target != "line":
            key = {"object": "reference_id", "top": "top_reference_id", "bottom": "bottom_reference_id"}[target]
            selected = getattr(item.ranges, key)
            if selected:
                self._direction_reference_id = selected
        self._direction_sync_ranges()
        self._direction_options_changed()

    def _direction_edit_reference(self):
        item = self._direction_range_object()
        return next((ref for ref in item.references if ref.id == self._direction_reference_id), None) if item else None

    def _direction_reference_binding(self, key):
        if self._direction_range_syncing:
            return
        self._direction_cancel_calibration()
        item = self._direction_range_object()
        if item is None:
            return
        selected = self.direction_reference_controls[key][0].currentData()
        if key != "edit":
            setattr(item.ranges, key, selected)
        self._direction_reference_id = selected
        self._direction_ranges_updated()

    def _direction_reference_setting(self, *_args):
        if self._direction_range_syncing:
            return
        ref = self._direction_edit_reference()
        if ref is None:
            return
        unit = self.direction_reference_unit.currentText()
        length = convert_length(ref.length, ref.unit, unit) if unit != ref.unit else self.direction_reference_length.value()
        for item in self.direction_masks:
            for candidate in item.references:
                if candidate.id == ref.id:
                    candidate.length, candidate.unit = length, unit
        self._direction_ranges_updated()

    def _direction_create_reference(self, name):
        ref = LengthReference(name)
        for mask in self.direction_masks:
            mask.references.append(copy.deepcopy(ref))
        item = self._direction_range_object()
        keys = ("reference_id",) if item.top is None else ("top_reference_id", "bottom_reference_id")
        for key in keys:
            if getattr(item.ranges, key) is None:
                setattr(item.ranges, key, ref.id)
        self._direction_reference_id = ref.id
        return ref

    def _direction_new_reference(self):
        item = self._direction_range_object()
        if item is None:
            return
        name, ok = QInputDialog.getText(self, "New Direction Reference", "Name", text=f"Reference {len(item.references) + 1}")
        if not ok or not name.strip():
            return
        if any(ref.name == name.strip() for ref in item.references):
            QMessageBox.information(self, "Reference", "Choose a unique reference name.")
            return
        self._direction_cancel_calibration()
        self._direction_create_reference(name.strip())
        self._direction_reference_target = "line"
        self._direction_ranges_updated()

    def _direction_rename_reference(self):
        ref = self._direction_edit_reference()
        if ref:
            name, ok = QInputDialog.getText(self, "Rename Reference", "Name", text=ref.name)
            if ok and name.strip():
                if any(r.id != ref.id and r.name == name.strip() for r in self._direction_range_object().references):
                    QMessageBox.information(self, "Reference", "Choose a unique reference name.")
                    return
                for mask in self.direction_masks:
                    for candidate in mask.references:
                        if candidate.id == ref.id:
                            candidate.name = name.strip()
                self._direction_sync_ranges()

    def _direction_delete_reference(self):
        ref = self._direction_edit_reference()
        if ref is None:
            return
        if any(ref.id in (m.ranges.reference_id, m.ranges.top_reference_id, m.ranges.bottom_reference_id) for m in self.direction_masks):
            QMessageBox.information(self, "Reference", "This reference is in use. Assign another reference before deleting it.")
            return
        for mask in self.direction_masks:
            mask.references = [r for r in mask.references if r.id != ref.id]
        self._direction_sync_ranges()

    def _direction_edit_line(self, _checked=False, *, redraw=False):
        self._direction_cancel_calibration()
        item, ref = self._direction_range_object(), self._direction_edit_reference()
        if not self._direction_active() or item is None or ref is None:
            return
        self._direction_reference_target = "line"
        self._direction_calibration = {"kind": "line", "object": item.name, "reference": ref.id,
                                       "pending": redraw or not self._direction_reference_measured(ref)}
        self.frame_viewer.reference_pick_points = [] if redraw else list(ref.points)
        self._direction_sync_ranges()
        self._direction_options_changed()

    def _direction_edit_quad(self, _checked=False):
        if self._direction_calibration is not None:
            self._direction_cancel_calibration()
        else:
            item = self._direction_range_object()
            if item is None or not self._direction_active():
                return
            self._direction_calibration = {"kind": "quad", "object": item.name}
            self.frame_viewer.direction_quad_points = []
            self.frame_viewer.set_mode("direction_quad")
        self._direction_sync_ranges()
        self._refresh_output_ui()

    def _direction_cancel_calibration(self):
        self._direction_calibration = None
        if hasattr(self, "frame_viewer"):
            viewer = self.frame_viewer
            viewer.reference_pick_active = False
            viewer._reference_drag_index = None
            viewer.direction_quad_points = []
            viewer._direction_quad_drag = None
            if self._direction_active():
                viewer.set_mode("direction")

    def _direction_reference_line(self, points):
        context = self._direction_calibration
        if not self._direction_active() or not context or context["kind"] != "line":
            return
        for mask in self.direction_masks:
            for ref in mask.references:
                if ref.id == context["reference"]:
                    ref.points = list(points)
                    mask.ranges.calibration_video_key = self._direction_video_key
        self._direction_cancel_calibration()
        self._direction_ranges_updated()

    def _direction_quad_completed(self, points):
        context = self._direction_calibration
        if not self._direction_active() or not context or context["kind"] != "quad":
            return
        item = next((m for m in self.direction_masks if m.name == context["object"]), None)
        if item:
            item.ranges.quad_points = list(points)
        self._direction_cancel_calibration()
        self._direction_ranges_updated()

    def _direction_sync_calibration_viewer(self):
        if not self._direction_active():
            return
        item = self._direction_range_object()
        viewer = self.frame_viewer
        visible = bool(item and item.ranges.mode != "pixels" and self.direction_calibration_toggle.isChecked())
        viewer.direction_show_calibration = visible
        ref = self._direction_edit_reference()
        viewer.reference_lines = [(r.id, r.name, list(r.points)) for r in item.references] if item else []
        viewer.reference_selected_id = ref.id if ref else None
        editing = visible and item.ranges.mode == "physical" and self._direction_reference_target == "line" and ref is not None
        if editing:
            context = self._direction_calibration
            if context is None or context["kind"] != "line" or context["reference"] != ref.id:
                self._direction_calibration = {"kind": "line", "object": item.name, "reference": ref.id,
                                               "pending": not self._direction_reference_measured(ref)}
                viewer.reference_pick_points = list(ref.points)
            viewer.reference_pick_active = True
            viewer.set_mode("reference_line")
        elif self._direction_calibration and self._direction_calibration["kind"] == "line":
            self._direction_cancel_calibration()
        if item and item.ranges.mode == "geometric" and self._direction_calibration is None:
            viewer.direction_quad_points = list(item.ranges.quad_points)
        viewer.update()
