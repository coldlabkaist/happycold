from __future__ import annotations

import copy
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPushButton, QRadioButton, QScrollArea, QSlider,
    QSizePolicy, QVBoxLayout, QWidget,
)

from mask_io import read_mask_png
from mask_tools import build_circle_mask, build_polygon_mask, merge_mask, paint_brush_segment, parse_draw_shape_payload
from occlusion_3d import (
    LengthReference, ObjectFace, ObjectWorkspace, OcclusionObject, UNITS,
    convert_length, load_bundle, new_id, save_bundle,
)
from shared import MASK_PALETTE, MaskRecord, MaskTransformSource, build_occlusion_dataframe, mask_polygon_geometry
from ui_controls import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox


class Occlusion3DTabMixin:
    def _build_occlusion_3d_tab(self) -> QWidget:
        self.object3d_workspace = ObjectWorkspace()
        self._object3d_videos = {}
        self._object3d_video_key = None
        self._object3d_selected_id = None
        self._object3d_face = "top"
        self._object3d_edit_target = "line"
        self._object3d_refreshing = False
        self._object3d_stroke_open = False
        self._object3d_last_brush_refresh = 0.0
        self._object3d_transform_source = None
        self._object3d_render_cache = {}
        self._object3d_undo = []

        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(10, 10, 10, 10)
        scroll = QScrollArea()
        self.object3d_scroll = scroll
        scroll.setObjectName("occlusion3DControlsScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        objects = QGroupBox("1. Objects")
        self.object3d_objects_group = objects
        form = QFormLayout(objects)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.object3d_combo = NoWheelComboBox()
        self.object3d_combo.setToolTip("Choose the object that receives edits.")
        form.addRow(self.object3d_combo)
        actions = QGridLayout()
        for index, (label, handler) in enumerate((
            ("Add Object", self._object3d_add_object), ("Duplicate", self._object3d_duplicate_object),
            ("Rename", self._object3d_rename_object), ("Delete", self._object3d_delete_object),
        )):
            button = QPushButton(label)
            button.clicked.connect(handler)
            actions.addWidget(button, 0, index)
        form.addRow(actions)
        self.object3d_include = QCheckBox("Include in CSV")
        form.addRow(self.object3d_include)
        outer.addWidget(objects)

        target_row = QHBoxLayout()
        self.object3d_target_group = QButtonGroup(self)
        self.object3d_target_buttons = {}
        for target, label in (("line", "Line"), ("top", "Top"), ("bottom", "Bottom")):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setProperty("object3dTarget", target)
            button.setToolTip(f"Edit the selected reference line." if target == "line" else f"Edit the selected object's {target} face.")
            button.setChecked(target == "line")
            self.object3d_target_group.addButton(button)
            self.object3d_target_buttons[target] = button
            button.clicked.connect(lambda _checked=False, key=target: self._object3d_set_edit_target(key))
            target_row.addWidget(button)
        outer.addLayout(target_row)

        def text(message):
            label = QLabel(message)
            label.setWordWrap(True)
            return label

        references = QGroupBox("2. Reference Line")
        self.object3d_references_group = references
        form = QFormLayout(references)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.object3d_reference_combo = NoWheelComboBox()
        self.object3d_reference_combo.setToolTip("Choose Line, then click two endpoints in the video. Drag a handle to adjust the line.")
        form.addRow("Current Reference", self.object3d_reference_combo)
        actions = QGridLayout()
        for index, (label, handler) in enumerate((
            ("Add Reference", self._object3d_add_reference),
            ("Rename", self._object3d_rename_reference),
            ("Delete", self._object3d_delete_reference),
        )):
            button = QPushButton(label)
            button.clicked.connect(handler)
            actions.addWidget(button, 0, index)
        form.addRow(actions)
        row = QHBoxLayout()
        self.object3d_reference_length = NoWheelDoubleSpinBox()
        self.object3d_reference_length.setDecimals(5)
        self.object3d_reference_length.setKeyboardTracking(False)
        self.object3d_reference_length.setRange(0.00001, 100000)
        self.object3d_reference_unit = NoWheelComboBox()
        self.object3d_reference_unit.addItems(list(UNITS))
        row.addWidget(self.object3d_reference_length, 1)
        row.addWidget(self.object3d_reference_unit)
        form.addRow("Real Distance", row)
        self.object3d_redraw_button = QPushButton("Redraw Selected Line")
        form.addRow(self.object3d_redraw_button)
        self.object3d_reference_status = text("Add a reference to begin.")
        form.addRow(self.object3d_reference_status)
        layout.addWidget(references)

        self.object3d_face_controls = {}
        for key in ("top", "bottom"):
            group = QGroupBox(f"2. {key.title()} Face")
            face_form = QFormLayout(group)
            face_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
            draw = QRadioButton("Draw")
            transform = QRadioButton("Transform")
            draw.setChecked(True)
            draw.setToolTip("Draw mode (D).")
            transform.setToolTip("Transform mode (T): drag to move, E/R to resize, Ctrl+E/R to rotate.")
            mode_group = QButtonGroup(group)
            mode_group.addButton(draw)
            mode_group.addButton(transform)
            mode_row = QHBoxLayout()
            mode_row.addWidget(draw)
            mode_row.addWidget(transform)
            mode_row.addStretch(1)
            face_form.addRow("Edit Mode", mode_row)
            tool = NoWheelComboBox()
            for label, mode in (("Rectangle Drag", "occ_rect_drag"), ("Square Drag", "occ_rect_drag"),
                                ("Rectangle 4 Points", "occ_rect"), ("Polygon", "occ_polygon"),
                                ("Circle Drag", "occ_circle"), ("Brush", "occ_free")):
                tool.addItem(label, mode)
            tool.setToolTip("Hold Shift while dragging a rectangle for a square. Right-click to finish a polygon.")
            face_form.addRow("Mask Tool", tool)
            add = QRadioButton("Add")
            erase = QRadioButton("Erase")
            add.setChecked(True)
            action_group = QButtonGroup(group)
            action_group.addButton(add)
            action_group.addButton(erase)
            action_row = QHBoxLayout()
            action_row.addWidget(add)
            action_row.addWidget(erase)
            action_row.addStretch(1)
            face_form.addRow("Draw Action", action_row)
            brush = NoWheelSpinBox()
            brush.setRange(1, 200)
            brush.setValue(12)
            brush.setSuffix(" px")
            brush.setFixedWidth(88)
            brush_slider = QSlider(Qt.Orientation.Horizontal)
            brush_slider.setRange(1, 200)
            brush_slider.setValue(12)
            brush_row = QHBoxLayout()
            brush_row.addWidget(brush_slider, 1)
            brush_row.addWidget(brush)
            face_form.addRow("Brush Size", brush_row)
            reference = NoWheelComboBox()
            reference.setToolTip(f"Choose the reference for this {key} face independently.")
            face_form.addRow("Reference", reference)
            margin = NoWheelDoubleSpinBox()
            margin.setDecimals(5)
            margin.setKeyboardTracking(False)
            margin.setRange(-100000, 100000)
            unit = NoWheelComboBox()
            unit.addItems(list(UNITS))
            row = QHBoxLayout()
            row.addWidget(margin, 1)
            row.addWidget(unit)
            margin.setToolTip("Positive margins expand the face; negative margins shrink it.")
            face_form.addRow("Mask Margin", row)
            status = text("Draw this face to continue.")
            face_form.addRow(status)
            clear = QPushButton(f"Clear {key.title()} Face")
            face_form.addRow(clear)
            self.object3d_face_controls[key] = dict(
                group=group, tool=tool, draw=draw, transform=transform, add=add, erase=erase,
                mode_group=mode_group, action_group=action_group, brush=brush, brush_slider=brush_slider,
                reference=reference, margin=margin, unit=unit, status=status,
            )
            for radio in (draw, transform, add, erase):
                radio.toggled.connect(lambda checked, face=key: self._object3d_activate_face(face) if checked else None)
            tool.currentIndexChanged.connect(lambda _value, face=key: self._object3d_activate_face(face))
            brush.valueChanged.connect(brush_slider.setValue)
            brush_slider.valueChanged.connect(brush.setValue)
            brush.valueChanged.connect(lambda _value, face=key: self._object3d_activate_face(face))
            reference.currentIndexChanged.connect(lambda _index, face=key: self._object3d_face_setting(face))
            margin.valueChanged.connect(lambda _value, face=key: self._object3d_face_setting(face))
            unit.currentIndexChanged.connect(lambda _index, face=key: self._object3d_face_setting(face))
            clear.clicked.connect(lambda _checked=False, face=key: self._object3d_clear_face(face))
            layout.addWidget(group)

        self.object3d_summary = QLabel("Not ready")
        layout.addWidget(self.object3d_summary)
        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        row = QGridLayout()
        for index, (label, handler, tooltip) in enumerate((
            ("Import Face", self._object3d_import_png, "Import one PNG into the active face."),
            ("Import Folder", self._object3d_import_folder, "Import editable 3D objects and references."),
            ("Export Masks", self._object3d_export, "Export final masks, editable faces, and references."),
        )):
            button = QPushButton(label)
            button.setToolTip(tooltip)
            button.clicked.connect(handler)
            if index == 2:
                row.addWidget(button, 1, 0, 1, 2)
            else:
                row.addWidget(button, 0, index)
        outer.addLayout(row)
        self.object3d_reference_combo.currentIndexChanged.connect(self._object3d_reference_selected)
        self.object3d_reference_length.valueChanged.connect(self._object3d_reference_setting)
        self.object3d_reference_unit.currentIndexChanged.connect(self._object3d_reference_setting)
        self.object3d_redraw_button.clicked.connect(lambda: self._object3d_measure(replace=True))
        self.object3d_combo.currentIndexChanged.connect(self._object3d_object_selected)
        self.object3d_include.toggled.connect(self._object3d_include_changed)
        for combo in tab.findChildren(NoWheelComboBox):
            combo.setMinimumWidth(0)
            combo.setMinimumContentsLength(16)
            combo.setSizeAdjustPolicy(NoWheelComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        for combo in (self.object3d_reference_unit, *[c["unit"] for c in self.object3d_face_controls.values()]):
            combo.setMinimumContentsLength(2)
            combo.setMinimumWidth(64)
            combo.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        return tab

    def _object3d_active(self):
        return hasattr(self, "mode_tabs") and self.mode_tabs.currentIndex() == self.TAB_OCCLUSION_3D

    def _object3d_object(self):
        return next((o for o in self.object3d_workspace.objects if o.id == self._object3d_selected_id), None)

    def _object3d_reference(self):
        key = self.object3d_reference_combo.currentData()
        return next((r for r in self.object3d_workspace.references if r.id == key), None)

    def _object3d_push_undo(self):
        self._object3d_undo.append((copy.deepcopy(self.object3d_workspace), self._object3d_selected_id))
        del self._object3d_undo[:-self._mask_undo_limit]

    def _object3d_undo_edit(self):
        if self._object3d_undo:
            self.frame_viewer.reference_pick_active = False
            self.frame_viewer._reference_drag_index = None
            self.frame_viewer.clear_occ_rect_points()
            self.frame_viewer.clear_occ_circle()
            self._object3d_stroke_open = False
            self.object3d_workspace, self._object3d_selected_id = self._object3d_undo.pop()
            self._object3d_changed()
            self._sync_object3d_mode()
        else:
            self.statusBar().showMessage("Nothing to undo in this tab.", 2500)

    def _object3d_require_video(self):
        if self.video_state is None:
            self.statusBar().showMessage("Load a video first.", 3000)
            return False
        return True

    def _object3d_name(self, title, initial, names):
        name, ok = QInputDialog.getText(self, title, "Name", text=initial)
        if not ok:
            return None
        name = name.strip()
        if not name or name in names:
            QMessageBox.information(self, title, "Enter a non-empty, unique name.")
            return None
        return name

    def _object3d_add_reference(self):
        if not self._object3d_require_video():
            return
        refs = self.object3d_workspace.references
        name = self._object3d_name("Add Reference", f"Reference {len(refs) + 1}", [r.name for r in refs])
        if name:
            self._object3d_push_undo()
            reference = LengthReference(name)
            refs.append(reference)
            self._object3d_changed(reference_id=reference.id)
            self._object3d_set_edit_target("line")

    def _object3d_rename_reference(self):
        reference = self._object3d_reference()
        if reference:
            name = self._object3d_name("Rename Reference", reference.name,
                                       [r.name for r in self.object3d_workspace.references if r is not reference])
            if name:
                self._object3d_push_undo()
                reference.name = name
                self._object3d_changed()

    def _object3d_delete_reference(self):
        reference = self._object3d_reference()
        if reference is None:
            return
        if any(face.reference_id == reference.id for o in self.object3d_workspace.objects for face in (o.top, o.bottom)):
            self.statusBar().showMessage("This reference is in use. Assign another reference to its faces before deleting it.", 6000)
            return
        self._object3d_push_undo()
        self.object3d_workspace.references.remove(reference)
        self._object3d_changed()
        self._sync_object3d_mode()

    def _object3d_reference_selected(self, _index=0):
        if not self._object3d_refreshing:
            self.frame_viewer.reference_pick_active = False
            self._object3d_refresh()
            self._sync_object3d_mode()

    def _object3d_reference_setting(self, _value=None):
        reference = self._object3d_reference()
        if reference is None or self._object3d_refreshing:
            return
        self._object3d_push_undo()
        unit = self.object3d_reference_unit.currentText()
        reference.length = (convert_length(reference.length, reference.unit, unit)
                            if unit != reference.unit else self.object3d_reference_length.value())
        reference.unit = unit
        self._object3d_changed()

    def _object3d_measure(self, _checked=False, *, replace=False):
        """Enter the shared Line target; redrawing also invalidates the old measurement."""
        reference = self._object3d_reference()
        if reference is None or not self._object3d_active():
            return
        if replace:
            self._object3d_push_undo()
            reference.points = []
            self._object3d_render_cache.clear()
        self._object3d_set_edit_target("line")

    def _object3d_reference_line(self, points):
        reference = self._object3d_reference()
        if self._object3d_active() and self._object3d_edit_target == "line" and reference:
            self._object3d_push_undo()
            reference.points = list(points)
            self._object3d_changed()

    def _object3d_add_object(self):
        if not self._object3d_require_video():
            return
        ws = self.object3d_workspace
        name = self._object3d_name("Add Object", f"Object {len(ws.objects) + 1}", [o.name for o in ws.objects])
        if name:
            self._object3d_push_undo()
            reference = self._object3d_reference()
            shape = (ws.height, ws.width)
            top = ObjectFace(np.zeros(shape, np.uint8), reference_id=reference.id if reference else None)
            bottom = copy.deepcopy(top)
            obj = OcclusionObject(name, top, bottom, MASK_PALETTE[len(ws.objects) % len(MASK_PALETTE)].name())
            ws.objects.append(obj)
            self._object3d_selected_id = obj.id
            self._object3d_changed()
            self._object3d_activate_face("top")

    def _object3d_duplicate_object(self):
        obj = self._object3d_object()
        if obj:
            name = self._object3d_name("Duplicate Object", f"{obj.name} copy", [o.name for o in self.object3d_workspace.objects])
            if name:
                self._object3d_push_undo()
                duplicate = copy.deepcopy(obj)
                duplicate.id, duplicate.name = new_id(), name
                duplicate.color = MASK_PALETTE[len(self.object3d_workspace.objects) % len(MASK_PALETTE)].name()
                self.object3d_workspace.objects.append(duplicate)
                self._object3d_selected_id = duplicate.id
                self._object3d_changed()

    def _object3d_rename_object(self):
        obj = self._object3d_object()
        if obj:
            name = self._object3d_name("Rename Object", obj.name, [o.name for o in self.object3d_workspace.objects if o is not obj])
            if name:
                self._object3d_push_undo()
                obj.name = name
                self._object3d_changed()

    def _object3d_delete_object(self):
        obj = self._object3d_object()
        if obj:
            self._object3d_push_undo()
            self.object3d_workspace.objects.remove(obj)
            self._object3d_selected_id = None
            self._object3d_changed()

    def _object3d_object_selected(self, _index=0):
        if not self._object3d_refreshing:
            self.frame_viewer.reference_pick_active = False
            self.frame_viewer.clear_occ_rect_points()
            self.frame_viewer.clear_occ_circle()
            self._object3d_selected_id = self.object3d_combo.currentData()
            self._object3d_transform_source = None
            self._object3d_stroke_open = False
            self._object3d_refresh()
            self._sync_object3d_mode()

    def _object3d_include_changed(self, checked):
        obj = self._object3d_object()
        if obj and not self._object3d_refreshing:
            self._object3d_push_undo()
            obj.included = checked
            self._object3d_changed()

    def _object3d_face_setting(self, face):
        obj = self._object3d_object()
        if obj is None or self._object3d_refreshing:
            return
        self._object3d_push_undo()
        controls = self.object3d_face_controls[face]
        item = getattr(obj, face)
        item.reference_id = controls["reference"].currentData()
        unit = controls["unit"].currentText()
        item.margin = (convert_length(item.margin, item.unit, unit)
                       if unit != item.unit else controls["margin"].value())
        item.unit = unit
        obj.same_reference = False
        self._object3d_changed(object_id=obj.id)

    def _object3d_activate_face(self, face):
        self._object3d_set_edit_target(face)

    def _object3d_set_edit_target(self, target):
        if self._object3d_refreshing or target not in {"line", "top", "bottom"}:
            return
        self._object3d_edit_target = target
        if target != "line":
            self._object3d_face = target
        self._object3d_transform_source = None
        self._object3d_stroke_open = False
        viewer = self.frame_viewer
        viewer._reference_drag_index = None
        viewer._free_dragging = False
        viewer._free_last_point = None
        viewer._occ_transform_erase_dragging = False
        viewer._occ_transform_erase_last_point = None
        viewer._cancel_occ_transform_drag()
        viewer.clear_occ_rect_points()
        viewer.clear_occ_circle()
        self._sync_object3d_mode()
        self._object3d_refresh()
        self._refresh_output_ui()

    def _object3d_refresh_edit_target(self):
        obj = self._object3d_object()
        for target, button in self.object3d_target_buttons.items():
            # Line also opens the reference list when it is empty, so an object
            # created first can still add its first measurement here.
            button.setEnabled(self.video_state is not None if target == "line" else obj is not None)
            button.setChecked(target == self._object3d_edit_target)
        self.object3d_references_group.setVisible(self._object3d_edit_target == "line")
        for face, controls in self.object3d_face_controls.items():
            controls["group"].setVisible(self._object3d_edit_target == face)

    def _object3d_clear_face(self, face):
        obj = self._object3d_object()
        if obj:
            self._object3d_push_undo()
            getattr(obj, face).mask[:] = 0
            self._object3d_changed()

    def _object3d_changed(self, *, reference_id=None, object_id=None):
        if object_id is None:
            self._object3d_render_cache.clear()
        else:
            self._object3d_render_cache.pop(object_id, None)
        self._object3d_transform_source = None
        self._object3d_refresh(reference_id=reference_id)
        self._refresh_output_ui()

    def _object3d_result(self, obj):
        if obj.id not in self._object3d_render_cache:
            errors = self.object3d_workspace.errors(obj)
            if errors:
                self._object3d_render_cache[obj.id] = (errors, None, None)
            else:
                mask, outline = self.object3d_workspace.render_with_outline(obj)
                self._object3d_render_cache[obj.id] = ([], mask, mask_polygon_geometry(outline, shape="polygon"))
        return self._object3d_render_cache[obj.id]

    @staticmethod
    def _object3d_configure_length_input(widget, unit, *, positive=False):
        # A single physical range/precision across units prevents display clamping.
        factor = UNITS[unit]
        widget.setDecimals({"m": 8, "cm": 6, "mm": 5}[unit])
        widget.setRange(1e-8 / factor if positive else -100000 / factor, 100000 / factor)
        widget.setSingleStep(0.01 if unit == "m" else 0.1)

    def _object3d_refresh(self, *, reference_id=None):
        if self._object3d_refreshing:
            return
        self._object3d_refreshing = True
        try:
            ws = self.object3d_workspace
            self.object3d_references_group.setToolTip(
                f"References for {self.video_state.path.name}." if self.video_state else "Load a video first.")
            reference_id = reference_id or self.object3d_reference_combo.currentData()
            self.object3d_reference_combo.clear()
            for reference in ws.references:
                try:
                    reference.pixels_per_metre
                    status = "Measured"
                except ValueError:
                    status = "Needs measurement"
                self.object3d_reference_combo.addItem(f"{reference.name} — {status}", reference.id)
            index = self.object3d_reference_combo.findData(reference_id)
            self.object3d_reference_combo.setCurrentIndex(max(0, index) if ws.references else -1)
            reference = self._object3d_reference()
            for widget in (self.object3d_reference_length, self.object3d_reference_unit):
                widget.setEnabled(reference is not None)
            self.object3d_redraw_button.setEnabled(reference is not None and bool(reference.points))
            if reference:
                self.object3d_reference_unit.setCurrentText(reference.unit)
                self._object3d_configure_length_input(self.object3d_reference_length, reference.unit, positive=True)
                self.object3d_reference_length.setValue(reference.length)
                try:
                    reference.pixels_per_metre
                    self.object3d_reference_status.setText(f"{np.linalg.norm(np.subtract(*reference.points)):.2f} px = {reference.length:g} {reference.unit}.")
                except ValueError as exc:
                    self.object3d_reference_status.setText(str(exc))
            else:
                self.object3d_reference_status.setText("Add a reference to begin.")
            self.object3d_combo.clear()
            for obj in ws.objects:
                errors, _mask, _geometry = self._object3d_result(obj)
                status = "Excluded" if not obj.included else "Needs setup" if errors else "Ready"
                self.object3d_combo.addItem(f"{obj.name} — {status}", obj.id)
            index = self.object3d_combo.findData(self._object3d_selected_id)
            self.object3d_combo.setCurrentIndex(max(0, index) if ws.objects else -1)
            self._object3d_selected_id = self.object3d_combo.currentData()
            obj = self._object3d_object()
            self.object3d_include.setEnabled(obj is not None)
            self.object3d_include.setChecked(obj.included if obj else False)
            self._object3d_refresh_edit_target()
            for key, controls in self.object3d_face_controls.items():
                controls["group"].setEnabled(obj is not None)
                combo = controls["reference"]
                combo.clear()
                combo.addItem("Select a reference…", None)
                for item in ws.references:
                    combo.addItem(item.name, item.id)
                if obj:
                    face = getattr(obj, key)
                    controls["unit"].setCurrentText(face.unit)
                    self._object3d_configure_length_input(controls["margin"], face.unit)
                    controls["margin"].setValue(face.margin)
                    combo.setCurrentIndex(max(0, combo.findData(face.reference_id)))
                    combo.setEnabled(True)
                    try:
                        ws.adjusted_face(obj, key)
                        controls["status"].setText(f"Margin: {ws.margin_pixels(obj, key):+.2f} px")
                    except ValueError as exc:
                        controls["status"].setText(str(exc))
            self._object3d_refresh_readiness()
            self._object3d_refresh_viewer()
        finally:
            self._object3d_refreshing = False

    def _object3d_refresh_viewer(self, *, refresh=True):
        if not self._object3d_active():
            return
        records = {}
        selected = self._object3d_object()
        for obj in self.object3d_workspace.objects:
            errors, result, geometry = self._object3d_result(obj)
            if result is not None and obj.included:
                records[f"object:{obj.id}"] = MaskRecord(obj.name, QColor(obj.color), result, geometry=geometry)
        if selected:
            for face, color in (("top", "#06b6d4"), ("bottom", "#f59e0b")):
                key = f"face:{face}"
                # The transform source is the unadjusted face; margins are rebuilt after edits.
                records[key] = MaskRecord(f"{selected.name} / {face.title()} Face", QColor(color), getattr(selected, face).mask)
        selected_face = f"face:{self._object3d_face}" if selected and self._object3d_edit_target != "line" else None
        self.frame_viewer.set_mask_records(records, selected_face, refresh=refresh)
        self.frame_viewer.reference_lines = [(r.id, r.name, list(r.points)) for r in self.object3d_workspace.references]
        reference = self._object3d_reference()
        self.frame_viewer.reference_selected_id = reference.id if reference else None
        self.frame_viewer.update()

    def _sync_object3d_mode(self):
        if not self._object3d_active():
            return
        controls = self.object3d_face_controls[self._object3d_face]
        viewer = self.frame_viewer
        viewer.set_occ_margin_pick_mode(False)
        viewer.object3d_active = True
        viewer.set_margin_value(0)
        editing_line = self._object3d_edit_target == "line"
        reference = self._object3d_reference()
        viewer.reference_pick_active = editing_line and reference is not None
        viewer.reference_pick_points = list(reference.points) if reference and editing_line else []
        viewer.square_drag_enabled = not editing_line and controls["tool"].currentText() == "Square Drag"
        viewer.set_mode("reference_line" if editing_line else controls["tool"].currentData())
        viewer.set_occ_transform_mode(not editing_line and controls["transform"].isChecked())
        viewer.set_occlusion_draw_mode(add=controls["add"].isChecked(),
                                       brush_radius=controls["brush"].value())
        self._object3d_refresh_viewer()

    def _object3d_video_changed(self, video_path, width, height):
        if self._object3d_video_key is not None:
            self._object3d_videos[self._object3d_video_key] = (
                copy.deepcopy(self.object3d_workspace), self._object3d_selected_id,
                copy.deepcopy(self._object3d_undo),
            )
        key = str(Path(video_path).resolve())
        if key in self._object3d_videos:
            self.object3d_workspace, self._object3d_selected_id, self._object3d_undo = copy.deepcopy(self._object3d_videos[key])
            if (self.object3d_workspace.width, self.object3d_workspace.height) != (width, height):
                self.object3d_workspace = self.object3d_workspace.for_new_video(width, height)
                self._object3d_undo = []
        else:
            ws = self.object3d_workspace
            self.object3d_workspace = ws.for_new_video(width, height) if ws.width else ObjectWorkspace(width, height)
            self._object3d_undo = []
        self._object3d_video_key = key
        self.object3d_workspace.video_key = key
        self._object3d_stroke_open = False
        self.frame_viewer.reference_pick_active = False
        self._object3d_changed()
        self._sync_object3d_mode()

    def _dispatch_occlusion_edit(self, method, *args):
        if not self._object3d_active():
            return getattr(self, method)(*args)
        obj = self._object3d_object()
        if obj is None or self._object3d_edit_target == "line":
            return
        face = getattr(obj, self._object3d_face)
        controls = self.object3d_face_controls[self._object3d_face]
        if method in {"_finalize_mask_draw", "_finalize_occ_transform"}:
            self._object3d_stroke_open = False
            self._object3d_changed(object_id=obj.id)
            return
        shape = None
        if method == "apply_occ_rect_mask":
            if controls["transform"].isChecked():
                return
            parsed = parse_draw_shape_payload(args[0], default_add=controls["add"].isChecked())
            built = build_polygon_mask(face.mask, parsed.points, shape_kind=parsed.shape_kind) if parsed else None
            if built is None:
                self.statusBar().showMessage("Draw a valid polygon or four distinct corners.", 3000)
                return
            shape, add = built[0], parsed.add
        elif method == "apply_occ_circle_mask":
            payload = args[0]
            shape = build_circle_mask(face.mask, payload[2], payload[3])
            add = bool(payload[4])
        if shape is not None:
            self._object3d_push_undo()
            face.mask = merge_mask(face.mask, shape, add=add)
            if method == "apply_occ_circle_mask":
                self.frame_viewer.clear_occ_circle()
        elif method in {"apply_occ_free_segment", "erase_occ_transform_point", "erase_occ_transform_segment"}:
            if not self._object3d_stroke_open:
                self._object3d_push_undo()
                self._object3d_stroke_open = True
            if method == "apply_occ_free_segment":
                start, end, add = args[0]
            elif method == "erase_occ_transform_point":
                start = end = args[0]
                add = False
            else:
                start, end = args[0]
                add = False
            paint_brush_segment(face.mask, start, end, radius=controls["brush"].value(), add=add)
            self._object3d_render_cache.pop(obj.id, None)
            now = time.monotonic()
            if now - self._object3d_last_brush_refresh < 0.03:
                return
            self._object3d_last_brush_refresh = now
        elif method == "translate_selected_mask":
            dx, dy = self._clamp_mask_shift(face.mask, *args[0])
            if not (dx or dy):
                return
            self._object3d_push_undo()
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            face.mask = cv2.warpAffine(face.mask, matrix, (face.mask.shape[1], face.mask.shape[0]), flags=cv2.INTER_NEAREST)
        else:
            return
        self._object3d_changed(object_id=obj.id)

    def _object3d_affine(self, *, scale=1.0, angle=0.0):
        obj = self._object3d_object()
        if not self._object3d_active() or self._object3d_edit_target == "line" or obj is None or not self.object3d_face_controls[self._object3d_face]["transform"].isChecked():
            return
        face = getattr(obj, self._object3d_face)
        if not np.any(face.mask):
            return
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(angle):
            return
        source = self._object3d_transform_source
        if source is None:
            source = [MaskTransformSource.from_mask(face.mask), 1.0, 0.0]
        next_scale = max(0.05, min(20.0, source[1] * scale))
        next_angle = source[2] + angle
        if next_scale == source[1] and next_angle == source[2]:
            return
        transformed = source[0].render(next_angle, next_scale)
        if not np.any(transformed):
            self.statusBar().showMessage("The face must retain at least one pixel. Transform skipped.", 3000)
            return
        self._object3d_push_undo()
        face.mask = transformed
        self._object3d_changed(object_id=obj.id)
        self._object3d_transform_source = [source[0], next_scale, next_angle]

    def _object3d_save_errors(self):
        included = [o for o in self.object3d_workspace.objects if o.included]
        if not included:
            return ["Include at least one 3D object."]
        return [f"{o.name}: {error}" for o in included for error in self._object3d_result(o)[0]]

    def _object3d_refresh_readiness(self):
        errors = self._object3d_save_errors()
        if self.video_state is None:
            errors.insert(0, "Load a video first.")
        if self.csv_df is None or self.csv_path is None or not self.bodyparts:
            errors.insert(0, "Load a CSV with bodypart coordinates.")
        ready = not errors
        self.object3d_summary.setText("Ready" if ready else "Not ready")
        self.object3d_summary.setStyleSheet("color: #15803d;" if ready else "color: #b45309;")
        self.object3d_summary.setToolTip("\n".join(errors) if errors else "Ready to save this video's CSV.")
        return ready

    def _object3d_save_task(self):
        errors = self._object3d_save_errors()
        if self.video_state is None or self.csv_df is None or self.csv_path is None or not self.bodyparts or errors:
            QMessageBox.warning(self, "Save 3D Objects", errors[0] if errors else "Load a video and a CSV with bodypart coordinates.")
            return None
        masks = [MaskRecord(o.name, QColor(o.color), self._object3d_result(o)[1].copy())
                 for o in self.object3d_workspace.objects if o.included]
        source, bodyparts = self.csv_df.copy(), list(self.bodyparts)
        width, height = self.video_state.width, self.video_state.height
        output = self._occlusion_output_path()
        def export(*, should_cancel=None):
            from batch import raise_if_cancelled
            raise_if_cancelled(should_cancel)
            result = build_occlusion_dataframe(source, bodyparts, masks, width, height)
            return self._write_csv_output(output, result, should_cancel)
        def completed(result):
            self.statusBar().showMessage("3D object CSV save cancelled." if getattr(result, "cancelled", False)
                                         else f"Occlusion CSV saved: {output}")
        return "Save CSV Progress", "Saving 3D object occlusion CSV...", export, completed

    def _object3d_import_png(self):
        if not self._object3d_require_video() or self._object3d_object() is None or self._object3d_edit_target == "line":
            self.statusBar().showMessage("Select an object and choose Top or Bottom before importing a face.", 3000)
            return
        path, _ = QFileDialog.getOpenFileName(self, f"Import {self._object3d_face.title()} Face", "", "PNG Files (*.png)")
        if not path:
            return
        mask = read_mask_png(path)
        if mask is None:
            QMessageBox.warning(self, "Import Face", "Could not read this mask PNG.")
            return
        self._object3d_push_undo()
        ws = self.object3d_workspace
        getattr(self._object3d_object(), self._object3d_face).mask = (cv2.resize(mask, (ws.width, ws.height), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        self._object3d_changed()

    def _object3d_import_folder(self):
        if not self._object3d_require_video():
            return
        folder = QFileDialog.getExistingDirectory(self, "Import Editable 3D Objects", str(self.save_folder))
        if not folder:
            return
        try:
            ws = load_bundle(Path(folder), self.video_state.width, self.video_state.height,
                             video_key=self._object3d_video_key)
        except (ValueError, OSError, KeyError, TypeError, cv2.error) as exc:
            QMessageBox.warning(self, "Import 3D Objects", str(exc))
            return
        self._object3d_push_undo()
        # Measurements are restored only when the bundle identifies this exact video.
        remap = {}
        reference_names = {r.name for r in self.object3d_workspace.references}
        for reference in ws.references:
            previous = reference.id
            reference.id = new_id()
            remap[previous] = reference.id
            while reference.name in reference_names:
                reference.name += " imported"
            reference_names.add(reference.name)
        for obj in ws.objects:
            obj.id = new_id()
            for face in (obj.top, obj.bottom):
                face.reference_id = remap.get(face.reference_id)
            existing = {o.name for o in self.object3d_workspace.objects}
            while obj.name in existing:
                obj.name += " imported"
            self.object3d_workspace.objects.append(obj)
        self.object3d_workspace.references.extend(ws.references)
        if ws.objects:
            self._object3d_selected_id = ws.objects[-1].id
        self._object3d_changed()
        self._sync_object3d_mode()
        self.statusBar().showMessage("Objects imported. Check their references and the preview before saving.", 6000)

    def _object3d_export(self):
        if not self.object3d_workspace.objects:
            self.statusBar().showMessage("Add an object before exporting.", 3000)
            return
        folder = self._mask_export_folder()
        try:
            save_bundle(self.object3d_workspace, folder)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Export Masks", str(exc))
            return
        self.statusBar().showMessage(f"Masks exported to {folder}", 5000)
        QMessageBox.information(self, "Export Masks", f"Masks exported to:\n{folder}")
