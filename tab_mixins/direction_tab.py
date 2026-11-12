from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QSlider,
    QSizePolicy, QVBoxLayout, QWidget,
)

from direction import (
    DirectionConfig, DirectionMask, DirectionTarget, build_direction_dataframe, direction_mask_from_object,
    direction_label, export_direction_masks, load_direction_masks, prepare_direction_pose,
)
from shared import MaskRecord
from .direction_ranges import DirectionRangesMixin
from ui_controls import NoWheelComboBox, NoWheelDoubleSpinBox


class DirectionTabMixin(DirectionRangesMixin):
    def _build_direction_tab(self):
        self.direction_masks = []
        self._direction_targets = {}
        self._direction_video_key = None
        self._direction_video_lines = {}
        self._direction_refreshing = False
        self._direction_pose_cache = None
        self._direction_viewer_dirty = True
        saved = self.settings.get("direction", {})
        saved = saved if isinstance(saved, dict) else {}
        self._direction_point_names = dict(saved.get("points", {})) if isinstance(saved.get("points"), dict) else {}
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(10, 10, 10, 10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        objects = QGroupBox("1. Objects")
        box = QVBoxLayout(objects)
        row = QHBoxLayout()
        self.direction_source = NoWheelComboBox()
        for label, key in (("Occlusion Tab", "occlusion"), ("3D Object Tab", "3d")):
            self.direction_source.addItem(label, key)
        self.direction_load = QPushButton("Load Masks")
        self.direction_load.setToolTip("Load a read-only snapshot. Load again after changing the source masks.")
        row.addWidget(self.direction_source, 1)
        row.addWidget(self.direction_load)
        box.addLayout(row)
        self.direction_objects = QListWidget()
        self.direction_objects.setMinimumHeight(48)
        self.direction_objects.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.direction_objects.setMaximumHeight(105)
        self.direction_objects.setToolTip("Check objects to include in the CSV. Select an object to inspect it.")
        box.addWidget(self.direction_objects)
        layout.addWidget(objects)

        layout.addWidget(self._build_direction_ranges())
        heading = QGroupBox("3. Direction")
        form = QFormLayout(heading)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.direction_method = NoWheelComboBox()
        for label, key in (("Ear Midpoint to Nose", "ears"), ("Neck to Nose", "neck"), ("Body Center to Nose", "body")):
            self.direction_method.addItem(label, key)
        self.direction_method.setCurrentIndex(max(0, self.direction_method.findData(saved.get("method", "ears"))))
        form.addRow("Method", self.direction_method)
        self.direction_point_controls = {"distance": (self.direction_distance_node, self.direction_distance_label)}
        self.direction_distance_node.currentIndexChanged.connect(lambda: self._direction_point_changed("distance"))
        for key, label in (("nose", "Nose"), ("left_ear", "Left Ear"), ("right_ear", "Right Ear"), ("anchor", "Origin")):
            combo = NoWheelComboBox()
            form.addRow(label, combo)
            self.direction_point_controls[key] = (combo, form.labelForField(combo))
            combo.currentIndexChanged.connect(lambda _index, point=key: self._direction_point_changed(point))
        self.direction_coordinates = NoWheelComboBox()
        for label, key in (("Auto", "auto"), ("Pixels", "pixels"), ("Normalized (0-1)", "normalized")):
            self.direction_coordinates.addItem(label, key)
        self.direction_coordinates.setCurrentIndex(max(0, self.direction_coordinates.findData(saved.get("coordinates", "auto"))))
        form.addRow("Coordinates", self.direction_coordinates)
        layout.addWidget(heading)

        row = QHBoxLayout()
        self.direction_angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.direction_angle_slider.setRange(0, 900)
        self.direction_angle = NoWheelDoubleSpinBox()
        self.direction_angle.setRange(0, 90)
        self.direction_angle.setDecimals(1)
        self.direction_angle.setSingleStep(1)
        self.direction_angle.setPrefix("± ")
        self.direction_angle.setSuffix("°")
        self.direction_angle.setFixedWidth(105)
        try:
            initial_angle = float(saved.get("angle", 15))
        except (ValueError, TypeError):
            initial_angle = 15
        self.direction_angle.setValue(initial_angle if np.isfinite(initial_angle) else 15)
        self.direction_angle_slider.setValue(round(self.direction_angle.value() * 10))
        self.direction_angle.setToolTip("Half-angle on each side of the heading. Zero tests a forward ray from the nose.")
        row.addWidget(self.direction_angle_slider, 1)
        row.addWidget(self.direction_angle)
        form.addRow("Angle Margin", row)
        self.direction_angle_label = QLabel()
        form.addRow(self.direction_angle_label)

        review = QGroupBox("4. Review")
        review_layout = QVBoxLayout(review)
        self.direction_instance = NoWheelComboBox()
        self.direction_instance.setToolTip("Choose the CSV row or animal shown in this frame. All rows are saved.")
        review_layout.addWidget(self.direction_instance)
        self.direction_result = QLabel("Unknown")
        self.direction_result.setWordWrap(True)
        review_layout.addWidget(self.direction_result)
        self.direction_ready = QLabel("Not ready")
        review_layout.addWidget(self.direction_ready)
        layout.addWidget(review)
        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        imports = QHBoxLayout()
        self.direction_import_mask = QPushButton("Import Mask")
        self.direction_import_folder = QPushButton("Import Folder")
        self.direction_import_mask.setToolTip("Import a PNG base mask and any saved Direction recipe.")
        self.direction_import_folder.setToolTip("Import Occlusion masks, editable 3D faces, or exported Direction masks.")
        self.direction_import_mask.clicked.connect(lambda: self._direction_import_masks(folder=False))
        self.direction_import_folder.clicked.connect(lambda: self._direction_import_masks(folder=True))
        imports.addWidget(self.direction_import_mask)
        imports.addWidget(self.direction_import_folder)
        outer.addLayout(imports)
        export = QPushButton("Export Masks")
        export.clicked.connect(self._direction_export)
        outer.addWidget(export)
        self.direction_load.clicked.connect(self._direction_load_masks)
        self.direction_objects.itemChanged.connect(self._direction_options_changed)
        self.direction_objects.currentRowChanged.connect(self._direction_selection_changed)
        self.direction_method.currentIndexChanged.connect(self._direction_method_changed)
        self.direction_coordinates.currentIndexChanged.connect(self._direction_options_changed)
        self.direction_angle_slider.valueChanged.connect(lambda value: self.direction_angle.setValue(value / 10))
        self.direction_angle.valueChanged.connect(lambda value: self.direction_angle_slider.setValue(round(value * 10)))
        self.direction_angle.valueChanged.connect(self._direction_options_changed)
        self.direction_instance.currentIndexChanged.connect(self._direction_selection_changed)
        for combo in tab.findChildren(NoWheelComboBox):
            combo.setMinimumWidth(0)
            combo.setMinimumContentsLength(8)
            combo.setSizeAdjustPolicy(NoWheelComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        for combo in (self.direction_range_unit, self.direction_reference_unit):
            combo.setMinimumContentsLength(2)
            combo.setMinimumWidth(80)
            combo.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        return tab

    def _direction_active(self):
        return hasattr(self, "mode_tabs") and self.mode_tabs.currentIndex() == self.TAB_DIRECTION

    def _direction_method_changed(self, *_args):
        self._direction_refresh_columns()
        self._direction_options_changed()

    def _direction_refresh_columns(self):
        if not hasattr(self, "direction_method"):
            return
        self._direction_refreshing = True
        aliases = {"distance": ("nose", "snout"), "nose": ("nose", "snout"), "left_ear": ("ear_l", "left_ear", "ear_left", "l_ear"),
                   "right_ear": ("ear_r", "right_ear", "ear_right", "r_ear"),
                   "neck": ("neck",), "body": ("body_c", "body_center", "body_centre", "centroid")}
        try:
            parts = list(self.bodyparts)
            available = {name.lower().replace(" ", "_").replace("-", "_"): name for name in parts}
            method = self.direction_method.currentData()
            for key, (combo, label) in self.direction_point_controls.items():
                storage_key = method if key == "anchor" else key
                selected = self._direction_point_names.get(storage_key)
                if not selected:
                    selected = next((available[a] for a in aliases.get(storage_key, ()) if a in available), None)
                combo.clear()
                combo.addItem("Select a bodypart...", None)
                for name in parts:
                    combo.addItem(name, name)
                combo.setCurrentIndex(max(0, combo.findData(selected)))
                if parts and selected in parts:
                    self._direction_point_names[storage_key] = combo.currentData()
                visible = (key in {"left_ear", "right_ear"}) == (method == "ears") if key not in {"nose", "distance"} else True
                combo.setVisible(visible)
                label.setVisible(visible)
                if key == "anchor":
                    label.setText("Neck" if method == "neck" else "Body Center")
        finally:
            self._direction_refreshing = False
        self._direction_pose_cache = None

    def _direction_point_changed(self, key):
        if self._direction_refreshing:
            return
        storage_key = self.direction_method.currentData() if key == "anchor" else key
        self._direction_point_names[storage_key] = self.direction_point_controls[key][0].currentData()
        self._direction_pose_cache = None
        self._direction_options_changed()

    def _direction_config(self):
        return DirectionConfig(self.direction_method.currentData(), self.direction_point_controls["nose"][0].currentData(),
            self.direction_point_controls["anchor"][0].currentData(), self.direction_point_controls["left_ear"][0].currentData(),
            self.direction_point_controls["right_ear"][0].currentData(), self.direction_angle.value(), self.direction_coordinates.currentData(), self.direction_point_controls["distance"][0].currentData())

    def _direction_settings_payload(self):
        return {"direction": {"method": self.direction_method.currentData(), "points": self._direction_point_names,
                               "coordinates": self.direction_coordinates.currentData(), "angle": self.direction_angle.value()}}

    def _direction_selected_masks(self):
        return [item for index, item in enumerate(self.direction_masks)
                if self.direction_objects.item(index).checkState() == Qt.CheckState.Checked]

    def _direction_set_masks(self, masks, checked=None, selected=None):
        if len({item.name for item in masks}) != len(masks):
            raise ValueError("Object names must be unique.")
        # Validate raw geometry before replacing a usable workspace.
        for item in masks:
            DirectionTarget(item.mask)
        self._direction_cancel_calibration()
        self.direction_masks = masks
        self._direction_reuse_video_references()
        self._direction_rebuild_ranges()
        self.direction_objects.setFixedHeight(min(105, max(48, len(masks) * 24 + 8)))
        self._direction_refreshing = True
        try:
            self.direction_objects.clear()
            for mask in masks:
                item = QListWidgetItem(mask.name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if checked is None or mask.name in checked else Qt.CheckState.Unchecked)
                item.setToolTip("Video-specific 3D mask" if mask.video_key else "Read-only object mask")
                self.direction_objects.addItem(item)
            if masks:
                self.direction_objects.setCurrentRow(next((i for i, m in enumerate(masks) if m.name == selected), 0))
        finally:
            self._direction_refreshing = False
        self._direction_viewer_dirty = True
        self._direction_sync_ranges()
        self._direction_options_changed()

    def _direction_import_masks(self, *, folder):
        self._direction_load_masks(source="folder" if folder else "png")

    def _direction_load_masks(self, _checked=False, *, source=None):
        if self.video_state is None:
            QMessageBox.information(self, "Direction", "Load a video first.")
            return
        width, height = self.video_state.width, self.video_state.height
        try:
            source = source or self.direction_source.currentData()
            if source == "occlusion":
                masks = [DirectionMask(record.name, record.mask.copy(), record.color.name()) for record in self.mask_records.values()]
                for mask in masks:
                    mask.ranges.quad_points = list(self.frame_viewer.occ_margin_points)
            elif source == "3d":
                masks = [direction_mask_from_object(obj, self.object3d_workspace.references, self._object3d_video_key)
                         for obj in self.object3d_workspace.objects if obj.included]
            else:
                if source == "png":
                    path, _ = QFileDialog.getOpenFileName(self, "Import Direction Mask", "", "PNG Files (*.png)")
                else:
                    path = QFileDialog.getExistingDirectory(self, "Import Direction Masks", str(self.save_folder))
                if not path:
                    return
                masks = load_direction_masks(Path(path), width, height, self._direction_video_key)
            if not masks:
                raise ValueError("Prepare at least one object mask in the selected source.")
            if any(item.mask.shape != (height, width) for item in masks):
                raise ValueError("Mask dimensions do not match this video.")
            if source in {"occlusion", "3d"}:
                previous = {m.name: m for m in self.direction_masks}
                for mask in masks:
                    old = previous.get(mask.name)
                    if old is not None and (old.top is None) == (mask.top is None):
                        mask.ranges = copy.deepcopy(old.ranges)
                        mask.references = copy.deepcopy(old.references)
            self._direction_set_masks(masks)
            self.statusBar().showMessage(f"Loaded {len(masks)} read-only direction masks.", 4000)
        except (OSError, ValueError, TypeError, cv2.error) as exc:
            QMessageBox.warning(self, "Load Direction Masks", str(exc))

    def _direction_options_changed(self, *_args):
        if self._direction_refreshing:
            return
        angle = self.direction_angle.value()
        self.direction_angle_label.setText("Forward ray from nose" if angle == 0 else f"Total angle: {2 * angle:g}°")
        if hasattr(self, "save_current_button"):
            self._refresh_output_ui()
            self._direction_refresh_preview()
            self._save_settings()

    def _direction_selection_changed(self, *_args):
        if not self._direction_refreshing:
            self._direction_cancel_calibration()
            self._direction_sync_ranges()
            self._direction_refresh_preview()
            self._refresh_output_ui()

    def _direction_errors(self):
        if self.video_state is None or self.csv_df is None or self.csv_path is None:
            return ["Load a video and its CSV."]
        if not self._direction_selected_masks():
            return ["Load and check at least one object."]
        if self._direction_calibration is not None and self._direction_calibration.get("pending", True):
            return ["Finish or cancel calibration."]
        try:
            config = self._direction_config()
            parts = config.required_bodyparts() + [config.distance_node]
            if not config.distance_node:
                return ["Choose a Distance Node."]
        except ValueError as exc:
            return [str(exc)]
        if any(f"{part}.{axis}" not in self.csv_df for part in parts for axis in ("x", "y")):
            return ["Choose bodyparts present in this CSV."]
        if any(item.video_key and item.video_key != self._direction_video_key for item in self._direction_selected_masks()):
            return ["Reload 3D masks measured for this video."]
        errors = [f"{m.name}: {self._direction_range_errors[m.name]}" for m in self._direction_selected_masks() if m.name in self._direction_range_errors]
        return errors

    def _direction_batch_scoped(self):
        return any(m.top is not None or m.video_key or m.ranges.mode == "physical" for m in self._direction_selected_masks())

    def _direction_refresh_readiness(self):
        errors = self._direction_errors()
        self.direction_ready.setText("Not ready" if errors else "Ready")
        self.direction_ready.setToolTip("\n".join(errors) if errors else "Ready to save direction columns.")
        return not errors

    def _direction_refresh_preview(self):
        if not self._direction_active():
            return
        viewer = self.frame_viewer
        viewer.direction_preview = None
        viewer.direction_distance_point = None
        viewer.direction_range_overlay = None
        rows = np.array([], dtype=int)
        if self.csv_df is not None:
            if self._node_frame_values is None:
                row = self.current_frame_number - 1
                if 0 <= row < len(self.csv_df):
                    rows = np.array([row])
            else:
                target = self.current_frame_number - int(self._node_frame_zero_based)
                rows = np.flatnonzero(self._node_frame_values.to_numpy() == target)
        current = self.direction_instance.currentData()
        self.direction_instance.blockSignals(True)
        self.direction_instance.clear()
        identity = next((c for c in ("instance.id", "instance", "instance_id", "track", "track_id")
                         if self.csv_df is not None and c in self.csv_df), None)
        for row in rows:
            label = f"{identity}: {self.csv_df.iloc[row][identity]}" if identity else f"Row {row + 1}"
            self.direction_instance.addItem(label, int(row))
        self.direction_instance.setCurrentIndex(max(0, self.direction_instance.findData(current)) if len(rows) else -1)
        self.direction_instance.setVisible(len(rows) > 1)
        self.direction_instance.blockSignals(False)
        values = {}
        row = self.direction_instance.currentData()
        if not self._direction_errors() and row is not None:
            config = self._direction_config()
            key = (id(self.csv_df), config.method, config.nose, config.anchor, config.left_ear,
                   config.right_ear, config.coordinates, config.distance_node, self.video_state.width, self.video_state.height)
            if self._direction_pose_cache is None or self._direction_pose_cache[0] != key:
                self._direction_pose_cache = (key, prepare_direction_pose(self.csv_df, config, self.video_state.width, self.video_state.height))
            pose = self._direction_pose_cache[1].subset(slice(row, row + 1))
            if pose.distance_valid[0]:
                viewer.direction_distance_point = (tuple(pose.distance[0]), pose.distance_name)
            if pose.valid[0]:
                viewer.direction_preview = (tuple(pose.origin[0]), tuple(pose.nose[0]), config.angle)
            for mask in self._direction_selected_masks():
                states, reasons = self._direction_targets[mask.name].evaluate(pose, config.angle)
                values[mask.name] = (states[0], reasons[0])
        selected = self.direction_objects.currentRow()
        selected_name = self.direction_masks[selected].name if 0 <= selected < len(self.direction_masks) else None
        selected_masks = self._direction_selected_masks()
        checked = {mask.name for mask in selected_masks}
        def result_for(name):
            reason = "not_selected" if name not in checked else "no_frame_data" if row is None else ""
            return values.get(name, ("Unknown", reason))
        self.direction_objects.blockSignals(True)
        for index, mask in enumerate(self.direction_masks):
            item = self.direction_objects.item(index)
            item.setText(f"{mask.name}: {direction_label(*result_for(mask.name))}")
            source = "Video-specific 3D mask" if mask.video_key else "Read-only object mask"
            item.setToolTip(f"{item.text()}\n{source}")
        self.direction_objects.blockSignals(False)
        self.direction_result.setText(f"{selected_name}: {direction_label(*result_for(selected_name))}" if selected_name else "Unknown")
        self.direction_result.setToolTip("Distance Node defines Overlap / Near / Far. Heading is tested against the Object region only when Near.")
        colors = {"Overlap": "#f59e0b", "NearFacing": "#22c55e", "NearNotFacing": "#3b82f6", "Far": "#94a3b8", "Unknown": "#9ca3af"}
        records = {mask.name: MaskRecord(mask.name, QColor(colors[result_for(mask.name)[0]]),
                   self._direction_targets[mask.name].object_mask if mask.name in self._direction_targets else mask.mask)
                   for mask in selected_masks}
        zones = self._direction_targets.get(selected_name)
        if zones is not None:
            if selected_name not in self._direction_range_paths:
                ring = ((zones.near_mask > 0) & (zones.object_mask == 0)).astype(np.uint8)
                self._direction_range_paths[selected_name] = (viewer._mask_to_path(zones.near_mask), viewer._mask_to_path(ring)) if np.any(ring) else (None, None)
            viewer.direction_range_overlay = self._direction_range_paths[selected_name]
        # Single-pixel/line masks use colored pixmaps rather than vector paths.
        # Refresh those pixmaps when the Facing color changes between frames.
        recolor = any(name not in viewer._mask_fill_path_cache
                      and name in viewer.mask_records and viewer.mask_records[name].color != record.color
                      for name, record in records.items())
        viewer.set_mask_records(records, selected_name, refresh=self._direction_viewer_dirty or recolor)
        self._direction_viewer_dirty = False
        self._direction_sync_calibration_viewer()
        viewer.update()

    def _direction_video_changed(self, video_path, width, height):
        if self._direction_video_key and self.direction_masks:
            old_height, old_width = self.direction_masks[0].mask.shape
            lines = {ref.id: list(ref.points) for mask in self.direction_masks for ref in mask.references}
            self._direction_video_lines[self._direction_video_key] = ((old_width, old_height), lines)
        checked = {m.name for m in self._direction_selected_masks()}
        current = self._direction_range_object()
        selected = current.name if current else None
        key = str(Path(video_path).resolve())
        self._direction_video_key = key
        masks = [m.resized(width, height) for m in self.direction_masks]
        dimensions, lines = self._direction_video_lines.get(key, (None, {}))
        if dimensions != (width, height):
            lines = {}
        for mask in masks:
            if mask.top is not None or mask.video_key:
                mask.video_key = key
            for reference in mask.references:
                reference.points = list(lines.get(reference.id, []))
            mask.ranges.calibration_video_key = key if any(r.points for r in mask.references) else None
        self._direction_pose_cache = None
        self._direction_set_masks(masks, checked, selected)

    def _direction_output_path(self, csv_path=None):
        path = csv_path or self.csv_path
        return None if path is None else self._apply_current_output_affixes(self.save_folder / f"{path.stem}_direction.csv")

    def _direction_save_task(self):
        errors = self._direction_errors()
        if errors:
            QMessageBox.warning(self, "Save Direction", errors[0])
            return None
        config, masks = self._direction_config(), copy.deepcopy(self._direction_selected_masks())
        source = self.csv_df.copy()
        width, height = self.video_state.width, self.video_state.height
        output = self._direction_output_path()
        def export(*, should_cancel=None):
            result = build_direction_dataframe(source, config, masks, width, height, should_cancel)
            return self._write_csv_output(output, result, should_cancel)
        return "Save CSV Progress", "Saving direction CSV...", export, None

    def _direction_save_multiple(self):
        errors = self._direction_errors()
        if errors:
            QMessageBox.warning(self, "Save Direction", errors[0])
            return
        masks = copy.deepcopy(self._direction_selected_masks())
        if self._direction_batch_scoped():
            QMessageBox.information(self, "Direction Batch Unavailable", "3D and physical ranges are video-specific. Measure references and save each video separately.")
            return
        videos = self._select_videos_for_batch()
        if not videos:
            return
        config = self._direction_config()
        prefix, suffix = self._current_output_affixes()
        folder = Path(self.save_folder)
        def export(item, should_cancel=None):
            scaled = [m.resized(item.width, item.height) for m in masks]
            result = build_direction_dataframe(item.source_df, config, scaled, item.width, item.height, should_cancel)
            path = self._apply_output_affixes(folder / f"{item.csv_path.stem}_direction.csv", prefix, suffix)
            return self._write_csv_output(path, result, should_cancel)
        self._start_batch_export(title="Direction Batch Save", activity_text="Saving direction CSVs...",
            selected_videos=videos, source_width=self.video_state.width, source_height=self.video_state.height,
            export_item=export, on_completed=lambda result: self.statusBar().showMessage(f"Direction CSVs saved: {result.saved_count}"))

    def _direction_export(self):
        masks = self._direction_selected_masks()
        if not masks:
            QMessageBox.information(self, "Export Masks", "Load and check at least one object.")
            return
        try:
            folder = self._mask_export_folder()
            export_direction_masks(masks, folder, self._direction_config())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Export Masks", str(exc))
            return
        QMessageBox.information(self, "Export Masks", f"Masks exported to:\n{folder}")
