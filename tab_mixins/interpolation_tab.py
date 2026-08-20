from __future__ import annotations

import time

import numpy as np
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from interpolation import build_interpolation_pipeline_dataframe
from mask_tools import build_circle_mask, build_polygon_mask, merge_mask, paint_brush_segment, parse_draw_shape_payload
from shared import MaskTransformSource
from ui_controls import NoWheelComboBox, NoWheelSpinBox


class InterpolationTabMixin:
    def _build_interpolation_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setObjectName("interpolationScroll")
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 12)
        layout.setSpacing(10)

        info = QLabel(
            "Step 1 optionally removes complete "
            "skeletons using one region and anchor-node rule. Step 2 fills missing coordinates."
        )
        info.setWordWrap(True)
        info.setProperty("muted", True)
        layout.addWidget(info)

        automatic_group = QGroupBox("1. Automatic Removal (Optional)")
        automatic_layout = QVBoxLayout(automatic_group)
        automatic_layout.setSpacing(8)
        automatic_help = QLabel(
            "Define the region and removal rule together. Select No automatic removal to skip "
            "this whole step and run interpolation only."
        )
        automatic_help.setWordWrap(True)
        automatic_help.setProperty("muted", True)
        automatic_layout.addWidget(automatic_help)

        region_heading = QLabel("Region")
        region_heading.setProperty("sectionTitle", True)
        automatic_layout.addWidget(region_heading)
        region_layout = QFormLayout()
        region_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        region_layout.setHorizontalSpacing(8)
        region_layout.setVerticalSpacing(7)
        self.interpolation_shape_combo = NoWheelComboBox()
        self.interpolation_shape_combo.addItem("Rectangle Drag", "interp_rect_drag")
        self.interpolation_shape_combo.addItem("Rectangle 4 Points", "interp_rect")
        self.interpolation_shape_combo.addItem("Polygon", "interp_polygon")
        self.interpolation_shape_combo.addItem("Circle Drag", "interp_circle")
        self.interpolation_shape_combo.addItem("Brush", "interp_free")
        region_layout.addRow("Shape", self.interpolation_shape_combo)

        edit_mode_row = QHBoxLayout()
        self.interpolation_region_draw_radio = QRadioButton("Draw")
        self.interpolation_region_transform_radio = QRadioButton("Transform")
        self.interpolation_region_draw_radio.setChecked(True)
        self.interpolation_region_edit_group = QButtonGroup(self)
        self.interpolation_region_edit_group.addButton(self.interpolation_region_draw_radio)
        self.interpolation_region_edit_group.addButton(
            self.interpolation_region_transform_radio
        )
        edit_mode_row.addWidget(self.interpolation_region_draw_radio)
        edit_mode_row.addWidget(self.interpolation_region_transform_radio)
        edit_mode_row.addStretch(1)
        region_layout.addRow("Edit", edit_mode_row)

        transform_help = QLabel(
            "Same as Annotate masks: D draw 쨌 T transform 쨌 drag to move 쨌 "
            "[ / ] or E / R to scale 쨌 Ctrl+[ / ] or Ctrl+E / R to rotate 쨌 Ctrl+drag to erase."
        )
        transform_help.setWordWrap(True)
        transform_help.setProperty("muted", True)
        region_layout.addRow("", transform_help)

        free_draw_row = QHBoxLayout()
        self.interpolation_draw_add_radio = QRadioButton("Draw")
        self.interpolation_draw_erase_radio = QRadioButton("Erase")
        self.interpolation_draw_add_radio.setChecked(True)
        self.interpolation_draw_mode_group = QButtonGroup(self)
        self.interpolation_draw_mode_group.addButton(self.interpolation_draw_add_radio)
        self.interpolation_draw_mode_group.addButton(self.interpolation_draw_erase_radio)
        self.interpolation_brush_spinbox = NoWheelSpinBox()
        self.interpolation_brush_spinbox.setRange(1, 200)
        self.interpolation_brush_spinbox.setValue(20)
        self.interpolation_brush_spinbox.setSuffix(" px")
        free_draw_row.addWidget(self.interpolation_draw_add_radio)
        free_draw_row.addWidget(self.interpolation_draw_erase_radio)
        free_draw_row.addStretch(1)
        free_draw_row.addWidget(QLabel("Brush"))
        free_draw_row.addWidget(self.interpolation_brush_spinbox)
        region_layout.addRow("Draw Action", free_draw_row)

        self.interpolation_region_label = QLabel("No region selected.")
        self.interpolation_region_label.setWordWrap(True)
        self.interpolation_region_label.setProperty("muted", True)
        self.interpolation_reset_button = QPushButton("Reset Region")
        region_layout.addRow("", self.interpolation_region_label)
        region_layout.addRow("", self.interpolation_reset_button)
        automatic_layout.addLayout(region_layout)

        removal_heading = QLabel("Removal rule")
        removal_heading.setProperty("sectionTitle", True)
        automatic_layout.addWidget(removal_heading)
        removal_layout = QFormLayout()
        removal_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        removal_layout.setHorizontalSpacing(8)
        removal_layout.setVerticalSpacing(7)
        self.interpolation_removal_combo = NoWheelComboBox()
        self.interpolation_removal_combo.addItem("No automatic removal", "none")
        self.interpolation_removal_combo.addItem(
            "Remove skeletons outside the region", "outside"
        )
        self.interpolation_removal_combo.addItem(
            "Remove skeletons inside the region", "inside"
        )
        self.interpolation_anchor_combo = NoWheelComboBox()
        self.interpolation_anchor_combo.addItem("Select anchor node...", None)
        self.interpolation_removal_help_label = QLabel(
            "Each row is one skeleton. The anchor node determines whether that row's entire "
            "skeleton payload is removed."
        )
        self.interpolation_removal_help_label.setWordWrap(True)
        self.interpolation_removal_help_label.setProperty("muted", True)
        removal_layout.addRow("Remove", self.interpolation_removal_combo)
        removal_layout.addRow("Anchor node", self.interpolation_anchor_combo)
        removal_layout.addRow("", self.interpolation_removal_help_label)
        automatic_layout.addLayout(removal_layout)

        interpolation_group = QGroupBox("2. Interpolation")
        interpolation_layout = QVBoxLayout(interpolation_group)
        self.interpolation_enabled_checkbox = QCheckBox(
            "Interpolate missing coordinates after automatic removal"
        )
        self.interpolation_enabled_checkbox.setChecked(True)
        self.interpolation_extrapolate_checkbox = QCheckBox(
            "Extrapolate missing coordinates at track start / end"
        )
        self.interpolation_extrapolate_checkbox.setChecked(
            bool(self.settings.get("interpolation_extrapolate", False))
        )
        self.interpolation_enabled_help_label = QLabel(
            "Internal gaps use linear interpolation. Optional start/end extrapolation uses "
            "the nearest valid coordinate in each track."
        )
        self.interpolation_enabled_help_label.setWordWrap(True)
        self.interpolation_enabled_help_label.setProperty("muted", True)
        interpolation_layout.addWidget(self.interpolation_enabled_checkbox)
        interpolation_layout.addWidget(self.interpolation_extrapolate_checkbox)
        interpolation_layout.addWidget(self.interpolation_enabled_help_label)

        layout.addWidget(automatic_group)
        layout.addWidget(interpolation_group)
        layout.addStretch(1)
        scroll.setWidget(content)
        tab_layout.addWidget(scroll)

        self._interpolation_transform_source: MaskTransformSource | None = None
        self._interpolation_transform_angle = 0.0
        self._interpolation_transform_scale = 1.0
        self.interpolation_shape_combo.currentIndexChanged.connect(self._sync_interpolation_mode)
        self.interpolation_removal_combo.currentIndexChanged.connect(
            self._on_interpolation_removal_changed
        )
        self.interpolation_anchor_combo.currentIndexChanged.connect(
            self._refresh_interpolation_ui
        )
        self.interpolation_enabled_checkbox.toggled.connect(self._refresh_interpolation_ui)
        self.interpolation_enabled_checkbox.toggled.connect(self._save_settings)
        self.interpolation_extrapolate_checkbox.toggled.connect(
            self._refresh_interpolation_ui
        )
        self.interpolation_extrapolate_checkbox.toggled.connect(self._save_settings)
        self.interpolation_region_draw_radio.toggled.connect(
            self._sync_interpolation_mode
        )
        self.interpolation_region_transform_radio.toggled.connect(
            self._sync_interpolation_mode
        )
        self.interpolation_draw_add_radio.toggled.connect(
            self._sync_interpolation_draw_mode
        )
        self.interpolation_draw_erase_radio.toggled.connect(
            self._sync_interpolation_draw_mode
        )
        self.interpolation_brush_spinbox.valueChanged.connect(
            self._sync_interpolation_draw_mode
        )
        self.interpolation_reset_button.clicked.connect(self.reset_interpolation_region)
        self._refresh_interpolation_ui()
        return tab

    def _selected_interpolation_removal_mode(self) -> str:
        return str(self.interpolation_removal_combo.currentData())

    def _selected_interpolation_anchor(self) -> str | None:
        value = self.interpolation_anchor_combo.currentData()
        return None if value is None else str(value)

    def _interpolation_enabled(self) -> bool:
        return self.interpolation_enabled_checkbox.isChecked()

    def _interpolation_extrapolation_enabled(self) -> bool:
        return self.interpolation_extrapolate_checkbox.isChecked()

    def _interpolation_settings_payload(self) -> dict:
        return {
            "interpolation_extrapolate": self._interpolation_extrapolation_enabled(),
        }

    def _has_interpolation_region(self) -> bool:
        return self.interpolation_mask is not None and bool(np.any(self.interpolation_mask))

    def _set_interpolation_removal_mode(self, mode: str) -> None:
        index = self.interpolation_removal_combo.findData(mode)
        if index < 0 or index == self.interpolation_removal_combo.currentIndex():
            return
        self.interpolation_removal_combo.blockSignals(True)
        self.interpolation_removal_combo.setCurrentIndex(index)
        self.interpolation_removal_combo.blockSignals(False)

    def _populate_interpolation_anchor_combo(self) -> None:
        expected_values: list[str | None] = [None, *self.bodyparts]
        current_values = [
            self.interpolation_anchor_combo.itemData(index)
            for index in range(self.interpolation_anchor_combo.count())
        ]
        if current_values == expected_values:
            return
        selected_anchor = self._selected_interpolation_anchor()
        self.interpolation_anchor_combo.blockSignals(True)
        self.interpolation_anchor_combo.clear()
        self.interpolation_anchor_combo.addItem("Select anchor node...", None)
        for bodypart in self.bodyparts:
            self.interpolation_anchor_combo.addItem(bodypart, bodypart)
        selected_index = self.interpolation_anchor_combo.findData(selected_anchor)
        self.interpolation_anchor_combo.setCurrentIndex(max(0, selected_index))
        self.interpolation_anchor_combo.blockSignals(False)

    def _clear_interpolation_region(self) -> None:
        self._invalidate_interpolation_transform_source()
        self.interpolation_mask = None
        self.frame_viewer.clear_interpolation_rect_points()
        self.frame_viewer.clear_interpolation_circle()
        self.frame_viewer.set_interpolation_mask(None)

    def _on_interpolation_removal_changed(self, _index: int) -> None:
        if self._selected_interpolation_removal_mode() == "none":
            self._clear_interpolation_region()
        self._refresh_interpolation_ui()

    def _ensure_interpolation_mask(self) -> np.ndarray | None:
        if self.video_state is None:
            return None
        expected_shape = (self.video_state.height, self.video_state.width)
        if self.interpolation_mask is None or self.interpolation_mask.shape != expected_shape:
            self.interpolation_mask = np.zeros(expected_shape, dtype=np.uint8)
        return self.interpolation_mask

    def _set_interpolation_mask(
        self,
        mask: np.ndarray | None,
        refresh: bool = True,
        reset_transform_source: bool = True,
    ) -> None:
        if reset_transform_source:
            self._invalidate_interpolation_transform_source()
        normalized_mask = None if mask is None else (mask > 0).astype(np.uint8)
        self.interpolation_mask = normalized_mask if normalized_mask is not None and np.any(normalized_mask) else None
        if self._has_interpolation_region() and self._selected_interpolation_removal_mode() == "none":
            self._set_interpolation_removal_mode("outside")
        elif not self._has_interpolation_region():
            self._set_interpolation_removal_mode("none")
        if refresh:
            self.frame_viewer.set_interpolation_mask(self.interpolation_mask)
        self._refresh_interpolation_ui()

    def _sync_interpolation_mode(self) -> None:
        if self.mode_tabs.currentIndex() != self.TAB_INTERPOLATION:
            return
        mode = str(self.interpolation_shape_combo.currentData())
        self.frame_viewer.clear_interpolation_rect_points()
        self.frame_viewer.clear_interpolation_circle()
        self.frame_viewer.set_mode(mode)
        self.frame_viewer.set_interpolation_transform_mode(
            self.interpolation_region_transform_radio.isChecked()
            and self._has_interpolation_region()
        )
        self._sync_interpolation_draw_mode()
        self._refresh_interpolation_ui()

    def _sync_interpolation_draw_mode(self) -> None:
        if not hasattr(self, "interpolation_draw_add_radio"):
            return
        self.frame_viewer.set_interpolation_draw_mode(
            add=self.interpolation_draw_add_radio.isChecked(),
            brush_radius=int(self.interpolation_brush_spinbox.value()),
        )

    def _refresh_interpolation_ui(self) -> None:
        if not hasattr(self, "interpolation_region_label"):
            return
        self._populate_interpolation_anchor_combo()
        has_region = self._has_interpolation_region()
        if not has_region:
            self._set_interpolation_removal_mode("none")
            if self.interpolation_region_transform_radio.isChecked():
                self.interpolation_region_draw_radio.setChecked(True)
        self.interpolation_region_transform_radio.setEnabled(has_region)
        mask_pixels = 0 if not has_region else int(np.count_nonzero(self.interpolation_mask))
        removal_mode = self._selected_interpolation_removal_mode()
        anchor = self._selected_interpolation_anchor()
        if has_region:
            removal_label = "inside" if removal_mode == "inside" else "outside"
            anchor_label = anchor if anchor is not None else "select an anchor node"
            self.interpolation_region_label.setText(
                f"Region: {mask_pixels:,} pixels | remove {removal_label} using {anchor_label}"
            )
        else:
            self.interpolation_region_label.setText("No region selected. Automatic removal is off.")
        removal_active = has_region and removal_mode != "none"
        self.interpolation_removal_combo.setEnabled(has_region)
        self.interpolation_anchor_combo.setEnabled(removal_active and bool(self.bodyparts))
        selected_shape = str(self.interpolation_shape_combo.currentData())
        draw_controls_enabled = self.interpolation_region_draw_radio.isChecked()
        brush_enabled = selected_shape == "interp_free" and draw_controls_enabled
        self.interpolation_draw_add_radio.setEnabled(draw_controls_enabled)
        self.interpolation_draw_erase_radio.setEnabled(draw_controls_enabled)
        self.interpolation_brush_spinbox.setEnabled(brush_enabled)
        interpolation_enabled = self._interpolation_enabled()
        self.interpolation_extrapolate_checkbox.setEnabled(interpolation_enabled)
        if interpolation_enabled and self._interpolation_extrapolation_enabled():
            self.interpolation_enabled_help_label.setText(
                "Enabled: internal gaps use linear interpolation; start/end gaps use the nearest valid coordinate."
            )
        elif interpolation_enabled:
            self.interpolation_enabled_help_label.setText(
                "Enabled: internal gaps are interpolated; missing coordinates at track start/end remain missing."
            )
        else:
            self.interpolation_enabled_help_label.setText(
                "Disabled: removed skeleton coordinates remain missing in the saved CSV."
            )
        if hasattr(self, "save_current_button"):
            self._refresh_output_ui()

    def _invalidate_interpolation_transform_source(self) -> None:
        self._interpolation_transform_source = None
        self._interpolation_transform_angle = 0.0
        self._interpolation_transform_scale = 1.0

    def _ensure_interpolation_transform_source(self) -> bool:
        if not self._has_interpolation_region():
            return False
        if (
            self._interpolation_transform_source is not None
            and self._interpolation_transform_source.mask.shape == self.interpolation_mask.shape
        ):
            return True
        try:
            self._interpolation_transform_source = MaskTransformSource.from_mask(
                self.interpolation_mask
            )
        except ValueError:
            self._invalidate_interpolation_transform_source()
            return False
        self._interpolation_transform_angle = 0.0
        self._interpolation_transform_scale = 1.0
        return True

    def translate_interpolation_region(self, shift: tuple[int, int]) -> None:
        if not self._has_interpolation_region():
            return
        dx, dy = self._clamp_mask_shift(self.interpolation_mask, *shift)
        if dx == 0 and dy == 0:
            return
        translated = np.zeros_like(self.interpolation_mask)
        src_x0 = max(0, -dx)
        src_x1 = self.interpolation_mask.shape[1] - max(0, dx)
        src_y0 = max(0, -dy)
        src_y1 = self.interpolation_mask.shape[0] - max(0, dy)
        dst_x0 = max(0, dx)
        dst_y0 = max(0, dy)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        translated[dst_y0:dst_y1, dst_x0:dst_x1] = self.interpolation_mask[
            src_y0:src_y1, src_x0:src_x1
        ]
        if hasattr(self, "_push_interpolation_undo"):
            self._push_interpolation_undo("move interpolation region")
        self._set_interpolation_mask(translated)

    def _apply_interpolation_affine_transform(
        self,
        *,
        angle_delta: float = 0.0,
        scale_multiplier: float = 1.0,
    ) -> None:
        if scale_multiplier <= 0 or not self._ensure_interpolation_transform_source():
            return
        source = self._interpolation_transform_source
        if source is None:
            return
        next_angle = (self._interpolation_transform_angle + float(angle_delta)) % 360.0
        next_scale = self._interpolation_transform_scale * float(scale_multiplier)
        if next_scale < 0.1 or next_scale > 10.0:
            return
        transformed = source.render(next_angle, next_scale)
        if not np.any(transformed):
            return
        if hasattr(self, "_push_interpolation_undo"):
            label = "rotate interpolation region" if abs(angle_delta) > 1e-6 else "scale interpolation region"
            self._push_interpolation_undo(label)
        self._interpolation_transform_angle = next_angle
        self._interpolation_transform_scale = next_scale
        self._set_interpolation_mask(
            transformed,
            reset_transform_source=False,
        )

    def scale_interpolation_region(self, scale_factor: float) -> None:
        self._apply_interpolation_affine_transform(scale_multiplier=scale_factor)

    def rotate_interpolation_region(self, angle_degrees: float) -> None:
        if abs(angle_degrees) < 1e-6:
            return
        self._apply_interpolation_affine_transform(angle_delta=angle_degrees)

    def apply_interpolation_rect(self, payload: object) -> None:
        if hasattr(self, "video_state") and self.video_state is None:
            return
        parsed = parse_draw_shape_payload(payload, default_add=self.interpolation_draw_add_radio.isChecked(), default_shape="rectangle")
        if parsed is None:
            return
        points, add, shape_kind = parsed.points, parsed.add, parsed.shape_kind
        if hasattr(self, "_push_interpolation_undo"):
            self._push_interpolation_undo("draw interpolation region")
        mask = self._merge_interpolation_polygon(points, add=add, shape_kind=shape_kind)
        if mask is not None:
            self._set_interpolation_mask(mask)

    def _merge_interpolation_polygon(
        self,
        points: list[tuple[float, float]],
        *,
        add: bool = True,
        shape_kind: str = "rectangle",
    ) -> np.ndarray | None:
        mask = self._ensure_interpolation_mask()
        if mask is None:
            return None
        built = build_polygon_mask(mask, points, shape_kind=shape_kind)
        if built is None:
            message = "Polygon needs at least three points." if shape_kind == "polygon" else "Rectangle needs four distinct corner points."
            self.statusBar().showMessage(message, 3000)
            self.frame_viewer.clear_interpolation_rect_points()
            return None
        shape_mask, _ordered_points, _geometry_shape = built
        mask[:] = merge_mask(mask, shape_mask, add=add)
        return mask
    def apply_interpolation_circle(self, payload: object) -> None:
        if hasattr(self, "video_state") and self.video_state is None:
            return
        if not isinstance(payload, tuple) or len(payload) < 2:
            return
        add = self.interpolation_draw_add_radio.isChecked()
        center, radius = payload[:2]
        if len(payload) >= 5 and isinstance(payload[4], bool):
            add = payload[4]
        if hasattr(self, "_push_interpolation_undo"):
            self._push_interpolation_undo("draw interpolation circle")
        mask = self._ensure_interpolation_mask()
        if mask is None:
            return
        start = payload[2] if len(payload) >= 4 else (center[0] - radius, center[1])
        end = payload[3] if len(payload) >= 4 else (center[0] + radius, center[1])
        shape_mask = build_circle_mask(mask, start, end)
        mask[:] = merge_mask(mask, shape_mask, add=add)
        self._set_interpolation_mask(mask)
    def apply_interpolation_free_segment(
        self,
        payload: tuple[tuple[float, float], tuple[float, float], bool],
    ) -> None:
        if hasattr(self, "video_state") and self.video_state is None:
            return
        if not getattr(self, "_interpolation_free_undo_open", False):
            if hasattr(self, "_push_interpolation_undo"):
                self._push_interpolation_undo("brush interpolation region")
            self._interpolation_free_undo_open = True
        mask = self._ensure_interpolation_mask()
        if mask is None:
            return
        start, end, add = payload
        paint_brush_segment(
            mask,
            start,
            end,
            radius=int(self.interpolation_brush_spinbox.value()),
            add=bool(add),
        )
        now = time.monotonic()
        if now - self._last_interpolation_preview_refresh >= 0.03:
            self._last_interpolation_preview_refresh = now
            self.frame_viewer.set_interpolation_mask(mask, high_quality=False)

    def _finalize_interpolation_free_draw(self) -> None:
        self._set_interpolation_mask(self.interpolation_mask)
        self._interpolation_free_undo_open = False

    def reset_interpolation_region(self) -> None:
        if self._has_interpolation_region() and hasattr(self, "_push_interpolation_undo"):
            self._push_interpolation_undo("clear interpolation region")
        self._clear_interpolation_region()
        self._set_interpolation_removal_mode("none")
        self._refresh_interpolation_ui()

    def save_interpolated_csv(self) -> None:
        if self.csv_df is None or self.video_state is None or not self.bodyparts:
            QMessageBox.warning(self, "Interpolation", "Load a video and CSV first.")
            return
        removal_mode = self._selected_interpolation_removal_mode()
        anchor = self._selected_interpolation_anchor()
        interpolate = self._interpolation_enabled()
        if removal_mode != "none" and not self._has_interpolation_region():
            QMessageBox.warning(self, "Interpolation", "Draw an automatic-removal region first.")
            return
        if removal_mode != "none" and anchor is None:
            QMessageBox.warning(self, "Interpolation", "Select an anchor node for automatic removal.")
            return
        if removal_mode == "none" and not interpolate:
            QMessageBox.warning(self, "Interpolation", "Enable automatic removal or interpolation first.")
            return
        output_path = self._interpolation_output_path()
        if output_path is None:
            return
        try:
            result_df = build_interpolation_pipeline_dataframe(
                self.csv_df,
                self.bodyparts,
                self.interpolation_mask,
                self.video_state.width,
                self.video_state.height,
                removal_mode,
                anchor,
                interpolate,
                self._interpolation_extrapolation_enabled(),
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_path, index=False)
        except Exception as exc:
            QMessageBox.warning(self, "Interpolation", f"Could not save interpolated CSV.\n\n{exc}")
            return
        self.statusBar().showMessage(f"Removal/interpolation pipeline CSV saved: {output_path}")
