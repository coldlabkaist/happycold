from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_API", "pyqt6")

import cv2
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QKeySequence, QMouseEvent, QPainter, QPen, QPixmap, QPolygonF, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


APP_NAME = "happycold"
DEFAULT_SAVE_DIR = Path.cwd() / "output"
APP_ICON_PATH = Path(__file__).resolve().parent / "CoLD_icon.png"


def get_settings_path() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        settings_dir = Path(local_appdata) / APP_NAME
    else:
        settings_dir = Path.home() / f".{APP_NAME}"
    return settings_dir / "happycold_setting.json"


SETTINGS_PATH = get_settings_path()

from happycold_shared import (
    MaskRecord,
    PinRecord,
    RoomRecord,
    VideoState,
    adjust_mask_by_mode,
    build_chamber_mark_dataframe,
    build_circle_detection_dataframe,
    build_normalized_dataframe,
    build_occlusion_dataframe,
    bodyparts_from_dataframe,
    discover_videos,
    infer_pixel_scale,
)
from tab_mixins import ChamberTabMixin, CircleTabMixin, OcclusionTabMixin, PinTabMixin, SquareTabMixin


class TrajectoryPreviewDialog(QDialog):
    TRACK_COLUMN_CANDIDATES = ("track", "track_id", "track id")
    FRAME_COLUMN_CANDIDATES = ("frame idx", "frame_idx", "frame index", "frame_index", "frame")

    def __init__(
        self,
        df: pd.DataFrame,
        bodyparts: list[str],
        normalized: bool,
        frame_width: int | None,
        frame_height: int | None,
        video_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        mode_label = "Normalized" if normalized else "Raw"
        self.setWindowTitle(f"{APP_NAME} - {mode_label} Trajectory Preview")
        self.resize(1100, 760)
        self._default_export_name = f"{video_name or 'trajectory'}_trajectory"

        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(10, 7), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.canvas)

        if not bodyparts:
            empty_label = QLabel("No bodyparts found in this CSV.")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(empty_label)
            return

        colors = ["#2563eb", "#ea580c", "#16a34a", "#dc2626", "#7c3aed", "#0891b2", "#4f46e5", "#a16207"]
        columns = min(3, max(1, len(bodyparts)))
        rows = math.ceil(len(bodyparts) / columns)
        track_col = self._find_matching_column(df, self.TRACK_COLUMN_CANDIDATES)
        frame_col = self._find_matching_column(df, self.FRAME_COLUMN_CANDIDATES)
        x_limit, y_limit = self._plot_limits(df, bodyparts, normalized, frame_width, frame_height)

        for index, bodypart in enumerate(bodyparts, start=1):
            ax = self.figure.add_subplot(rows, columns, index)
            self._plot_bodypart(ax, df, bodypart, track_col, frame_col, colors)
            ax.set_title(bodypart, fontsize=10)
            ax.set_xlim(0, x_limit)
            ax.set_ylim(y_limit, 0)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
        self.canvas.draw_idle()

    def _show_context_menu(self, position: QPoint) -> None:
        menu = QMenu(self)
        save_action = menu.addAction("Save As...")
        selected_action = menu.exec(self.canvas.mapToGlobal(position))
        if selected_action == save_action:
            self._save_figure_as()

    def _save_figure_as(self) -> None:
        start_dir = str(Path.cwd() / f"{self._default_export_name}.png")
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Trajectory Image",
            start_dir,
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;SVG Image (*.svg);;PDF Document (*.pdf)",
        )
        if not file_path:
            return

        output_path = Path(file_path)
        if output_path.suffix == "" and selected_filter:
            if "PNG" in selected_filter:
                output_path = output_path.with_suffix(".png")
            elif "JPEG" in selected_filter:
                output_path = output_path.with_suffix(".jpg")
            elif "SVG" in selected_filter:
                output_path = output_path.with_suffix(".svg")
            elif "PDF" in selected_filter:
                output_path = output_path.with_suffix(".pdf")
        self.figure.savefig(output_path)

    @staticmethod
    def _canonical_column_name(name: str) -> str:
        return "".join(character.lower() for character in str(name) if character.isalnum())

    @classmethod
    def _find_matching_column(cls, df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
        lookup = {cls._canonical_column_name(column): column for column in df.columns}
        for candidate in candidates:
            match = lookup.get(cls._canonical_column_name(candidate))
            if match is not None:
                return match
        return None

    @staticmethod
    def _plot_limits(
        df: pd.DataFrame,
        bodyparts: list[str],
        normalized: bool,
        frame_width: int | None,
        frame_height: int | None,
    ) -> tuple[float, float]:
        if normalized or frame_width is None or frame_height is None:
            return 1.0, 1.0
        x_columns = [f"{bodypart}.x" for bodypart in bodyparts if f"{bodypart}.x" in df.columns]
        y_columns = [f"{bodypart}.y" for bodypart in bodyparts if f"{bodypart}.y" in df.columns]
        return (
            TrajectoryPreviewDialog._axis_extent(df, x_columns, frame_width),
            TrajectoryPreviewDialog._axis_extent(df, y_columns, frame_height),
        )

    @staticmethod
    def _axis_extent(df: pd.DataFrame, columns: list[str], frame_extent: int) -> float:
        if not columns:
            return float(frame_extent)
        has_pixel_scale = any(infer_pixel_scale(df[column], frame_extent) == 1.0 for column in columns)
        return float(frame_extent if has_pixel_scale else 1.0)

    @staticmethod
    def _track_groups(df: pd.DataFrame, track_col: str | None, frame_col: str | None) -> list[tuple[str | None, pd.DataFrame]]:
        if track_col is None:
            return [(None, df)]

        groups: list[tuple[str | None, pd.DataFrame]] = []
        for track_value, group_df in df.groupby(track_col, dropna=False, sort=False):
            if frame_col is not None:
                group_df = (
                    group_df.assign(_trajectory_order=pd.to_numeric(group_df[frame_col], errors="coerce"))
                    .sort_values("_trajectory_order", kind="stable")
                    .drop(columns="_trajectory_order")
                )
            groups.append((None if pd.isna(track_value) else str(track_value), group_df))
        return groups

    def _plot_bodypart(
        self,
        ax,
        df: pd.DataFrame,
        bodypart: str,
        track_col: str | None,
        frame_col: str | None,
        colors: list[str],
    ) -> None:
        groups = self._track_groups(df, track_col, frame_col)
        for index, (track_label, group_df) in enumerate(groups):
            ax.plot(
                pd.to_numeric(group_df[f"{bodypart}.x"], errors="coerce"),
                pd.to_numeric(group_df[f"{bodypart}.y"], errors="coerce"),
                color=colors[index % len(colors)],
                linewidth=1.1,
                label=track_label,
            )
        if track_col is not None and len(groups) > 1:
            ax.legend(fontsize=7, loc="best")


class FrameViewer(QWidget):
    square_points_changed = pyqtSignal()
    chamber_rect_points_changed = pyqtSignal()
    chamber_rect_completed = pyqtSignal(object)
    chamber_circle_changed = pyqtSignal()
    chamber_circle_completed = pyqtSignal(object)
    circle_changed = pyqtSignal()
    view_changed = pyqtSignal()
    pin_added = pyqtSignal(object)
    occ_rect_points_changed = pyqtSignal()
    occ_rect_completed = pyqtSignal(object)
    occ_circle_changed = pyqtSignal()
    occ_circle_completed = pyqtSignal(object)
    occ_margin_points_changed = pyqtSignal()
    free_draw_segment = pyqtSignal(object)
    free_draw_finished = pyqtSignal()
    occ_transform_requested = pyqtSignal(object)
    occ_scale_requested = pyqtSignal(float)
    occ_transform_erase_requested = pyqtSignal(object)
    occ_transform_erase_segment_requested = pyqtSignal(object)
    occ_transform_finished = pyqtSignal()
    occ_mask_double_clicked = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(920, 580)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._pixmap: QPixmap | None = None
        self._frame_width = 0
        self._frame_height = 0
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0.0, 0.0)

        self.mode = "square"
        self.margin_value = 0.0
        self.free_draw_add = True

        self.square_points: list[tuple[float, float]] = []
        self._square_drag_index: int | None = None
        self.chamber_rect_points: list[tuple[float, float]] = []
        self.chamber_circle_start: tuple[float, float] | None = None
        self.chamber_circle_end: tuple[float, float] | None = None
        self._chamber_circle_dragging = False
        self._chamber_circle_current: tuple[float, float] | None = None
        self.circle_start: tuple[float, float] | None = None
        self.circle_end: tuple[float, float] | None = None
        self._circle_dragging = False
        self._circle_current: tuple[float, float] | None = None
        self._circle_move_dragging = False
        self._circle_move_last_point: tuple[float, float] | None = None

        self.occ_rect_points: list[tuple[float, float]] = []
        self.occ_margin_points: list[tuple[float, float]] = []
        self.occ_circle_start: tuple[float, float] | None = None
        self.occ_circle_end: tuple[float, float] | None = None
        self._occ_circle_dragging = False
        self._occ_circle_current: tuple[float, float] | None = None
        self.occ_margin_pick_mode = False
        self._occ_margin_drag_index: int | None = None
        self._free_dragging = False
        self._free_last_point: tuple[float, float] | None = None
        self._occ_rect_draw_override_add: bool | None = None
        self._occ_circle_draw_override_add: bool | None = None
        self.occ_transform_mode = False
        self._occ_transform_dragging = False
        self._occ_transform_last_point: tuple[float, float] | None = None
        self._occ_transform_erase_dragging = False
        self._occ_transform_erase_last_point: tuple[float, float] | None = None

        self._pan_dragging = False
        self._pan_drag_start = QPoint()
        self._pan_start_offset = QPointF(0.0, 0.0)

        self.pin_records: list[PinRecord] = []
        self.mask_records: dict[str, MaskRecord] = {}
        self.selected_mask_name: str | None = None
        self.chamber_mask: np.ndarray | None = None
        self.room_records: dict[str, RoomRecord] = {}
        self.selected_room_name: str | None = None
        self._mask_fill_cache: dict[str, QPixmap] = {}
        self._mask_margin_cache: dict[str, QPixmap] = {}
        self._chamber_base_cache: QPixmap | None = None
        self._chamber_fill_cache: QPixmap | None = None
        self._chamber_edge_cache: QPixmap | None = None

    def set_frame(self, frame_rgb: np.ndarray) -> None:
        frame_rgb = np.ascontiguousarray(frame_rgb)
        height, width, channels = frame_rgb.shape
        image = QImage(frame_rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self._frame_width = width
        self._frame_height = height
        self.update()

    def has_frame(self) -> bool:
        return self._pixmap is not None and self._frame_width > 0 and self._frame_height > 0

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.update()

    def set_margin_value(self, margin_value: float) -> None:
        self.margin_value = margin_value
        self.update()
        self.circle_changed.emit()
        self.occ_circle_changed.emit()

    def set_occ_transform_mode(self, enabled: bool) -> None:
        self.occ_transform_mode = enabled

    def set_occ_margin_pick_mode(self, enabled: bool) -> None:
        self.occ_margin_pick_mode = enabled
        if not enabled:
            self._occ_margin_drag_index = None
        self.update()

    def set_pin_records(self, pins: list[PinRecord]) -> None:
        self.pin_records = list(pins)
        self.update()

    def set_chamber_records(
        self,
        chamber_mask: np.ndarray | None,
        rooms: dict[str, RoomRecord],
        selected_name: str | None,
        refresh: bool = False,
    ) -> None:
        previous_names = set(self.room_records.keys())
        previous_selected = self.selected_room_name
        previous_has_chamber = self.chamber_mask is not None and bool(np.any(self.chamber_mask))
        next_has_chamber = chamber_mask is not None and bool(np.any(chamber_mask))
        self.chamber_mask = None if chamber_mask is None else chamber_mask.astype(np.uint8)
        self.room_records = rooms
        self.selected_room_name = selected_name
        if refresh or previous_names != set(rooms.keys()) or previous_selected != selected_name or previous_has_chamber != next_has_chamber:
            self._rebuild_chamber_cache()
        self.update()

    def set_mask_records(self, masks: dict[str, MaskRecord], selected_name: str | None, refresh: bool = False) -> None:
        previous_names = set(self.mask_records.keys())
        previous_selected = self.selected_mask_name
        self.mask_records = masks
        self.selected_mask_name = selected_name
        if refresh or previous_names != set(masks.keys()) or previous_selected != selected_name:
            self._rebuild_mask_cache()
        self.update()

    def refresh_mask_record(self, name: str, include_margin: bool = True) -> None:
        record = self.mask_records.get(name)
        if record is None:
            self._mask_fill_cache.pop(name, None)
            self._mask_margin_cache.pop(name, None)
            self.update()
            return
        self._rebuild_single_mask_cache(name, record, include_margin=include_margin)
        self.update()

    def _rebuild_mask_cache(self) -> None:
        self._mask_fill_cache = {}
        self._mask_margin_cache = {}
        if self._frame_width <= 0 or self._frame_height <= 0:
            return

        for name, record in self.mask_records.items():
            self._rebuild_single_mask_cache(name, record)

    def _rebuild_single_mask_cache(self, name: str, record: MaskRecord, include_margin: bool = True) -> None:
        if self._frame_width <= 0 or self._frame_height <= 0:
            return
        base = record.mask.astype(bool)
        selected = name == self.selected_mask_name
        alpha = 105 if selected else 55

        image = np.zeros((self._frame_height, self._frame_width, 4), dtype=np.uint8)
        image[base, 0] = record.color.red()
        image[base, 1] = record.color.green()
        image[base, 2] = record.color.blue()
        image[base, 3] = alpha
        qimg = QImage(image.data, self._frame_width, self._frame_height, self._frame_width * 4, QImage.Format.Format_RGBA8888).copy()
        self._mask_fill_cache[name] = QPixmap.fromImage(qimg)

        self._mask_margin_cache.pop(name, None)
        if not include_margin:
            return
        try:
            adjusted = adjust_mask_by_mode(record.mask, record.margin, record.margin_mode, self.occ_margin_points).astype(np.uint8)
        except ValueError:
            adjusted = None
        if adjusted is not None and adjusted.any():
            edge = cv2.morphologyEx(adjusted, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
            edge_alpha = 220 if selected else 180
            edge_image = np.zeros((self._frame_height, self._frame_width, 4), dtype=np.uint8)
            edge_image[edge > 0, 0] = record.color.red()
            edge_image[edge > 0, 1] = record.color.green()
            edge_image[edge > 0, 2] = record.color.blue()
            edge_image[edge > 0, 3] = edge_alpha
            edge_qimg = QImage(edge_image.data, self._frame_width, self._frame_height, self._frame_width * 4, QImage.Format.Format_RGBA8888).copy()
            self._mask_margin_cache[name] = QPixmap.fromImage(edge_qimg)

    def _rebuild_chamber_cache(self) -> None:
        self._chamber_base_cache = None
        self._chamber_fill_cache = None
        self._chamber_edge_cache = None
        if self._frame_width <= 0 or self._frame_height <= 0:
            return

        if self.chamber_mask is not None and np.any(self.chamber_mask):
            base_image = np.zeros((self._frame_height, self._frame_width, 4), dtype=np.uint8)
            chamber = self.chamber_mask.astype(bool)
            base_image[chamber, 0] = 209
            base_image[chamber, 1] = 213
            base_image[chamber, 2] = 219
            base_image[chamber, 3] = 72
            qimg = QImage(base_image.data, self._frame_width, self._frame_height, self._frame_width * 4, QImage.Format.Format_RGBA8888).copy()
            self._chamber_base_cache = QPixmap.fromImage(qimg)

        if not self.room_records:
            return

        fill_image = np.zeros((self._frame_height, self._frame_width, 4), dtype=np.uint8)
        edge_image = np.zeros((self._frame_height, self._frame_width, 4), dtype=np.uint8)
        for name, record in self.room_records.items():
            base = record.mask.astype(bool)
            if not np.any(base):
                continue
            selected = name == self.selected_room_name
            fill_alpha = 118 if selected else 86
            fill_image[base, 0] = record.color.red()
            fill_image[base, 1] = record.color.green()
            fill_image[base, 2] = record.color.blue()
            fill_image[base, 3] = fill_alpha

            edge = cv2.morphologyEx(record.mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
            edge_alpha = 245 if selected else 205
            edge_image[edge > 0, 0] = record.color.red()
            edge_image[edge > 0, 1] = record.color.green()
            edge_image[edge > 0, 2] = record.color.blue()
            edge_image[edge > 0, 3] = edge_alpha

        fill_qimg = QImage(fill_image.data, self._frame_width, self._frame_height, self._frame_width * 4, QImage.Format.Format_RGBA8888).copy()
        self._chamber_fill_cache = QPixmap.fromImage(fill_qimg)
        edge_qimg = QImage(edge_image.data, self._frame_width, self._frame_height, self._frame_width * 4, QImage.Format.Format_RGBA8888).copy()
        self._chamber_edge_cache = QPixmap.fromImage(edge_qimg)

    def reset_view(self) -> None:
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0.0, 0.0)
        self.update()
        self.view_changed.emit()

    def clear_square_points(self) -> None:
        self.square_points.clear()
        self.update()
        self.square_points_changed.emit()

    def clear_chamber_rect_points(self) -> None:
        self.chamber_rect_points.clear()
        self.update()
        self.chamber_rect_points_changed.emit()

    def clear_chamber_circle(self) -> None:
        self.chamber_circle_start = None
        self.chamber_circle_end = None
        self._chamber_circle_current = None
        self._chamber_circle_dragging = False
        self.update()
        self.chamber_circle_changed.emit()

    def clear_circle(self) -> None:
        self.circle_start = None
        self.circle_end = None
        self._circle_current = None
        self._circle_dragging = False
        self._circle_move_dragging = False
        self._circle_move_last_point = None
        self.update()
        self.circle_changed.emit()

    def clear_occ_rect_points(self) -> None:
        self.occ_rect_points.clear()
        self._occ_rect_draw_override_add = None
        self.update()
        self.occ_rect_points_changed.emit()

    def clear_occ_margin_points(self) -> None:
        self.occ_margin_points.clear()
        self._occ_margin_drag_index = None
        self.update()
        self.occ_margin_points_changed.emit()

    def clear_occ_circle(self) -> None:
        self.occ_circle_start = None
        self.occ_circle_end = None
        self._occ_circle_current = None
        self._occ_circle_dragging = False
        self._occ_circle_draw_override_add = None
        self.update()
        self.occ_circle_changed.emit()

    def _fit_rect(self) -> QRectF | None:
        if not self._pixmap:
            return None
        scaled = self._pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRectF((self.width() - scaled.width()) / 2.0, (self.height() - scaled.height()) / 2.0, scaled.width(), scaled.height())

    def _clamped_pan_offset(self, pan_offset: QPointF | None = None, zoom_factor: float | None = None) -> QPointF:
        fit_rect = self._fit_rect()
        if fit_rect is None:
            return QPointF(0.0, 0.0)

        offset = QPointF(self.pan_offset if pan_offset is None else pan_offset)
        zoom = self.zoom_factor if zoom_factor is None else zoom_factor
        if zoom <= 1.0:
            return QPointF(0.0, 0.0)

        max_x = max(0.0, (fit_rect.width() * zoom - fit_rect.width()) / 2.0)
        max_y = max(0.0, (fit_rect.height() * zoom - fit_rect.height()) / 2.0)
        offset.setX(max(-max_x, min(max_x, offset.x())))
        offset.setY(max(-max_y, min(max_y, offset.y())))
        return offset

    def _image_rect(self) -> QRectF | None:
        fit_rect = self._fit_rect()
        if fit_rect is None:
            return None
        center = fit_rect.center() + self._clamped_pan_offset()
        width = fit_rect.width() * self.zoom_factor
        height = fit_rect.height() * self.zoom_factor
        return QRectF(center.x() - width / 2.0, center.y() - height / 2.0, width, height)

    def _widget_to_image(self, position: QPointF) -> tuple[float, float] | None:
        rect = self._image_rect()
        if rect is None or rect.width() <= 0 or rect.height() <= 0 or not rect.contains(position):
            return None
        return (
            ((position.x() - rect.left()) / rect.width()) * self._frame_width,
            ((position.y() - rect.top()) / rect.height()) * self._frame_height,
        )

    def _image_to_widget(self, point: tuple[float, float]) -> QPointF | None:
        rect = self._image_rect()
        if rect is None or self._frame_width <= 0 or self._frame_height <= 0:
            return None
        return QPointF(rect.left() + (point[0] / self._frame_width) * rect.width(), rect.top() + (point[1] / self._frame_height) * rect.height())

    def _apply_zoom(self, new_zoom: float, anchor_widget: QPointF | None = None, anchor_image: tuple[float, float] | None = None) -> None:
        new_zoom = max(1.0, min(new_zoom, 8.0))
        if abs(new_zoom - self.zoom_factor) < 1e-6:
            return
        if anchor_widget is not None and anchor_image is not None:
            self.zoom_factor = new_zoom
            rect = self._image_rect()
            if rect is not None:
                target_x = rect.left() + (anchor_image[0] / self._frame_width) * rect.width()
                target_y = rect.top() + (anchor_image[1] / self._frame_height) * rect.height()
                self.pan_offset += QPointF(anchor_widget.x() - target_x, anchor_widget.y() - target_y)
        else:
            self.zoom_factor = new_zoom
        self.pan_offset = self._clamped_pan_offset()
        self.update()
        self.view_changed.emit()

    def _point_hit_index(self, position: QPointF, points: list[tuple[float, float]], radius: float = 12.0) -> int | None:
        best_index: int | None = None
        best_distance = radius
        for index, point in enumerate(points):
            widget_point = self._image_to_widget(point)
            if widget_point is None:
                continue
            distance = math.hypot(position.x() - widget_point.x(), position.y() - widget_point.y())
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        return best_index

    @staticmethod
    def _mask_hit_distance(mask: np.ndarray, point: tuple[float, float], margin: int) -> float | None:
        if mask.ndim != 2 or not np.any(mask):
            return None
        height, width = mask.shape
        x = int(round(point[0]))
        y = int(round(point[1]))
        if not (0 <= x < width and 0 <= y < height):
            return None
        if mask[y, x] > 0:
            return 0.0
        radius = max(0, int(margin))
        if radius == 0:
            return None
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        roi = mask[y0:y1, x0:x1]
        ys, xs = np.where(roi > 0)
        if len(xs) == 0:
            return None
        dx = (xs + x0).astype(np.float32) - float(x)
        dy = (ys + y0).astype(np.float32) - float(y)
        min_distance_sq = float(np.min(dx * dx + dy * dy))
        if min_distance_sq <= float(radius * radius):
            return math.sqrt(min_distance_sq)
        return None

    def _mask_name_at_point(self, point: tuple[float, float], margin: int = 8) -> str | None:
        if not self.mask_records:
            return None
        best_name: str | None = None
        best_distance = float("inf")
        ordered_names = list(self.mask_records.keys())
        if self.selected_mask_name in self.mask_records:
            ordered_names = [self.selected_mask_name] + [name for name in ordered_names if name != self.selected_mask_name]
        for name in ordered_names:
            record = self.mask_records.get(name)
            if record is None:
                continue
            distance = self._mask_hit_distance(record.mask.astype(np.uint8), point, margin)
            if distance is None:
                continue
            if distance < best_distance:
                best_distance = distance
                best_name = name
                if distance <= 0.0:
                    break
        return best_name

    @staticmethod
    def _control_pressed(modifiers: Qt.KeyboardModifiers) -> bool:
        return bool(modifiers & Qt.KeyboardModifier.ControlModifier)

    def _effective_draw_add_with_modifiers(self, modifiers: Qt.KeyboardModifiers) -> bool:
        if self._control_pressed(modifiers):
            return False
        return self.free_draw_add

    def wheelEvent(self, event) -> None:
        if not self.has_frame():
            return
        anchor_widget = event.position()
        anchor_image = self._widget_to_image(anchor_widget)
        factor = 1.15 if event.angleDelta().y() > 0 else (1 / 1.15)
        self._apply_zoom(self.zoom_factor * factor, anchor_widget, anchor_image)
        event.accept()

    def circle_geometry(self) -> tuple[tuple[float, float], float, float] | None:
        end_point = self._circle_current if self._circle_dragging and self._circle_current else self.circle_end
        if not self.circle_start or not end_point:
            return None
        center = ((self.circle_start[0] + end_point[0]) / 2.0, (self.circle_start[1] + end_point[1]) / 2.0)
        base_radius = math.dist(self.circle_start, end_point) / 2.0
        return center, base_radius, max(1.0, base_radius + self.margin_value)

    def chamber_circle_geometry(self) -> tuple[tuple[float, float], float, tuple[float, float], tuple[float, float]] | None:
        end_point = self._chamber_circle_current if self._chamber_circle_dragging and self._chamber_circle_current else self.chamber_circle_end
        if not self.chamber_circle_start or not end_point:
            return None
        center = ((self.chamber_circle_start[0] + end_point[0]) / 2.0, (self.chamber_circle_start[1] + end_point[1]) / 2.0)
        base_radius = math.dist(self.chamber_circle_start, end_point) / 2.0
        return center, base_radius, self.chamber_circle_start, end_point

    def occ_circle_geometry(self) -> tuple[tuple[float, float], float, float] | None:
        end_point = self._occ_circle_current if self._occ_circle_dragging and self._occ_circle_current else self.occ_circle_end
        if not self.occ_circle_start or not end_point:
            return None
        center = ((self.occ_circle_start[0] + end_point[0]) / 2.0, (self.occ_circle_start[1] + end_point[1]) / 2.0)
        base_radius = math.dist(self.occ_circle_start, end_point) / 2.0
        return center, base_radius, max(1.0, base_radius + self.margin_value)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus()
        if not self.has_frame():
            return

        if event.button() == Qt.MouseButton.RightButton:
            self._pan_dragging = True
            self._pan_drag_start = event.position().toPoint()
            self._pan_start_offset = QPointF(self.pan_offset)
            return

        image_point = self._widget_to_image(event.position())
        if image_point is None or event.button() != Qt.MouseButton.LeftButton:
            return

        if self.mode.startswith("occ_") and self.occ_margin_pick_mode:
            hit_index = self._point_hit_index(event.position(), self.occ_margin_points)
            if hit_index is not None:
                self._occ_margin_drag_index = hit_index
            elif len(self.occ_margin_points) < 4:
                self.occ_margin_points.append(image_point)
                self.update()
                self.occ_margin_points_changed.emit()
        elif self.mode.startswith("occ_") and self.occ_transform_mode:
            if bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self._occ_transform_erase_dragging = True
                self._occ_transform_erase_last_point = image_point
                self.occ_transform_erase_requested.emit(image_point)
                return
            self._occ_transform_dragging = True
            self._occ_transform_last_point = image_point
        elif self.mode == "pin":
            self.pin_added.emit(image_point)
        elif self.mode == "chamber_rect":
            self.chamber_rect_points.append(image_point)
            self.update()
            self.chamber_rect_points_changed.emit()
            if len(self.chamber_rect_points) == 4:
                self.chamber_rect_completed.emit(list(self.chamber_rect_points))
                self.chamber_rect_points.clear()
                self.update()
                self.chamber_rect_points_changed.emit()
        elif self.mode == "chamber_circle":
            self._chamber_circle_dragging = True
            self.chamber_circle_start = image_point
            self.chamber_circle_end = None
            self._chamber_circle_current = image_point
            self.update()
            self.chamber_circle_changed.emit()
        elif self.mode == "square":
            hit_index = self._point_hit_index(event.position(), self.square_points)
            if hit_index is not None:
                self._square_drag_index = hit_index
            elif len(self.square_points) < 4:
                self.square_points.append(image_point)
                self.update()
                self.square_points_changed.emit()
        elif self.mode == "circle":
            geometry = self.circle_geometry()
            if geometry is not None:
                center, base_radius, _ = geometry
                if math.dist(center, image_point) <= max(8.0, base_radius):
                    self._circle_move_dragging = True
                    self._circle_move_last_point = image_point
                    return
            self._circle_dragging = True
            self.circle_start = image_point
            self.circle_end = None
            self._circle_current = image_point
            self.update()
            self.circle_changed.emit()
        elif self.mode == "occ_rect":
            if len(self.occ_rect_points) == 0:
                self._occ_rect_draw_override_add = False if self._control_pressed(event.modifiers()) else None
            elif self._occ_rect_draw_override_add is None and self._control_pressed(event.modifiers()):
                self._occ_rect_draw_override_add = False
            self.occ_rect_points.append(image_point)
            self.update()
            self.occ_rect_points_changed.emit()
            if len(self.occ_rect_points) == 4:
                add_value = self.free_draw_add if self._occ_rect_draw_override_add is None else self._occ_rect_draw_override_add
                self.occ_rect_completed.emit((list(self.occ_rect_points), bool(add_value)))
                self.occ_rect_points.clear()
                self._occ_rect_draw_override_add = None
                self.update()
                self.occ_rect_points_changed.emit()
        elif self.mode == "occ_circle":
            self._occ_circle_dragging = True
            self.occ_circle_start = image_point
            self.occ_circle_end = None
            self._occ_circle_current = image_point
            self._occ_circle_draw_override_add = False if self._control_pressed(event.modifiers()) else None
            self.update()
            self.occ_circle_changed.emit()
        elif self.mode == "occ_free":
            self._free_dragging = True
            self._free_last_point = image_point
            self.free_draw_segment.emit((image_point, image_point, self._effective_draw_add_with_modifiers(event.modifiers())))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._circle_move_dragging and not (event.buttons() & Qt.MouseButton.LeftButton):
            self._circle_move_dragging = False
            self._circle_move_last_point = None
            self.circle_changed.emit()
        if self._square_drag_index is not None and not (event.buttons() & Qt.MouseButton.LeftButton):
            self._square_drag_index = None
            self.square_points_changed.emit()
        if self._occ_margin_drag_index is not None and not (event.buttons() & Qt.MouseButton.LeftButton):
            self._occ_margin_drag_index = None
            self.occ_margin_points_changed.emit()
        if self._occ_transform_dragging and not (event.buttons() & Qt.MouseButton.LeftButton):
            self._occ_transform_dragging = False
            self._occ_transform_last_point = None
            self.occ_transform_finished.emit()
        if self._occ_transform_erase_dragging and not (event.buttons() & Qt.MouseButton.LeftButton):
            self._occ_transform_erase_dragging = False
            self._occ_transform_erase_last_point = None

        if self._pan_dragging:
            delta = event.position().toPoint() - self._pan_drag_start
            self.pan_offset = self._clamped_pan_offset(self._pan_start_offset + QPointF(delta.x(), delta.y()))
            self.update()
            self.view_changed.emit()
            return

        image_point = self._widget_to_image(event.position())
        if image_point is None:
            return

        if self.mode.startswith("occ_") and self.occ_margin_pick_mode and self._occ_margin_drag_index is not None:
            self.occ_margin_points[self._occ_margin_drag_index] = image_point
            self.update()
            self.occ_margin_points_changed.emit()
        elif self.mode == "circle" and self._circle_dragging:
            self._circle_current = image_point
            self.update()
            self.circle_changed.emit()
        elif self.mode == "chamber_circle" and self._chamber_circle_dragging:
            self._chamber_circle_current = image_point
            self.update()
            self.chamber_circle_changed.emit()
        elif self.mode == "circle" and self._circle_move_dragging and self._circle_move_last_point is not None:
            dx = image_point[0] - self._circle_move_last_point[0]
            dy = image_point[1] - self._circle_move_last_point[1]
            if self.circle_start is not None and self.circle_end is not None and (dx != 0 or dy != 0):
                self.circle_start = (self.circle_start[0] + dx, self.circle_start[1] + dy)
                self.circle_end = (self.circle_end[0] + dx, self.circle_end[1] + dy)
                self._circle_move_last_point = image_point
                self.update()
                self.circle_changed.emit()
        elif self.mode == "occ_circle" and self._occ_circle_dragging:
            self._occ_circle_current = image_point
            self.update()
            self.occ_circle_changed.emit()
        elif self.mode == "occ_free" and self._free_dragging and self._free_last_point is not None:
            self.free_draw_segment.emit((self._free_last_point, image_point, self._effective_draw_add_with_modifiers(event.modifiers())))
            self._free_last_point = image_point
        elif self.mode == "square" and self._square_drag_index is not None:
            self.square_points[self._square_drag_index] = image_point
            self.update()
            self.square_points_changed.emit()
        elif self.mode.startswith("occ_") and self.occ_transform_mode and self._occ_transform_erase_dragging:
            if self._occ_transform_erase_last_point is not None:
                self.occ_transform_erase_segment_requested.emit((self._occ_transform_erase_last_point, image_point))
            else:
                self.occ_transform_erase_requested.emit(image_point)
            self._occ_transform_erase_last_point = image_point
        elif self.mode.startswith("occ_") and self.occ_transform_mode and self._occ_transform_dragging and self._occ_transform_last_point is not None:
            dx = int(round(image_point[0] - self._occ_transform_last_point[0]))
            dy = int(round(image_point[1] - self._occ_transform_last_point[1]))
            if dx != 0 or dy != 0:
                self.occ_transform_requested.emit((dx, dy))
                self._occ_transform_last_point = image_point

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self._pan_dragging = False
            return

        image_point = self._widget_to_image(event.position())
        if self.mode == "circle" and self._circle_dragging and event.button() == Qt.MouseButton.LeftButton:
            self.circle_end = image_point if image_point is not None else self._circle_current
            self._circle_current = None
            self._circle_dragging = False
            self.update()
            self.circle_changed.emit()
        elif self.mode == "chamber_circle" and self._chamber_circle_dragging and event.button() == Qt.MouseButton.LeftButton:
            self.chamber_circle_end = image_point if image_point is not None else self._chamber_circle_current
            self._chamber_circle_current = None
            self._chamber_circle_dragging = False
            self.update()
            geometry = self.chamber_circle_geometry()
            self.chamber_circle_changed.emit()
            if geometry is not None:
                self.chamber_circle_completed.emit(geometry)
        elif self.mode == "circle" and self._circle_move_dragging and event.button() == Qt.MouseButton.LeftButton:
            self._circle_move_dragging = False
            self._circle_move_last_point = None
            self.update()
            self.circle_changed.emit()
        elif self.mode == "occ_circle" and self._occ_circle_dragging and event.button() == Qt.MouseButton.LeftButton:
            self.occ_circle_end = image_point if image_point is not None else self._occ_circle_current
            self._occ_circle_current = None
            self._occ_circle_dragging = False
            self.update()
            geometry = self.occ_circle_geometry()
            self.occ_circle_changed.emit()
            if geometry is not None:
                center, base_radius, _ = geometry
                add_value = self.free_draw_add if self._occ_circle_draw_override_add is None else self._occ_circle_draw_override_add
                self.occ_circle_completed.emit((center, base_radius, self.occ_circle_start, self.occ_circle_end, bool(add_value)))
            self._occ_circle_draw_override_add = None
        elif self.mode == "occ_free" and event.button() == Qt.MouseButton.LeftButton:
            self._free_dragging = False
            self._free_last_point = None
            self.free_draw_finished.emit()
        elif self.mode == "square" and event.button() == Qt.MouseButton.LeftButton:
            self._square_drag_index = None
        elif self.mode.startswith("occ_") and self.occ_margin_pick_mode and event.button() == Qt.MouseButton.LeftButton:
            self._occ_margin_drag_index = None
        elif self.mode.startswith("occ_") and self.occ_transform_mode and event.button() == Qt.MouseButton.LeftButton:
            self._occ_transform_dragging = False
            self._occ_transform_last_point = None
            self._occ_transform_erase_dragging = False
            self._occ_transform_erase_last_point = None
            self.occ_transform_finished.emit()

    def leaveEvent(self, event) -> None:
        self._pan_dragging = False
        self._chamber_circle_dragging = False
        self._chamber_circle_current = None
        self._circle_dragging = False
        self._circle_current = None
        self._circle_move_dragging = False
        self._circle_move_last_point = None
        self._square_drag_index = None
        self._occ_margin_drag_index = None
        self._free_dragging = False
        self._free_last_point = None
        self._occ_rect_draw_override_add = None
        self._occ_circle_dragging = False
        self._occ_circle_current = None
        self._occ_circle_draw_override_add = None
        self._occ_transform_dragging = False
        self._occ_transform_last_point = None
        self._occ_transform_erase_dragging = False
        self._occ_transform_erase_last_point = None
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.has_frame()
            and self.mode.startswith("occ_")
            and not self.occ_margin_pick_mode
        ):
            image_point = self._widget_to_image(event.position())
            if image_point is not None:
                name = self._mask_name_at_point(image_point, margin=8)
                if name is not None:
                    self._square_drag_index = None
                    self._occ_margin_drag_index = None
                    self._occ_transform_dragging = False
                    self._occ_transform_last_point = None
                    self._occ_transform_erase_dragging = False
                    self._occ_transform_erase_last_point = None
                    self._free_dragging = False
                    self._free_last_point = None
                    self._occ_rect_draw_override_add = None
                    self._occ_circle_draw_override_add = None
                    if self.occ_rect_points:
                        self.occ_rect_points.clear()
                        self._occ_rect_draw_override_add = None
                        self.occ_rect_points_changed.emit()
                    self.occ_mask_double_clicked.emit(name)
                    event.accept()
                    return
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#11161c"))

        if not self._pixmap:
            painter.setPen(QColor("#d0d7de"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a video to see frames here.")
            return

        rect = self._image_rect()
        if rect is None:
            return

        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
        painter.setPen(QPen(QColor("#5b6b7c"), 1))
        painter.drawRect(rect)
        self._draw_chamber_overlay(painter)
        self._draw_mask_overlays(painter)
        self._draw_pin_overlays(painter)
        self._draw_square_overlay(painter)
        self._draw_circle_overlay(painter)
        self._draw_occ_shape_overlay(painter)

    def _draw_chamber_overlay(self, painter: QPainter) -> None:
        if not self.mode.startswith("chamber_"):
            return
        rect = self._image_rect()
        if rect is None:
            return

        painter.save()
        if self._chamber_base_cache is not None:
            painter.drawPixmap(rect, self._chamber_base_cache, QRectF(self._chamber_base_cache.rect()))
        if self._chamber_fill_cache is not None:
            painter.drawPixmap(rect, self._chamber_fill_cache, QRectF(self._chamber_fill_cache.rect()))
        if self._chamber_edge_cache is not None:
            painter.drawPixmap(rect, self._chamber_edge_cache, QRectF(self._chamber_edge_cache.rect()))

        draft_color = QColor("#d1d5db")
        if self.selected_room_name is not None:
            selected = self.room_records.get(self.selected_room_name)
            if selected is not None:
                draft_color = selected.color

        if self.chamber_rect_points:
            widget_points = [self._image_to_widget(point) for point in self.chamber_rect_points]
            widget_points = [point for point in widget_points if point is not None]
            if len(widget_points) >= 3:
                painter.setBrush(QColor(draft_color.red(), draft_color.green(), draft_color.blue(), 55))
                painter.setPen(QPen(draft_color.lighter(145), 2))
                painter.drawPolygon(QPolygonF(widget_points))
            for index, point in enumerate(widget_points, start=1):
                painter.setBrush(draft_color)
                painter.setPen(QPen(QColor("#f8fafc"), 1))
                painter.drawEllipse(point, 5.0, 5.0)
                painter.drawText(point + QPointF(7, -7), str(index))

        geometry = self.chamber_circle_geometry()
        if geometry is not None:
            center, base_radius, _, _ = geometry
            center_point = self._image_to_widget(center)
            if center_point is not None:
                scale = min(rect.width() / self._frame_width, rect.height() / self._frame_height)
                painter.setBrush(QColor(draft_color.red(), draft_color.green(), draft_color.blue(), 55))
                painter.setPen(QPen(draft_color.lighter(145), 2))
                painter.drawEllipse(center_point, base_radius * scale, base_radius * scale)
        painter.restore()

    def _draw_mask_overlays(self, painter: QPainter) -> None:
        if not self.mask_records or not self.mode.startswith("occ_"):
            return
        painter.save()
        rect = self._image_rect()
        if rect is None:
            painter.restore()
            return
        for name, record in self.mask_records.items():
            fill_pixmap = self._mask_fill_cache.get(name)
            if fill_pixmap is not None:
                painter.drawPixmap(rect, fill_pixmap, QRectF(fill_pixmap.rect()))
            margin_pixmap = self._mask_margin_cache.get(name)
            if margin_pixmap is not None:
                painter.drawPixmap(rect, margin_pixmap, QRectF(margin_pixmap.rect()))
        painter.restore()

    def _draw_pin_overlays(self, painter: QPainter) -> None:
        if not self.pin_records or self.mode != "pin":
            return
        colors = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6", "#8b5cf6"]
        painter.save()
        painter.setPen(QPen(QColor("#ffffff"), 2))
        for index, pin in enumerate([pin for pin in self.pin_records if pin.frame == getattr(self, "current_frame_number", pin.frame)]):
            point = self._image_to_widget((pin.x, pin.y))
            if point is None:
                continue
            color = QColor(colors[index % len(colors)])
            painter.setBrush(color)
            painter.drawEllipse(point, 6.0, 6.0)
            painter.drawText(point + QPointF(8, -8), pin.pin_id)
        painter.restore()

    def _draw_square_overlay(self, painter: QPainter) -> None:
        if self.mode != "square":
            return
        widget_points = [self._image_to_widget(point) for point in self.square_points]
        widget_points = [point for point in widget_points if point is not None]
        if not widget_points:
            return
        painter.save()
        if len(widget_points) >= 3:
            painter.setBrush(QColor(37, 99, 235, 65))
            painter.setPen(QPen(QColor("#bfdbfe"), 2))
            painter.drawPolygon(QPolygonF(widget_points))
        painter.setPen(QPen(QColor("#f8fafc"), 2))
        for index, point in enumerate(widget_points, start=1):
            painter.setBrush(QColor("#1d4ed8"))
            painter.drawEllipse(point, 6.0, 6.0)
            painter.drawText(point + QPointF(8, -8), str(index))
        painter.restore()

    def _draw_circle_overlay(self, painter: QPainter) -> None:
        if self.mode != "circle":
            return
        geometry = self.circle_geometry()
        if geometry is None:
            return
        center, base_radius, adjusted_radius = geometry
        center_point = self._image_to_widget(center)
        rect = self._image_rect()
        if center_point is None or rect is None:
            return
        scale = min(rect.width() / self._frame_width, rect.height() / self._frame_height)
        painter.save()
        painter.setBrush(QColor(239, 68, 68, 65))
        painter.setPen(QPen(QColor("#fecaca"), 2))
        painter.drawEllipse(center_point, base_radius * scale, base_radius * scale)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#f97316"), 2, Qt.PenStyle.DashLine))
        painter.drawEllipse(center_point, adjusted_radius * scale, adjusted_radius * scale)
        painter.restore()

    def _draw_occ_shape_overlay(self, painter: QPainter) -> None:
        if not self.mode.startswith("occ_"):
            return
        painter.save()
        selected = self.mask_records.get(self.selected_mask_name) if self.selected_mask_name is not None else None
        selected_color = selected.color if selected is not None else QColor("#0891b2")
        if self.occ_margin_points:
            widget_points = [self._image_to_widget(point) for point in self.occ_margin_points]
            widget_points = [point for point in widget_points if point is not None]
            if len(widget_points) >= 3:
                painter.setBrush(QColor(16, 185, 129, 40))
                painter.setPen(QPen(QColor("#a7f3d0"), 2, Qt.PenStyle.DashLine))
                painter.drawPolygon(QPolygonF(widget_points))
            painter.setPen(QPen(QColor("#ecfdf5"), 2))
            for index, point in enumerate(widget_points, start=1):
                painter.setBrush(QColor("#10b981"))
                painter.drawEllipse(point, 5.0, 5.0)
                painter.drawText(point + QPointF(7, -7), str(index))
        if self.occ_rect_points:
            widget_points = [self._image_to_widget(point) for point in self.occ_rect_points]
            widget_points = [point for point in widget_points if point is not None]
            if len(widget_points) >= 3:
                painter.setBrush(QColor(selected_color.red(), selected_color.green(), selected_color.blue(), 55))
                painter.setPen(QPen(selected_color.lighter(140), 2))
                painter.drawPolygon(QPolygonF(widget_points))
            for index, point in enumerate(widget_points, start=1):
                painter.setBrush(selected_color)
                painter.drawEllipse(point, 5.0, 5.0)
                painter.drawText(point + QPointF(7, -7), str(index))
        geometry = self.occ_circle_geometry()
        if geometry is not None:
            center, base_radius, adjusted_radius = geometry
            center_point = self._image_to_widget(center)
            rect = self._image_rect()
            if center_point is not None and rect is not None:
                scale = min(rect.width() / self._frame_width, rect.height() / self._frame_height)
                painter.setBrush(QColor(selected_color.red(), selected_color.green(), selected_color.blue(), 55))
                painter.setPen(QPen(selected_color.lighter(140), 2))
                painter.drawEllipse(center_point, base_radius * scale, base_radius * scale)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(selected_color.lighter(170), 2, Qt.PenStyle.DashLine))
                painter.drawEllipse(center_point, adjusted_radius * scale, adjusted_radius * scale)
        painter.restore()


class MainWindow(SquareTabMixin, ChamberTabMixin, CircleTabMixin, PinTabMixin, OcclusionTabMixin, QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        if APP_ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.resize(1820, 1000)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.video_state: VideoState | None = None
        self.csv_path: Path | None = None
        self.csv_df: pd.DataFrame | None = None
        self.bodyparts: list[str] = []
        self.current_folder = Path()
        self.save_folder = DEFAULT_SAVE_DIR
        self.current_frame_number = 1
        self.current_frame_rgb: np.ndarray | None = None
        self._loading_slider = False
        self.settings = self._load_settings()
        self.csv_auto_candidates: list[Path] = []
        csv_manual_dir = str(self.settings.get("csv_manual_dir", "")).strip()
        self.csv_manual_folder: Path | None = None
        if csv_manual_dir:
            candidate = Path(csv_manual_dir)
            if candidate.exists():
                self.csv_manual_folder = candidate
        self._last_mask_preview_refresh = 0.0
        self._last_transform_preview_refresh = 0.0
        self._trajectory_preview_windows: list[TrajectoryPreviewDialog] = []

        self.pins: list[PinRecord] = []
        self.pin_counter = 0
        self.chamber_mask: np.ndarray | None = None
        self.room_records: dict[str, RoomRecord] = {}
        self.selected_room_name: str | None = None
        self.mask_records: dict[str, MaskRecord] = {}
        self.selected_mask_name: str | None = None
        self.default_mask_margin = int(self.settings.get("occlusion_mask_margin", 0))
        self.default_circle_margin = int(self.settings.get("circle_detection_margin", 0))
        self.default_mask_brush = int(self.settings.get("occlusion_mask_brush", 12))
        self.default_mask_margin_mode = str(self.settings.get("occlusion_mask_margin_mode", "simple"))
        if self.default_mask_margin_mode not in {"simple", "geometric"}:
            self.default_mask_margin_mode = "simple"

        self._build_ui()
        self._refresh_csv_search_folder_ui()
        self._connect_signals()
        self._register_shortcuts()
        self._initialize_directories()
        self.mask_margin_slider.setValue(self.default_mask_margin)
        self.circle_margin_slider.setValue(self.default_circle_margin)
        self.mask_brush_slider.setValue(self.default_mask_brush)
        if self.default_mask_margin_mode == "geometric":
            self.mask_margin_geometric_radio.setChecked(True)
        else:
            self.mask_margin_simple_radio.setChecked(True)
        self.mode_tabs.setCurrentIndex(int(self.settings.get("last_tab_index", 0)))
        self._refresh_chamber_ui()
        self._refresh_output_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)
        self.setCentralWidget(root)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter)
        self.main_splitter = splitter

        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(8)

        viewer_box = QGroupBox("Viewer")
        viewer_box_layout = QFormLayout(viewer_box)
        self.viewer_help_label = QLabel("Wheel: zoom | right drag: pan | left/right arrow: frame")
        self.zoom_label = QLabel("100%")
        viewer_box_layout.addRow("Controls", self.viewer_help_label)
        viewer_box_layout.addRow("Zoom", self.zoom_label)

        self.frame_viewer = FrameViewer()

        slider_box = QGroupBox("Frame Navigation")
        slider_layout = QVBoxLayout(slider_box)
        row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setRange(1, 1)
        self.frame_slider.setTracking(True)
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setEnabled(False)
        self.frame_spinbox.setRange(1, 1)
        self.frame_spinbox.setSuffix(" fr")
        self.frame_spinbox.setFixedWidth(92)
        self.frame_position_label = QLabel("Frame 0 / 0")
        row.addWidget(self.frame_slider, stretch=1)
        row.addWidget(self.frame_spinbox)
        row.addWidget(self.frame_position_label)
        slider_layout.addLayout(row)

        viewer_layout.addWidget(viewer_box)
        viewer_layout.addWidget(self.frame_viewer, stretch=1)
        viewer_layout.addWidget(slider_box)
        splitter.addWidget(viewer_panel)
        viewer_panel.setMinimumWidth(480)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_panel.setMinimumWidth(260)

        file_group = QGroupBox("File Load")
        file_layout = QVBoxLayout(file_group)
        self.folder_label = QLabel("-")
        self.folder_label.setWordWrap(True)
        self.choose_folder_button = QPushButton("Open Video Folder")
        self.video_list = QListWidget()
        self.video_list.setAlternatingRowColors(True)
        file_layout.addWidget(self.folder_label)
        file_layout.addWidget(self.choose_folder_button)
        file_layout.addWidget(self.video_list, stretch=1)

        csv_group = QGroupBox("CSV Mapping")
        csv_layout = QFormLayout(csv_group)
        self.csv_auto_combo = QComboBox()
        self.csv_auto_combo.setEnabled(False)
        self.csv_auto_combo.setToolTip("Automatically discovered CSV candidates for the selected video.")
        self.load_csv_folder_button = QPushButton("Load Folder Manually")
        self.load_csv_folder_button.setToolTip("Select an additional CSV folder for persistent auto-detection.")
        self.load_csv_button = QPushButton("Load CSV Manually")
        self.load_csv_button.setToolTip("Choose any CSV file manually.")
        csv_row = QHBoxLayout()
        csv_row.addWidget(self.csv_auto_combo, stretch=1)
        csv_row.addWidget(self.load_csv_folder_button)
        csv_row.addWidget(self.load_csv_button)
        self.csv_path_label = QLabel("Select a video first.")
        self.csv_path_label.setWordWrap(True)
        self.csv_path_label.setToolTip("Full path of the currently selected CSV.")
        csv_layout.addRow("CSV", csv_row)
        csv_layout.addRow("Path", self.csv_path_label)

        self.mode_tabs = QTabWidget()
        self.mode_tabs.setMinimumHeight(340)
        self.mode_tabs.setMaximumHeight(420)
        self.square_tab = self._build_square_tab()
        self.chamber_tab = self._build_chamber_tab()
        self.circle_tab = self._build_circle_tab()
        self.pin_tab = self._build_pin_tab()
        self.occlusion_tab = self._build_occlusion_tab()
        self.mode_tabs.addTab(self.square_tab, "Square Norm/Trajectory")
        self.mode_tabs.addTab(self.chamber_tab, "Chamber Mark")
        self.mode_tabs.addTab(self.circle_tab, "Circle Detection")
        self.mode_tabs.addTab(self.pin_tab, "Pin Coordinates")
        self.mode_tabs.addTab(self.occlusion_tab, "Occlusion Detect")
        self.mode_tabs.setTabToolTip(0, "Choose four points for perspective normalization and geometric mask margin.")
        self.mode_tabs.setTabToolTip(1, "Define a chamber and named rooms, then export room masks and per-frame room membership.")
        self.mode_tabs.setTabToolTip(2, "Draw a circle and classify each bodypart as inside or outside.")
        self.mode_tabs.setTabToolTip(3, "Place pins to inspect absolute and normalized coordinates.")
        self.mode_tabs.setTabToolTip(4, "Create and adjust occlusion masks, including simple and geometric margins.")

        save_group = QGroupBox("Save Options And Save")
        save_group.setMinimumHeight(180)
        save_group.setMaximumHeight(240)
        save_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        save_layout = QFormLayout(save_group)
        self.save_folder_label = QLabel("-")
        self.save_folder_label.setWordWrap(True)
        self.choose_save_folder_button = QPushButton("Choose Save Folder")
        self.current_output_label = QLabel("-")
        self.current_output_label.setWordWrap(True)
        self.save_current_button = QPushButton("Save Current Result")
        self.save_multiple_button = QPushButton("Save Multiple CSVs")
        self.save_multiple_button.setToolTip("Apply current mode settings to multiple selected videos at once.")
        save_layout.addRow("Save Folder", self.save_folder_label)
        save_layout.addRow(self.choose_save_folder_button)
        save_layout.addRow("Current Output", self.current_output_label)
        save_layout.addRow(self.save_current_button)
        save_layout.addRow(self.save_multiple_button)

        right_layout.addWidget(file_group, stretch=6)
        right_layout.addWidget(csv_group, stretch=2)
        right_layout.addWidget(self.mode_tabs, stretch=4)
        right_layout.addWidget(save_group, stretch=0)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1180, 640])

        self.statusBar().showMessage("Open a folder or choose a video from the list.")

    def _connect_signals(self) -> None:
        self.choose_folder_button.clicked.connect(self.choose_folder)
        self.choose_save_folder_button.clicked.connect(self.choose_save_folder)
        self.load_csv_folder_button.clicked.connect(self.choose_csv_folder)
        self.load_csv_button.clicked.connect(self.choose_csv)
        self.csv_auto_combo.currentIndexChanged.connect(self._on_csv_auto_selection_changed)
        self.video_list.currentItemChanged.connect(self._on_video_item_changed)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        self.frame_spinbox.valueChanged.connect(self._on_slider_changed)
        self.mode_tabs.currentChanged.connect(self._on_mode_changed)

        self.square_reset_button.clicked.connect(self.frame_viewer.clear_square_points)
        self.square_preview_button.clicked.connect(self.preview_square_normalization)
        self.chamber_shape_combo.currentIndexChanged.connect(self._sync_chamber_mode)
        self.chamber_edit_chamber_radio.toggled.connect(self._refresh_chamber_ui)
        self.chamber_edit_room_radio.toggled.connect(self._refresh_chamber_ui)
        self.room_combo.currentIndexChanged.connect(self._on_room_selection_changed)
        self.room_add_button.clicked.connect(self.add_room)
        self.room_rename_button.clicked.connect(self.rename_room)
        self.room_delete_button.clicked.connect(self.delete_room)
        self.room_clear_button.clicked.connect(self.clear_selected_room)
        self.chamber_reset_button.clicked.connect(self.reset_chamber)
        self.import_chamber_mask_button.clicked.connect(self.import_chamber_mask)
        self.export_chamber_mask_button.clicked.connect(self.export_chamber_mask)
        self.circle_margin_slider.valueChanged.connect(self._on_circle_margin_changed)
        self.circle_margin_spinbox.valueChanged.connect(self._on_circle_margin_changed)
        self.circle_reset_button.clicked.connect(self.frame_viewer.clear_circle)
        self.pin_reset_button.clicked.connect(self.reset_pins)
        self.pin_remove_last_button.clicked.connect(self.remove_last_pin)

        self.mask_name_button.clicked.connect(self.add_mask)
        self.mask_rename_button.clicked.connect(self.rename_mask)
        self.mask_delete_button.clicked.connect(self.delete_mask)
        self.mask_clear_button.clicked.connect(self.clear_selected_mask)
        self.mask_combo.currentIndexChanged.connect(self._on_mask_selection_changed)
        self.mask_shape_combo.currentIndexChanged.connect(self._sync_occlusion_mode)
        self.mask_draw_radio.toggled.connect(self._sync_occlusion_mode)
        self.mask_transform_radio.toggled.connect(self._sync_occlusion_mode)
        self.mask_margin_simple_radio.toggled.connect(self._on_mask_margin_mode_changed)
        self.mask_margin_geometric_radio.toggled.connect(self._on_mask_margin_mode_changed)
        self.occ_margin_set_button.clicked.connect(self._on_occ_margin_set_clicked)
        self.mask_add_radio.toggled.connect(self._sync_draw_mode)
        self.mask_erase_radio.toggled.connect(self._sync_draw_mode)
        self.mask_brush_slider.valueChanged.connect(self._on_mask_brush_changed)
        self.mask_brush_spinbox.valueChanged.connect(self._on_mask_brush_changed)
        self.mask_margin_slider.valueChanged.connect(self._on_mask_margin_changed)
        self.mask_margin_spinbox.valueChanged.connect(self._on_mask_margin_changed)
        self.import_mask_button.clicked.connect(self.import_mask_png)
        self.import_mask_folder_button.clicked.connect(self.import_mask_folder)
        self.occlusion_help_button.clicked.connect(self.show_occlusion_controls_help)

        self.save_current_button.clicked.connect(self.save_current_mode_output)
        self.save_multiple_button.clicked.connect(self.save_multiple_mode_outputs)
        self.export_masks_button.clicked.connect(self.export_masks)

        self.frame_viewer.square_points_changed.connect(self._on_square_points_changed)
        self.frame_viewer.chamber_rect_completed.connect(self.apply_chamber_rect)
        self.frame_viewer.chamber_rect_points_changed.connect(self._refresh_chamber_ui)
        self.frame_viewer.chamber_circle_completed.connect(self.apply_chamber_circle)
        self.frame_viewer.chamber_circle_changed.connect(self._refresh_chamber_ui)
        self.frame_viewer.circle_changed.connect(self._refresh_circle_ui)
        self.frame_viewer.view_changed.connect(self._refresh_view_ui)
        self.frame_viewer.pin_added.connect(self.add_pin)
        self.frame_viewer.occ_rect_completed.connect(self.apply_occ_rect_mask)
        self.frame_viewer.occ_rect_points_changed.connect(self._refresh_mask_draft_ui)
        self.frame_viewer.occ_circle_completed.connect(self.apply_occ_circle_mask)
        self.frame_viewer.occ_circle_changed.connect(self._refresh_mask_draft_ui)
        self.frame_viewer.occ_margin_points_changed.connect(self._on_occ_margin_points_changed)
        self.frame_viewer.free_draw_segment.connect(self.apply_occ_free_segment)
        self.frame_viewer.free_draw_finished.connect(self._finalize_mask_draw)
        self.frame_viewer.occ_transform_requested.connect(self.translate_selected_mask)
        self.frame_viewer.occ_scale_requested.connect(self.scale_selected_mask)
        self.frame_viewer.occ_transform_erase_requested.connect(self.erase_occ_transform_point)
        self.frame_viewer.occ_transform_erase_segment_requested.connect(self.erase_occ_transform_segment)
        self.frame_viewer.occ_transform_finished.connect(self._finalize_occ_transform)
        self.frame_viewer.occ_mask_double_clicked.connect(self._on_occ_mask_double_clicked)

    def _register_shortcuts(self) -> None:
        QShortcut(QKeySequence("Left"), self, activated=lambda: self.step_frame(-1))
        QShortcut(QKeySequence("Right"), self, activated=lambda: self.step_frame(1))
        QShortcut(QKeySequence("D"), self, activated=lambda: self._switch_occlusion_mode_shortcut(transform_mode=False))
        QShortcut(QKeySequence("T"), self, activated=lambda: self._switch_occlusion_mode_shortcut(transform_mode=True))
        QShortcut(QKeySequence("E"), self, activated=self._handle_e_shortcut)
        QShortcut(QKeySequence("R"), self, activated=self._handle_r_shortcut)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self._handle_ctrl_e_shortcut)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._handle_ctrl_r_shortcut)
        QShortcut(QKeySequence("["), self, activated=lambda: self._scale_selected_mask_shortcut(0.96))
        QShortcut(QKeySequence("]"), self, activated=lambda: self._scale_selected_mask_shortcut(1.04))
        QShortcut(QKeySequence("Ctrl+["), self, activated=lambda: self._rotate_selected_mask_shortcut(-4.0))
        QShortcut(QKeySequence("Ctrl+]"), self, activated=lambda: self._rotate_selected_mask_shortcut(4.0))
        QShortcut(QKeySequence("F1"), self, activated=self._open_context_help)
        QShortcut(QKeySequence("1"), self, activated=lambda: self._select_mask_by_slot(0))
        QShortcut(QKeySequence("2"), self, activated=lambda: self._select_mask_by_slot(1))
        QShortcut(QKeySequence("3"), self, activated=lambda: self._select_mask_by_slot(2))
        QShortcut(QKeySequence("4"), self, activated=lambda: self._select_mask_by_slot(3))
        QShortcut(QKeySequence("5"), self, activated=lambda: self._select_mask_by_slot(4))
        QShortcut(QKeySequence("6"), self, activated=lambda: self._select_mask_by_slot(5))
        QShortcut(QKeySequence("7"), self, activated=lambda: self._select_mask_by_slot(6))
        QShortcut(QKeySequence("8"), self, activated=lambda: self._select_mask_by_slot(7))
        QShortcut(QKeySequence("9"), self, activated=lambda: self._select_mask_by_slot(8))
        QShortcut(QKeySequence("0"), self, activated=lambda: self._select_mask_by_slot(9))

    def _switch_occlusion_mode_shortcut(self, transform_mode: bool) -> None:
        if self.mode_tabs.currentIndex() != 4:
            return
        if transform_mode:
            self.mask_transform_radio.setChecked(True)
        else:
            self.mask_draw_radio.setChecked(True)

    def _handle_e_shortcut(self) -> None:
        if self._mask_transform_shortcuts_enabled():
            self._scale_selected_mask_shortcut(0.96)

    def _handle_r_shortcut(self) -> None:
        if self._mask_transform_shortcuts_enabled():
            self._scale_selected_mask_shortcut(1.04)

    def _handle_ctrl_e_shortcut(self) -> None:
        if self._mask_transform_shortcuts_enabled():
            self._rotate_selected_mask_shortcut(-4.0)

    def _handle_ctrl_r_shortcut(self) -> None:
        if self._mask_transform_shortcuts_enabled():
            self._rotate_selected_mask_shortcut(4.0)

    def _select_mask_by_slot(self, slot_index: int) -> None:
        if self.mode_tabs.currentIndex() != 4:
            return
        if slot_index < 0 or slot_index >= min(10, self.mask_combo.count()):
            return
        self.mask_combo.setCurrentIndex(slot_index)

    def _on_occ_mask_double_clicked(self, name: str) -> None:
        if not name:
            return
        index = self.mask_combo.findData(name)
        if index >= 0:
            self.mask_combo.setCurrentIndex(index)

    def _open_context_help(self) -> None:
        if self.mode_tabs.currentIndex() == 4:
            self.show_occlusion_controls_help()

    def _ensure_save_folder(self) -> None:
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.save_folder_label.setText(str(self.save_folder))
        self._refresh_output_ui()

    def _initialize_directories(self) -> None:
        input_dir = self.settings.get("input_dir")
        output_dir = self.settings.get("output_dir")

        input_path = Path(input_dir) if input_dir else None
        output_path = Path(output_dir) if output_dir else None

        if input_path and input_path.exists():
            self.current_folder = input_path
            if output_path and output_path.exists():
                self.save_folder = output_path
            else:
                self.save_folder = input_path
            self.folder_label.setText(str(self.current_folder))
            self._ensure_save_folder()
            self.load_video_list(self.current_folder)
            return

        self.video_list.clear()
        self.folder_label.setText("-")

        if output_path and output_path.exists():
            self.save_folder = output_path
            self._ensure_save_folder()
        else:
            self.save_folder = DEFAULT_SAVE_DIR
            self.save_folder_label.setText("-")
            self._refresh_output_ui()
        self.statusBar().showMessage("Open a video folder to populate the list.")

    def _load_settings(self) -> dict:
        if not SETTINGS_PATH.exists():
            return {}
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self) -> None:
        data = {
            "last_tab_index": self.mode_tabs.currentIndex() if hasattr(self, "mode_tabs") else 0,
            "occlusion_mask_margin": int(getattr(self, "default_mask_margin", 0)),
            "occlusion_mask_margin_mode": str(getattr(self, "default_mask_margin_mode", "simple")),
            "occlusion_mask_brush": int(getattr(self, "default_mask_brush", 12)),
            "circle_detection_margin": int(getattr(self, "default_circle_margin", 0)),
            "input_dir": str(self.current_folder) if getattr(self, "current_folder", None) and str(self.current_folder) not in {"", "."} else "",
            "output_dir": str(self.save_folder) if getattr(self, "save_folder", None) else "",
            "csv_manual_dir": str(self.csv_manual_folder) if getattr(self, "csv_manual_folder", None) else "",
        }
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _refresh_csv_search_folder_ui(self) -> None:
        if not hasattr(self, "load_csv_folder_button"):
            return
        if self.csv_manual_folder is None:
            self.load_csv_folder_button.setToolTip(
                "Select an additional CSV folder for persistent auto-detection. "
                "Default mapping search uses the selected video's folder."
            )
        else:
            self.load_csv_folder_button.setToolTip(
                f"Additional CSV auto-detection folder: {self.csv_manual_folder}"
            )

    def _normalized_output_path(self) -> Path | None:
        return None if self.csv_path is None else self._normalized_output_path_for(self.csv_path)

    def _normalized_output_path_for(self, csv_path: Path) -> Path:
        return self.save_folder / f"{csv_path.stem}_normalized.csv"

    def _chamber_output_stem(self) -> str | None:
        if self.csv_path is not None:
            return self.csv_path.stem
        if self.video_state is not None:
            return self.video_state.path.stem
        return None

    def _chamber_csv_output_path(self) -> Path | None:
        stem = self._chamber_output_stem()
        return None if stem is None or self.csv_path is None else self._chamber_csv_output_path_for(self.csv_path)

    def _chamber_csv_output_path_for(self, csv_path: Path) -> Path:
        return self.save_folder / f"{csv_path.stem}_chamber_mark.csv"

    def _chamber_mask_output_path(self) -> Path | None:
        stem = self._chamber_output_stem()
        return None if stem is None else self.save_folder / f"{stem}_chamber_mask.png"

    def _chamber_overlay_output_path(self) -> Path | None:
        stem = self._chamber_output_stem()
        return None if stem is None else self.save_folder / f"{stem}_chamber_mask_with_frame.png"

    def _chamber_manifest_output_path(self) -> Path | None:
        stem = self._chamber_output_stem()
        return None if stem is None else self.save_folder / f"{stem}_chamber_mask.json"

    def _circle_output_path(self) -> Path | None:
        return None if self.csv_path is None else self._circle_output_path_for(self.csv_path)

    def _circle_output_path_for(self, csv_path: Path) -> Path:
        return self.save_folder / f"{csv_path.stem}_detection.csv"

    def _occlusion_output_path(self) -> Path | None:
        return None if self.csv_path is None else self._occlusion_output_path_for(self.csv_path)

    def _occlusion_output_path_for(self, csv_path: Path) -> Path:
        return self.save_folder / f"{csv_path.stem}_occlusion.csv"

    def _mask_export_folder(self) -> Path:
        return self.save_folder / "masks"

    @staticmethod
    def _video_size(video_path: Path) -> tuple[int, int] | None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return None
        try:
            width = max(1, int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
            height = max(1, int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        finally:
            capture.release()
        return width, height

    def _select_videos_for_batch(self) -> list[Path]:
        if self.video_list.count() == 0:
            QMessageBox.information(self, "Batch Save", "No videos in the list. Open a video folder first.")
            return []

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Videos For Batch CSV Save")
        dialog.resize(620, 560)
        layout = QVBoxLayout(dialog)

        info = QLabel("Select videos using click, Ctrl/Shift-click, or drag.")
        info.setWordWrap(True)
        layout.addWidget(info)

        warning_label = QLabel(
            "Warning:\n"
            "- Only videos with auto-detected CSV candidates are converted.\n"
            "- The top auto-detected CSV candidate is used for each video.\n"
            "- The same condition is applied to all selected videos."
        )
        warning_label.setWordWrap(True)
        layout.addWidget(warning_label)

        list_widget = QListWidget()
        list_widget.setAlternatingRowColors(True)
        list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        list_widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        list_widget.setSelectionRectVisible(True)
        layout.addWidget(list_widget, stretch=1)

        any_selected = False
        current_path = self.video_state.path if self.video_state is not None else None
        for index in range(self.video_list.count()):
            source_item = self.video_list.item(index)
            item = QListWidgetItem(source_item.text())
            item.setData(Qt.ItemDataRole.UserRole, source_item.data(Qt.ItemDataRole.UserRole))
            list_widget.addItem(item)
            if source_item.isSelected():
                item.setSelected(True)
                any_selected = True
            elif not any_selected and current_path is not None and source_item.data(Qt.ItemDataRole.UserRole) == str(current_path):
                item.setSelected(True)
                any_selected = True

        controls = QHBoxLayout()
        select_all_button = QPushButton("Select All")
        clear_button = QPushButton("Clear Selection")
        cancel_button = QPushButton("Cancel")
        start_button = QPushButton("Start Batch Save")
        controls.addWidget(select_all_button)
        controls.addWidget(clear_button)
        controls.addStretch(1)
        controls.addWidget(cancel_button)
        controls.addWidget(start_button)
        layout.addLayout(controls)

        select_all_button.clicked.connect(list_widget.selectAll)
        clear_button.clicked.connect(list_widget.clearSelection)
        cancel_button.clicked.connect(dialog.reject)

        def _accept_if_selected() -> None:
            if not list_widget.selectedItems():
                QMessageBox.information(dialog, "Batch Save", "Select at least one video.")
                return
            dialog.accept()

        start_button.clicked.connect(_accept_if_selected)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return []
        selected_paths: list[Path] = []
        for item in list_widget.selectedItems():
            data = item.data(Qt.ItemDataRole.UserRole)
            if data:
                selected_paths.append(Path(data))
        return selected_paths

    def save_multiple_mode_outputs(self) -> None:
        mode_index = self.mode_tabs.currentIndex()
        if mode_index not in {0, 1, 2, 4}:
            QMessageBox.information(self, "Batch Save", "Batch CSV save is not available in Pin tab.")
            return
        if self.video_state is None:
            QMessageBox.warning(self, "Batch Save", "Load a reference video first.")
            return

        selected_videos = self._select_videos_for_batch()
        if not selected_videos:
            return

        source_width = max(1, int(self.video_state.width))
        source_height = max(1, int(self.video_state.height))
        source_square_points = [tuple(point) for point in self.frame_viewer.square_points]
        source_chamber_mask = None if self.chamber_mask is None else self.chamber_mask.copy().astype(np.uint8)
        source_rooms = [
            RoomRecord(
                name=room.name,
                color=QColor(room.color),
                mask=room.mask.copy().astype(np.uint8),
            )
            for room in self.room_records.values()
        ]
        source_circle_geometry = self.frame_viewer.circle_geometry()
        source_occ_margin_points = [tuple(point) for point in self.frame_viewer.occ_margin_points]
        source_masks = [
            MaskRecord(
                name=record.name,
                color=QColor(record.color),
                mask=record.mask.copy().astype(np.uint8),
                margin=int(record.margin),
                margin_mode=str(record.margin_mode),
            )
            for record in self.mask_records.values()
        ]

        if mode_index == 0 and len(source_square_points) != 4:
            QMessageBox.warning(self, "Batch Save", "Square mode needs four points.")
            return
        if mode_index == 1:
            if source_chamber_mask is None or not np.any(source_chamber_mask):
                QMessageBox.warning(self, "Batch Save", "Chamber mode needs a chamber mask.")
                return
            if not source_rooms:
                QMessageBox.warning(self, "Batch Save", "Chamber mode needs at least one room.")
                return
        if mode_index == 2 and source_circle_geometry is None:
            QMessageBox.warning(self, "Batch Save", "Circle mode needs a circle.")
            return
        if mode_index == 4:
            if not source_masks:
                QMessageBox.warning(self, "Batch Save", "Occlusion mode needs at least one mask.")
                return
            if any(record.margin_mode == "geometric" for record in source_masks) and len(source_occ_margin_points) != 4:
                QMessageBox.warning(self, "Batch Save", "Geometric margin mode needs four occlusion geometric points.")
                return

        saved_count = 0
        skipped_auto_missing: list[Path] = []
        failed: list[tuple[Path, str]] = []
        total = len(selected_videos)
        for index, video_path in enumerate(selected_videos, start=1):
            self.statusBar().showMessage(f"Batch saving... {index}/{total} ({video_path.name})")
            QApplication.processEvents()
            try:
                csv_candidates = self._matching_csv_candidates(video_path)
                if not csv_candidates:
                    skipped_auto_missing.append(video_path)
                    continue
                csv_path = csv_candidates[0]

                video_size = self._video_size(video_path)
                if video_size is None:
                    raise ValueError("Could not read video size.")
                width, height = video_size
                scale_x = width / source_width
                scale_y = height / source_height

                source_df = pd.read_csv(csv_path)
                bodyparts = bodyparts_from_dataframe(source_df)
                if mode_index == 0:
                    quad_points = [(x * scale_x, y * scale_y) for x, y in source_square_points]
                    result_df = build_normalized_dataframe(source_df, bodyparts, quad_points, width, height)
                    output_path = self._normalized_output_path_for(csv_path)
                elif mode_index == 1:
                    chamber_mask = cv2.resize(source_chamber_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                    scaled_rooms: list[RoomRecord] = []
                    for room in source_rooms:
                        resized_mask = cv2.resize(room.mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                        room_mask = np.logical_and(resized_mask > 0, chamber_mask > 0).astype(np.uint8)
                        scaled_rooms.append(RoomRecord(name=room.name, color=QColor(room.color), mask=room_mask))
                    result_df = build_chamber_mark_dataframe(source_df, bodyparts, scaled_rooms, width, height)
                    output_path = self._chamber_csv_output_path_for(csv_path)
                elif mode_index == 2:
                    center, _, adjusted_radius = source_circle_geometry
                    scaled_center = (center[0] * scale_x, center[1] * scale_y)
                    scaled_radius = max(1.0, adjusted_radius * ((scale_x + scale_y) / 2.0))
                    result_df = build_circle_detection_dataframe(source_df, bodyparts, scaled_center, scaled_radius, width, height)
                    output_path = self._circle_output_path_for(csv_path)
                else:
                    scaled_quad_points = [(x * scale_x, y * scale_y) for x, y in source_occ_margin_points]
                    scaled_masks: list[MaskRecord] = []
                    for record in source_masks:
                        resized_mask = cv2.resize(record.mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                        scaled_masks.append(
                            MaskRecord(
                                name=record.name,
                                color=QColor(record.color),
                                mask=(resized_mask > 0).astype(np.uint8),
                                margin=record.margin,
                                margin_mode=record.margin_mode,
                            )
                        )
                    result_df = build_occlusion_dataframe(source_df, bodyparts, scaled_masks, width, height, scaled_quad_points)
                    output_path = self._occlusion_output_path_for(csv_path)

                output_path.parent.mkdir(parents=True, exist_ok=True)
                result_df.to_csv(output_path, index=False)
                saved_count += 1
            except Exception as exc:
                failed.append((video_path, str(exc)))

        lines = [
            f"Saved: {saved_count}",
            f"Skipped (no auto detection CSV): {len(skipped_auto_missing)}",
            f"Failed: {len(failed)}",
        ]
        if skipped_auto_missing:
            lines.append("")
            lines.append("Skipped videos:")
            lines.extend(f"- {path.name}" for path in skipped_auto_missing[:10])
            if len(skipped_auto_missing) > 10:
                lines.append(f"- ... and {len(skipped_auto_missing) - 10} more")
        if failed:
            lines.append("")
            lines.append("Failed videos:")
            lines.extend(f"- {path.name}: {reason}" for path, reason in failed[:10])
            if len(failed) > 10:
                lines.append(f"- ... and {len(failed) - 10} more")

        message = "\n".join(lines)
        if failed:
            QMessageBox.warning(self, "Batch Save Completed With Warnings", message)
        else:
            QMessageBox.information(self, "Batch Save Completed", message)
        self.statusBar().showMessage(
            f"Batch save finished: saved={saved_count}, skipped={len(skipped_auto_missing)}, failed={len(failed)}"
        )

    def _show_normalized_preview(self, preview_df: pd.DataFrame, normalized: bool) -> None:
        frame_width = self.video_state.width if self.video_state is not None else None
        frame_height = self.video_state.height if self.video_state is not None else None
        video_name = self.video_state.path.stem if self.video_state is not None else None
        preview_dialog = TrajectoryPreviewDialog(preview_df, self.bodyparts, normalized, frame_width, frame_height, video_name, self)
        preview_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        preview_dialog.finished.connect(lambda _result, dialog=preview_dialog: self._on_trajectory_preview_closed(dialog))
        self._trajectory_preview_windows.append(preview_dialog)
        preview_dialog.show()
        preview_dialog.raise_()
        preview_dialog.activateWindow()

    def _on_trajectory_preview_closed(self, dialog: QDialog) -> None:
        try:
            self._trajectory_preview_windows.remove(dialog)  # keep refs only for open windows
        except ValueError:
            pass

    def _refresh_output_ui(self) -> None:
        current_index = self.mode_tabs.currentIndex() if hasattr(self, "mode_tabs") else 0
        batch_text = "Save Multiple CSVs"
        if current_index == 0:
            output = self._normalized_output_path()
            self.save_current_button.setText("Save Normalized CSV")
            self.save_current_button.setEnabled(output is not None and len(self.frame_viewer.square_points) == 4)
            batch_text = "Save Multiple Normalized CSVs"
        elif current_index == 1:
            output = self._chamber_csv_output_path()
            chamber_ready = self.chamber_mask is not None and bool(np.any(self.chamber_mask)) and bool(self.room_records)
            self.save_current_button.setText("Save Chamber Outputs")
            self.save_current_button.setEnabled(output is not None and chamber_ready and self.csv_df is not None and bool(self.bodyparts))
            batch_text = "Save Multiple Chamber CSVs"
        elif current_index == 2:
            output = self._circle_output_path()
            self.save_current_button.setText("Save Detection CSV")
            self.save_current_button.setEnabled(output is not None and self.frame_viewer.circle_geometry() is not None and bool(self.bodyparts))
            batch_text = "Save Multiple Detection CSVs"
        elif current_index == 3:
            output = None
            self.save_current_button.setText("Pin Mode Does Not Save")
            self.save_current_button.setEnabled(False)
            batch_text = "Batch CSV Save Not Available In Pin"
        else:
            output = self._occlusion_output_path()
            geometric_ready = all(
                record.margin_mode != "geometric" or len(self.frame_viewer.occ_margin_points) == 4
                for record in self.mask_records.values()
            )
            self.save_current_button.setText("Save Occlusion CSV")
            self.save_current_button.setEnabled(output is not None and bool(self.mask_records) and self.csv_df is not None and geometric_ready)
            batch_text = "Save Multiple Occlusion CSVs"
        batch_enabled = self.save_current_button.isEnabled()
        self.current_output_label.setText(str(output) if output is not None else "-")
        if hasattr(self, "save_multiple_button"):
            self.save_multiple_button.setText(batch_text)
            self.save_multiple_button.setEnabled(batch_enabled)
            if current_index == 3:
                self.save_multiple_button.setToolTip("Batch CSV save is not available in Pin tab.")
            else:
                self.save_multiple_button.setToolTip("Apply current mode settings to multiple selected videos at once.")
        if hasattr(self, "export_masks_button"):
            self.export_masks_button.setEnabled(bool(self.mask_records))

    def choose_folder(self) -> None:
        start_dir = str(self.current_folder) if str(self.current_folder) not in {"", "."} and self.current_folder.exists() else str(Path.cwd())
        folder = QFileDialog.getExistingDirectory(self, "Select video folder", start_dir)
        if folder:
            self.load_video_list(Path(folder))
            if not self.settings.get("output_dir"):
                self.save_folder = Path(folder)
                self._ensure_save_folder()
            self._save_settings()

    def choose_save_folder(self) -> None:
        start_dir = str(self.save_folder) if self.save_folder.exists() else (str(self.current_folder) if str(self.current_folder) not in {"", "."} and self.current_folder.exists() else str(Path.cwd()))
        folder = QFileDialog.getExistingDirectory(self, "Select save folder", start_dir)
        if folder:
            self.save_folder = Path(folder)
            self._ensure_save_folder()
            self._save_settings()

    def load_video_list(self, folder: Path) -> None:
        self.current_folder = folder
        self.folder_label.setText(str(folder))
        self.video_list.clear()
        videos = discover_videos(folder)
        if not videos:
            self.statusBar().showMessage("No videos found in the selected folder.")
            return
        for video_path in videos:
            item = QListWidgetItem(str(video_path.relative_to(folder)))
            item.setData(Qt.ItemDataRole.UserRole, str(video_path))
            self.video_list.addItem(item)
        self.video_list.setCurrentRow(0)
        self.statusBar().showMessage(f"Loaded {len(videos)} videos.")

    @staticmethod
    def _csv_candidate_patterns(video_name: str) -> list[tuple[str, int]]:
        return [
            (video_name, 0),
            (f"predict_{video_name}", 1),
            (f"predict__{video_name}", 1),
        ]

    def _csv_candidate_sort_key(self, path: Path, video_name: str) -> tuple[int, str, str]:
        stem = path.stem
        for prefix, exact_rank in self._csv_candidate_patterns(video_name):
            if stem == prefix:
                return (exact_rank, "", stem)
            if stem.startswith(f"{prefix}_"):
                suffix = stem[len(prefix) + 1 :]
                return (exact_rank + 2, suffix.lower(), stem)
        return (4, stem.lower(), stem)

    def _matching_csv_candidates(self, video_path: Path) -> list[Path]:
        video_name = video_path.stem
        search_dirs: list[Path] = [video_path.parent]
        if self.csv_manual_folder is not None and self.csv_manual_folder.exists():
            if self.csv_manual_folder != video_path.parent:
                search_dirs.append(self.csv_manual_folder)
        candidates: list[Path] = []
        seen_paths: set[str] = set()
        for search_dir in search_dirs:
            for path in search_dir.glob("*.csv"):
                unique_key = str(path.resolve())
                if unique_key in seen_paths:
                    continue
                stem = path.stem
                for prefix, _ in self._csv_candidate_patterns(video_name):
                    if stem == prefix or stem.startswith(f"{prefix}_"):
                        candidates.append(path)
                        seen_paths.add(unique_key)
                        break
        return sorted(candidates, key=lambda path: self._csv_candidate_sort_key(path, video_name))

    def _set_csv_auto_candidates(self, candidates: list[Path], selected_path: Path | None = None, allow_default: bool = True) -> None:
        self.csv_auto_candidates = list(candidates)
        self.csv_auto_combo.blockSignals(True)
        self.csv_auto_combo.clear()
        for path in self.csv_auto_candidates:
            self.csv_auto_combo.addItem(path.name, str(path))
        has_candidates = bool(self.csv_auto_candidates)
        self.csv_auto_combo.setEnabled(has_candidates)
        if has_candidates:
            selected_index = 0 if allow_default else -1
            if selected_path is not None:
                for index, path in enumerate(self.csv_auto_candidates):
                    if path == selected_path:
                        selected_index = index
                        break
            self.csv_auto_combo.setCurrentIndex(selected_index)
        else:
            self.csv_auto_combo.setCurrentIndex(-1)
        self.csv_auto_combo.blockSignals(False)

    def _update_csv_path_label(self, csv_path: Path | None) -> None:
        self.csv_path_label.setText(str(csv_path) if csv_path is not None else "No CSV selected.")

    def _on_csv_auto_selection_changed(self, index: int) -> None:
        if index < 0 or index >= len(getattr(self, "csv_auto_candidates", [])):
            return
        selected_path = self.csv_auto_candidates[index]
        if self.csv_path == selected_path:
            return
        self.load_csv(selected_path, auto_matched=True)

    def _on_video_item_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        if current is not None:
            path_str = current.data(Qt.ItemDataRole.UserRole)
            if path_str:
                self.load_video(Path(path_str))

    def _release_video_capture(self) -> None:
        if self.video_state is not None:
            self.video_state.capture.release()
            self.video_state = None

    def load_video(self, video_path: Path) -> None:
        previous_state = self.video_state
        self._release_video_capture()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            QMessageBox.warning(self, "Video Load Failed", f"Could not open video:\n{video_path}")
            return

        self.video_state = VideoState(
            path=video_path,
            capture=capture,
            frame_count=max(1, int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))),
            fps=float(capture.get(cv2.CAP_PROP_FPS)) or 0.0,
            width=max(1, int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))),
            height=max(1, int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))),
        )
        self._loading_slider = True
        self.frame_slider.setEnabled(True)
        self.frame_slider.setRange(1, self.video_state.frame_count)
        self.frame_slider.setValue(1)
        self.frame_spinbox.setEnabled(True)
        self.frame_spinbox.setRange(1, self.video_state.frame_count)
        self.frame_spinbox.setValue(1)
        self._loading_slider = False
        self.current_frame_number = 1
        self.current_frame_rgb = None
        if previous_state is not None and (previous_state.width != self.video_state.width or previous_state.height != self.video_state.height):
            self._rescale_annotations(previous_state.width, previous_state.height, self.video_state.width, self.video_state.height)
        self.frame_viewer.reset_view()
        self.frame_viewer.clear_chamber_rect_points()
        self.frame_viewer.clear_chamber_circle()
        self.frame_viewer.clear_occ_rect_points()
        self.frame_viewer.clear_occ_circle()
        self._load_matching_csv(video_path)
        self._load_frame(1)
        self._update_frame_label(1)
        self.statusBar().showMessage(f"Video loaded: {video_path.name}")

    def _load_matching_csv(self, video_path: Path) -> None:
        candidates = self._matching_csv_candidates(video_path)
        self._set_csv_auto_candidates(candidates)
        if candidates:
            self.load_csv(candidates[0], auto_matched=True)
        else:
            self.csv_path = None
            self.csv_df = None
            self.bodyparts = []
            self._update_csv_path_label(None)
            self._refresh_square_ui()
            self._refresh_circle_ui()

    def choose_csv(self) -> None:
        start_dir = self.video_state.path.parent if self.video_state is not None else (self.current_folder if self.current_folder.exists() else Path.cwd())
        csv_file, _ = QFileDialog.getOpenFileName(self, "Select CSV file", str(start_dir), "CSV Files (*.csv)")
        if csv_file:
            self.load_csv(Path(csv_file), auto_matched=False)

    def choose_csv_folder(self) -> None:
        start_dir = self.csv_manual_folder if self.csv_manual_folder is not None and self.csv_manual_folder.exists() else (
            self.video_state.path.parent if self.video_state is not None else (self.current_folder if self.current_folder.exists() else Path.cwd())
        )
        folder = QFileDialog.getExistingDirectory(self, "Select CSV Auto-Detection Folder", str(start_dir))
        if not folder:
            return
        self.csv_manual_folder = Path(folder)
        self._refresh_csv_search_folder_ui()
        self._save_settings()
        if self.video_state is not None:
            self._load_matching_csv(self.video_state.path)
        self.statusBar().showMessage(f"CSV auto-detection folder set: {self.csv_manual_folder}")

    def load_csv(self, csv_path: Path, auto_matched: bool) -> None:
        try:
            self.csv_df = pd.read_csv(csv_path)
        except Exception as exc:
            QMessageBox.critical(self, "CSV Load Failed", f"Could not read CSV:\n{exc}")
            return
        self.csv_path = csv_path
        self.bodyparts = bodyparts_from_dataframe(self.csv_df)
        self._update_csv_path_label(csv_path)
        if auto_matched:
            self._set_csv_auto_candidates(getattr(self, "csv_auto_candidates", []), csv_path)
        else:
            current_candidates = getattr(self, "csv_auto_candidates", [])
            selected = csv_path if csv_path in current_candidates else None
            self._set_csv_auto_candidates(current_candidates, selected, allow_default=selected is not None)
        self._refresh_square_ui()
        self._refresh_circle_ui()

    def _load_frame(self, frame_number: int) -> None:
        if self.video_state is None:
            return
        frame_number = max(1, min(frame_number, self.video_state.frame_count))
        self.video_state.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ok, frame = self.video_state.capture.read()
        if not ok:
            return
        self.current_frame_number = frame_number
        self.frame_viewer.current_frame_number = frame_number
        self.current_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_viewer.set_frame(self.current_frame_rgb)
        self.frame_viewer.set_chamber_records(self.chamber_mask, self.room_records, self.selected_room_name)
        self.frame_viewer.set_pin_records(self.pins)
        self.frame_viewer.set_mask_records(self.mask_records, self.selected_mask_name)
        self._refresh_view_ui()
        self._refresh_chamber_ui()
        self._refresh_pin_ui()
        self._update_frame_label(frame_number)
        if hasattr(self, "_refresh_square_current_buttons"):
            self._refresh_square_current_buttons()

    def _on_slider_changed(self, value: int) -> None:
        if not self._loading_slider:
            self._load_frame(value)

    def _update_frame_label(self, frame_number: int) -> None:
        total = self.video_state.frame_count if self.video_state else 0
        if self.frame_slider.value() != frame_number:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(frame_number)
            self.frame_slider.blockSignals(False)
        if self.frame_spinbox.value() != frame_number:
            self.frame_spinbox.blockSignals(True)
            self.frame_spinbox.setValue(frame_number)
            self.frame_spinbox.blockSignals(False)
        self.frame_position_label.setText(f"Frame {frame_number} / {total}")

    def step_frame(self, delta: int) -> None:
        if self.video_state is None:
            return
        self.frame_slider.setValue(max(1, min(self.current_frame_number + delta, self.video_state.frame_count)))

    def _rescale_annotations(self, old_width: int, old_height: int, new_width: int, new_height: int) -> None:
        if old_width <= 0 or old_height <= 0 or new_width <= 0 or new_height <= 0:
            return
        scale_x = new_width / old_width
        scale_y = new_height / old_height

        self.frame_viewer.square_points = [(x * scale_x, y * scale_y) for x, y in self.frame_viewer.square_points]
        self.frame_viewer.chamber_rect_points = [(x * scale_x, y * scale_y) for x, y in self.frame_viewer.chamber_rect_points]
        if self.frame_viewer.chamber_circle_start is not None:
            self.frame_viewer.chamber_circle_start = (self.frame_viewer.chamber_circle_start[0] * scale_x, self.frame_viewer.chamber_circle_start[1] * scale_y)
        if self.frame_viewer.chamber_circle_end is not None:
            self.frame_viewer.chamber_circle_end = (self.frame_viewer.chamber_circle_end[0] * scale_x, self.frame_viewer.chamber_circle_end[1] * scale_y)
        if self.frame_viewer.circle_start is not None:
            self.frame_viewer.circle_start = (self.frame_viewer.circle_start[0] * scale_x, self.frame_viewer.circle_start[1] * scale_y)
        if self.frame_viewer.circle_end is not None:
            self.frame_viewer.circle_end = (self.frame_viewer.circle_end[0] * scale_x, self.frame_viewer.circle_end[1] * scale_y)

        for pin in self.pins:
            pin.x *= scale_x
            pin.y *= scale_y

        # Different video resolutions can invalidate mask arrays, so clear masks instead of resizing.
        self.reset_chamber()
        self.reset_masks()
        self.frame_viewer.clear_occ_margin_points()
        self.frame_viewer.set_chamber_records(self.chamber_mask, self.room_records, self.selected_room_name, refresh=True)
        self.frame_viewer.set_pin_records(self.pins)
        self.frame_viewer.set_mask_records(self.mask_records, self.selected_mask_name, refresh=True)
        self._refresh_square_ui()
        self._refresh_chamber_ui()
        self._refresh_circle_ui()
        self._refresh_pin_ui()
        self._refresh_mask_ui()

    def _refresh_view_ui(self) -> None:
        self.zoom_label.setText(f"{int(round(self.frame_viewer.zoom_factor * 100))}%")

    def _on_mode_changed(self, index: int) -> None:
        if index == 0:
            self.frame_viewer.set_mode("square")
            self.frame_viewer.set_margin_value(0.0)
            if hasattr(self, "square_preview_button") and self.square_preview_button.isEnabled():
                self.square_preview_button.setFocus()
        elif index == 1:
            self._sync_chamber_mode()
        elif index == 2:
            self.frame_viewer.set_mode("circle")
            self.frame_viewer.set_margin_value(float(self.circle_margin_slider.value()))
        elif index == 3:
            self.frame_viewer.set_mode("pin")
            self.frame_viewer.set_margin_value(0.0)
        else:
            self._sync_occlusion_mode()
        self._save_settings()
        self._refresh_output_ui()

    def _refresh_mask_draft_ui(self) -> None:
        self._refresh_output_ui()

    @staticmethod
    def _clamp_mask_shift(mask: np.ndarray, dx: int, dy: int) -> tuple[int, int]:
        if mask.ndim != 2:
            return 0, 0
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return 0, 0
        height, width = mask.shape
        min_x = int(xs.min())
        max_x = int(xs.max())
        min_y = int(ys.min())
        max_y = int(ys.max())
        min_dx = -min_x
        max_dx = (width - 1) - max_x
        min_dy = -min_y
        max_dy = (height - 1) - max_y
        clamped_dx = int(max(min_dx, min(max_dx, int(dx))))
        clamped_dy = int(max(min_dy, min(max_dy, int(dy))))
        return clamped_dx, clamped_dy

    def translate_selected_mask(self, shift: tuple[int, int]) -> None:
        current = self._selected_mask()
        if current is None:
            return
        dx, dy = shift
        if dx == 0 and dy == 0:
            return
        dx, dy = self._clamp_mask_shift(current.mask.astype(np.uint8), dx, dy)
        if dx == 0 and dy == 0:
            return
        translated = np.zeros_like(current.mask)
        src_x0 = max(0, -dx)
        src_x1 = current.mask.shape[1] - max(0, dx)
        src_y0 = max(0, -dy)
        src_y1 = current.mask.shape[0] - max(0, dy)
        dst_x0 = max(0, dx)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y0 = max(0, dy)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        if src_x1 > src_x0 and src_y1 > src_y0:
            translated[dst_y0:dst_y1, dst_x0:dst_x1] = current.mask[src_y0:src_y1, src_x0:src_x1]
            current.mask = translated
            # During dragging, refresh only the active mask fill at a capped rate.
            now = time.monotonic()
            if now - getattr(self, "_last_transform_preview_refresh", 0.0) >= 0.02:
                self._last_transform_preview_refresh = now
                self.frame_viewer.refresh_mask_record(current.name, include_margin=False)

    def scale_selected_mask(self, scale_factor: float) -> None:
        current = self._selected_mask()
        if current is None or scale_factor <= 0:
            return
        ys, xs = np.where(current.mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return
        center_x = float(xs.mean())
        center_y = float(ys.mean())
        matrix = np.array(
            [
                [scale_factor, 0.0, center_x - scale_factor * center_x],
                [0.0, scale_factor, center_y - scale_factor * center_y],
            ],
            dtype=np.float32,
        )
        new_mask = cv2.warpAffine(
            current.mask.astype(np.uint8),
            matrix,
            (current.mask.shape[1], current.mask.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if new_mask.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            current.mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)
            self.frame_viewer.refresh_mask_record(current.name, include_margin=True)

    def rotate_selected_mask(self, angle_degrees: float) -> None:
        current = self._selected_mask()
        if current is None or abs(angle_degrees) < 1e-6:
            return
        ys, xs = np.where(current.mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return
        center_x = float(xs.mean())
        center_y = float(ys.mean())
        matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_degrees, 1.0)
        new_mask = cv2.warpAffine(
            current.mask.astype(np.uint8),
            matrix,
            (current.mask.shape[1], current.mask.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if new_mask.any():
            current.mask = new_mask.astype(np.uint8)
            self.frame_viewer.refresh_mask_record(current.name, include_margin=True)

    def _mask_transform_shortcuts_enabled(self) -> bool:
        return (
            self.mode_tabs.currentIndex() == 4
            and self.mask_transform_radio.isChecked()
            and self._selected_mask() is not None
        )

    def _scale_selected_mask_shortcut(self, scale_factor: float) -> None:
        if not self._mask_transform_shortcuts_enabled():
            return
        self.scale_selected_mask(scale_factor)
        self._refresh_mask_ui()

    def _rotate_selected_mask_shortcut(self, angle_degrees: float) -> None:
        if not self._mask_transform_shortcuts_enabled():
            return
        self.rotate_selected_mask(angle_degrees)
        self._refresh_mask_ui()

    def save_current_mode_output(self) -> None:
        index = self.mode_tabs.currentIndex()
        if index == 0:
            self.save_normalized_csv()
        elif index == 1:
            self.save_chamber_outputs()
        elif index == 2:
            self.save_circle_detection_csv()
        elif index == 4:
            self.save_occlusion_csv()

    def closeEvent(self, event) -> None:
        self._save_settings()
        self._release_video_capture()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    if APP_ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
