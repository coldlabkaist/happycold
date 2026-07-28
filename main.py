from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("QT_API", "pyqt6")

import cv2
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QPoint, QPointF, QRectF, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QIcon, QImage, QKeySequence, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QFrame,
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
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


APP_NAME = "happycold"
DEFAULT_SAVE_DIR = Path.cwd() / "output"
APP_ICON_PATH = Path(__file__).resolve().parent / "CoLD_icon.png"
NODE_INSTANCE_COLUMN_CANDIDATES = (
    "instance",
    "instance_id",
    "instance id",
    "track",
    "track_id",
    "track id",
)


@dataclass(frozen=True)
class NodeOverlayPoint:
    label: str
    x: float
    y: float
    instance_key: str
    bodypart: str
    review_role: str = ""


@dataclass(frozen=True)
class MaskClipboardItem:
    label: str
    source: str
    mask: np.ndarray
    color_name: str = "#06b6d4"
    margin: int = 0
    margin_mode: str = "simple"
    geometry: dict | None = None
    circle_center: tuple[float, float] | None = None
    circle_base_radius: float | None = None
    circle_margin: int = 0


@dataclass(frozen=True)
class MaskUndoSnapshot:
    tab_key: str
    label: str
    payload: object


@dataclass(frozen=True)
class MaskHoverInfo:
    title: str
    source_label: str
    lines: tuple[str, ...]
    anchor: tuple[float, float]
    marker_points: tuple[tuple[float, float], ...] = ()
    marker_circle: tuple[tuple[float, float], float] | None = None
    color_name: str = "#f8fafc"


def get_settings_path() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        settings_dir = Path(local_appdata) / APP_NAME
    else:
        settings_dir = Path.home() / f".{APP_NAME}"
    return settings_dir / "happycold_setting.json"


SETTINGS_PATH = get_settings_path()

from batch import BatchItem, BatchRunResult, raise_if_cancelled
from batch_ui import (
    BatchExportWorker,
    BatchProgressDialog,
    BatchVideoChoice,
    BatchVideoSelectionDialog,
    SingleExportWorker,
    SingleSaveProgressDialog,
)
from shared import (
    MASK_PALETTE,
    MaskRecord,
    MaskTransformSource,
    PinRecord,
    RoomRecord,
    VideoState,
    adjust_mask_by_mode,
    build_chamber_mark_dataframe,
    build_circle_detection_dataframe,
    build_normalized_dataframe,
    build_occlusion_dataframe,
    build_rectified_geometry,
    bodyparts_from_dataframe,
    circle_mask_geometry,
    clone_mask_geometry,
    discover_videos,
    infer_mask_geometry,
    infer_pixel_scale,
    mask_geometry_for_export,
    mask_geometry_is_exact,
    mask_polygon_geometry,
    order_quad_points,
    scale_mask_geometry,
    smooth_binary_mask_low,
    translate_mask_geometry,
)
from interpolation import build_interpolation_pipeline_dataframe
from trajectory import resolve_bodypart_coordinate_columns
from ui_controls import NoWheelComboBox, NoWheelSpinBox
from ui_theme import build_app_stylesheet
from tab_mixins import (
    ChamberTabMixin,
    CircleTabMixin,
    InterpolationTabMixin,
    OcclusionTabMixin,
    PinTabMixin,
    PipelinePanelMixin,
    SquareTabMixin,
    TrackingRepairTabMixin,
)
from tab_mixins.circle_tab import (
    build_circle_mask,
    infer_circular_mask_geometry,
)


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
        normalized_display_size: tuple[int, int] | None = None,
        video_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        mode_label = "Normalized" if normalized else "Raw"
        self.setWindowTitle(f"{APP_NAME} - {mode_label} Trajectory Preview")
        self.resize(1100, 760)
        self._default_export_name = f"{video_name or 'trajectory'}_trajectory"
        self._normalized_display_size = normalized_display_size if normalized else None
        self._normalized = normalized

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
        x_limit, y_limit = self._plot_limits(
            df,
            bodyparts,
            normalized,
            frame_width,
            frame_height,
            normalized_display_size,
        )

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
        normalized_display_size: tuple[int, int] | None = None,
    ) -> tuple[float, float]:
        if normalized:
            if normalized_display_size is not None:
                return float(normalized_display_size[0]), float(normalized_display_size[1])
            return 1.0, 1.0
        if frame_width is None or frame_height is None:
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
            x_values, y_values = self._scaled_bodypart_xy(group_df, bodypart)
            ax.plot(
                x_values,
                y_values,
                color=colors[index % len(colors)],
                linewidth=1.1,
                label=track_label,
            )
        if track_col is not None and len(groups) > 1:
            ax.legend(fontsize=7, loc="best")

    def _scaled_bodypart_xy(self, df: pd.DataFrame, bodypart: str) -> tuple[pd.Series, pd.Series]:
        coordinate_columns = resolve_bodypart_coordinate_columns(df, bodypart, normalized=self._normalized)
        if coordinate_columns is None:
            return pd.Series(float("nan"), index=df.index), pd.Series(float("nan"), index=df.index)
        x_col, y_col = coordinate_columns
        x_values = pd.to_numeric(df[x_col], errors="coerce")
        y_values = pd.to_numeric(df[y_col], errors="coerce")
        if self._normalized_display_size is None:
            return x_values, y_values
        return (
            x_values * float(self._normalized_display_size[0]),
            y_values * float(self._normalized_display_size[1]),
        )

class FrameViewer(QWidget):
    interpolation_rect_points_changed = pyqtSignal()
    interpolation_rect_completed = pyqtSignal(object)
    interpolation_circle_changed = pyqtSignal()
    interpolation_circle_completed = pyqtSignal(object)
    interpolation_free_segment = pyqtSignal(object)
    interpolation_free_finished = pyqtSignal()
    interpolation_transform_requested = pyqtSignal(object)
    interpolation_transform_finished = pyqtSignal()
    square_points_changed = pyqtSignal()
    trajectory_region_changed = pyqtSignal()
    chamber_rect_points_changed = pyqtSignal()
    chamber_rect_completed = pyqtSignal(object)
    chamber_circle_changed = pyqtSignal()
    chamber_circle_completed = pyqtSignal(object)
    chamber_transform_requested = pyqtSignal(object)
    chamber_transform_finished = pyqtSignal()
    circle_changed = pyqtSignal()
    circle_edit_started = pyqtSignal()
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
    annotate_context_menu_requested = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(320, 180)
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
        self.interpolation_rect_points: list[tuple[float, float]] = []
        self.interpolation_circle_start: tuple[float, float] | None = None
        self.interpolation_circle_end: tuple[float, float] | None = None
        self._interpolation_circle_dragging = False
        self._interpolation_circle_current: tuple[float, float] | None = None
        self._interpolation_free_dragging = False
        self._interpolation_free_last_point: tuple[float, float] | None = None
        self.interpolation_draw_add = True
        self.interpolation_brush_radius = 20
        self._interpolation_mask_cache: QPixmap | None = None
        self._interpolation_mask_path_cache: QPainterPath | None = None
        self._interpolation_mask: np.ndarray | None = None
        self.interpolation_transform_mode = False
        self._interpolation_transform_dragging = False
        self._interpolation_transform_last_point: tuple[float, float] | None = None
        self._interpolation_transform_preview_shift = (0, 0)
        self._interpolation_transform_shift_limits = (0, 0, 0, 0)


        self.square_points: list[tuple[float, float]] = []
        self._square_drag_index: int | None = None
        self.trajectory_region_start: tuple[float, float] | None = None
        self.trajectory_region_end: tuple[float, float] | None = None
        self._trajectory_region_current: tuple[float, float] | None = None
        self._trajectory_region_dragging = False
        self.chamber_rect_points: list[tuple[float, float]] = []
        self.chamber_circle_start: tuple[float, float] | None = None
        self.chamber_circle_end: tuple[float, float] | None = None
        self._chamber_circle_dragging = False
        self._chamber_circle_current: tuple[float, float] | None = None
        self.chamber_transform_mode = False
        self._chamber_transform_target_key: str | None = None
        self._chamber_transform_target_mask: np.ndarray | None = None
        self._chamber_transform_dragging = False
        self._chamber_transform_last_point: tuple[float, float] | None = None
        self._chamber_transform_preview_shift = (0, 0)
        self._chamber_transform_shift_limits = (0, 0, 0, 0)
        self.circle_start: tuple[float, float] | None = None
        self.circle_end: tuple[float, float] | None = None
        self.circle_geometry_source = "exact"
        self._circle_dragging = False
        self._circle_current: tuple[float, float] | None = None
        self._circle_move_dragging = False
        self._circle_move_last_point: tuple[float, float] | None = None
        self.circle_transform_mode = False

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
        self._occ_transform_preview_shift = (0, 0)
        self._occ_transform_shift_limits = (0, 0, 0, 0)
        self._occ_transform_erase_dragging = False
        self._occ_transform_erase_last_point: tuple[float, float] | None = None

        self._pan_dragging = False
        self._pan_drag_start = QPoint()
        self._pan_start_offset = QPointF(0.0, 0.0)
        self._right_button_drag_moved = False

        self.pin_records: list[PinRecord] = []
        self.show_pins_outside_pin_mode = False
        self.snap_to_nearby_pins = False
        self.pin_snap_radius = 20.0
        self.mask_records: dict[str, MaskRecord] = {}
        self.selected_mask_name: str | None = None
        self.chamber_mask: np.ndarray | None = None
        self.chamber_geometry: dict | None = None
        self.room_records: dict[str, RoomRecord] = {}
        self.selected_room_name: str | None = None
        self._mask_hover_info: MaskHoverInfo | None = None
        self._mask_hover_widget_position = QPointF(0.0, 0.0)
        self._hover_geometry_cache: dict[str, dict | None] = {}
        self._mask_fill_cache: dict[str, QPixmap] = {}
        self._mask_margin_cache: dict[str, QPixmap] = {}
        self._mask_fill_path_cache: dict[str, QPainterPath] = {}
        self._mask_margin_path_cache: dict[str, QPainterPath] = {}
        self._chamber_base_cache: QPixmap | None = None
        self._chamber_fill_cache: QPixmap | None = None
        self._chamber_edge_cache: QPixmap | None = None
        self._chamber_base_path_cache: QPainterPath | None = None
        self._chamber_room_fill_cache: dict[str, QPixmap] = {}
        self._chamber_room_edge_cache: dict[str, QPixmap] = {}
        self._chamber_room_path_cache: dict[str, QPainterPath] = {}
        self.node_overlay_points: list[NodeOverlayPoint] = []
        self.node_overlay_color_mode = "white"
        self._node_instance_color_indexes: dict[str, int] = {}
        self._node_bodypart_color_indexes: dict[str, int] = {}

    def set_frame(self, frame_rgb: np.ndarray) -> None:
        frame_rgb = np.ascontiguousarray(frame_rgb)
        height, width, channels = frame_rgb.shape
        image = QImage(frame_rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self._frame_width = width
        self._frame_height = height
        self.update()

    def set_node_overlay_points(self, points: list[NodeOverlayPoint]) -> None:
        self.node_overlay_points = list(points)
        self.update()

    def set_node_overlay_color_mode(self, mode: str) -> None:
        if mode not in {"white", "instance", "bodypart"}:
            mode = "white"
        self.node_overlay_color_mode = mode
        self.update()

    def reset_node_overlay_color_assignments(self) -> None:
        self._node_instance_color_indexes.clear()
        self._node_bodypart_color_indexes.clear()
        self.update()

    @staticmethod
    def _node_palette() -> list[str]:
        return [
            "#22d3ee",
            "#f97316",
            "#a3e635",
            "#f472b6",
            "#facc15",
            "#818cf8",
            "#34d399",
            "#fb7185",
            "#60a5fa",
            "#c084fc",
            "#2dd4bf",
            "#f59e0b",
        ]

    def _node_overlay_color(self, point: NodeOverlayPoint) -> QColor:
        if self.node_overlay_color_mode == "white":
            return QColor("#ffffff")
        if self.node_overlay_color_mode == "instance":
            color_indexes = self._node_instance_color_indexes
            key = point.instance_key
        else:
            color_indexes = self._node_bodypart_color_indexes
            key = point.bodypart
        if key not in color_indexes:
            color_indexes[key] = len(color_indexes)
        palette = self._node_palette()
        return QColor(palette[color_indexes[key] % len(palette)])

    def set_interpolation_draw_mode(self, add: bool, brush_radius: int) -> None:
        self.interpolation_draw_add = bool(add)
        self.interpolation_brush_radius = max(1, int(brush_radius))

    def set_interpolation_transform_mode(self, enabled: bool) -> None:
        self.interpolation_transform_mode = bool(enabled)
        if not enabled:
            self._cancel_interpolation_transform_drag()
        self.update()

    def _begin_interpolation_transform_drag(self, point: tuple[float, float]) -> None:
        mask = self._interpolation_mask
        if mask is None or mask.ndim != 2 or not np.any(mask):
            return
        nonzero = cv2.findNonZero((mask > 0).astype(np.uint8))
        if nonzero is None:
            return
        x, y, width, height = cv2.boundingRect(nonzero)
        mask_height, mask_width = mask.shape
        self._interpolation_transform_dragging = True
        self._interpolation_transform_last_point = point
        self._interpolation_transform_preview_shift = (0, 0)
        self._interpolation_transform_shift_limits = (
            -x,
            mask_width - (x + width),
            -y,
            mask_height - (y + height),
        )

    def _update_interpolation_transform_preview(self, point: tuple[float, float]) -> None:
        if (
            not self._interpolation_transform_dragging
            or self._interpolation_transform_last_point is None
        ):
            return
        dx = int(round(point[0] - self._interpolation_transform_last_point[0]))
        dy = int(round(point[1] - self._interpolation_transform_last_point[1]))
        min_dx, max_dx, min_dy, max_dy = self._interpolation_transform_shift_limits
        self._interpolation_transform_preview_shift = (
            max(min_dx, min(max_dx, dx)),
            max(min_dy, min(max_dy, dy)),
        )
        self.update()

    def _cancel_interpolation_transform_drag(self) -> None:
        self._interpolation_transform_dragging = False
        self._interpolation_transform_last_point = None
        self._interpolation_transform_preview_shift = (0, 0)
        self._interpolation_transform_shift_limits = (0, 0, 0, 0)
        self.update()

    def _finish_interpolation_transform_drag(self) -> None:
        if not self._interpolation_transform_dragging:
            return
        shift = self._interpolation_transform_preview_shift
        self._cancel_interpolation_transform_drag()
        if shift != (0, 0):
            self.interpolation_transform_requested.emit(shift)
        self.interpolation_transform_finished.emit()

    def set_chamber_transform_mode(
        self,
        enabled: bool,
        target_key: str | None,
        target_mask: np.ndarray | None,
    ) -> None:
        self.chamber_transform_mode = bool(enabled)
        self._chamber_transform_target_key = target_key
        self._chamber_transform_target_mask = target_mask
        if not enabled:
            self._cancel_chamber_transform_drag()
        self.update()

    def _begin_chamber_transform_drag(self, point: tuple[float, float]) -> None:
        mask = self._chamber_transform_target_mask
        if mask is None or mask.ndim != 2 or not np.any(mask):
            return
        nonzero = cv2.findNonZero((mask > 0).astype(np.uint8))
        if nonzero is None:
            return
        x, y, width, height = cv2.boundingRect(nonzero)
        mask_height, mask_width = mask.shape
        self._chamber_transform_dragging = True
        self._chamber_transform_last_point = point
        self._chamber_transform_preview_shift = (0, 0)
        self._chamber_transform_shift_limits = (
            -x,
            mask_width - (x + width),
            -y,
            mask_height - (y + height),
        )

    def _update_chamber_transform_preview(self, point: tuple[float, float]) -> None:
        if not self._chamber_transform_dragging or self._chamber_transform_last_point is None:
            return
        dx = int(round(point[0] - self._chamber_transform_last_point[0]))
        dy = int(round(point[1] - self._chamber_transform_last_point[1]))
        min_dx, max_dx, min_dy, max_dy = self._chamber_transform_shift_limits
        self._chamber_transform_preview_shift = (
            max(min_dx, min(max_dx, dx)),
            max(min_dy, min(max_dy, dy)),
        )
        self.update()

    def _cancel_chamber_transform_drag(self) -> None:
        self._chamber_transform_dragging = False
        self._chamber_transform_last_point = None
        self._chamber_transform_preview_shift = (0, 0)
        self._chamber_transform_shift_limits = (0, 0, 0, 0)
        self.update()

    def _finish_chamber_transform_drag(self) -> None:
        if not self._chamber_transform_dragging:
            return
        shift = self._chamber_transform_preview_shift
        self._cancel_chamber_transform_drag()
        if shift != (0, 0):
            self.chamber_transform_requested.emit(shift)
        self.chamber_transform_finished.emit()

    def set_circle_transform_mode(self, enabled: bool) -> None:
        self.circle_transform_mode = bool(enabled)
        if not enabled:
            self._circle_move_dragging = False
            self._circle_move_last_point = None
        self.update()

    def set_interpolation_mask(
        self,
        mask: np.ndarray | None,
        high_quality: bool = True,
    ) -> None:
        self._interpolation_mask = mask
        self._interpolation_mask_cache = None
        self._interpolation_mask_path_cache = None
        if mask is None:
            self._cancel_interpolation_transform_drag()
        if (
            mask is not None
            and self._frame_width > 0
            and self._frame_height > 0
            and mask.shape == (self._frame_height, self._frame_width)
            and np.any(mask)
        ):
            self._interpolation_mask_path_cache = self._mask_to_path(
                mask,
                high_quality=high_quality,
            )
            base = mask.astype(bool)
            edge = cv2.morphologyEx(
                mask.astype(np.uint8),
                cv2.MORPH_GRADIENT,
                np.ones((3, 3), dtype=np.uint8),
            )
            image = np.zeros((self._frame_height, self._frame_width, 4), dtype=np.uint8)
            image[base, 0] = 16
            image[base, 1] = 185
            image[base, 2] = 129
            image[base, 3] = 72
            image[edge > 0, 0] = 167
            image[edge > 0, 1] = 243
            image[edge > 0, 2] = 208
            image[edge > 0, 3] = 225
            qimage = QImage(
                image.data,
                self._frame_width,
                self._frame_height,
                self._frame_width * 4,
                QImage.Format.Format_RGBA8888,
            ).copy()
            self._interpolation_mask_cache = QPixmap.fromImage(qimage)
        self.update()

    def has_frame(self) -> bool:
        return self._pixmap is not None and self._frame_width > 0 and self._frame_height > 0

    def set_mode(self, mode: str) -> None:
        if self.mode != mode:
            self._set_mask_hover_info(None)
        self.mode = mode
        self.update()

    def set_margin_value(self, margin_value: float) -> None:
        self.margin_value = margin_value
        self.update()
        self.circle_changed.emit()
        self.occ_circle_changed.emit()

    def set_occ_transform_mode(self, enabled: bool) -> None:
        self.occ_transform_mode = enabled
        if not enabled:
            self._cancel_occ_transform_drag()

    def _begin_occ_transform_drag(self, point: tuple[float, float]) -> None:
        self._occ_transform_dragging = True
        self._occ_transform_last_point = point
        self._occ_transform_preview_shift = (0, 0)
        self._occ_transform_shift_limits = (0, 0, 0, 0)
        record = self.mask_records.get(self.selected_mask_name) if self.selected_mask_name else None
        if record is None or record.mask.ndim != 2:
            return
        nonzero = cv2.findNonZero((record.mask > 0).astype(np.uint8))
        if nonzero is None:
            return
        x, y, width, height = cv2.boundingRect(nonzero)
        mask_height, mask_width = record.mask.shape
        self._occ_transform_shift_limits = (
            -x,
            mask_width - (x + width),
            -y,
            mask_height - (y + height),
        )

    def _update_occ_transform_preview(self, point: tuple[float, float]) -> None:
        if not self._occ_transform_dragging or self._occ_transform_last_point is None:
            return
        dx = int(round(point[0] - self._occ_transform_last_point[0]))
        dy = int(round(point[1] - self._occ_transform_last_point[1]))
        min_dx, max_dx, min_dy, max_dy = self._occ_transform_shift_limits
        self._occ_transform_preview_shift = (
            max(min_dx, min(max_dx, dx)),
            max(min_dy, min(max_dy, dy)),
        )
        self.update()

    def _cancel_occ_transform_drag(self) -> None:
        self._occ_transform_dragging = False
        self._occ_transform_last_point = None
        self._occ_transform_preview_shift = (0, 0)
        self._occ_transform_shift_limits = (0, 0, 0, 0)
        self.update()

    def _finish_occ_transform_drag(self) -> None:
        if not self._occ_transform_dragging:
            return
        shift = self._occ_transform_preview_shift
        self._cancel_occ_transform_drag()
        if shift != (0, 0):
            self.occ_transform_requested.emit(shift)
        self.occ_transform_finished.emit()

    def set_occ_margin_pick_mode(self, enabled: bool) -> None:
        self.occ_margin_pick_mode = enabled
        if not enabled:
            self._occ_margin_drag_index = None
        self.update()

    def set_pin_records(self, pins: list[PinRecord]) -> None:
        self.pin_records = list(pins)
        self.update()

    def set_pin_overlay_options(
        self,
        show_outside_pin_mode: bool,
        snap_to_nearby_pins: bool,
        snap_radius: int | float = 20,
    ) -> None:
        self.show_pins_outside_pin_mode = bool(show_outside_pin_mode)
        self.snap_to_nearby_pins = bool(snap_to_nearby_pins)
        self.pin_snap_radius = max(0.0, float(snap_radius))
        self.update()

    def _pin_overlay_supported_in_current_mode(self) -> bool:
        if self.mode == "pin":
            return True
        if self.mode in {"inspect", "trajectory_region", "circle"}:
            return True
        if self.mode.startswith("chamber_") or self.mode.startswith("occ_"):
            return True
        return False

    def _pin_snap_supported_in_current_mode(self) -> bool:
        if self.mode in {
            "trajectory_region",
            "interp_rect",
            "interp_circle",
            "chamber_rect",
            "chamber_circle",
            "circle",
            "occ_rect",
            "occ_circle",
            "square",
        }:
            return True
        return False

    def _nearest_pin_for_snap(self, image_point: tuple[float, float]) -> PinRecord | None:
        if not self.pin_records or self.pin_snap_radius <= 0:
            return None
        current_frame = getattr(self, "current_frame_number", None)
        best_pin: PinRecord | None = None
        best_distance_sq = float(self.pin_snap_radius) * float(self.pin_snap_radius)
        for pin in self.pin_records:
            if current_frame is not None and pin.frame != current_frame:
                continue
            dx = float(pin.x) - float(image_point[0])
            dy = float(pin.y) - float(image_point[1])
            distance_sq = dx * dx + dy * dy
            if distance_sq <= best_distance_sq:
                best_distance_sq = distance_sq
                best_pin = pin
        return best_pin

    def _snapped_drawing_point(self, image_point: tuple[float, float]) -> tuple[float, float]:
        if not self.snap_to_nearby_pins or not self._pin_snap_supported_in_current_mode():
            return image_point
        pin = self._nearest_pin_for_snap(image_point)
        if pin is None:
            return image_point
        x = max(0.0, min(float(self._frame_width - 1), float(pin.x))) if self._frame_width > 0 else float(pin.x)
        y = max(0.0, min(float(self._frame_height - 1), float(pin.y))) if self._frame_height > 0 else float(pin.y)
        return (x, y)

    def set_chamber_records(
        self,
        chamber_mask: np.ndarray | None,
        rooms: dict[str, RoomRecord],
        selected_name: str | None,
        refresh: bool = False,
        chamber_geometry: dict | None = None,
    ) -> None:
        previous_names = set(self.room_records.keys())
        previous_selected = self.selected_room_name
        previous_has_chamber = self.chamber_mask is not None and bool(np.any(self.chamber_mask))
        next_has_chamber = chamber_mask is not None and bool(np.any(chamber_mask))
        self.chamber_mask = None if chamber_mask is None else chamber_mask.astype(np.uint8)
        self.chamber_geometry = clone_mask_geometry(chamber_geometry)
        self.room_records = rooms
        self.selected_room_name = selected_name
        self._hover_geometry_cache.clear()
        if refresh or previous_names != set(rooms.keys()) or previous_selected != selected_name or previous_has_chamber != next_has_chamber:
            self._rebuild_chamber_cache()
        self.update()

    def set_mask_records(self, masks: dict[str, MaskRecord], selected_name: str | None, refresh: bool = False) -> None:
        previous_names = set(self.mask_records.keys())
        previous_selected = self.selected_mask_name
        self.mask_records = masks
        self.selected_mask_name = selected_name
        self._hover_geometry_cache.clear()
        if refresh or previous_names != set(masks.keys()) or previous_selected != selected_name:
            self._rebuild_mask_cache()
        self.update()

    def refresh_mask_record(self, name: str, include_margin: bool = True) -> None:
        self._hover_geometry_cache.pop(f"occ:{name}", None)
        record = self.mask_records.get(name)
        if record is None:
            self._mask_fill_cache.pop(name, None)
            self._mask_margin_cache.pop(name, None)
            self._mask_fill_path_cache.pop(name, None)
            self._mask_margin_path_cache.pop(name, None)
            self.update()
            return
        self._rebuild_single_mask_cache(name, record, include_margin=include_margin)
        self.update()

    def _rebuild_mask_cache(self) -> None:
        self._hover_geometry_cache.clear()
        self._mask_fill_cache = {}
        self._mask_margin_cache = {}
        self._mask_fill_path_cache = {}
        self._mask_margin_path_cache = {}
        if self._frame_width <= 0 or self._frame_height <= 0:
            return

        for name, record in self.mask_records.items():
            self._rebuild_single_mask_cache(name, record)

    @staticmethod
    def _colored_mask_pixmap(mask: np.ndarray, color: QColor, alpha: int) -> QPixmap:
        alpha_source = (mask > 0).astype(np.uint8) * np.uint8(max(0, min(255, alpha)))
        soft_alpha = cv2.GaussianBlur(
            alpha_source,
            (3, 3),
            0.0,
            borderType=cv2.BORDER_REPLICATE,
        )
        height, width = mask.shape
        image = np.empty((height, width, 4), dtype=np.uint8)
        image[:, :, 0] = color.red()
        image[:, :, 1] = color.green()
        image[:, :, 2] = color.blue()
        image[:, :, 3] = soft_alpha
        qimage = QImage(
            image.data,
            width,
            height,
            width * 4,
            QImage.Format.Format_RGBA8888,
        ).copy()
        return QPixmap.fromImage(qimage)

    @staticmethod
    def _signed_polygon_area(points: list[tuple[float, float]]) -> float:
        if len(points) < 3:
            return 0.0
        area = 0.0
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            area += float(point[0]) * float(next_point[1]) - float(next_point[0]) * float(point[1])
        return area * 0.5

    @staticmethod
    def _line_intersection(
        point_a: np.ndarray,
        direction_a: np.ndarray,
        point_b: np.ndarray,
        direction_b: np.ndarray,
    ) -> tuple[float, float] | None:
        cross = float(direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0])
        if abs(cross) < 1e-6:
            return None
        delta = point_b - point_a
        t = float(delta[0] * direction_b[1] - delta[1] * direction_b[0]) / cross
        intersection = point_a + direction_a * t
        if not np.all(np.isfinite(intersection)):
            return None
        return float(intersection[0]), float(intersection[1])

    def _offset_polygon_points(
        self,
        points: list[tuple[float, float]],
        margin: float,
    ) -> list[tuple[float, float]] | None:
        if len(points) < 3 or abs(float(margin)) < 1e-6:
            return list(points)
        signed_area = self._signed_polygon_area(points)
        if abs(signed_area) < 1e-6:
            return None
        normal_sign = 1.0 if signed_area > 0 else -1.0
        offset_lines: list[tuple[np.ndarray, np.ndarray]] = []
        for index, point in enumerate(points):
            current = np.array(point, dtype=np.float64)
            next_point = np.array(points[(index + 1) % len(points)], dtype=np.float64)
            direction = next_point - current
            length = float(np.hypot(direction[0], direction[1]))
            if length < 1e-6:
                return None
            unit = direction / length
            outward_normal = np.array([unit[1], -unit[0]], dtype=np.float64) * normal_sign
            offset_lines.append((current + outward_normal * float(margin), unit))

        offset_points: list[tuple[float, float]] = []
        for index in range(len(offset_lines)):
            previous_point, previous_direction = offset_lines[index - 1]
            current_point, current_direction = offset_lines[index]
            intersection = self._line_intersection(
                previous_point,
                previous_direction,
                current_point,
                current_direction,
            )
            if intersection is None:
                return None
            offset_points.append(intersection)
        if abs(self._signed_polygon_area(offset_points)) < 0.5:
            return None
        return offset_points

    def _geometry_to_path(
        self,
        geometry: dict | None,
        margin: int | float = 0,
    ) -> QPainterPath | None:
        if not isinstance(geometry, dict):
            return None
        if str(geometry.get("source", "")).lower() != "exact":
            return None
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.OddEvenFill)
        kind = str(geometry.get("kind", "")).lower()
        if kind == "circle":
            center = self._geometry_point(geometry.get("center"))
            radius = self._geometry_radius(
                geometry.get("adjusted_radius", geometry.get("base_radius", geometry.get("radius")))
            )
            if center is None or radius is None:
                return None
            radius = float(radius) + float(margin)
            if radius <= 0:
                return None
            path.addEllipse(
                QRectF(
                    float(center[0] - radius),
                    float(center[1] - radius),
                    float(radius * 2.0),
                    float(radius * 2.0),
                )
            )
            return None if path.isEmpty() else path

        raw_points = geometry.get("points")
        points: list[tuple[float, float]] = []
        if isinstance(raw_points, list):
            for value in raw_points:
                point = self._geometry_point(value)
                if point is not None:
                    points.append(point)
        if len(points) < 3:
            return None
        points = self._offset_polygon_points(points, float(margin))
        if points is None or len(points) < 3:
            return None
        path.moveTo(float(points[0][0]), float(points[0][1]))
        for point in points[1:]:
            path.lineTo(float(point[0]), float(point[1]))
        path.closeSubpath()
        return None if path.isEmpty() else path

    @staticmethod
    def _add_display_contour(
        path: QPainterPath,
        contour: np.ndarray,
        *,
        high_quality: bool,
    ) -> None:
        """Append an antialiased vector contour without changing its geometry."""
        if len(contour) == 0:
            return
        if len(contour) < 3 or abs(float(cv2.contourArea(contour))) < 0.5:
            x, y, width, height = cv2.boundingRect(contour)
            path.addRect(QRectF(float(x), float(y), float(max(1, width)), float(max(1, height))))
            return

        epsilon = 0.35 if high_quality else 1.0
        approximated = cv2.approxPolyDP(contour, epsilon, True)
        if len(approximated) < 4 and len(contour) >= 4 and abs(float(cv2.contourArea(contour))) >= 8.0:
            safer_epsilon = 0.08 if high_quality else 0.18
            safer = cv2.approxPolyDP(contour, safer_epsilon, True)
            approximated = safer if len(safer) >= 4 else contour
        points = approximated.reshape(-1, 2).astype(np.float64)

        # A rasterized axis-aligned rectangle has a perfectly rectangular contour.
        # Use its pixel bounds so the display remains an exact rectangle at every zoom.
        if len(points) == 4 and cv2.isContourConvex(approximated):
            x, y, width, height = cv2.boundingRect(approximated)
            expected_area = float(max(0, width - 1) * max(0, height - 1))
            contour_area = abs(float(cv2.contourArea(approximated)))
            if expected_area > 0 and abs(contour_area - expected_area) <= max(1.0, expected_area * 0.002):
                path.addRect(QRectF(float(x), float(y), float(width), float(height)))
                return

        points += 0.5
        path.moveTo(float(points[0, 0]), float(points[0, 1]))
        for point in points[1:]:
            path.lineTo(float(point[0]), float(point[1]))
        path.closeSubpath()

    @classmethod
    def _mask_to_path(
        cls,
        mask: np.ndarray | None,
        high_quality: bool = True,
    ) -> QPainterPath | None:
        """Build a faithful resolution-independent path from the original binary mask."""
        if mask is None:
            return None
        binary = (mask > 0).astype(np.uint8)
        if not np.any(binary):
            return None
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_NONE,
        )
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.OddEvenFill)
        for contour in contours:
            cls._add_display_contour(path, contour, high_quality=high_quality)
        return None if path.isEmpty() else path

    def _draw_mask_path(
        self,
        painter: QPainter,
        mask_path: QPainterPath | None,
        target_rect: QRectF,
        *,
        fill_color: QColor | None = None,
        fill_alpha: int = 0,
        outline_color: QColor | None = None,
        outline_alpha: int = 255,
        outline_width: float = 2.0,
        outline_style: Qt.PenStyle = Qt.PenStyle.SolidLine,
    ) -> bool:
        if (
            mask_path is None
            or mask_path.isEmpty()
            or self._frame_width <= 0
            or self._frame_height <= 0
            or target_rect.width() <= 0
            or target_rect.height() <= 0
        ):
            return False

        scale_x = target_rect.width() / max(1, self._frame_width)
        scale_y = target_rect.height() / max(1, self._frame_height)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.translate(target_rect.left(), target_rect.top())
        painter.scale(scale_x, scale_y)

        if fill_color is not None and fill_alpha > 0:
            fill = QColor(fill_color)
            fill.setAlpha(max(0, min(255, int(fill_alpha))))
            painter.setBrush(fill)
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)

        if outline_color is not None and outline_alpha > 0 and outline_width > 0:
            outline = QColor(outline_color)
            outline.setAlpha(max(0, min(255, int(outline_alpha))))
            pen = QPen(outline, max(0.01, float(outline_width)), outline_style)
            pen.setCosmetic(True)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            painter.setPen(pen)
        else:
            painter.setPen(Qt.PenStyle.NoPen)

        painter.drawPath(mask_path)
        painter.restore()
        return True

    def _rebuild_single_mask_cache(self, name: str, record: MaskRecord, include_margin: bool = True) -> None:
        if self._frame_width <= 0 or self._frame_height <= 0:
            return
        selected = name == self.selected_mask_name
        alpha = 105 if selected else 55
        self._mask_fill_cache[name] = self._colored_mask_pixmap(
            record.mask,
            record.color,
            alpha,
        )
        mask_path = self._geometry_to_path(record.geometry) or self._mask_to_path(
            record.mask,
            high_quality=include_margin,
        )
        if mask_path is not None:
            self._mask_fill_path_cache[name] = mask_path
        else:
            self._mask_fill_path_cache.pop(name, None)

        self._mask_margin_cache.pop(name, None)
        self._mask_margin_path_cache.pop(name, None)
        if not include_margin:
            return
        try:
            adjusted = adjust_mask_by_mode(record.mask, record.margin, record.margin_mode, self.occ_margin_points).astype(np.uint8)
        except ValueError:
            adjusted = None
        if adjusted is not None and adjusted.any():
            geometry_margin = (
                float(record.margin)
                if str(record.margin_mode) == "simple"
                else 0.0 if int(record.margin) == 0 else None
            )
            margin_path = (
                self._geometry_to_path(record.geometry, margin=geometry_margin)
                if geometry_margin is not None
                else None
            ) or self._mask_to_path(adjusted)
            if margin_path is not None:
                self._mask_margin_path_cache[name] = margin_path
            edge = cv2.morphologyEx(adjusted, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
            edge_alpha = 220 if selected else 180
            self._mask_margin_cache[name] = self._colored_mask_pixmap(
                edge,
                record.color,
                edge_alpha,
            )

    def _rebuild_chamber_cache(self) -> None:
        self._hover_geometry_cache.clear()
        self._chamber_base_cache = None
        self._chamber_fill_cache = None
        self._chamber_edge_cache = None
        self._chamber_base_path_cache = None
        self._chamber_room_fill_cache = {}
        self._chamber_room_edge_cache = {}
        self._chamber_room_path_cache = {}
        if self._frame_width <= 0 or self._frame_height <= 0:
            return

        if self.chamber_mask is not None and np.any(self.chamber_mask):
            self._chamber_base_cache = self._colored_mask_pixmap(
                self.chamber_mask,
                QColor("#d1d5db"),
                72,
            )
            self._chamber_base_path_cache = self._geometry_to_path(self.chamber_geometry) or self._mask_to_path(self.chamber_mask)

        for name, record in self.room_records.items():
            if not np.any(record.mask):
                continue
            selected = name == self.selected_room_name
            self._chamber_room_fill_cache[name] = self._colored_mask_pixmap(
                record.mask,
                record.color,
                118 if selected else 86,
            )
            room_path = self._geometry_to_path(record.geometry) or self._mask_to_path(record.mask)
            if room_path is not None:
                self._chamber_room_path_cache[name] = room_path
            edge = cv2.morphologyEx(
                record.mask.astype(np.uint8),
                cv2.MORPH_GRADIENT,
                np.ones((3, 3), dtype=np.uint8),
            )
            self._chamber_room_edge_cache[name] = self._colored_mask_pixmap(
                edge,
                record.color,
                245 if selected else 205,
            )

    def reset_view(self) -> None:
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0.0, 0.0)
        self.update()
        self.view_changed.emit()

    def clear_square_points(self) -> None:
        self.square_points.clear()
        self.update()
        self.square_points_changed.emit()

    def clear_trajectory_region(self) -> None:
        self.trajectory_region_start = None
        self.trajectory_region_end = None
        self._trajectory_region_current = None
        self._trajectory_region_dragging = False
        self.update()
        self.trajectory_region_changed.emit()

    def trajectory_region_bounds(self) -> tuple[float, float, float, float] | None:
        end = (
            self._trajectory_region_current
            if self._trajectory_region_dragging and self._trajectory_region_current is not None
            else self.trajectory_region_end
        )
        if self.trajectory_region_start is None or end is None:
            return None
        x1, y1 = self.trajectory_region_start
        x2, y2 = end
        if abs(x2 - x1) < 1.0 or abs(y2 - y1) < 1.0:
            return None
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def clear_interpolation_rect_points(self) -> None:
        self.interpolation_rect_points.clear()
        self.update()
        self.interpolation_rect_points_changed.emit()

    def clear_interpolation_circle(self) -> None:
        self.interpolation_circle_start = None
        self.interpolation_circle_end = None
        self._interpolation_circle_current = None
        self._interpolation_circle_dragging = False
        self.update()
        self.interpolation_circle_changed.emit()

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
        self.circle_geometry_source = "exact"
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

    def _set_mask_hover_info(
        self,
        info: MaskHoverInfo | None,
        widget_position: QPointF | None = None,
    ) -> None:
        position_changed = False
        if widget_position is not None:
            position_changed = (
                abs(widget_position.x() - self._mask_hover_widget_position.x()) > 0.5
                or abs(widget_position.y() - self._mask_hover_widget_position.y()) > 0.5
            )
            self._mask_hover_widget_position = QPointF(widget_position)
        if self._mask_hover_info != info or (info is not None and position_changed):
            self._mask_hover_info = info
            self.update()

    @staticmethod
    def _format_hover_point(point: tuple[float, float] | list[float]) -> str:
        return f"({float(point[0]):.1f}, {float(point[1]):.1f})"

    @staticmethod
    def _geometry_source_label(geometry: dict) -> str:
        return "exact" if str(geometry.get("source", "exact")).lower() == "exact" else "inferred"

    def _format_cursor_hover_lines(self, image_point: tuple[float, float]) -> list[str]:
        x, y = float(image_point[0]), float(image_point[1])
        lines = [f"cursor=({x:.1f}, {y:.1f}) px"]
        if self._frame_width > 0 and self._frame_height > 0:
            lines.append(f"normalized=({x / self._frame_width:.4f}, {y / self._frame_height:.4f})")
        return lines

    @staticmethod
    def _mask_pixel_stats(mask: np.ndarray | None) -> dict[str, int | tuple[int, int, int, int]] | None:
        if mask is None or mask.ndim != 2:
            return None
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        left = int(xs.min())
        right = int(xs.max())
        top = int(ys.min())
        bottom = int(ys.max())
        return {
            "area": int(len(xs)),
            "width": int(right - left + 1),
            "height": int(bottom - top + 1),
            "bbox": (left, top, right, bottom),
        }

    def _hover_mask_from_geometry(self, geometry: dict | None) -> np.ndarray | None:
        if not isinstance(geometry, dict) or self._frame_width <= 0 or self._frame_height <= 0:
            return None
        mask = np.zeros((self._frame_height, self._frame_width), dtype=np.uint8)
        kind = str(geometry.get("kind", "")).lower()
        if kind == "circle":
            center = self._geometry_point(geometry.get("center"))
            radius = self._geometry_radius(
                geometry.get("adjusted_radius", geometry.get("base_radius", geometry.get("radius")))
            )
            if center is None or radius is None:
                return None
            cv2.circle(mask, (int(round(center[0])), int(round(center[1]))), int(round(radius)), 1, thickness=-1)
            return mask
        raw_points = geometry.get("points")
        points: list[tuple[float, float]] = []
        if isinstance(raw_points, list):
            for value in raw_points:
                point = self._geometry_point(value)
                if point is not None:
                    points.append(point)
        if len(points) >= 3:
            contour = np.array(points, dtype=np.float32).round().astype(np.int32)
            cv2.fillPoly(mask, [contour], 1)
            return mask
        return None

    @staticmethod
    def _geometry_point(value: object) -> tuple[float, float] | None:
        try:
            return float(value[0]), float(value[1])  # type: ignore[index]
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _geometry_radius(value: object) -> float | None:
        try:
            radius = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(radius) or radius <= 0:
            return None
        return radius

    def _cached_hover_geometry(
        self,
        key: str,
        geometry: dict | None,
        mask: np.ndarray | None,
    ) -> dict | None:
        cloned = clone_mask_geometry(geometry)
        if cloned is not None:
            cloned.setdefault("source", "exact")
            return cloned
        if mask is None:
            return None
        if key not in self._hover_geometry_cache:
            self._hover_geometry_cache[key] = infer_mask_geometry(mask)
        return clone_mask_geometry(self._hover_geometry_cache.get(key))

    def _point_in_polygon(
        self,
        points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
        point: tuple[float, float],
        margin: float = 6.0,
    ) -> bool:
        if len(points) < 3:
            return False
        contour = np.array(points, dtype=np.float32)
        try:
            return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True) >= -float(margin)
        except cv2.error:
            return False

    @staticmethod
    def _point_in_circle(
        center: tuple[float, float],
        radius: float,
        point: tuple[float, float],
        margin: float = 6.0,
    ) -> bool:
        return math.hypot(point[0] - center[0], point[1] - center[1]) <= radius + margin

    def _room_name_at_point(self, point: tuple[float, float], margin: int = 8) -> str | None:
        if not self.room_records:
            return None
        best_name: str | None = None
        best_distance = float("inf")
        ordered_names = list(self.room_records.keys())
        if self.selected_room_name in self.room_records:
            ordered_names = [self.selected_room_name] + [name for name in ordered_names if name != self.selected_room_name]
        for name in ordered_names:
            record = self.room_records.get(name)
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

    def _format_geometry_hover(
        self,
        title: str,
        geometry: dict | None,
        anchor: tuple[float, float],
        color_name: str,
        mask: np.ndarray | None = None,
        cursor: tuple[float, float] | None = None,
    ) -> MaskHoverInfo | None:
        if not isinstance(geometry, dict):
            return None
        source_label = self._geometry_source_label(geometry)
        source_is_exact = source_label == "exact"
        kind = str(geometry.get("kind", "")).lower()
        lines: list[str] = []
        marker_points: list[tuple[float, float]] = []
        marker_circle: tuple[tuple[float, float], float] | None = None

        if cursor is not None:
            lines.extend(self._format_cursor_hover_lines(cursor))

        stats = self._mask_pixel_stats(mask)
        fallback_bbox_line: str | None = None
        if kind == "circle":
            center = self._geometry_point(geometry.get("center"))
            radius = self._geometry_radius(
                geometry.get("adjusted_radius", geometry.get("base_radius", geometry.get("radius")))
            )
            base_radius = self._geometry_radius(geometry.get("base_radius"))
            if center is not None:
                anchor = center
            if base_radius is not None and radius is not None and abs(base_radius - radius) > 0.05:
                lines.append(f"base radius{'=' if source_is_exact else '~='}{base_radius:.1f}px")
                lines.append(f"adjusted radius{'=' if source_is_exact else '~='}{radius:.1f}px")
            elif radius is not None:
                lines.append(f"radius{'=' if source_is_exact else '~='}{radius:.1f}px")
            if center is not None and radius is not None:
                pass
        else:
            raw_points = geometry.get("points")
            points: list[tuple[float, float]] = []
            if isinstance(raw_points, list):
                for value in raw_points:
                    point = self._geometry_point(value)
                    if point is not None:
                        points.append(point)
            if points and source_is_exact:
                anchor = points[0]
            else:
                centroid = self._geometry_point(geometry.get("centroid"))
                if centroid is not None:
                    anchor = centroid
                bbox = geometry.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    try:
                        left, top, right, bottom = [float(value) for value in bbox]
                        fallback_bbox_line = f"bbox~=({left:.1f}, {top:.1f})-({right:.1f}, {bottom:.1f})"
                    except (TypeError, ValueError):
                        pass

        if stats is not None:
            bbox = stats.get("bbox")
            lines.append(f"area={stats['area']} px")
            lines.append(f"width={stats['width']} px | height={stats['height']} px")
            if isinstance(bbox, tuple) and len(bbox) == 4:
                left, top, right, bottom = bbox
                lines.append(f"bbox=({left}, {top})-({right}, {bottom})")
        else:
            area = geometry.get("area")
            try:
                if area is not None:
                    lines.append(f"area~={int(round(float(area)))} px")
            except (TypeError, ValueError):
                pass
            if fallback_bbox_line is not None:
                lines.append(fallback_bbox_line)

        if not lines:
            return None
        return MaskHoverInfo(
            title=title,
            source_label=source_label,
            lines=tuple(lines[:8]),
            anchor=anchor,
            marker_points=tuple(marker_points),
            marker_circle=marker_circle,
            color_name=color_name,
        )

    def _build_mask_hover_info(self, image_point: tuple[float, float]) -> MaskHoverInfo | None:
        if self.mode == "square" and len(self.square_points) == 4:
            ordered_points = order_quad_points(self.square_points).tolist()
            polygon_points = [(float(x), float(y)) for x, y in ordered_points]
            if self._point_in_polygon(polygon_points, image_point):
                geometry = mask_polygon_geometry(ordered_points, source="exact", shape="rectangle")
                return self._format_geometry_hover(
                    "Square region",
                    geometry,
                    image_point,
                    "#60a5fa",
                    mask=self._hover_mask_from_geometry(geometry),
                    cursor=None,
                )

        if self.mode == "circle":
            circle = self.circle_geometry()
            if circle is not None:
                center, base_radius, adjusted_radius = circle
                if self._point_in_circle(center, adjusted_radius, image_point):
                    geometry = circle_mask_geometry(
                        center,
                        base_radius,
                        adjusted_radius,
                        source=getattr(self, "circle_geometry_source", "exact"),
                    )
                    return self._format_geometry_hover(
                        "Circle mask",
                        geometry,
                        center,
                        "#ef4444",
                        mask=self._hover_mask_from_geometry(geometry),
                        cursor=None,
                    )

        if self.mode.startswith("chamber_"):
            room_name = self._room_name_at_point(image_point, margin=8)
            if room_name is not None:
                record = self.room_records.get(room_name)
                if record is not None:
                    geometry = self._cached_hover_geometry(f"room:{room_name}", record.geometry, record.mask)
                    return self._format_geometry_hover(
                        f"Room: {room_name}",
                        geometry,
                        image_point,
                        record.color.name(),
                        mask=record.mask,
                        cursor=None,
                    )
            if self.chamber_mask is not None and self._mask_hit_distance(self.chamber_mask.astype(np.uint8), image_point, 8) is not None:
                geometry = self._cached_hover_geometry("chamber", self.chamber_geometry, self.chamber_mask)
                return self._format_geometry_hover(
                    "Chamber",
                    geometry,
                    image_point,
                    "#d1d5db",
                    mask=self.chamber_mask,
                    cursor=None,
                )

        if self.mode.startswith("occ_"):
            mask_name = self._mask_name_at_point(image_point, margin=8)
            if mask_name is not None:
                record = self.mask_records.get(mask_name)
                if record is not None:
                    geometry = self._cached_hover_geometry(f"occ:{mask_name}", record.geometry, record.mask)
                    return self._format_geometry_hover(
                        f"Mask: {mask_name}",
                        geometry,
                        image_point,
                        record.color.name(),
                        mask=record.mask,
                        cursor=None,
                    )
        return None

    def _build_cursor_hover_info(self, image_point: tuple[float, float]) -> MaskHoverInfo:
        return MaskHoverInfo(
            title="Cursor",
            source_label="",
            lines=tuple(self._format_cursor_hover_lines(image_point)),
            anchor=image_point,
            color_name="#94a3b8",
        )

    def _update_mask_hover(self, widget_position: QPointF, image_point: tuple[float, float] | None) -> None:
        if image_point is None or self._pan_dragging:
            self._set_mask_hover_info(None, widget_position)
            return
        info = self._build_mask_hover_info(image_point)
        if info is None and self.mode == "pin":
            info = self._build_cursor_hover_info(image_point)
        self._set_mask_hover_info(info, widget_position)

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

    def interpolation_circle_geometry(self) -> tuple[tuple[float, float], float] | None:
        end_point = (
            self._interpolation_circle_current
            if self._interpolation_circle_dragging and self._interpolation_circle_current
            else self.interpolation_circle_end
        )
        if not self.interpolation_circle_start or not end_point:
            return None
        center = ((self.interpolation_circle_start[0] + end_point[0]) / 2.0, (self.interpolation_circle_start[1] + end_point[1]) / 2.0)
        return center, math.dist(self.interpolation_circle_start, end_point) / 2.0

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
            self._set_mask_hover_info(None, event.position())
            self._pan_dragging = True
            self._right_button_drag_moved = False
            self._pan_drag_start = event.position().toPoint()
            self._pan_start_offset = QPointF(self.pan_offset)
            return

        image_point = self._widget_to_image(event.position())
        if image_point is None or event.button() != Qt.MouseButton.LeftButton:
            self._set_mask_hover_info(None, event.position())
            return
        self._set_mask_hover_info(None, event.position())

        if self.mode == "trajectory_region":
            draw_point = self._snapped_drawing_point(image_point)
            self.trajectory_region_start = draw_point
            self.trajectory_region_end = None
            self._trajectory_region_current = draw_point
            self._trajectory_region_dragging = True
            self.update()
            self.trajectory_region_changed.emit()
        elif self.mode.startswith("interp_") and self.interpolation_transform_mode:
            self._begin_interpolation_transform_drag(image_point)
        elif self.mode == "interp_rect":
            draw_point = self._snapped_drawing_point(image_point)
            self.interpolation_rect_points.append(draw_point)
            self.update()
            self.interpolation_rect_points_changed.emit()
            if len(self.interpolation_rect_points) == 4:
                self.interpolation_rect_completed.emit(list(self.interpolation_rect_points))
                self.interpolation_rect_points.clear()
                self.update()
                self.interpolation_rect_points_changed.emit()
        elif self.mode == "interp_circle":
            draw_point = self._snapped_drawing_point(image_point)
            self._interpolation_circle_dragging = True
            self.interpolation_circle_start = draw_point
            self.interpolation_circle_end = None
            self._interpolation_circle_current = draw_point
            self.update()
            self.interpolation_circle_changed.emit()
        elif self.mode == "interp_free":
            self._interpolation_free_dragging = True
            self._interpolation_free_last_point = image_point
            self.interpolation_free_segment.emit(
                (image_point, image_point, self.interpolation_draw_add)
            )
        elif self.mode.startswith("occ_") and self.occ_margin_pick_mode:
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
            self._begin_occ_transform_drag(image_point)
        elif self.mode == "pin":
            self.pin_added.emit(image_point)
        elif self.mode.startswith("chamber_") and self.chamber_transform_mode:
            self._begin_chamber_transform_drag(image_point)
        elif self.mode == "chamber_rect":
            draw_point = self._snapped_drawing_point(image_point)
            self.chamber_rect_points.append(draw_point)
            self.update()
            self.chamber_rect_points_changed.emit()
            if len(self.chamber_rect_points) == 4:
                self.chamber_rect_completed.emit(list(self.chamber_rect_points))
                self.chamber_rect_points.clear()
                self.update()
                self.chamber_rect_points_changed.emit()
        elif self.mode == "chamber_circle":
            draw_point = self._snapped_drawing_point(image_point)
            self._chamber_circle_dragging = True
            self.chamber_circle_start = draw_point
            self.chamber_circle_end = None
            self._chamber_circle_current = draw_point
            self.update()
            self.chamber_circle_changed.emit()
        elif self.mode == "square":
            hit_index = self._point_hit_index(event.position(), self.square_points)
            if hit_index is not None:
                self._square_drag_index = hit_index
            elif len(self.square_points) < 4:
                draw_point = self._snapped_drawing_point(image_point)
                self.square_points.append(draw_point)
                self.update()
                self.square_points_changed.emit()
        elif self.mode == "circle":
            geometry = self.circle_geometry()
            if self.circle_transform_mode and geometry is not None:
                self.circle_edit_started.emit()
                self._circle_move_dragging = True
                self._circle_move_last_point = image_point
                return
            self.circle_edit_started.emit()
            draw_point = self._snapped_drawing_point(image_point)
            self.circle_geometry_source = "exact"
            self._circle_dragging = True
            self.circle_start = draw_point
            self.circle_end = None
            self._circle_current = draw_point
            self.update()
            self.circle_changed.emit()
        elif self.mode == "occ_rect":
            if len(self.occ_rect_points) == 0:
                self._occ_rect_draw_override_add = False if self._control_pressed(event.modifiers()) else None
            elif self._occ_rect_draw_override_add is None and self._control_pressed(event.modifiers()):
                self._occ_rect_draw_override_add = False
            draw_point = self._snapped_drawing_point(image_point)
            self.occ_rect_points.append(draw_point)
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
            draw_point = self._snapped_drawing_point(image_point)
            self._occ_circle_dragging = True
            self.occ_circle_start = draw_point
            self.occ_circle_end = None
            self._occ_circle_current = draw_point
            self._occ_circle_draw_override_add = False if self._control_pressed(event.modifiers()) else None
            self.update()
            self.occ_circle_changed.emit()
        elif self.mode == "occ_free":
            self._free_dragging = True
            self._free_last_point = image_point
            self.free_draw_segment.emit((image_point, image_point, self._effective_draw_add_with_modifiers(event.modifiers())))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._trajectory_region_dragging and not (event.buttons() & Qt.MouseButton.LeftButton):
            self.trajectory_region_end = self._trajectory_region_current
            self._trajectory_region_current = None
            self._trajectory_region_dragging = False
            self.update()
            self.trajectory_region_changed.emit()
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
            self._finish_occ_transform_drag()
        if self._interpolation_transform_dragging and not (event.buttons() & Qt.MouseButton.LeftButton):
            self._finish_interpolation_transform_drag()
        if self._chamber_transform_dragging and not (event.buttons() & Qt.MouseButton.LeftButton):
            self._finish_chamber_transform_drag()
        if self._interpolation_circle_dragging and not (event.buttons() & Qt.MouseButton.LeftButton):
            self._interpolation_circle_dragging = False
            self._interpolation_circle_current = None
            self.interpolation_circle_changed.emit()
        if self._interpolation_free_dragging and not (event.buttons() & Qt.MouseButton.LeftButton):
            self._interpolation_free_dragging = False
            self._interpolation_free_last_point = None
            self.interpolation_free_finished.emit()
        if self._occ_transform_erase_dragging and not (event.buttons() & Qt.MouseButton.LeftButton):
            self._occ_transform_erase_dragging = False
            self._occ_transform_erase_last_point = None

        if self._pan_dragging:
            delta = event.position().toPoint() - self._pan_drag_start
            if delta.manhattanLength() > 4:
                self._right_button_drag_moved = True
            self.pan_offset = self._clamped_pan_offset(self._pan_start_offset + QPointF(delta.x(), delta.y()))
            self.update()
            self.view_changed.emit()
            return

        image_point = self._widget_to_image(event.position())
        if image_point is None:
            self._set_mask_hover_info(None, event.position())
            return

        if self.mode == "trajectory_region" and self._trajectory_region_dragging:
            self._trajectory_region_current = image_point
            self.update()
            self.trajectory_region_changed.emit()
        elif (
            self.mode.startswith("interp_")
            and self.interpolation_transform_mode
            and self._interpolation_transform_dragging
        ):
            self._update_interpolation_transform_preview(image_point)
        elif self.mode == "interp_circle" and self._interpolation_circle_dragging:
            self._interpolation_circle_current = image_point
            self.update()
            self.interpolation_circle_changed.emit()
        elif self.mode == "interp_free" and self._interpolation_free_dragging and self._interpolation_free_last_point is not None:
            self.interpolation_free_segment.emit((self._interpolation_free_last_point, image_point, self.interpolation_draw_add))
            self._interpolation_free_last_point = image_point
        elif self.mode.startswith("occ_") and self.occ_margin_pick_mode and self._occ_margin_drag_index is not None:
            self.occ_margin_points[self._occ_margin_drag_index] = image_point
            self.update()
            self.occ_margin_points_changed.emit()
        elif self.mode == "circle" and self._circle_dragging:
            self._circle_current = image_point
            self.update()
            self.circle_changed.emit()
        elif (
            self.mode.startswith("chamber_")
            and self.chamber_transform_mode
            and self._chamber_transform_dragging
        ):
            self._update_chamber_transform_preview(image_point)
        elif self.mode == "chamber_circle" and self._chamber_circle_dragging:
            self._chamber_circle_current = image_point
            self.update()
            self.chamber_circle_changed.emit()
        elif self.mode == "circle" and self._circle_move_dragging and self._circle_move_last_point is not None:
            dx = image_point[0] - self._circle_move_last_point[0]
            dy = image_point[1] - self._circle_move_last_point[1]
            if self.circle_start is not None and self.circle_end is not None and (dx != 0 or dy != 0):
                min_dx = -min(self.circle_start[0], self.circle_end[0])
                max_dx = (self._frame_width - 1) - max(self.circle_start[0], self.circle_end[0])
                min_dy = -min(self.circle_start[1], self.circle_end[1])
                max_dy = (self._frame_height - 1) - max(self.circle_start[1], self.circle_end[1])
                dx = max(min_dx, min(max_dx, dx))
                dy = max(min_dy, min(max_dy, dy))
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
            self._update_occ_transform_preview(image_point)

        if event.buttons() & Qt.MouseButton.LeftButton:
            self._set_mask_hover_info(None, event.position())
        else:
            self._update_mask_hover(event.position(), image_point)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            was_context_click = self._pan_dragging and not self._right_button_drag_moved
            self._pan_dragging = False
            self._right_button_drag_moved = False
            if was_context_click:
                self.annotate_context_menu_requested.emit(
                    (
                        self.mapToGlobal(event.position().toPoint()),
                        self._widget_to_image(event.position()),
                    )
                )
                event.accept()
            return

        image_point = self._widget_to_image(event.position())
        if (
            self.mode == "trajectory_region"
            and self._trajectory_region_dragging
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self.trajectory_region_end = (
                image_point if image_point is not None else self._trajectory_region_current
            )
            self._trajectory_region_current = None
            self._trajectory_region_dragging = False
            self.update()
            self.trajectory_region_changed.emit()
        elif (
            self.mode.startswith("interp_")
            and self.interpolation_transform_mode
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._finish_interpolation_transform_drag()
        elif self.mode == "interp_circle" and self._interpolation_circle_dragging and event.button() == Qt.MouseButton.LeftButton:
            self.interpolation_circle_end = image_point if image_point is not None else self._interpolation_circle_current
            self._interpolation_circle_current = None
            self._interpolation_circle_dragging = False
            self.update()
            geometry = self.interpolation_circle_geometry()
            self.interpolation_circle_changed.emit()
            if geometry is not None:
                self.interpolation_circle_completed.emit(geometry)
        elif self.mode == "interp_free" and event.button() == Qt.MouseButton.LeftButton:
            self._interpolation_free_dragging = False
            self._interpolation_free_last_point = None
            self.interpolation_free_finished.emit()
        elif self.mode == "circle" and self._circle_dragging and event.button() == Qt.MouseButton.LeftButton:
            self.circle_end = image_point if image_point is not None else self._circle_current
            self._circle_current = None
            self._circle_dragging = False
            self.update()
            self.circle_changed.emit()
        elif (
            self.mode.startswith("chamber_")
            and self.chamber_transform_mode
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._finish_chamber_transform_drag()
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
            was_transform_dragging = self._occ_transform_dragging
            was_erase_dragging = self._occ_transform_erase_dragging
            if was_transform_dragging:
                self._finish_occ_transform_drag()
            self._occ_transform_erase_dragging = False
            self._occ_transform_erase_last_point = None
            if not was_transform_dragging and was_erase_dragging:
                self.occ_transform_finished.emit()

    def leaveEvent(self, event) -> None:
        was_interpolation_free_dragging = self._interpolation_free_dragging
        if self._trajectory_region_dragging:
            self.trajectory_region_end = self._trajectory_region_current
            self._trajectory_region_current = None
            self._trajectory_region_dragging = False
            self.trajectory_region_changed.emit()
        was_free_dragging = self._free_dragging
        was_transform_erase_dragging = self._occ_transform_erase_dragging
        self._pan_dragging = False
        self._chamber_circle_dragging = False
        self._chamber_circle_current = None
        self._circle_dragging = False
        self._circle_current = None
        self._circle_move_dragging = False
        self._circle_move_last_point = None
        self._interpolation_circle_dragging = False
        self._interpolation_circle_current = None
        self._interpolation_free_dragging = False
        self._interpolation_free_last_point = None

        self._square_drag_index = None
        self._occ_margin_drag_index = None
        self._free_dragging = False
        self._free_last_point = None
        self._occ_rect_draw_override_add = None
        self._occ_circle_dragging = False
        self._occ_circle_current = None
        self._occ_circle_draw_override_add = None
        if self._occ_transform_dragging:
            self._finish_occ_transform_drag()
        if self._interpolation_transform_dragging:
            self._finish_interpolation_transform_drag()
        if self._chamber_transform_dragging:
            self._finish_chamber_transform_drag()
        self._occ_transform_erase_dragging = False
        self._occ_transform_erase_last_point = None
        if was_interpolation_free_dragging:
            self.interpolation_free_finished.emit()
        if was_free_dragging:
            self.free_draw_finished.emit()
        if was_transform_erase_dragging:
            self.occ_transform_finished.emit()
        self._set_mask_hover_info(None)
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
                    self._cancel_occ_transform_drag()
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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
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
        self._draw_interpolation_overlay(painter)
        self._draw_chamber_overlay(painter)
        self._draw_mask_overlays(painter)
        self._draw_pin_overlays(painter)
        self._draw_square_overlay(painter)
        self._draw_trajectory_region_overlay(painter)
        self._draw_circle_overlay(painter)
        self._draw_occ_shape_overlay(painter)
        self._draw_node_overlay(painter)
        self._draw_mask_hover_overlay(painter)

    def _draw_node_overlay(self, painter: QPainter) -> None:
        if not self.node_overlay_points:
            return
        image_rect = self._image_rect()
        if image_rect is None:
            return

        painter.save()
        painter.setClipRect(image_rect)
        for node in self.node_overlay_points:
            widget_point = self._image_to_widget((node.x, node.y))
            if widget_point is None:
                continue
            if node.review_role == "delete":
                color = QColor("#ef4444")
                radius = 7.0
            elif node.review_role == "compare":
                color = QColor("#f59e0b")
                radius = 6.0
            else:
                color = self._node_overlay_color(node)
                radius = 5.0
            painter.setPen(QPen(QColor("#0f172a"), 2))
            painter.setBrush(color)
            painter.drawEllipse(widget_point, radius, radius)
        painter.restore()

    def _draw_mask_hover_overlay(self, painter: QPainter) -> None:
        info = self._mask_hover_info
        if info is None:
            return
        image_rect = self._image_rect()
        if image_rect is None:
            return
        text_lines = [info.title]
        if info.source_label:
            text_lines.append(f"source: {info.source_label}")
        text_lines.extend(info.lines)
        metrics = painter.fontMetrics()
        text_width = max(metrics.horizontalAdvance(line) for line in text_lines)
        line_height = metrics.height()
        padding_x = 10.0
        padding_y = 7.0
        box_width = float(text_width) + padding_x * 2.0
        box_height = float(line_height * len(text_lines)) + padding_y * 2.0
        x = self._mask_hover_widget_position.x() + 14.0
        y = self._mask_hover_widget_position.y() + 14.0
        if x + box_width > self.width() - 8.0:
            x = self._mask_hover_widget_position.x() - box_width - 14.0
        if y + box_height > self.height() - 8.0:
            y = self._mask_hover_widget_position.y() - box_height - 14.0
        x = max(8.0, min(x, self.width() - box_width - 8.0))
        y = max(8.0, min(y, self.height() - box_height - 8.0))
        box = QRectF(x, y, box_width, box_height)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QColor(15, 23, 42, 228))
        painter.setPen(QPen(QColor(226, 232, 240, 190), 1))
        painter.drawRoundedRect(box, 6.0, 6.0)
        painter.setPen(QColor("#f8fafc"))
        text_x = x + padding_x
        text_y = y + padding_y + metrics.ascent()
        for index, line in enumerate(text_lines):
            painter.setPen(QColor("#f8fafc") if index == 0 else QColor("#cbd5e1"))
            painter.drawText(QPointF(text_x, text_y + index * line_height), line)
        painter.restore()

    def _draw_interpolation_overlay(self, painter: QPainter) -> None:
        if not self.mode.startswith("interp_"):
            return
        image_rect = self._image_rect()
        if image_rect is None:
            return
        painter.save()
        if (
            self._interpolation_mask_path_cache is not None
            or self._interpolation_mask_cache is not None
        ):
            target_rect = image_rect
            preview_dx, preview_dy = self._interpolation_transform_preview_shift
            if self._interpolation_transform_dragging and (preview_dx != 0 or preview_dy != 0):
                target_rect = QRectF(image_rect)
                target_rect.translate(
                    preview_dx * image_rect.width() / max(1, self._frame_width),
                    preview_dy * image_rect.height() / max(1, self._frame_height),
                )
            if not self._draw_mask_path(
                painter,
                self._interpolation_mask_path_cache,
                target_rect,
                fill_color=QColor("#10b981"),
                fill_alpha=72,
                outline_color=QColor("#a7f3d0"),
                outline_alpha=225,
                outline_width=2.0,
            ) and self._interpolation_mask_cache is not None:
                painter.drawPixmap(
                    target_rect,
                    self._interpolation_mask_cache,
                    QRectF(self._interpolation_mask_cache.rect()),
                )

        if self.interpolation_rect_points:
            widget_points = [self._image_to_widget(point) for point in self.interpolation_rect_points]
            widget_points = [point for point in widget_points if point is not None]
            if len(widget_points) >= 3:
                painter.setBrush(QColor(16, 185, 129, 45))
                painter.setPen(QPen(QColor("#a7f3d0"), 2))
                painter.drawPolygon(QPolygonF(widget_points))
            painter.setPen(QPen(QColor("#ecfdf5"), 2))
            for index, point in enumerate(widget_points, start=1):
                painter.setBrush(QColor("#10b981"))
                painter.drawEllipse(point, 5.0, 5.0)
                painter.drawText(point + QPointF(7, -7), str(index))

        geometry = self.interpolation_circle_geometry()
        if geometry is not None:
            center, radius = geometry
            center_point = self._image_to_widget(center)
            if center_point is not None:
                scale = min(
                    image_rect.width() / self._frame_width,
                    image_rect.height() / self._frame_height,
                )
                painter.setBrush(QColor(16, 185, 129, 45))
                painter.setPen(QPen(QColor("#a7f3d0"), 2))
                painter.drawEllipse(center_point, radius * scale, radius * scale)
        painter.restore()

    def _draw_chamber_overlay(self, painter: QPainter) -> None:
        if not self.mode.startswith("chamber_"):
            return
        rect = self._image_rect()
        if rect is None:
            return

        painter.save()
        painter.setClipRect(rect)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        preview_dx, preview_dy = self._chamber_transform_preview_shift

        chamber_target_rect = rect
        previewing_chamber = (
            self._chamber_transform_dragging
            and self._chamber_transform_target_key == "chamber"
            and (preview_dx != 0 or preview_dy != 0)
        )
        if previewing_chamber:
            chamber_target_rect = QRectF(rect)
            chamber_target_rect.translate(
                preview_dx * rect.width() / max(1, self._frame_width),
                preview_dy * rect.height() / max(1, self._frame_height),
            )
        if not self._draw_mask_path(
            painter,
            self._chamber_base_path_cache,
            chamber_target_rect,
            fill_color=QColor("#d1d5db"),
            fill_alpha=72,
            outline_color=QColor("#e5e7eb"),
            outline_alpha=185,
            outline_width=1.6,
        ) and self._chamber_base_cache is not None:
            painter.drawPixmap(
                chamber_target_rect,
                self._chamber_base_cache,
                QRectF(self._chamber_base_cache.rect()),
            )

        for name, record in self.room_records.items():
            room_target_rect = rect
            previewing_room = (
                self._chamber_transform_dragging
                and self._chamber_transform_target_key == f"room:{name}"
                and (preview_dx != 0 or preview_dy != 0)
            )
            if previewing_room:
                room_target_rect = QRectF(rect)
                room_target_rect.translate(
                    preview_dx * rect.width() / max(1, self._frame_width),
                    preview_dy * rect.height() / max(1, self._frame_height),
                )
            selected = name == self.selected_room_name
            room_path = self._chamber_room_path_cache.get(name)
            if not self._draw_mask_path(
                painter,
                room_path,
                room_target_rect,
                fill_color=record.color,
                fill_alpha=118 if selected else 86,
                outline_color=record.color,
                outline_alpha=245 if selected else 205,
                outline_width=2.0,
            ):
                fill_pixmap = self._chamber_room_fill_cache.get(name)
                if fill_pixmap is not None:
                    painter.drawPixmap(room_target_rect, fill_pixmap, QRectF(fill_pixmap.rect()))
                edge_pixmap = self._chamber_room_edge_cache.get(name)
                if edge_pixmap is not None:
                    painter.drawPixmap(room_target_rect, edge_pixmap, QRectF(edge_pixmap.rect()))

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
        painter.setClipRect(rect)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        preview_dx, preview_dy = self._occ_transform_preview_shift
        for name, record in self.mask_records.items():
            previewing_selected = (
                self._occ_transform_dragging
                and name == self.selected_mask_name
                and (preview_dx != 0 or preview_dy != 0)
            )
            target_rect = rect
            if previewing_selected:
                target_rect = QRectF(rect)
                target_rect.translate(
                    preview_dx * rect.width() / max(1, self._frame_width),
                    preview_dy * rect.height() / max(1, self._frame_height),
                )
            selected = name == self.selected_mask_name
            fill_path = self._mask_fill_path_cache.get(name)
            if not self._draw_mask_path(
                painter,
                fill_path,
                target_rect,
                fill_color=record.color,
                fill_alpha=105 if selected else 55,
            ):
                fill_pixmap = self._mask_fill_cache.get(name)
                if fill_pixmap is not None:
                    painter.drawPixmap(target_rect, fill_pixmap, QRectF(fill_pixmap.rect()))
            if previewing_selected:
                continue
            margin_path = self._mask_margin_path_cache.get(name)
            edge_alpha = 220 if selected else 180
            if not self._draw_mask_path(
                painter,
                margin_path,
                rect,
                outline_color=record.color,
                outline_alpha=edge_alpha,
                outline_width=2.0,
            ):
                margin_pixmap = self._mask_margin_cache.get(name)
                if margin_pixmap is not None:
                    painter.drawPixmap(rect, margin_pixmap, QRectF(margin_pixmap.rect()))
        painter.restore()

    def _draw_pin_overlays(self, painter: QPainter) -> None:
        if not self.pin_records:
            return
        if self.mode != "pin" and not (self.show_pins_outside_pin_mode and self._pin_overlay_supported_in_current_mode()):
            return
        colors = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6", "#8b5cf6"]
        current_frame = getattr(self, "current_frame_number", None)
        visible_pins = [pin for pin in self.pin_records if current_frame is None or pin.frame == current_frame]
        if not visible_pins:
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for index, pin in enumerate(visible_pins):
            point = self._image_to_widget((pin.x, pin.y))
            if point is None:
                continue
            color = QColor(colors[index % len(colors)])
            painter.setPen(QPen(QColor("#0f172a"), 1.5))
            painter.setBrush(color)
            painter.drawEllipse(point, 4.5, 4.5)
            painter.setPen(QPen(QColor("#f8fafc"), 1))
            painter.drawText(point + QPointF(7, -7), pin.pin_id)
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

    def _draw_trajectory_region_overlay(self, painter: QPainter) -> None:
        if self.mode != "trajectory_region":
            return
        bounds = self.trajectory_region_bounds()
        if bounds is None:
            return
        left, top, right, bottom = bounds
        top_left = self._image_to_widget((left, top))
        bottom_right = self._image_to_widget((right, bottom))
        image_rect = self._image_rect()
        if top_left is None or bottom_right is None or image_rect is None:
            return
        painter.save()
        painter.setClipRect(image_rect)
        painter.setBrush(QColor(37, 99, 235, 48))
        pen = QPen(QColor("#93c5fd"), 2.0, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawRect(QRectF(top_left, bottom_right).normalized())
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


class MainWindow(
    TrackingRepairTabMixin,
    InterpolationTabMixin,
    SquareTabMixin,
    ChamberTabMixin,
    CircleTabMixin,
    PinTabMixin,
    OcclusionTabMixin,
    PipelinePanelMixin,
    QMainWindow,
):
    TAB_TRACKING_REPAIR = 0
    TAB_INTERPOLATION = 1
    TAB_SQUARE = 2
    TAB_CHAMBER = 3
    TAB_CIRCLE = 4
    TAB_OCCLUSION = 5
    TAB_TRAJECTORY = 6
    TAB_PIN = 7
    TAB_PIPELINE = 8
    CATEGORY_PREPARE = 0
    CATEGORY_ANNOTATE = 1
    CATEGORY_INSPECT = 2
    CATEGORY_PIPELINE = 3
    FILE_SIDEBAR_COLLAPSED_THRESHOLD = 8
    FILE_SIDEBAR_DEFAULT_WIDTH = 230
    FILE_SIDEBAR_MIN_RESTORE_WIDTH = 160
    FILE_SIDEBAR_MAX_WIDTH = 360
    TAB_VISIBILITY_SCHEMA_VERSION = 2
    PERSISTENT_SPINBOX_SETTINGS_KEY = "spinbox_values"
    PERSISTENT_SPINBOX_NAMES = (
        "tracking_duplicate_criteria_spinbox",
        "tracking_zscore_high_spinbox",
        "tracking_zscore_low_spinbox",
        "interpolation_brush_spinbox",
        "pin_snap_radius_spinbox",
        "mask_brush_spinbox",
        "mask_margin_spinbox",
        "circle_margin_spinbox",
        "square_start_spinbox",
        "square_duration_spinbox",
        "square_end_spinbox",
    )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setStyleSheet(build_app_stylesheet())
        if APP_ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.setMinimumSize(860, 560)
        available_screen = QApplication.primaryScreen()
        if available_screen is None:
            self.resize(1820, 1000)
        else:
            available_geometry = available_screen.availableGeometry()
            self.resize(
                min(1820, max(860, int(available_geometry.width() * 0.94))),
                min(1000, max(560, int(available_geometry.height() * 0.90))),
            )
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.video_state: VideoState | None = None
        self.csv_path: Path | None = None
        self.csv_df: pd.DataFrame | None = None
        self.bodyparts: list[str] = []
        self._node_frame_column: str | None = None
        self._node_frame_values: pd.Series | None = None
        self._node_frame_zero_based = False
        self._node_coordinate_scales: dict[str, tuple[float, float]] = {}
        self.current_folder = Path()
        self.save_folder = DEFAULT_SAVE_DIR

        self.current_frame_number = 1
        self.current_frame_rgb: np.ndarray | None = None
        self._loading_slider = False
        self.settings = self._load_settings()
        try:
            saved_sidebar_width = int(
                self.settings.get("file_sidebar_width", self.FILE_SIDEBAR_DEFAULT_WIDTH)
            )
        except (TypeError, ValueError):
            saved_sidebar_width = self.FILE_SIDEBAR_DEFAULT_WIDTH
        self._last_file_sidebar_width = max(
            self.FILE_SIDEBAR_MIN_RESTORE_WIDTH,
            min(self.FILE_SIDEBAR_MAX_WIDTH, saved_sidebar_width),
        )
        self._setting_file_sidebar_size = False
        self.csv_auto_candidates: list[Path] = []
        csv_manual_dir = str(self.settings.get("csv_manual_dir", "")).strip()
        self.csv_manual_folder: Path | None = None
        if csv_manual_dir:
            candidate = Path(csv_manual_dir)
            if candidate.exists():
                self.csv_manual_folder = candidate
        self._last_mask_preview_refresh = 0.0
        self._last_transform_preview_refresh = 0.0
        self._mask_transform_source_name: str | None = None
        self._mask_transform_source: MaskTransformSource | None = None
        self._mask_transform_angle = 0.0
        self._mask_transform_scale = 1.0
        self._trajectory_preview_windows: list[TrajectoryPreviewDialog] = []
        self._last_interpolation_preview_refresh = 0.0
        self.interpolation_mask: np.ndarray | None = None


        self.pins: list[PinRecord] = []
        self.selected_pin_index: int | None = None
        self.pin_counter = 0
        self.chamber_mask: np.ndarray | None = None
        self.chamber_geometry: dict | None = None
        self.chamber_boundary_mode = "unset"
        self.room_records: dict[str, RoomRecord] = {}
        self.selected_room_name: str | None = None
        self.mask_records: dict[str, MaskRecord] = {}
        self.selected_mask_name: str | None = None
        self._mask_clipboard: MaskClipboardItem | None = None
        self._mask_undo_limit = 10
        self._mask_undo_stacks: dict[str, list[MaskUndoSnapshot]] = {
            "interpolation": [],
            "chamber": [],
            "circle": [],
            "occlusion": [],
        }
        self._mask_undo_restoring = False
        self._interpolation_free_undo_open = False
        self._occlusion_free_undo_open = False
        self._occlusion_transform_undo_open = False
        self.default_mask_margin = int(self.settings.get("occlusion_mask_margin", 0))
        self.default_circle_margin = int(self.settings.get("circle_detection_margin", 0))
        self.default_mask_brush = int(self.settings.get("occlusion_mask_brush", 12))
        self.default_mask_margin_mode = str(self.settings.get("occlusion_mask_margin_mode", "simple"))
        if self.default_mask_margin_mode not in {"simple", "geometric"}:
            self.default_mask_margin_mode = "simple"
        self._mode_tab_memory_suspended = False
        self._settings_save_suspended = False
        self._restoring_persistent_spinbox_values = False
        self.last_mode_tab_by_workflow = self._load_last_mode_tab_by_workflow()

        self._build_ui()
        self._refresh_csv_search_folder_ui()
        self._connect_signals()
        self._restore_file_sidebar_state()
        QTimer.singleShot(0, self._restore_file_sidebar_state)
        QTimer.singleShot(0, self._restore_controls_splitter_state)
        self._register_shortcuts()
        self._initialize_directories()
        self.mask_margin_slider.setValue(self.default_mask_margin)
        self.circle_margin_slider.setValue(self.default_circle_margin)
        self.mask_brush_slider.setValue(self.default_mask_brush)
        if self.default_mask_margin_mode == "geometric":
            self.mask_margin_geometric_radio.setChecked(True)
        else:
            self.mask_margin_simple_radio.setChecked(True)
        self._restore_persistent_spinbox_values()
        self._restore_last_visible_mode_tab()
        self._refresh_chamber_ui()
        self._on_mode_changed(self.mode_tabs.currentIndex())
        self._refresh_output_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        root_layout.addWidget(splitter)
        self.main_splitter = splitter

        file_sidebar = QFrame()
        file_sidebar.setObjectName("fileSidebar")
        file_sidebar.setMinimumWidth(0)
        file_sidebar.setMaximumWidth(self.FILE_SIDEBAR_MAX_WIDTH)
        file_sidebar.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        file_layout = QVBoxLayout(file_sidebar)
        file_layout.setContentsMargins(12, 12, 12, 12)
        file_layout.setSpacing(8)

        file_header = QHBoxLayout()
        file_title = QLabel("VIDEO FILES")
        file_title.setProperty("sectionTitle", True)
        self.video_count_label = QLabel("0 videos")
        self.video_count_label.setProperty("muted", True)
        file_header.addWidget(file_title)
        file_header.addStretch(1)
        file_header.addWidget(self.video_count_label)

        self.choose_folder_button = QPushButton("Open Video Folder")
        self.choose_folder_button.setProperty("primary", True)
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setObjectName("folderPathLabel")
        self.folder_label.setWordWrap(True)
        self.folder_label.setProperty("muted", True)
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.video_list = QListWidget()
        self.video_list.setAlternatingRowColors(False)
        self.video_list.setUniformItemSizes(True)
        self.video_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)

        csv_group = QGroupBox("CSV Mapping")
        csv_layout = QVBoxLayout(csv_group)
        csv_layout.setContentsMargins(8, 10, 8, 8)
        csv_layout.setSpacing(5)
        csv_caption = QLabel("Detected CSV")
        csv_caption.setProperty("muted", True)
        self.csv_auto_combo = NoWheelComboBox()
        self.csv_auto_combo.setEnabled(False)
        self.csv_auto_combo.setToolTip("Automatically discovered CSV candidates for the selected video.")
        self.load_csv_folder_button = QPushButton("Search Folder")
        self.load_csv_folder_button.setToolTip("Select an additional CSV folder for persistent auto-detection.")
        self.load_csv_button = QPushButton("Choose CSV")
        self.load_csv_button.setToolTip("Choose any CSV file manually.")
        csv_button_row = QHBoxLayout()
        csv_button_row.setSpacing(5)
        csv_button_row.addWidget(self.load_csv_folder_button, stretch=1)
        csv_button_row.addWidget(self.load_csv_button, stretch=1)
        csv_path_caption = QLabel("Current path")
        csv_path_caption.setProperty("muted", True)
        self.csv_path_label = QLabel("Select a video first.")
        self.csv_path_label.setWordWrap(True)
        self.csv_path_label.setMaximumHeight(42)
        self.csv_path_label.setProperty("muted", True)
        self.csv_path_label.setToolTip("Full path of the currently selected CSV.")
        csv_layout.addWidget(csv_caption)
        csv_layout.addWidget(self.csv_auto_combo)
        csv_layout.addLayout(csv_button_row)
        csv_layout.addWidget(csv_path_caption)
        csv_layout.addWidget(self.csv_path_label)

        file_layout.addLayout(file_header)
        file_layout.addWidget(self.choose_folder_button)
        file_layout.addWidget(self.folder_label)
        file_layout.addWidget(self.video_list, stretch=1)
        file_layout.addWidget(csv_group)
        splitter.addWidget(file_sidebar)
        self.file_sidebar = file_sidebar

        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(8)

        viewer_header = QFrame()
        viewer_header.setObjectName("viewerHeader")
        viewer_header_layout = QHBoxLayout(viewer_header)
        viewer_header_layout.setContentsMargins(12, 8, 12, 8)
        self.file_sidebar_toggle_label = QLabel()
        self.file_sidebar_toggle_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.LinksAccessibleByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        )
        self.file_sidebar_toggle_label.setOpenExternalLinks(False)
        self.file_sidebar_toggle_label.setToolTip(
            "Hide or restore the Video Files panel (Ctrl+B)."
        )
        viewer_title = QLabel("VIEWER")
        viewer_title.setProperty("sectionTitle", True)
        self.viewer_help_label = QLabel("Wheel: zoom  ·  Right drag: pan  ·  ← / →: frame")
        self.viewer_help_label.setProperty("muted", True)
        self.viewer_help_label.setMinimumWidth(0)
        self.viewer_help_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self.viewer_help_label.setToolTip(self.viewer_help_label.text())
        zoom_caption = QLabel("Zoom")
        zoom_caption.setProperty("muted", True)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setProperty("sectionTitle", True)
        viewer_header_layout.addWidget(self.file_sidebar_toggle_label)
        viewer_header_layout.addSpacing(4)
        viewer_header_layout.addWidget(viewer_title)
        viewer_header_layout.addSpacing(10)
        viewer_header_layout.addWidget(self.viewer_help_label)
        viewer_header_layout.addStretch(1)
        viewer_header_layout.addWidget(zoom_caption)
        viewer_header_layout.addWidget(self.zoom_label)

        self.frame_viewer = FrameViewer()
        self.frame_viewer.setObjectName("frameViewer")
        self.frame_viewer.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        node_controls = QFrame()
        node_controls.setObjectName("viewerNodeControls")
        node_controls_layout = QHBoxLayout(node_controls)
        node_controls_layout.setContentsMargins(12, 8, 12, 8)
        node_title = QLabel("NODES")
        node_title.setProperty("sectionTitle", True)
        self.show_nodes_checkbox = QCheckBox("Show nodes")
        self.show_nodes_checkbox.setEnabled(False)
        self.show_nodes_checkbox.setToolTip("Show the node coordinates from the loaded CSV on the current frame.")
        node_color_label = QLabel("Color")
        node_color_label.setProperty("muted", True)
        self.node_color_combo = NoWheelComboBox()
        self.node_color_combo.addItem("White", "white")
        self.node_color_combo.addItem("Color by instance", "instance")
        self.node_color_combo.addItem("Color by bodypart", "bodypart")
        self.node_color_combo.setEnabled(False)
        self.node_color_combo.setToolTip(
            "Use one color for all nodes, one color per tracked instance, or one color per bodypart."
        )
        node_controls_layout.addWidget(node_title)
        node_controls_layout.addSpacing(10)
        node_controls_layout.addWidget(self.show_nodes_checkbox)
        node_controls_layout.addStretch(1)
        node_controls_layout.addWidget(node_color_label)
        node_controls_layout.addWidget(self.node_color_combo)

        frame_navigation = QFrame()
        frame_navigation.setObjectName("frameNavigation")
        frame_navigation_layout = QHBoxLayout(frame_navigation)
        frame_navigation_layout.setContentsMargins(12, 8, 12, 8)
        navigation_title = QLabel("FRAME")
        navigation_title.setProperty("sectionTitle", True)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setRange(1, 1)
        self.frame_slider.setTracking(True)
        self.frame_spinbox = NoWheelSpinBox()
        self.frame_spinbox.setEnabled(False)
        self.frame_spinbox.setRange(1, 1)
        self.frame_spinbox.setSuffix(" fr")
        self.frame_spinbox.setFixedWidth(92)
        self.frame_position_label = QLabel("Frame 0 / 0")
        self.frame_position_label.setProperty("muted", True)
        frame_navigation_layout.addWidget(navigation_title)
        frame_navigation_layout.addWidget(self.frame_slider, stretch=1)
        frame_navigation_layout.addWidget(self.frame_spinbox)
        frame_navigation_layout.addWidget(self.frame_position_label)

        viewer_layout.addWidget(viewer_header)
        viewer_layout.addWidget(node_controls)
        viewer_layout.addWidget(self.frame_viewer, stretch=1)
        viewer_layout.addWidget(frame_navigation)
        splitter.addWidget(viewer_panel)
        viewer_panel.setMinimumWidth(320)
        self.viewer_panel = viewer_panel

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 2, 0)
        right_layout.setSpacing(8)
        right_panel.setMinimumWidth(330)

        self.workflow_category_bar = QTabBar()
        self.workflow_category_bar.setObjectName("workflowCategoryBar")
        self.workflow_category_bar.setDrawBase(False)
        self.workflow_category_bar.setExpanding(True)
        self.workflow_category_bar.addTab("1. Prepare")
        self.workflow_category_bar.addTab("2. Annotate")
        self.workflow_category_bar.addTab("3. Inspect")
        self.workflow_category_bar.addTab("4. Pipeline")
        self.workflow_category_bar.setTabToolTip(
            self.CATEGORY_PREPARE,
            "Remove invalid tracking rows, repair gaps, and create normalized coordinates.",
        )
        self.workflow_category_bar.setTabToolTip(
            self.CATEGORY_ANNOTATE,
            "Append chamber, circle, and occlusion analysis columns.",
        )
        self.workflow_category_bar.setTabToolTip(
            self.CATEGORY_INSPECT,
            "Inspect coordinates, trajectories, and heatmaps without changing the CSV.",
        )
        self.workflow_category_bar.setTabToolTip(
            self.CATEGORY_PIPELINE,
            "Combine enabled preparation and annotation stages into one output CSV.",
        )

        self.mode_tabs = QTabWidget()
        self.mode_tabs.setObjectName("modeTabs")
        self.mode_tabs.tabBar().setObjectName("modeToolTabBar")
        self.mode_tabs.tabBar().setExpanding(False)
        self.mode_tabs.setMinimumHeight(300)
        self.mode_tabs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.tracking_repair_tab = self._build_tracking_repair_tab()
        self.interpolation_tab = self._build_interpolation_tab()
        self.square_tab = self._build_square_tab()
        self.chamber_tab = self._build_chamber_tab()
        self.circle_tab = self._build_circle_tab()
        self.occlusion_tab = self._build_occlusion_tab()
        self.trajectory_tab = self._build_trajectory_tab()
        self.pin_tab = self._build_pin_tab()
        self.pipeline_tab = self._build_pipeline_tab()
        self.mode_tabs.addTab(self.tracking_repair_tab, "1. Tracking Repair")
        self.mode_tabs.addTab(self.interpolation_tab, "2. Region / Interpolate")
        self.mode_tabs.addTab(self.square_tab, "3. Normalize")
        self.mode_tabs.addTab(self.chamber_tab, "Chamber")
        self.mode_tabs.addTab(self.circle_tab, "Circle")
        self.mode_tabs.addTab(self.occlusion_tab, "Occlusion")
        self.mode_tabs.addTab(self.trajectory_tab, "Trajectory")
        self.mode_tabs.addTab(self.pin_tab, "Pin Coordinates")
        self.mode_tabs.addTab(self.pipeline_tab, "Multi Pipeline")
        self.mode_tabs.setTabToolTip(
            self.TAB_TRACKING_REPAIR,
            "Remove duplicate skeletons first, then robust Z-score length outliers.",
        )
        self.mode_tabs.setTabToolTip(
            self.TAB_INTERPOLATION,
            "Remove skeletons by an anchor node and region, then optionally interpolate missing coordinates.",
        )
        self.mode_tabs.setTabToolTip(self.TAB_SQUARE, "Choose four points and append normalized coordinates without replacing source coordinates.")
        self.mode_tabs.setTabToolTip(self.TAB_TRAJECTORY, "Preview raw or normalized trajectories and heatmaps without changing the CSV.")
        self.mode_tabs.setTabToolTip(self.TAB_CHAMBER, "Define a chamber and named rooms, then export room masks and per-frame room membership.")
        self.mode_tabs.setTabToolTip(self.TAB_CIRCLE, "Draw a circle and classify each bodypart as inside or outside.")
        self.mode_tabs.setTabToolTip(self.TAB_PIN, "Place pins to inspect absolute and normalized coordinates.")
        self.mode_tabs.setTabToolTip(self.TAB_OCCLUSION, "Create and adjust occlusion masks, including simple and geometric margins.")
        self.mode_tabs.setTabToolTip(self.TAB_PIPELINE, "Select multiple configured stages and write one combined CSV.")
        self.annotate_help_button = QPushButton("?")
        self.annotate_help_button.setObjectName("annotateHelpButton")
        self.annotate_help_button.setFixedSize(40, 26)
        self.annotate_help_button.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        help_button_font = self.annotate_help_button.font()
        help_button_font.setBold(True)
        help_button_font.setPixelSize(17)
        self.annotate_help_button.setFont(help_button_font)
        self.annotate_help_button.setAccessibleName("Annotate keyboard shortcuts")
        self.annotate_help_button.setToolTip(
            "Annotate shortcuts: D Draw | T Transform | E/R Scale | "
            "Ctrl+E/R Rotate | F1 Help"
        )
        self._setup_tab_visibility_controls()
        try:
            saved_category = int(self.settings.get("workflow_category", self.CATEGORY_PREPARE))
        except (TypeError, ValueError):
            saved_category = self.CATEGORY_PREPARE
        self.workflow_category_bar.setCurrentIndex(
            max(self.CATEGORY_PREPARE, min(saved_category, self.CATEGORY_PIPELINE))
        )
        self.workflow_category_bar.currentChanged.connect(self._on_workflow_category_changed)
        self.workflow_category_bar.tabBarClicked.connect(self._on_workflow_category_clicked)
        self._apply_workflow_category_visibility(select_first=True)

        self.workflow_tools_label = QLabel()
        self.workflow_tools_label.setObjectName("workflowToolsLabel")
        self._refresh_workflow_hierarchy_ui()

        tool_workspace = QWidget()
        tool_layout = QVBoxLayout(tool_workspace)
        tool_layout.setContentsMargins(0, 0, 0, 0)
        tool_layout.setSpacing(6)
        workflow_heading = QLabel("WORKFLOW")
        workflow_heading.setObjectName("workflowHeading")
        tool_layout.addWidget(workflow_heading)
        tool_layout.addWidget(self.workflow_category_bar)
        tool_layout.addSpacing(2)
        workflow_tools_row = QWidget()
        workflow_tools_row_layout = QHBoxLayout(workflow_tools_row)
        workflow_tools_row_layout.setContentsMargins(0, 0, 0, 0)
        workflow_tools_row_layout.setSpacing(8)
        workflow_tools_row_layout.addWidget(self.workflow_tools_label, stretch=1)
        workflow_tools_row_layout.addWidget(self.annotate_help_button)
        tool_layout.addWidget(workflow_tools_row)
        tool_layout.addWidget(self.mode_tabs, stretch=1)
        self.output_workspace = self._build_output_workspace()
        self.controls_splitter = QSplitter(Qt.Orientation.Vertical)
        self.controls_splitter.setObjectName("controlsSplitter")
        self.controls_splitter.setChildrenCollapsible(False)
        self.controls_splitter.setHandleWidth(8)
        self.controls_splitter.addWidget(tool_workspace)
        self.controls_splitter.addWidget(self.output_workspace)
        self.controls_splitter.setStretchFactor(0, 3)
        self.controls_splitter.setStretchFactor(1, 2)
        right_layout.addWidget(self.controls_splitter, stretch=1)

        right_scroll = QScrollArea()
        right_scroll.setObjectName("controlsScroll")
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(360)
        right_scroll.setWidget(right_panel)
        splitter.addWidget(right_scroll)
        self.controls_scroll = right_scroll

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)
        splitter.setSizes([230, 1050, 500])

        self.statusBar().showMessage("Open a folder or choose a video from the list.")

    def _restore_controls_splitter_state(self) -> None:
        if not hasattr(self, "controls_splitter"):
            return
        saved_sizes = self.settings.get("controls_splitter_sizes", [610, 280])
        if (
            isinstance(saved_sizes, list)
            and len(saved_sizes) == 2
            and all(isinstance(value, (int, float)) and value >= 0 for value in saved_sizes)
        ):
            self.controls_splitter.setSizes([int(saved_sizes[0]), int(saved_sizes[1])])
        else:
            self.controls_splitter.setSizes([610, 280])

    def _file_sidebar_is_collapsed(self) -> bool:
        if not hasattr(self, "main_splitter"):
            return bool(self.settings.get("file_sidebar_collapsed", False))
        sizes = self.main_splitter.sizes()
        return not sizes or sizes[0] <= self.FILE_SIDEBAR_COLLAPSED_THRESHOLD

    def _update_file_sidebar_toggle_ui(self) -> None:
        if not hasattr(self, "file_sidebar_toggle_label"):
            return
        collapsed = self._file_sidebar_is_collapsed()
        link_text = "Show files" if collapsed else "Hide files"
        self.file_sidebar_toggle_label.setText(
            f'<a href="toggle-files">{link_text}</a>'
        )
        action = "Restore" if collapsed else "Hide"
        self.file_sidebar_toggle_label.setToolTip(
            f"{action} the Video Files panel (Ctrl+B)."
        )

    def _on_main_splitter_moved(self, _position: int, _index: int) -> None:
        sizes = self.main_splitter.sizes()
        if (
            not self._setting_file_sidebar_size
            and sizes
            and sizes[0] > self.FILE_SIDEBAR_COLLAPSED_THRESHOLD
        ):
            self._last_file_sidebar_width = max(
                self.FILE_SIDEBAR_MIN_RESTORE_WIDTH,
                min(self.FILE_SIDEBAR_MAX_WIDTH, sizes[0]),
            )
        self._update_file_sidebar_toggle_ui()

    def _hide_file_sidebar(self) -> None:
        sizes = self.main_splitter.sizes()
        if len(sizes) < 3:
            return
        sidebar_width, viewer_width, controls_width = sizes[:3]
        if sidebar_width > self.FILE_SIDEBAR_COLLAPSED_THRESHOLD:
            self._last_file_sidebar_width = max(
                self.FILE_SIDEBAR_MIN_RESTORE_WIDTH,
                min(self.FILE_SIDEBAR_MAX_WIDTH, sidebar_width),
            )
        self._setting_file_sidebar_size = True
        try:
            self.main_splitter.setSizes(
                [0, max(1, viewer_width + sidebar_width), max(1, controls_width)]
            )
        finally:
            self._setting_file_sidebar_size = False
        self._update_file_sidebar_toggle_ui()

    def _show_file_sidebar(self) -> None:
        sizes = self.main_splitter.sizes()
        if len(sizes) < 3:
            return
        sidebar_width, viewer_width, controls_width = sizes[:3]
        if sidebar_width > self.FILE_SIDEBAR_COLLAPSED_THRESHOLD:
            self._update_file_sidebar_toggle_ui()
            return

        desired_width = max(
            self.FILE_SIDEBAR_MIN_RESTORE_WIDTH,
            min(self.FILE_SIDEBAR_MAX_WIDTH, self._last_file_sidebar_width),
        )
        viewer_minimum = self.viewer_panel.minimumWidth()
        controls_minimum = self.controls_scroll.minimumWidth()
        total_width = max(0, sum(sizes))
        available_width = max(0, total_width - viewer_minimum - controls_minimum)
        target_width = min(desired_width, available_width) if available_width else desired_width

        viewer_take = min(target_width, max(0, viewer_width - viewer_minimum))
        controls_take = min(
            target_width - viewer_take,
            max(0, controls_width - controls_minimum),
        )
        restored_width = viewer_take + controls_take
        if restored_width <= self.FILE_SIDEBAR_COLLAPSED_THRESHOLD:
            restored_width = target_width

        self._setting_file_sidebar_size = True
        try:
            self.main_splitter.setSizes(
                [
                    restored_width,
                    max(viewer_minimum, viewer_width - viewer_take),
                    max(controls_minimum, controls_width - controls_take),
                ]
            )
        finally:
            self._setting_file_sidebar_size = False
        self._update_file_sidebar_toggle_ui()

    def _toggle_file_sidebar(self) -> None:
        if self._file_sidebar_is_collapsed():
            self._show_file_sidebar()
        else:
            self._hide_file_sidebar()

    def _restore_file_sidebar_state(self) -> None:
        if bool(self.settings.get("file_sidebar_collapsed", False)):
            self._hide_file_sidebar()
            return

        sizes = self.main_splitter.sizes()
        if len(sizes) < 3 or sum(sizes) <= 0:
            self._setting_file_sidebar_size = True
            try:
                self.main_splitter.setSizes(
                    [self._last_file_sidebar_width, 1050, 500]
                )
            finally:
                self._setting_file_sidebar_size = False
            self._update_file_sidebar_toggle_ui()
            return

        sidebar_width, viewer_width, controls_width = sizes[:3]
        available_width = max(
            0,
            sum(sizes)
            - self.viewer_panel.minimumWidth()
            - self.controls_scroll.minimumWidth(),
        )
        target_width = min(self._last_file_sidebar_width, available_width)
        width_delta = target_width - sidebar_width
        if width_delta >= 0:
            viewer_take = min(
                width_delta,
                max(0, viewer_width - self.viewer_panel.minimumWidth()),
            )
            controls_take = min(
                width_delta - viewer_take,
                max(0, controls_width - self.controls_scroll.minimumWidth()),
            )
            target_width = sidebar_width + viewer_take + controls_take
            viewer_width -= viewer_take
            controls_width -= controls_take
        else:
            viewer_width += -width_delta

        self._setting_file_sidebar_size = True
        try:
            self.main_splitter.setSizes(
                [target_width, viewer_width, controls_width]
            )
        finally:
            self._setting_file_sidebar_size = False
        self._update_file_sidebar_toggle_ui()

    def _mode_tab_specs(self) -> tuple[tuple[int, str, str], ...]:
        return (
            (self.TAB_TRACKING_REPAIR, "tracking_repair", "Tracking Repair"),
            (self.TAB_INTERPOLATION, "interpolation", "Region / Interpolate"),
            (self.TAB_SQUARE, "square", "Normalize"),
            (self.TAB_CHAMBER, "chamber", "Chamber"),
            (self.TAB_CIRCLE, "circle", "Circle"),
            (self.TAB_OCCLUSION, "occlusion", "Occlusion"),
            (self.TAB_TRAJECTORY, "trajectory", "Trajectory"),
            (self.TAB_PIN, "pin", "Pin Coordinates"),
            (self.TAB_PIPELINE, "pipeline", "Multi Pipeline"),
        )

    def _mode_tab_category(self, tab_index: int) -> int:
        if tab_index in {
            self.TAB_TRACKING_REPAIR,
            self.TAB_INTERPOLATION,
            self.TAB_SQUARE,
        }:
            return self.CATEGORY_PREPARE
        if tab_index in {self.TAB_CHAMBER, self.TAB_CIRCLE, self.TAB_OCCLUSION}:
            return self.CATEGORY_ANNOTATE
        if tab_index == self.TAB_PIPELINE:
            return self.CATEGORY_PIPELINE
        return self.CATEGORY_INSPECT

    def _default_mode_tab_by_workflow(self) -> dict[int, int]:
        return {
            self.CATEGORY_PREPARE: self.TAB_TRACKING_REPAIR,
            self.CATEGORY_ANNOTATE: self.TAB_CHAMBER,
            self.CATEGORY_INSPECT: self.TAB_TRAJECTORY,
            self.CATEGORY_PIPELINE: self.TAB_PIPELINE,
        }

    def _valid_mode_tab_index(self, tab_index: int) -> bool:
        return tab_index in {index for index, _key, _label in self._mode_tab_specs()}

    def _load_last_mode_tab_by_workflow(self) -> dict[int, int]:
        tab_by_workflow = self._default_mode_tab_by_workflow()
        raw_map = self.settings.get("last_mode_tab_by_workflow")
        if isinstance(raw_map, dict):
            for category_key, tab_value in raw_map.items():
                try:
                    category = int(category_key)
                    tab_index = int(tab_value)
                except (TypeError, ValueError):
                    continue
                if (
                    category in tab_by_workflow
                    and self._valid_mode_tab_index(tab_index)
                    and self._mode_tab_category(tab_index) == category
                ):
                    tab_by_workflow[category] = tab_index

        try:
            legacy_last_tab = int(self.settings.get("last_tab_index", -1))
        except (TypeError, ValueError):
            legacy_last_tab = -1
        if self._valid_mode_tab_index(legacy_last_tab):
            tab_by_workflow[self._mode_tab_category(legacy_last_tab)] = legacy_last_tab
        return tab_by_workflow

    def _remember_mode_tab_by_workflow(self, tab_index: int, *, force: bool = False) -> None:
        if getattr(self, "_mode_tab_memory_suspended", False) and not force:
            return
        if not self._valid_mode_tab_index(tab_index):
            return
        if not hasattr(self, "last_mode_tab_by_workflow"):
            self.last_mode_tab_by_workflow = self._default_mode_tab_by_workflow()
        self.last_mode_tab_by_workflow[self._mode_tab_category(tab_index)] = tab_index

    def _preferred_mode_tab_for_workflow(self, category: int, visible_indexes: list[int]) -> int:
        if not visible_indexes:
            return -1
        remembered_index = getattr(self, "last_mode_tab_by_workflow", {}).get(category)
        if remembered_index in visible_indexes:
            return int(remembered_index)
        default_index = self._default_mode_tab_by_workflow().get(category)
        if default_index in visible_indexes:
            return int(default_index)
        return int(visible_indexes[0])

    def _apply_workflow_category_visibility(self, *, select_first: bool) -> None:
        if not hasattr(self, "workflow_category_bar"):
            return
        category = self.workflow_category_bar.currentIndex()
        if hasattr(self, "annotate_help_button"):
            self.annotate_help_button.setVisible(category == self.CATEGORY_ANNOTATE)

        visible_indexes: list[int] = []
        selected_index: int | None = None
        was_suspended = bool(getattr(self, "_mode_tab_memory_suspended", False))
        self._mode_tab_memory_suspended = True
        try:
            for tab_index, _key, _label in self._mode_tab_specs():
                action = getattr(self, "tab_visibility_actions", {}).get(tab_index)
                enabled_by_user = action is None or action.isChecked()
                visible = enabled_by_user and self._mode_tab_category(tab_index) == category
                self.mode_tabs.setTabVisible(tab_index, visible)
                if visible:
                    visible_indexes.append(tab_index)

            if visible_indexes:
                current_index = self.mode_tabs.currentIndex()
                if select_first or current_index not in visible_indexes:
                    preferred_index = self._preferred_mode_tab_for_workflow(category, visible_indexes)
                    if preferred_index >= 0:
                        self.mode_tabs.setCurrentIndex(preferred_index)
                        selected_index = preferred_index
                else:
                    selected_index = current_index
        finally:
            self._mode_tab_memory_suspended = was_suspended

        if selected_index is None:
            selected_index = self.mode_tabs.currentIndex()
        if selected_index in visible_indexes:
            self._remember_mode_tab_by_workflow(selected_index, force=True)

    def _refresh_workflow_hierarchy_ui(self) -> None:
        if not hasattr(self, "workflow_tools_label"):
            return
        category_titles = {
            self.CATEGORY_PREPARE: "PREPARE TOOLS",
            self.CATEGORY_ANNOTATE: "ANNOTATE TOOLS",
            self.CATEGORY_INSPECT: "INSPECT TOOLS",
            self.CATEGORY_PIPELINE: "PIPELINE TOOLS",
        }
        current_category = self.workflow_category_bar.currentIndex()
        self.workflow_tools_label.setText(
            category_titles.get(current_category, "TOOLS")
        )

    def _on_workflow_category_changed(self, _category_index: int) -> None:
        self._refresh_workflow_hierarchy_ui()
        self._apply_workflow_category_visibility(select_first=True)
        self._save_settings()

    def _on_workflow_category_clicked(self, category_index: int) -> None:
        if category_index != self.workflow_category_bar.currentIndex():
            return
        self._refresh_workflow_hierarchy_ui()
        self._apply_workflow_category_visibility(select_first=True)
        self._save_settings()

    def _setup_tab_visibility_controls(self) -> None:
        self._tab_visibility_updating = True
        self.tab_visibility_actions: dict[int, QAction] = {}

        view_menu = self.menuBar().addMenu("&View")
        self.visible_tabs_menu = view_menu.addMenu("Visible Tabs")
        saved_keys = self.settings.get("visible_mode_tabs")
        valid_keys = {key for _, key, _ in self._mode_tab_specs()}
        if isinstance(saved_keys, list):
            visible_keys = {str(key) for key in saved_keys} & valid_keys
        else:
            visible_keys = set(valid_keys)
        try:
            visibility_schema_version = int(self.settings.get("tab_visibility_schema_version", 1))
        except (TypeError, ValueError):
            visibility_schema_version = 1
        if visibility_schema_version < self.TAB_VISIBILITY_SCHEMA_VERSION:
            visible_keys.add("trajectory")
        if not visible_keys:
            visible_keys = set(valid_keys)
        default_key_by_category = {
            self.CATEGORY_PREPARE: "tracking_repair",
            self.CATEGORY_ANNOTATE: "chamber",
            self.CATEGORY_INSPECT: "trajectory",
            self.CATEGORY_PIPELINE: "pipeline",
        }
        key_category = {
            key: self._mode_tab_category(tab_index)
            for tab_index, key, _label in self._mode_tab_specs()
        }
        for category, default_key in default_key_by_category.items():
            if not any(key_category.get(key) == category for key in visible_keys):
                visible_keys.add(default_key)

        for tab_index, key, label in self._mode_tab_specs():
            action = QAction(label, self)
            action.setCheckable(True)
            is_visible = key in visible_keys
            action.setChecked(is_visible)
            action.toggled.connect(
                lambda checked, index=tab_index: self._set_mode_tab_visible(index, checked)
            )
            self.tab_visibility_actions[tab_index] = action
            self.visible_tabs_menu.addAction(action)
            self.mode_tabs.setTabVisible(tab_index, is_visible)

        self.visible_tabs_menu.addSeparator()
        self.show_all_tabs_action = QAction("Show All Tabs", self)
        self.show_all_tabs_action.triggered.connect(self._show_all_mode_tabs)
        self.visible_tabs_menu.addAction(self.show_all_tabs_action)

        tab_bar = self.mode_tabs.tabBar()
        tab_bar.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        tab_bar.setToolTip("Right-click to choose which tabs are visible.")
        tab_bar.customContextMenuRequested.connect(self._show_tab_visibility_context_menu)
        self._tab_visibility_updating = False
        self._refresh_tab_visibility_action_states()

    def _visible_mode_tab_keys(self) -> list[str]:
        if not hasattr(self, "tab_visibility_actions"):
            return [key for _, key, _ in self._mode_tab_specs()]
        return [
            key
            for tab_index, key, _ in self._mode_tab_specs()
            if self.tab_visibility_actions[tab_index].isChecked()
        ]

    def _refresh_tab_visibility_action_states(self) -> None:
        visible_by_category: dict[int, list[int]] = {}
        for tab_index, action in self.tab_visibility_actions.items():
            if action.isChecked():
                visible_by_category.setdefault(self._mode_tab_category(tab_index), []).append(tab_index)
        for tab_index, action in self.tab_visibility_actions.items():
            category_indexes = visible_by_category.get(self._mode_tab_category(tab_index), [])
            action.setEnabled(not (action.isChecked() and len(category_indexes) == 1))
        visible_count = sum(len(indexes) for indexes in visible_by_category.values())
        self.show_all_tabs_action.setEnabled(visible_count < len(self.tab_visibility_actions))

    def _set_mode_tab_visible(self, tab_index: int, checked: bool) -> None:
        if self._tab_visibility_updating:
            return
        action = self.tab_visibility_actions[tab_index]
        if not checked and not any(
            other_action.isChecked()
            for other_index, other_action in self.tab_visibility_actions.items()
            if other_index != tab_index
            and self._mode_tab_category(other_index) == self._mode_tab_category(tab_index)
        ):
            self._tab_visibility_updating = True
            action.setChecked(True)
            self._tab_visibility_updating = False
            self.statusBar().showMessage("Each workflow category needs at least one visible tool.", 3000)
            return

        self._apply_workflow_category_visibility(select_first=not checked)
        self._refresh_tab_visibility_action_states()
        self._save_settings()

    def _show_all_mode_tabs(self) -> None:
        self._tab_visibility_updating = True
        for _tab_index, action in self.tab_visibility_actions.items():
            action.setChecked(True)
        self._tab_visibility_updating = False
        self._apply_workflow_category_visibility(select_first=False)
        self._refresh_tab_visibility_action_states()
        self._save_settings()

    def _show_tab_visibility_context_menu(self, position: QPoint) -> None:
        menu = QMenu(self)
        title_action = menu.addAction("Visible Tabs")
        title_action.setEnabled(False)
        menu.addSeparator()
        for tab_index, _, _ in self._mode_tab_specs():
            menu.addAction(self.tab_visibility_actions[tab_index])
        menu.addSeparator()
        menu.addAction(self.show_all_tabs_action)
        menu.exec(self.mode_tabs.tabBar().mapToGlobal(position))

    def _restore_last_visible_mode_tab(self) -> None:
        try:
            requested_index = int(self.settings.get("last_tab_index", self.TAB_TRACKING_REPAIR))
        except (TypeError, ValueError):
            requested_index = self.TAB_TRACKING_REPAIR
        if not self._valid_mode_tab_index(requested_index):
            requested_index = self._preferred_mode_tab_for_workflow(
                self.workflow_category_bar.currentIndex(),
                [
                    tab_index
                    for tab_index, _, _ in self._mode_tab_specs()
                    if self._mode_tab_category(tab_index) == self.workflow_category_bar.currentIndex()
                    and self.mode_tabs.isTabVisible(tab_index)
                ],
            )
        if requested_index < 0:
            requested_index = self.TAB_TRACKING_REPAIR
        requested_category = self._mode_tab_category(requested_index)
        self.workflow_category_bar.setCurrentIndex(requested_category)
        self._apply_workflow_category_visibility(select_first=False)
        visible_indexes = [
            tab_index
            for tab_index, _, _ in self._mode_tab_specs()
            if self.mode_tabs.isTabVisible(tab_index)
        ]
        if visible_indexes:
            selected_index = (
                requested_index
                if requested_index in visible_indexes
                else self._preferred_mode_tab_for_workflow(requested_category, visible_indexes)
            )
            if selected_index >= 0:
                self.mode_tabs.setCurrentIndex(selected_index)
                self._remember_mode_tab_by_workflow(selected_index, force=True)

    def _connect_signals(self) -> None:

        self.file_sidebar_toggle_label.linkActivated.connect(
            lambda _href: self._toggle_file_sidebar()
        )
        self.main_splitter.splitterMoved.connect(self._on_main_splitter_moved)
        self.choose_save_folder_button.clicked.connect(self.choose_save_folder)
        self.choose_folder_button.clicked.connect(self.choose_folder)
        self.load_csv_folder_button.clicked.connect(self.choose_csv_folder)
        self.load_csv_button.clicked.connect(self.choose_csv)
        self.csv_auto_combo.currentIndexChanged.connect(self._on_csv_auto_selection_changed)
        self.video_list.currentItemChanged.connect(self._on_video_item_changed)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        self.frame_spinbox.valueChanged.connect(self._on_slider_changed)
        self.show_nodes_checkbox.toggled.connect(self._refresh_node_overlay)
        self.node_color_combo.currentIndexChanged.connect(self._on_node_color_mode_changed)
        self.mode_tabs.currentChanged.connect(self._on_mode_changed)

        self.square_reset_button.clicked.connect(self.frame_viewer.clear_square_points)
        self.square_preview_button.clicked.connect(self.preview_square_normalization)
        self.chamber_shape_combo.currentIndexChanged.connect(self._sync_chamber_mode)
        self.chamber_edit_chamber_radio.toggled.connect(
            self._on_chamber_target_or_mode_changed
        )
        self.chamber_edit_room_radio.toggled.connect(
            self._on_chamber_target_or_mode_changed
        )
        self.chamber_draw_radio.toggled.connect(self._on_chamber_target_or_mode_changed)
        self.chamber_transform_radio.toggled.connect(
            self._on_chamber_target_or_mode_changed
        )
        self.room_combo.currentIndexChanged.connect(self._on_room_selection_changed)
        self.room_add_button.clicked.connect(self.add_room)
        self.room_rename_button.clicked.connect(self.rename_room)
        self.room_delete_button.clicked.connect(self.delete_room)
        self.room_clear_button.clicked.connect(self.clear_selected_room)
        self.chamber_full_frame_button.clicked.connect(self.set_chamber_to_full_frame)
        self.chamber_reset_button.clicked.connect(self.reset_chamber)
        self.import_chamber_mask_button.clicked.connect(self.import_chamber_mask)
        self.export_chamber_mask_button.clicked.connect(self.export_chamber_mask)
        self.circle_draw_radio.toggled.connect(self._sync_circle_mode)
        self.circle_transform_radio.toggled.connect(self._sync_circle_mode)
        self.circle_margin_slider.valueChanged.connect(self._on_circle_margin_changed)
        self.circle_margin_spinbox.valueChanged.connect(self._on_circle_margin_changed)
        self.circle_reset_button.clicked.connect(self.clear_circle_with_undo)
        self.import_circle_mask_button.clicked.connect(self.import_circle_mask)
        self.export_circle_mask_button.clicked.connect(self.export_circle_mask)
        self.pin_reset_button.clicked.connect(self.reset_pins)
        self.pin_remove_last_button.clicked.connect(self.remove_last_pin)
        self.pin_import_button.clicked.connect(self.import_pins_metadata)
        self.pin_export_button.clicked.connect(self.export_pins_metadata)
        self.pin_show_outside_tab_checkbox.toggled.connect(self._on_pin_option_changed)
        self.pin_snap_checkbox.toggled.connect(self._on_pin_option_changed)
        self.pin_snap_radius_spinbox.valueChanged.connect(self._on_pin_option_changed)

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
        self.annotate_help_button.clicked.connect(self.show_annotate_controls_help)

        self.save_current_button.clicked.connect(self.save_current_mode_output)
        self.save_multiple_button.clicked.connect(self.save_multiple_mode_outputs)
        self.export_masks_button.clicked.connect(self.export_masks)

        self.frame_viewer.square_points_changed.connect(self._on_square_points_changed)
        self.frame_viewer.interpolation_rect_completed.connect(self.apply_interpolation_rect)
        self.frame_viewer.interpolation_rect_points_changed.connect(self._refresh_interpolation_ui)
        self.frame_viewer.interpolation_circle_completed.connect(self.apply_interpolation_circle)
        self.frame_viewer.interpolation_circle_changed.connect(self._refresh_interpolation_ui)
        self.frame_viewer.interpolation_free_segment.connect(self.apply_interpolation_free_segment)
        self.frame_viewer.interpolation_free_finished.connect(self._finalize_interpolation_free_draw)
        self.frame_viewer.interpolation_transform_requested.connect(
            self.translate_interpolation_region
        )
        self.frame_viewer.chamber_rect_completed.connect(self.apply_chamber_rect)
        self.frame_viewer.chamber_rect_points_changed.connect(self._refresh_chamber_ui)
        self.frame_viewer.chamber_circle_completed.connect(self.apply_chamber_circle)
        self.frame_viewer.chamber_circle_changed.connect(self._refresh_chamber_ui)
        self.frame_viewer.chamber_transform_requested.connect(
            self.translate_selected_chamber_layer
        )
        self.frame_viewer.circle_changed.connect(self._refresh_circle_ui)
        self.frame_viewer.circle_edit_started.connect(self._push_circle_undo)
        self.frame_viewer.view_changed.connect(self._refresh_view_ui)
        self.frame_viewer.annotate_context_menu_requested.connect(
            self._show_annotate_mask_context_menu
        )
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
        self._connect_persistent_spinbox_settings()

    def _register_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+B"), self, activated=self._toggle_file_sidebar)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo_active_mask_edit)
        QShortcut(QKeySequence("Left"), self, activated=lambda: self.step_frame(-1))
        QShortcut(QKeySequence("Right"), self, activated=lambda: self.step_frame(1))
        QShortcut(QKeySequence("D"), self, activated=lambda: self._switch_region_edit_mode_shortcut(transform_mode=False))
        QShortcut(QKeySequence("T"), self, activated=lambda: self._switch_region_edit_mode_shortcut(transform_mode=True))
        QShortcut(QKeySequence("E"), self, activated=self._handle_e_shortcut)
        QShortcut(QKeySequence("R"), self, activated=self._handle_r_shortcut)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self._handle_ctrl_e_shortcut)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._handle_ctrl_r_shortcut)
        QShortcut(QKeySequence("["), self, activated=lambda: self._scale_active_region_shortcut(0.96))
        QShortcut(QKeySequence("]"), self, activated=lambda: self._scale_active_region_shortcut(1.04))
        QShortcut(QKeySequence("Ctrl+["), self, activated=lambda: self._rotate_active_region_shortcut(-4.0))
        QShortcut(QKeySequence("Ctrl+]"), self, activated=lambda: self._rotate_active_region_shortcut(4.0))
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

    def _switch_region_edit_mode_shortcut(self, transform_mode: bool) -> None:
        current_tab = self.mode_tabs.currentIndex()
        if current_tab == self.TAB_INTERPOLATION:
            if transform_mode and self._has_interpolation_region():
                self.interpolation_region_transform_radio.setChecked(True)
            elif not transform_mode:
                self.interpolation_region_draw_radio.setChecked(True)
        elif current_tab == self.TAB_CHAMBER:
            if transform_mode and self._selected_chamber_layer() is not None:
                self.chamber_transform_radio.setChecked(True)
            elif not transform_mode:
                self.chamber_draw_radio.setChecked(True)
        elif current_tab == self.TAB_CIRCLE:
            if transform_mode and self.frame_viewer.circle_geometry() is not None:
                self.circle_transform_radio.setChecked(True)
            elif not transform_mode:
                self.circle_draw_radio.setChecked(True)
        elif current_tab == self.TAB_OCCLUSION:
            self._switch_occlusion_mode_shortcut(transform_mode)

    def _switch_occlusion_mode_shortcut(self, transform_mode: bool) -> None:
        if transform_mode:
            self.mask_transform_radio.setChecked(True)
        else:
            self.mask_draw_radio.setChecked(True)

    def _handle_e_shortcut(self) -> None:
        self._scale_active_region_shortcut(0.96)

    def _handle_r_shortcut(self) -> None:
        self._scale_active_region_shortcut(1.04)

    def _handle_ctrl_e_shortcut(self) -> None:
        self._rotate_active_region_shortcut(-4.0)

    def _handle_ctrl_r_shortcut(self) -> None:
        self._rotate_active_region_shortcut(4.0)

    def _select_mask_by_slot(self, slot_index: int) -> None:
        if self.mode_tabs.currentIndex() != self.TAB_OCCLUSION:
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
        if self.mode_tabs.currentIndex() in {
            self.TAB_CHAMBER,
            self.TAB_CIRCLE,
            self.TAB_OCCLUSION,
        }:
            self.show_annotate_controls_help()

    def _ensure_save_folder(self) -> None:
        self.save_folder.mkdir(parents=True, exist_ok=True)
        self.save_folder_label.setText(str(self.save_folder))
        if hasattr(self, "pipeline_save_folder_label"):
            self.pipeline_save_folder_label.setText(str(self.save_folder))
        if hasattr(self, "trajectory_output_folder_label"):
            self.trajectory_output_folder_label.setText(str(self.save_folder))
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
        self.video_count_label.setText("0 videos")
        self.folder_label.setText("No folder selected")
        self.folder_label.setToolTip("")

        if output_path and output_path.exists():
            self.save_folder = output_path
            self._ensure_save_folder()
        else:
            self.save_folder = DEFAULT_SAVE_DIR
            self.save_folder_label.setText("-")
            if hasattr(self, "trajectory_output_folder_label"):
                self.trajectory_output_folder_label.setText("-")
            self._refresh_output_ui()
        self.statusBar().showMessage("Open a video folder to populate the list.")

    def _load_settings(self) -> dict:
        if not SETTINGS_PATH.exists():
            return {}
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _persistent_spinbox_widgets(self) -> dict[str, QSpinBox | QDoubleSpinBox]:
        widgets: dict[str, QSpinBox | QDoubleSpinBox] = {}
        for name in self.PERSISTENT_SPINBOX_NAMES:
            widget = getattr(self, name, None)
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widgets[name] = widget
        return widgets

    def _persistent_spinbox_settings_payload(self) -> dict:
        values: dict[str, int | float] = {}
        for name, widget in self._persistent_spinbox_widgets().items():
            value = widget.value()
            values[name] = float(value) if isinstance(widget, QDoubleSpinBox) else int(value)
        return {self.PERSISTENT_SPINBOX_SETTINGS_KEY: values}

    def _restore_persistent_spinbox_values(self) -> None:
        saved_values = self.settings.get(self.PERSISTENT_SPINBOX_SETTINGS_KEY, {})
        if not isinstance(saved_values, dict):
            return
        widgets = self._persistent_spinbox_widgets()
        if not widgets:
            return
        self._restoring_persistent_spinbox_values = True
        self._settings_save_suspended = True
        try:
            for name, widget in widgets.items():
                if name not in saved_values:
                    continue
                raw_value = saved_values.get(name)
                try:
                    if isinstance(widget, QDoubleSpinBox):
                        value = float(raw_value)
                    else:
                        value = int(round(float(raw_value)))
                except (TypeError, ValueError):
                    continue
                value = max(widget.minimum(), min(widget.maximum(), value))
                widget.setValue(value)
        finally:
            self._settings_save_suspended = False
            self._restoring_persistent_spinbox_values = False

    def _connect_persistent_spinbox_settings(self) -> None:
        for widget in self._persistent_spinbox_widgets().values():
            widget.valueChanged.connect(self._on_persistent_spinbox_value_changed)

    def _on_persistent_spinbox_value_changed(self, *_args) -> None:
        if getattr(self, "_restoring_persistent_spinbox_values", False):
            return
        self._save_settings()

    def _save_settings(self) -> None:
        if getattr(self, "_settings_save_suspended", False):
            return
        if hasattr(self, "mode_tabs"):
            self._remember_mode_tab_by_workflow(self.mode_tabs.currentIndex())
        data = {
            "last_tab_index": self.mode_tabs.currentIndex() if hasattr(self, "mode_tabs") else 0,
            "last_mode_tab_by_workflow": {
                str(category): int(tab_index)
                for category, tab_index in getattr(self, "last_mode_tab_by_workflow", {}).items()
                if self._valid_mode_tab_index(int(tab_index))
            },
            "workflow_category": (
                self.workflow_category_bar.currentIndex()
                if hasattr(self, "workflow_category_bar")
                else self.CATEGORY_PREPARE
            ),
            "file_sidebar_collapsed": self._file_sidebar_is_collapsed(),
            "file_sidebar_width": int(
                getattr(self, "_last_file_sidebar_width", self.FILE_SIDEBAR_DEFAULT_WIDTH)
            ),
            "occlusion_mask_margin": int(getattr(self, "default_mask_margin", 0)),
            "visible_mode_tabs": self._visible_mode_tab_keys(),
            "tab_visibility_schema_version": self.TAB_VISIBILITY_SCHEMA_VERSION,
            "occlusion_mask_margin_mode": str(getattr(self, "default_mask_margin_mode", "simple")),
            "occlusion_mask_brush": int(getattr(self, "default_mask_brush", 12)),
            "circle_detection_margin": int(getattr(self, "default_circle_margin", 0)),
            "input_dir": str(self.current_folder) if getattr(self, "current_folder", None) and str(self.current_folder) not in {"", "."} else "",
            "output_dir": str(self.save_folder) if getattr(self, "save_folder", None) else "",
            "csv_manual_dir": str(self.csv_manual_folder) if getattr(self, "csv_manual_folder", None) else "",
            "controls_splitter_sizes": (
                self.controls_splitter.sizes()
                if hasattr(self, "controls_splitter")
                else [610, 280]
            ),
        }
        data.update(self._pipeline_settings_payload())
        data.update(self._tracking_repair_settings_payload())
        data.update(self._interpolation_settings_payload())
        data.update(self._pin_settings_payload())
        data.update(self._persistent_spinbox_settings_payload())
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
        return self._apply_current_output_affixes(self.save_folder / f"{csv_path.stem}_normalized.csv")

    def _chamber_output_stem(self) -> str | None:
        if self.csv_path is not None:
            return self.csv_path.stem
        if self.video_state is not None:
            return self.video_state.path.stem
        return None

    def _chamber_csv_output_path(self) -> Path | None:
        stem = self._chamber_output_stem()
        return None if stem is None or self.csv_path is None else self._chamber_csv_output_path_for(self.csv_path)

    def _interpolation_output_path(self) -> Path | None:
        if self.csv_path is None:
            return None
        removal_mode = self._selected_interpolation_removal_mode()
        interpolate = self._interpolation_enabled()
        if removal_mode == "none" and not interpolate:
            return None
        return self._interpolation_output_path_for(self.csv_path, removal_mode, interpolate)

    def _interpolation_output_path_for(
        self,
        csv_path: Path,
        removal_mode: str = "none",
        interpolate: bool = True,
    ) -> Path:
        if removal_mode != "none" and interpolate:
            suffix = "auto_removed_interpolated"
        elif removal_mode != "none":
            suffix = "auto_removed"
        else:
            suffix = "interpolated"
        return self._apply_current_output_affixes(self.save_folder / f"{csv_path.stem}_{suffix}.csv")


    def _chamber_csv_output_path_for(self, csv_path: Path) -> Path:
        return self._apply_current_output_affixes(self.save_folder / f"{csv_path.stem}_chamber_mark.csv")

    def _chamber_mask_output_path(self) -> Path | None:
        stem = self._chamber_output_stem()
        return None if stem is None else self._apply_current_output_affixes(self.save_folder / f"{stem}_chamber_mask.png")

    def _chamber_overlay_output_path(self) -> Path | None:
        stem = self._chamber_output_stem()
        return None if stem is None else self._apply_current_output_affixes(self.save_folder / f"{stem}_chamber_mask_with_frame.png")

    def _chamber_manifest_output_path(self) -> Path | None:
        stem = self._chamber_output_stem()
        return None if stem is None else self._apply_current_output_affixes(self.save_folder / f"{stem}_chamber_mask.json")

    def _circle_output_stem(self) -> str | None:
        if self.csv_path is not None:
            return self.csv_path.stem
        if self.video_state is not None:
            return self.video_state.path.stem
        return None

    def _circle_mask_output_path(self) -> Path | None:
        stem = self._circle_output_stem()
        return (
            None
            if stem is None
            else self._apply_current_output_affixes(
                self.save_folder / f"{stem}_circle_mask.png"
            )
        )

    def _circle_manifest_output_path(self) -> Path | None:
        mask_path = self._circle_mask_output_path()
        return None if mask_path is None else mask_path.with_suffix(".json")

    def _circle_output_path(self) -> Path | None:
        return None if self.csv_path is None else self._circle_output_path_for(self.csv_path)

    def _circle_output_path_for(self, csv_path: Path) -> Path:
        return self._apply_current_output_affixes(self.save_folder / f"{csv_path.stem}_detection.csv")

    def _occlusion_output_path(self) -> Path | None:
        return None if self.csv_path is None else self._occlusion_output_path_for(self.csv_path)

    def _occlusion_output_path_for(self, csv_path: Path) -> Path:
        return self._apply_current_output_affixes(self.save_folder / f"{csv_path.stem}_occlusion.csv")

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

    def _batch_job_is_running(self) -> bool:
        active_thread = getattr(self, "_active_batch_thread", None)
        return active_thread is not None and active_thread.isRunning()

    def _single_save_job_is_running(self) -> bool:
        active_thread = getattr(self, "_active_single_save_thread", None)
        return active_thread is not None and active_thread.isRunning()

    def _save_job_is_running(self) -> bool:
        return self._single_save_job_is_running() or self._batch_job_is_running()

    def _focus_active_batch_progress(self) -> bool:
        if not self._batch_job_is_running():
            return False
        dialog = getattr(self, "_active_batch_dialog", None)
        if dialog is not None:
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
        return True

    def _focus_active_single_save_progress(self) -> bool:
        if not self._single_save_job_is_running():
            return False
        dialog = getattr(self, "_active_single_save_dialog", None)
        if dialog is not None:
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
        return True

    def _focus_active_save_progress(self) -> bool:
        return self._focus_active_single_save_progress() or self._focus_active_batch_progress()

    def _batch_action_controls(self) -> tuple[QPushButton, ...]:
        return tuple(
            control
            for control in self.findChildren(QPushButton)
            if bool(control.property("batchAction"))
        )

    def _save_action_controls(self) -> tuple[QPushButton, ...]:
        controls: list[QPushButton] = list(self._batch_action_controls())
        for name in (
            "save_current_button",
            "save_multiple_button",
            "pipeline_run_current_button",
            "pipeline_run_batch_button",
        ):
            control = getattr(self, name, None)
            if control is not None:
                controls.append(control)
        unique_controls: list[QPushButton] = []
        seen: set[int] = set()
        for control in controls:
            marker = id(control)
            if marker in seen:
                continue
            seen.add(marker)
            unique_controls.append(control)
        return tuple(unique_controls)

    def _set_save_controls_busy(self, busy: bool) -> None:
        if busy:
            for control in self._save_action_controls():
                control.setEnabled(False)
            return
        self._refresh_output_ui()
        if hasattr(self, "square_batch_heatmap_overlay_button"):
            self._refresh_square_ui()

    def _set_batch_controls_busy(self, busy: bool) -> None:
        self._set_save_controls_busy(busy)

    def _start_single_save_export(
        self,
        *,
        title: str,
        activity_text: str,
        export_item,
        on_completed=None,
    ) -> bool:
        if self._focus_active_save_progress():
            return False

        dialog = SingleSaveProgressDialog(
            title,
            activity_text,
            parent=self,
        )
        worker = SingleExportWorker(export_item=export_item)
        thread = QThread(self)
        worker.moveToThread(thread)

        self._active_single_save_dialog = dialog
        self._active_single_save_worker = worker
        self._active_single_save_thread = thread
        self._active_single_save_completion = on_completed

        thread.started.connect(worker.run)
        worker.completed.connect(self._on_single_save_completed)
        worker.crashed.connect(self._on_single_save_crashed)
        worker.completed.connect(thread.quit)
        worker.crashed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.crashed.connect(worker.deleteLater)
        thread.finished.connect(self._on_single_save_thread_finished)
        thread.finished.connect(thread.deleteLater)
        dialog.cancel_requested.connect(self._request_active_single_save_cancel)

        self._set_save_controls_busy(True)
        self.statusBar().showMessage(f"{activity_text} The main window remains available.")
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        thread.start()
        return True

    def _request_active_single_save_cancel(self) -> None:
        worker = getattr(self, "_active_single_save_worker", None)
        if worker is not None:
            worker.request_cancel()
            self.statusBar().showMessage(
                "Save cancellation requested; the current step will stop before writing if possible."
            )

    def _on_single_save_completed(self, result) -> None:
        dialog = getattr(self, "_active_single_save_dialog", None)
        if dialog is not None:
            dialog.mark_completed(result)
        callback = getattr(self, "_active_single_save_completion", None)
        self._active_single_save_completion = None
        if callback is not None:
            callback(result)
        elif getattr(result, "cancelled", False):
            self.statusBar().showMessage("Save cancelled.")
        else:
            self.statusBar().showMessage(f"CSV saved: {getattr(result, 'output_path', '')}")

    def _on_single_save_crashed(self, message: str) -> None:
        dialog = getattr(self, "_active_single_save_dialog", None)
        if dialog is not None:
            dialog.mark_crashed(message)
        self._active_single_save_completion = None
        self.statusBar().showMessage(f"Save worker stopped: {message}")

    def _on_single_save_thread_finished(self) -> None:
        self._active_single_save_thread = None
        self._active_single_save_worker = None
        self._set_save_controls_busy(False)

    def _start_batch_export(
        self,
        *,
        title: str,
        activity_text: str,
        selected_videos: list[Path],
        source_width: int,
        source_height: int,
        export_item,
        on_completed,
        require_bodyparts: bool = True,
    ) -> bool:
        if self._focus_active_save_progress():
            return False

        dialog = BatchProgressDialog(
            title,
            activity_text,
            len(selected_videos),
            parent=self,
        )
        worker = BatchExportWorker(
            selected_videos=selected_videos,
            source_width=source_width,
            source_height=source_height,
            csv_candidates_for=self._matching_csv_candidates,
            video_size_for=self._video_size,
            export_item=export_item,
            require_bodyparts=require_bodyparts,
        )
        thread = QThread(self)
        worker.moveToThread(thread)

        self._active_batch_dialog = dialog
        self._active_batch_worker = worker
        self._active_batch_thread = thread
        self._active_batch_completion = on_completed

        thread.started.connect(worker.run)
        worker.item_started.connect(dialog.update_item_started)
        worker.item_finished.connect(dialog.update_item_finished)
        worker.completed.connect(self._on_batch_export_completed)
        worker.crashed.connect(self._on_batch_export_crashed)
        worker.completed.connect(thread.quit)
        worker.crashed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.crashed.connect(worker.deleteLater)
        thread.finished.connect(self._on_batch_thread_finished)
        thread.finished.connect(thread.deleteLater)
        dialog.cancel_requested.connect(self._request_active_batch_cancel)
        dialog.cancel_now_requested.connect(self._request_active_batch_cancel_now)

        self._set_save_controls_busy(True)
        self.statusBar().showMessage(f"{activity_text} The main window remains available.")
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        thread.start()
        return True

    def _request_active_batch_cancel(self) -> None:
        worker = getattr(self, "_active_batch_worker", None)
        if worker is not None:
            worker.request_cancel()
            self.statusBar().showMessage(
                "Batch cancellation requested; the current file will finish safely."
            )

    def _request_active_batch_cancel_now(self) -> None:
        worker = getattr(self, "_active_batch_worker", None)
        if worker is not None:
            if hasattr(worker, "request_cancel_now"):
                worker.request_cancel_now()
            else:
                worker.request_cancel()
            self.statusBar().showMessage(
                "Batch cancellation requested; the current file will stop before writing if possible."
            )

    def _on_batch_export_completed(self, result: BatchRunResult) -> None:
        dialog = getattr(self, "_active_batch_dialog", None)
        if dialog is not None:
            dialog.mark_completed(result)
        callback = getattr(self, "_active_batch_completion", None)
        self._active_batch_completion = None
        if callback is not None:
            callback(result)

    def _on_batch_export_crashed(self, message: str) -> None:
        dialog = getattr(self, "_active_batch_dialog", None)
        if dialog is not None:
            dialog.mark_crashed(message)
        self._active_batch_completion = None
        self.statusBar().showMessage(f"Batch worker stopped: {message}")

    def _on_batch_thread_finished(self) -> None:
        self._active_batch_thread = None
        self._active_batch_worker = None
        self._set_save_controls_busy(False)

    def _select_videos_for_batch(
        self,
        dialog_title: str = "Select Videos For Batch CSV Save",
        info_text: str = "Select videos using click, Ctrl/Shift-click, or drag.",
        warning_text: str | None = None,
        start_button_text: str = "Start Batch Save",
    ) -> list[Path]:
        if self.video_list.count() == 0:
            QMessageBox.information(
                self,
                "Batch Save",
                "No videos in the list. Open a video folder first.",
            )
            return []

        if warning_text is None:
            warning_text = (
                "Warning:\n"
                "- Only videos with auto-detected CSV candidates are converted.\n"
                "- The top auto-detected CSV candidate is used for each video.\n"
                "- The same condition is applied to all selected videos."
            )

        selected_source_paths = {
            str(item.data(Qt.ItemDataRole.UserRole))
            for item in self.video_list.selectedItems()
            if item.data(Qt.ItemDataRole.UserRole)
        }
        current_path = self.video_state.path if self.video_state is not None else None
        if not selected_source_paths and current_path is not None:
            selected_source_paths.add(str(current_path))

        choices: list[BatchVideoChoice] = []
        for index in range(self.video_list.count()):
            source_item = self.video_list.item(index)
            path_data = source_item.data(Qt.ItemDataRole.UserRole)
            if not path_data:
                continue
            video_path = Path(str(path_data))
            csv_candidates = self._matching_csv_candidates(video_path)
            choices.append(
                BatchVideoChoice(
                    path=video_path,
                    label=source_item.text(),
                    csv_path=csv_candidates[0] if csv_candidates else None,
                    selected=str(video_path) in selected_source_paths,
                )
            )

        dialog = BatchVideoSelectionDialog(
            title=dialog_title,
            info_text=info_text,
            warning_text=warning_text,
            start_button_text=start_button_text,
            choices=choices,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return []
        return dialog.selected_paths()

    def save_multiple_mode_outputs(self) -> None:
        if self._focus_active_save_progress():
            return
        mode_index = self.mode_tabs.currentIndex()
        supported_modes = {
            self.TAB_TRACKING_REPAIR,
            self.TAB_INTERPOLATION,
            self.TAB_SQUARE,
            self.TAB_CHAMBER,
            self.TAB_CIRCLE,
            self.TAB_OCCLUSION,
        }
        if mode_index not in supported_modes:
            QMessageBox.information(self, "Batch Save", "Batch CSV save is not available in this Inspect tab.")
            return
        if self.video_state is None:
            QMessageBox.warning(self, "Batch Save", "Load a reference video first.")
            return

        selected_videos = self._select_videos_for_batch()
        if not selected_videos:
            return

        source_width = max(1, int(self.video_state.width))
        source_height = max(1, int(self.video_state.height))
        source_interpolation_mask = None if self.interpolation_mask is None else self.interpolation_mask.copy().astype(np.uint8)
        source_interpolation_removal_mode = self._selected_interpolation_removal_mode()
        source_interpolation_anchor = self._selected_interpolation_anchor()
        source_interpolation_enabled = self._interpolation_enabled()
        source_interpolation_extrapolate = self._interpolation_extrapolation_enabled()

        source_square_points = [tuple(point) for point in self.frame_viewer.square_points]
        source_chamber_boundary_mode = getattr(self, "chamber_boundary_mode", "custom")
        source_chamber_mask = None if self.chamber_mask is None else self.chamber_mask.copy().astype(np.uint8)
        source_rooms = [
            RoomRecord(
                name=room.name,
                color=QColor(room.color),
                mask=room.mask.copy().astype(np.uint8),
                geometry=clone_mask_geometry(room.geometry),
            )
            for room in self._effective_room_records()
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
                geometry=clone_mask_geometry(record.geometry),
            )
            for record in self.mask_records.values()
        ]
        tracking_config = self._tracking_repair_config()
        save_folder = Path(self.save_folder)
        output_prefix, output_suffix = self._current_output_affixes()

        if mode_index == self.TAB_TRACKING_REPAIR:
            if not tracking_config.remove_duplicates and not tracking_config.remove_length_outliers:
                QMessageBox.warning(self, "Batch Save", "Enable duplicate or robust Z-score removal first.")
                return
        if mode_index == self.TAB_INTERPOLATION:
            if source_interpolation_removal_mode != "none" and (
                source_interpolation_mask is None or not np.any(source_interpolation_mask)
            ):
                QMessageBox.warning(self, "Batch Save", "Automatic removal needs a drawn region.")
                return
            if source_interpolation_removal_mode != "none" and source_interpolation_anchor is None:
                QMessageBox.warning(self, "Batch Save", "Automatic removal needs an anchor node.")
                return
            if source_interpolation_removal_mode == "none" and not source_interpolation_enabled:
                QMessageBox.warning(self, "Batch Save", "Enable automatic removal or interpolation first.")
                return
        if mode_index == self.TAB_SQUARE and len(source_square_points) != 4:
            QMessageBox.warning(self, "Batch Save", "Square mode needs four points.")
            return
        if mode_index == self.TAB_CHAMBER:
            if source_chamber_mask is None or not np.any(source_chamber_mask):
                QMessageBox.warning(self, "Batch Save", "Chamber mode needs a chamber mask.")
                return
            if not source_rooms:
                QMessageBox.warning(self, "Batch Save", "Chamber mode needs at least one room.")
                return
        if mode_index == self.TAB_CIRCLE and source_circle_geometry is None:
            QMessageBox.warning(self, "Batch Save", "Circle mode needs a circle.")
            return
        if mode_index == self.TAB_OCCLUSION:
            if not source_masks:
                QMessageBox.warning(self, "Batch Save", "Occlusion mode needs at least one mask.")
                return
            if any(record.margin_mode == "geometric" for record in source_masks) and len(source_occ_margin_points) != 4:
                QMessageBox.warning(self, "Batch Save", "Geometric margin mode needs four occlusion geometric points.")
                return

        def _batch_csv_output_path(csv_path: Path, analysis_suffix: str) -> Path:
            base_path = save_folder / f"{csv_path.stem}_{analysis_suffix}.csv"
            return self._apply_output_affixes(base_path, output_prefix, output_suffix)

        def _export_item(item: BatchItem, should_cancel=None) -> Path:
            raise_if_cancelled(should_cancel)
            if mode_index == self.TAB_TRACKING_REPAIR:
                repair_result = self._run_tracking_repair_for(
                    item.source_df,
                    item.bodyparts,
                    item.width,
                    item.height,
                    config=tracking_config,
                )
                result_df = repair_result.dataframe
                output_path = _batch_csv_output_path(item.csv_path, "tracking_postprocessed")
            elif mode_index == self.TAB_INTERPOLATION:
                interpolation_mask = None
                if source_interpolation_removal_mode != "none":
                    interpolation_mask = cv2.resize(
                        source_interpolation_mask.astype(np.uint8),
                        (item.width, item.height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                result_df = build_interpolation_pipeline_dataframe(
                    item.source_df,
                    item.bodyparts,
                    interpolation_mask,
                    item.width,
                    item.height,
                    source_interpolation_removal_mode,
                    source_interpolation_anchor,
                    source_interpolation_enabled,
                    source_interpolation_extrapolate,
                )
                if source_interpolation_removal_mode != "none" and source_interpolation_enabled:
                    analysis_suffix = "auto_removed_interpolated"
                elif source_interpolation_removal_mode != "none":
                    analysis_suffix = "auto_removed"
                else:
                    analysis_suffix = "interpolated"
                output_path = _batch_csv_output_path(item.csv_path, analysis_suffix)
            elif mode_index == self.TAB_SQUARE:
                quad_points = [(x * item.scale_x, y * item.scale_y) for x, y in source_square_points]
                result_df = build_normalized_dataframe(
                    item.source_df,
                    item.bodyparts,
                    quad_points,
                    item.width,
                    item.height,
                )
                output_path = _batch_csv_output_path(item.csv_path, "normalized")
            elif mode_index == self.TAB_CHAMBER:
                if source_chamber_boundary_mode == "full_frame":
                    chamber_mask = np.ones((item.height, item.width), dtype=np.uint8)
                else:
                    chamber_mask = cv2.resize(
                        source_chamber_mask.astype(np.uint8),
                        (item.width, item.height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                scaled_rooms: list[RoomRecord] = []
                for room in source_rooms:
                    resized_mask = cv2.resize(
                        room.mask.astype(np.uint8),
                        (item.width, item.height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    room_mask = np.logical_and(resized_mask > 0, chamber_mask > 0).astype(np.uint8)
                    scaled_rooms.append(
                        RoomRecord(
                            name=room.name,
                            color=room.color,
                            mask=room_mask,
                            geometry=scale_mask_geometry(room.geometry, item.scale_x, item.scale_y),
                        )
                    )
                result_df = build_chamber_mark_dataframe(
                    item.source_df,
                    item.bodyparts,
                    scaled_rooms,
                    item.width,
                    item.height,
                )
                output_path = _batch_csv_output_path(item.csv_path, "chamber_mark")
            elif mode_index == self.TAB_CIRCLE:
                center, _, adjusted_radius = source_circle_geometry
                scaled_center = (center[0] * item.scale_x, center[1] * item.scale_y)
                scaled_radius = max(1.0, adjusted_radius * ((item.scale_x + item.scale_y) / 2.0))
                result_df = build_circle_detection_dataframe(
                    item.source_df,
                    item.bodyparts,
                    scaled_center,
                    scaled_radius,
                    item.width,
                    item.height,
                )
                output_path = _batch_csv_output_path(item.csv_path, "detection")
            else:
                scaled_quad_points = [
                    (x * item.scale_x, y * item.scale_y)
                    for x, y in source_occ_margin_points
                ]
                scaled_masks: list[MaskRecord] = []
                for record in source_masks:
                    resized_mask = cv2.resize(
                        record.mask.astype(np.uint8),
                        (item.width, item.height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    scaled_masks.append(
                        MaskRecord(
                            name=record.name,
                            color=record.color,
                            mask=(resized_mask > 0).astype(np.uint8),
                            margin=record.margin,
                            margin_mode=record.margin_mode,
                            geometry=scale_mask_geometry(record.geometry, item.scale_x, item.scale_y),
                        )
                    )
                result_df = build_occlusion_dataframe(
                    item.source_df,
                    item.bodyparts,
                    scaled_masks,
                    item.width,
                    item.height,
                    scaled_quad_points,
                )
                output_path = _batch_csv_output_path(item.csv_path, "occlusion")

            raise_if_cancelled(should_cancel)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            raise_if_cancelled(should_cancel)
            result_df.to_csv(output_path, index=False)
            return output_path

        def _on_completed(result: BatchRunResult) -> None:
            state = "cancelled" if result.cancelled else "finished"
            self.statusBar().showMessage(
                f"Batch save {state}: saved={result.saved_count}, "
                f"skipped={len(result.skipped_auto_missing)}, failed={len(result.failed)}"
            )

        self._start_batch_export(
            title="Batch CSV Save Progress",
            activity_text="Saving CSV files in the background...",
            selected_videos=selected_videos,
            source_width=source_width,
            source_height=source_height,
            export_item=_export_item,
            on_completed=_on_completed,
        )

    def _show_normalized_preview(self, preview_df: pd.DataFrame, normalized: bool) -> None:
        frame_width = self.video_state.width if self.video_state is not None else None
        frame_height = self.video_state.height if self.video_state is not None else None
        video_name = self.video_state.path.stem if self.video_state is not None else None
        normalized_display_size = None
        if normalized and len(self.frame_viewer.square_points) == 4:
            try:
                _matrix, _inverse, normalized_display_size = build_rectified_geometry(self.frame_viewer.square_points)
            except Exception:
                normalized_display_size = None
        preview_dialog = TrajectoryPreviewDialog(
            preview_df,
            self.bodyparts,
            normalized,
            frame_width,
            frame_height,
            normalized_display_size,
            video_name,
            self,
        )
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
        self._set_output_workspace_mode(current_index)
        self.save_current_button.setText("Save CSV")
        if hasattr(self, "save_multiple_button"):
            self.save_multiple_button.setText("Save Multiple CSVs")

        if current_index == self.TAB_TRACKING_REPAIR:
            output = self._tracking_repair_output_path()
            config = self._tracking_repair_config()
            repair_ready = (
                output is not None
                and self.csv_df is not None
                and self.video_state is not None
                and bool(self.bodyparts)
                and (config.remove_duplicates or config.remove_length_outliers)
            )
            self.save_current_button.setEnabled(repair_ready)
        elif current_index == self.TAB_INTERPOLATION:
            output = self._interpolation_output_path()
            removal_mode = self._selected_interpolation_removal_mode()
            anchor = self._selected_interpolation_anchor()
            interpolate = self._interpolation_enabled()
            removal_ready = removal_mode == "none" or (
                self.interpolation_mask is not None
                and bool(np.any(self.interpolation_mask))
                and anchor is not None
            )
            interpolation_ready = (
                removal_ready
                and (removal_mode != "none" or interpolate)
                and self.csv_df is not None
                and bool(self.bodyparts)
            )
            self.save_current_button.setEnabled(output is not None and interpolation_ready)
        elif current_index == self.TAB_SQUARE:
            output = self._normalized_output_path()
            self.save_current_button.setEnabled(
                output is not None and len(self.frame_viewer.square_points) == 4
            )
        elif current_index == self.TAB_CHAMBER:
            output = self._chamber_csv_output_path()
            chamber_ready = (
                self.chamber_mask is not None
                and bool(np.any(self.chamber_mask))
                and bool(self.room_records)
            )
            self.save_current_button.setEnabled(
                output is not None
                and chamber_ready
                and self.csv_df is not None
                and bool(self.bodyparts)
            )
        elif current_index == self.TAB_CIRCLE:
            output = self._circle_output_path()
            self.save_current_button.setEnabled(
                output is not None
                and self.frame_viewer.circle_geometry() is not None
                and bool(self.bodyparts)
            )
        elif current_index in {self.TAB_TRAJECTORY, self.TAB_PIN, self.TAB_PIPELINE}:
            output = None
            self.save_current_button.setEnabled(False)
        else:
            output = self._occlusion_output_path()
            geometric_ready = all(
                record.margin_mode != "geometric"
                or len(self.frame_viewer.occ_margin_points) == 4
                for record in self.mask_records.values()
            )
            self.save_current_button.setEnabled(
                output is not None
                and bool(self.mask_records)
                and self.csv_df is not None
                and geometric_ready
            )

        batch_enabled = self.save_current_button.isEnabled()
        self.current_output_label.setText(str(output) if output is not None else "-")
        self.current_output_label.setToolTip(str(output) if output is not None else "")
        if hasattr(self, "save_multiple_button"):
            self.save_multiple_button.setEnabled(batch_enabled)
            if current_index in {self.TAB_TRAJECTORY, self.TAB_PIN}:
                self.save_multiple_button.setToolTip(
                    "CSV saving is not available in Inspect tabs."
                )
            elif current_index == self.TAB_PIPELINE:
                self.save_multiple_button.setToolTip(
                    "Use the Pipeline Output controls below."
                )
            else:
                self.save_multiple_button.setToolTip(
                    "Apply current mode settings to multiple selected videos at once."
                )
        if hasattr(self, "export_masks_button"):
            self.export_masks_button.setEnabled(bool(self.mask_records))
        self._refresh_pipeline_ui()
        if self._save_job_is_running():
            self._set_save_controls_busy(True)

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
        self.folder_label.setToolTip(str(folder))
        self.video_list.clear()
        videos = discover_videos(folder)
        self.video_count_label.setText(f"{len(videos)} video{'s' if len(videos) != 1 else ''}")
        if not videos:
            self.statusBar().showMessage("No videos found in the selected folder.")
            return
        for video_path in videos:
            item = QListWidgetItem(str(video_path.relative_to(folder)))
            item.setData(Qt.ItemDataRole.UserRole, str(video_path))
            item.setToolTip(str(video_path))
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
        cleared_mask_layers: tuple[str, ...] = ()
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
            cleared_mask_layers = self._rescale_annotations(
                previous_state.width,
                previous_state.height,
                self.video_state.width,
                self.video_state.height,
            )
        self.frame_viewer.reset_view()
        self.frame_viewer.clear_chamber_rect_points()
        self.frame_viewer.clear_chamber_circle()
        self.frame_viewer.clear_occ_rect_points()
        self.frame_viewer.clear_occ_circle()
        self._load_matching_csv(video_path)
        self._load_frame(1)
        self._update_frame_label(1)
        if previous_state is not None and previous_state.path != video_path:
            self._clear_mask_undo_stacks()
        if cleared_mask_layers:
            self._show_video_mask_reset_notice(
                previous_state.width,
                previous_state.height,
                self.video_state.width,
                self.video_state.height,
                cleared_mask_layers,
            )
        else:
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
            self._prepare_node_overlay_data()
            self._update_csv_path_label(None)
            self._refresh_node_overlay()
            self._refresh_tracking_repair_ui()
            self._refresh_square_ui()
            self._refresh_circle_ui()
            self._refresh_interpolation_ui()

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
        self._prepare_node_overlay_data()
        self._update_csv_path_label(csv_path)
        if auto_matched:
            self._set_csv_auto_candidates(getattr(self, "csv_auto_candidates", []), csv_path)
        else:
            current_candidates = getattr(self, "csv_auto_candidates", [])
            selected = csv_path if csv_path in current_candidates else None
            self._set_csv_auto_candidates(current_candidates, selected, allow_default=selected is not None)
        self._refresh_node_overlay()
        self._refresh_tracking_repair_ui()
        self._refresh_square_ui()
        self._refresh_circle_ui()
        self._refresh_interpolation_ui()

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
        self._refresh_chamber_viewer(refresh=False)
        self.frame_viewer.set_interpolation_mask(self.interpolation_mask)
        self.frame_viewer.set_pin_records(self.pins)
        self.frame_viewer.set_mask_records(self.mask_records, self.selected_mask_name)
        self._refresh_node_overlay()
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

    def _refresh_node_overlay(self, _checked: bool | None = None) -> None:
        available = self.video_state is not None and self.csv_df is not None and bool(self.bodyparts)
        self.show_nodes_checkbox.setEnabled(available)
        self.node_color_combo.setEnabled(available)
        if not available or not self.show_nodes_checkbox.isChecked():
            self.frame_viewer.set_node_overlay_points([])
            return
        points = self._node_overlay_points_for_frame(self.current_frame_number)
        self.frame_viewer.set_node_overlay_points(points)

    def _on_node_color_mode_changed(self, _index: int) -> None:
        mode = str(self.node_color_combo.currentData())
        self.frame_viewer.set_node_overlay_color_mode(mode)

    def _node_overlay_points_for_frame(self, frame_number: int) -> list[NodeOverlayPoint]:
        if self.csv_df is None or self.video_state is None:
            return []
        frame_rows = self._csv_rows_for_frame(frame_number)
        if frame_rows.empty:
            return []

        review_active = (
            self.mode_tabs.currentIndex() == self.TAB_TRACKING_REPAIR
            and getattr(self, "_tracking_repair_display_frame_number", None) == frame_number
            and getattr(self, "_tracking_repair_display_signature", None)
            == self._tracking_repair_review_signature()
        )
        review_rows = getattr(self, "_tracking_repair_display_row_indices", frozenset())
        deleted_rows = getattr(self, "_tracking_repair_deleted_row_indices", frozenset())
        if review_active and review_rows:
            frame_rows = frame_rows.loc[frame_rows.index.isin(review_rows)]
            if frame_rows.empty:
                return []

        instance_column = TrajectoryPreviewDialog._find_matching_column(
            frame_rows,
            NODE_INSTANCE_COLUMN_CANDIDATES,
        )
        show_instance = len(frame_rows) > 1
        instance_identities = self._node_instance_identities(frame_rows, instance_column)
        points: list[NodeOverlayPoint] = []
        for (row_index, row), (instance_label, instance_key) in zip(frame_rows.iterrows(), instance_identities):
            review_role = ""
            if review_active:
                review_role = "delete" if row_index in deleted_rows else "compare"
            for bodypart in self.bodyparts:
                try:
                    x = float(pd.to_numeric(row[f"{bodypart}.x"], errors="coerce"))
                    y = float(pd.to_numeric(row[f"{bodypart}.y"], errors="coerce"))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(x) or not math.isfinite(y):
                    continue
                x_scale, y_scale = self._node_coordinate_scales[bodypart]
                label = f"{bodypart} ({instance_label})" if show_instance else bodypart
                points.append(
                    NodeOverlayPoint(
                        label=label,
                        x=x * x_scale,
                        y=y * y_scale,
                        instance_key=instance_key,
                        bodypart=bodypart,
                        review_role=review_role,
                    )
                )
        return points

    @staticmethod
    def _node_instance_identities(
        frame_rows: pd.DataFrame,
        instance_column: str | None,
    ) -> list[tuple[str, str]]:
        raw_labels = [
            str(index)
            if instance_column is None or pd.isna(row[instance_column])
            else str(row[instance_column])
            for index, (_, row) in enumerate(frame_rows.iterrows(), start=1)
        ]
        label_counts: dict[str, int] = {}
        for label in raw_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        occurrences: dict[str, int] = {}
        identities: list[tuple[str, str]] = []
        for label in raw_labels:
            occurrences[label] = occurrences.get(label, 0) + 1
            occurrence = occurrences[label]
            display_label = label if label_counts[label] == 1 else f"{label} {occurrence}"
            key_label = label if label_counts[label] == 1 else f"{label}#{occurrence}"
            identities.append((display_label, f"{instance_column or 'row'}:{key_label}"))
        return identities

    def _prepare_node_overlay_data(self) -> None:
        if hasattr(self, "frame_viewer"):
            self.frame_viewer.reset_node_overlay_color_assignments()
        self._node_frame_column = None
        self._node_frame_values = None
        self._node_frame_zero_based = False
        self._node_coordinate_scales = {}
        if self.csv_df is None:
            return

        self._node_frame_column = TrajectoryPreviewDialog._find_matching_column(
            self.csv_df,
            TrajectoryPreviewDialog.FRAME_COLUMN_CANDIDATES,
        )
        if self._node_frame_column is not None:
            self._node_frame_values = pd.to_numeric(self.csv_df[self._node_frame_column], errors="coerce")
            self._node_frame_zero_based = bool((self._node_frame_values == 0).any())

        if self.video_state is not None:
            self._node_coordinate_scales = {
                bodypart: (
                    infer_pixel_scale(self.csv_df[f"{bodypart}.x"], self.video_state.width),
                    infer_pixel_scale(self.csv_df[f"{bodypart}.y"], self.video_state.height),
                )
                for bodypart in self.bodyparts
            }

    def _csv_rows_for_frame(self, frame_number: int) -> pd.DataFrame:
        if self.csv_df is None:
            return pd.DataFrame()
        if self._node_frame_column is None or self._node_frame_values is None:
            row_index = max(0, int(frame_number) - 1)
            return self.csv_df.iloc[row_index : row_index + 1]

        target_frame = int(frame_number) - 1 if self._node_frame_zero_based else int(frame_number)
        return self.csv_df.loc[self._node_frame_values == target_frame]

    def step_frame(self, delta: int) -> None:
        if self.video_state is None:
            return
        self.frame_slider.setValue(max(1, min(self.current_frame_number + delta, self.video_state.frame_count)))

    def _mask_layers_with_data(self) -> tuple[str, ...]:
        layers: list[str] = []
        if self._has_interpolation_region():
            layers.append("interpolation region")
        if (
            (self.chamber_mask is not None and bool(np.any(self.chamber_mask)))
            or any(bool(np.any(room.mask)) for room in self.room_records.values())
        ):
            layers.append("chamber/room masks")
        if self.mask_records:
            layers.append("occlusion masks")
        if self.frame_viewer.occ_margin_points:
            layers.append("geometric margin points")
        return tuple(layers)

    def _show_video_mask_reset_notice(
        self,
        old_width: int,
        old_height: int,
        new_width: int,
        new_height: int,
        cleared_layers: tuple[str, ...],
    ) -> None:
        layer_text = ", ".join(cleared_layers)
        self.statusBar().showMessage(
            f"Video size changed ({old_width}x{old_height} -> {new_width}x{new_height}); "
            f"incompatible data was cleared: {layer_text}.",
            6000,
        )

    def _rescale_annotations(
        self,
        old_width: int,
        old_height: int,
        new_width: int,
        new_height: int,
    ) -> tuple[str, ...]:
        if old_width <= 0 or old_height <= 0 or new_width <= 0 or new_height <= 0:
            return ()
        chamber_was_full_frame = getattr(self, "chamber_boundary_mode", "unset") == "full_frame"
        had_chamber_or_rooms = (
            (self.chamber_mask is not None and bool(np.any(self.chamber_mask)))
            or any(bool(np.any(room.mask)) for room in self.room_records.values())
        )
        had_room_masks = any(bool(np.any(room.mask)) for room in self.room_records.values())
        cleared_mask_layers = list(self._mask_layers_with_data())
        scale_x = new_width / old_width
        scale_y = new_height / old_height

        self.frame_viewer.square_points = [(x * scale_x, y * scale_y) for x, y in self.frame_viewer.square_points]
        if self.frame_viewer.trajectory_region_start is not None:
            x, y = self.frame_viewer.trajectory_region_start
            self.frame_viewer.trajectory_region_start = (x * scale_x, y * scale_y)
        if self.frame_viewer.trajectory_region_end is not None:
            x, y = self.frame_viewer.trajectory_region_end
            self.frame_viewer.trajectory_region_end = (x * scale_x, y * scale_y)
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
        self.reset_interpolation_region()
        if chamber_was_full_frame:
            self.reset_chamber_rooms_for_full_frame_video()
            if "chamber/room masks" in cleared_mask_layers:
                cleared_mask_layers.remove("chamber/room masks")
            if had_room_masks:
                cleared_mask_layers.append("room masks")
            elif had_chamber_or_rooms:
                self.statusBar().showMessage("Full-frame chamber resized to the new video.", 4000)
        else:
            self.reset_chamber()
        self.reset_masks()
        self.frame_viewer.clear_occ_margin_points()
        self._refresh_chamber_viewer(refresh=True)
        self.frame_viewer.set_pin_records(self.pins)
        self.frame_viewer.set_mask_records(self.mask_records, self.selected_mask_name, refresh=True)
        self._refresh_tracking_repair_ui()
        self._refresh_square_ui()
        self._refresh_chamber_ui()
        self._refresh_circle_ui()
        self._refresh_pin_ui()
        self._refresh_mask_ui()
        return tuple(cleared_mask_layers)

    def _refresh_view_ui(self) -> None:
        self.zoom_label.setText(f"{int(round(self.frame_viewer.zoom_factor * 100))}%")

    def _on_mode_changed(self, index: int) -> None:
        self._remember_mode_tab_by_workflow(index)
        self._sync_current_output_default_suffix(index)
        if index == self.TAB_TRACKING_REPAIR:
            self.frame_viewer.set_mode("inspect")
            self.frame_viewer.set_margin_value(0.0)
            self._refresh_tracking_repair_ui()
        elif index == self.TAB_INTERPOLATION:
            self._sync_interpolation_mode()
        elif index == self.TAB_SQUARE:
            self.frame_viewer.set_mode("square")
            self.frame_viewer.set_margin_value(0.0)
        elif index == self.TAB_CHAMBER:
            self._sync_chamber_mode()
        elif index == self.TAB_CIRCLE:
            self._sync_circle_mode()
        elif index == self.TAB_TRAJECTORY:
            self._sync_trajectory_region_mode()
            self.frame_viewer.set_margin_value(0.0)
            if self.square_preview_button.isEnabled():
                self.square_preview_button.setFocus()
        elif index == self.TAB_PIN:
            self.frame_viewer.set_mode("pin")
            self.frame_viewer.set_margin_value(0.0)
        elif index == self.TAB_PIPELINE:
            self.frame_viewer.set_mode("inspect")
            self.frame_viewer.set_margin_value(0.0)
        else:
            self._sync_occlusion_mode()
        self._save_settings()
        self._refresh_output_ui()


    @staticmethod
    def _copy_mask_payload(mask: np.ndarray | None) -> np.ndarray | None:
        return None if mask is None else mask.copy().astype(np.uint8)

    def _clear_mask_undo_stacks(self) -> None:
        for stack in self._mask_undo_stacks.values():
            stack.clear()
        self._interpolation_free_undo_open = False
        self._occlusion_free_undo_open = False
        self._occlusion_transform_undo_open = False

    def _active_mask_undo_key(self) -> str | None:
        return {
            self.TAB_INTERPOLATION: "interpolation",
            self.TAB_CHAMBER: "chamber",
            self.TAB_CIRCLE: "circle",
            self.TAB_OCCLUSION: "occlusion",
        }.get(self.mode_tabs.currentIndex())

    def _push_mask_undo_snapshot(self, tab_key: str, label: str, payload: object) -> None:
        if self._mask_undo_restoring:
            return
        stack = self._mask_undo_stacks.get(tab_key)
        if stack is None:
            return
        stack.append(MaskUndoSnapshot(tab_key=tab_key, label=label, payload=payload))
        overflow = len(stack) - self._mask_undo_limit
        if overflow > 0:
            del stack[:overflow]

    def _push_interpolation_undo(self, label: str = "edit interpolation region") -> None:
        self._push_mask_undo_snapshot(
            "interpolation",
            label,
            self._copy_mask_payload(self.interpolation_mask),
        )

    def _push_chamber_undo(self, label: str = "edit chamber mask") -> None:
        self._push_mask_undo_snapshot(
            "chamber",
            label,
            {
                "chamber_mask": self._copy_mask_payload(self.chamber_mask),
                "chamber_geometry": clone_mask_geometry(getattr(self, "chamber_geometry", None)),
                "boundary_mode": getattr(self, "chamber_boundary_mode", "unset"),
                "selected_room_name": self.selected_room_name,
                "rooms": {
                    name: {
                        "mask": room.mask.copy().astype(np.uint8),
                        "geometry": clone_mask_geometry(room.geometry),
                    }
                    for name, room in self.room_records.items()
                },
            },
        )

    def _circle_undo_payload(self) -> dict[str, object]:
        return {
            "circle_start": None if self.frame_viewer.circle_start is None else tuple(self.frame_viewer.circle_start),
            "circle_end": None if self.frame_viewer.circle_end is None else tuple(self.frame_viewer.circle_end),
            "margin": int(self.circle_margin_slider.value()),
            "source": getattr(self.frame_viewer, "circle_geometry_source", "exact"),
        }

    def _push_circle_undo(self, label: str = "edit circle mask") -> None:
        self._push_mask_undo_snapshot("circle", label, self._circle_undo_payload())

    def _push_occlusion_undo(self, label: str = "edit occlusion mask") -> None:
        self._push_mask_undo_snapshot(
            "occlusion",
            label,
            {
                "selected_mask_name": self.selected_mask_name,
                "masks": {
                    name: {
                        "mask": record.mask.copy().astype(np.uint8),
                        "geometry": clone_mask_geometry(record.geometry),
                    }
                    for name, record in self.mask_records.items()
                },
            },
        )

    def _restore_interpolation_undo(self, payload: object) -> None:
        mask = None if payload is None else np.asarray(payload).copy().astype(np.uint8)
        self._set_interpolation_mask(mask, refresh=True, reset_transform_source=True)

    def _restore_chamber_undo(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        chamber_mask = payload.get("chamber_mask")
        self.chamber_mask = (
            None
            if chamber_mask is None
            else (np.asarray(chamber_mask) > 0).astype(np.uint8)
        )
        self.chamber_geometry = clone_mask_geometry(payload.get("chamber_geometry"))
        self._set_chamber_boundary_mode(str(payload.get("boundary_mode", "unset")))
        rooms = payload.get("rooms", {})
        if isinstance(rooms, dict):
            for name, entry in rooms.items():
                room = self.room_records.get(str(name))
                if room is not None:
                    if isinstance(entry, dict):
                        mask = entry.get("mask")
                        room.geometry = clone_mask_geometry(entry.get("geometry"))
                    else:
                        mask = entry
                        room.geometry = None
                    room.mask = (np.asarray(mask) > 0).astype(np.uint8)
        selected_name = payload.get("selected_room_name")
        if isinstance(selected_name, str) and selected_name in self.room_records:
            self.selected_room_name = selected_name
        elif self.selected_room_name not in self.room_records:
            self.selected_room_name = sorted(self.room_records)[0] if self.room_records else None
        self._invalidate_chamber_transform_source()
        self._rebuild_room_list()
        self._refresh_chamber_viewer(refresh=True)
        self._refresh_chamber_ui()
        if self.mode_tabs.currentIndex() == self.TAB_CHAMBER:
            self._sync_chamber_mode()

    def _restore_circle_undo(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        margin = int(payload.get("margin", 0))
        margin = max(
            self.circle_margin_slider.minimum(),
            min(self.circle_margin_slider.maximum(), margin),
        )
        self._set_circle_margin_value(margin)
        start = payload.get("circle_start")
        end = payload.get("circle_end")
        self.frame_viewer.circle_start = None if start is None else tuple(start)
        self.frame_viewer.circle_end = None if end is None else tuple(end)
        source = str(payload.get("source", "exact"))
        self.frame_viewer.circle_geometry_source = source if source in {"exact", "inferred"} else "exact"
        self.frame_viewer._circle_current = None
        self.frame_viewer._circle_dragging = False
        self.frame_viewer._circle_move_dragging = False
        self.frame_viewer._circle_move_last_point = None
        self.frame_viewer.update()
        self.frame_viewer.circle_changed.emit()
        self._refresh_circle_ui()
        self._save_settings()

    def _restore_occlusion_undo(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        masks = payload.get("masks", {})
        if isinstance(masks, dict):
            for name, entry in masks.items():
                record = self.mask_records.get(str(name))
                if record is not None:
                    if isinstance(entry, dict):
                        mask = entry.get("mask")
                        record.geometry = clone_mask_geometry(entry.get("geometry"))
                    else:
                        mask = entry
                        record.geometry = None
                    record.mask = (np.asarray(mask) > 0).astype(np.uint8)
        selected_name = payload.get("selected_mask_name")
        if isinstance(selected_name, str) and selected_name in self.mask_records:
            self.selected_mask_name = selected_name
        elif self.selected_mask_name not in self.mask_records:
            self.selected_mask_name = sorted(self.mask_records)[0] if self.mask_records else None
        self._invalidate_mask_transform_source()
        self._rebuild_mask_list()
        self.frame_viewer.set_mask_records(self.mask_records, self.selected_mask_name, refresh=True)
        self._refresh_mask_ui()

    def undo_active_mask_edit(self) -> None:
        tab_key = self._active_mask_undo_key()
        if tab_key is None:
            self.statusBar().showMessage("Undo is available in Interpolation, Chamber, Circle, and Occlusion tabs.", 3000)
            return
        stack = self._mask_undo_stacks.get(tab_key, [])
        if not stack:
            self.statusBar().showMessage("Nothing to undo in this tab.", 2500)
            return
        snapshot = stack.pop()
        self._mask_undo_restoring = True
        try:
            if snapshot.tab_key == "interpolation":
                self._restore_interpolation_undo(snapshot.payload)
            elif snapshot.tab_key == "chamber":
                self._restore_chamber_undo(snapshot.payload)
            elif snapshot.tab_key == "circle":
                self._restore_circle_undo(snapshot.payload)
            elif snapshot.tab_key == "occlusion":
                self._restore_occlusion_undo(snapshot.payload)
        finally:
            self._mask_undo_restoring = False
            self._interpolation_free_undo_open = False
            self._occlusion_free_undo_open = False
            self._occlusion_transform_undo_open = False
        self.statusBar().showMessage(f"Undid {snapshot.label}.", 2500)

    def clear_circle_with_undo(self) -> None:
        if self.frame_viewer.circle_geometry() is not None:
            self._push_circle_undo("clear circle")
        self.frame_viewer.clear_circle()

    def _show_annotate_mask_context_menu(self, payload: object) -> None:
        if self.mode_tabs.currentIndex() not in {
            self.TAB_CHAMBER,
            self.TAB_CIRCLE,
            self.TAB_OCCLUSION,
        }:
            return
        global_pos = None
        image_point = None
        if isinstance(payload, tuple) and len(payload) >= 1:
            global_pos = payload[0]
            if len(payload) >= 2:
                image_point = payload[1]
        if not isinstance(global_pos, QPoint):
            global_pos = self.frame_viewer.mapToGlobal(self.frame_viewer.rect().center())

        if self.mode_tabs.currentIndex() == self.TAB_OCCLUSION and image_point is not None:
            mask_name = self.frame_viewer._mask_name_at_point(image_point, margin=8)
            if mask_name is not None:
                self._on_occ_mask_double_clicked(mask_name)

        copy_item = self._active_mask_clipboard_item()
        paste_label, paste_enabled = self._mask_paste_action_state()

        menu = QMenu(self)
        copy_action = menu.addAction(
            "Copy Mask" if copy_item is None else f"Copy Mask: {copy_item.label}"
        )
        copy_action.setEnabled(copy_item is not None)
        copy_action.triggered.connect(self.copy_active_mask_to_clipboard)
        paste_action = menu.addAction(paste_label)
        paste_action.setEnabled(paste_enabled)
        paste_action.triggered.connect(self.paste_mask_from_clipboard)
        menu.exec(global_pos)

    def _active_mask_clipboard_item(self) -> MaskClipboardItem | None:
        if self.video_state is None:
            return None
        tab_index = self.mode_tabs.currentIndex()
        if tab_index == self.TAB_CHAMBER:
            if self.chamber_edit_chamber_radio.isChecked():
                if self.chamber_mask is None or not np.any(self.chamber_mask):
                    return None
                return MaskClipboardItem(
                    label="chamber",
                    source="Chamber",
                    mask=self.chamber_mask.copy().astype(np.uint8),
                    color_name="#d1d5db",
                    geometry=clone_mask_geometry(getattr(self, "chamber_geometry", None)),
                )
            current = self._selected_room()
            if current is None:
                return None
            effective = self._effective_room_records_dict().get(current.name)
            mask = current.mask if effective is None else effective.mask
            geometry = current.geometry if effective is None else effective.geometry
            if not np.any(mask):
                return None
            return MaskClipboardItem(
                label=current.name,
                source="Chamber",
                mask=mask.copy().astype(np.uint8),
                color_name=current.color.name(),
                geometry=clone_mask_geometry(geometry),
            )

        if tab_index == self.TAB_CIRCLE:
            geometry = self.frame_viewer.circle_geometry()
            if geometry is None:
                return None
            center, base_radius, adjusted_radius = geometry
            return MaskClipboardItem(
                label="circle",
                source="Circle",
                mask=build_circle_mask(
                    self.video_state.width,
                    self.video_state.height,
                    center,
                    adjusted_radius,
                ),
                color_name="#ef4444",
                geometry=circle_mask_geometry(center, base_radius, adjusted_radius, source="exact"),
                circle_center=center,
                circle_base_radius=base_radius,
                circle_margin=int(self.circle_margin_slider.value()),
            )

        if tab_index == self.TAB_OCCLUSION:
            current = self._selected_mask()
            if current is None or not np.any(current.mask):
                return None
            return MaskClipboardItem(
                label=current.name,
                source="Occlusion",
                mask=current.mask.copy().astype(np.uint8),
                color_name=current.color.name(),
                margin=int(current.margin),
                margin_mode=current.margin_mode,
                geometry=clone_mask_geometry(current.geometry),
            )
        return None

    def copy_active_mask_to_clipboard(self) -> None:
        item = self._active_mask_clipboard_item()
        if item is None:
            self.statusBar().showMessage("No mask available to copy.", 3000)
            return
        self._mask_clipboard = item
        self.statusBar().showMessage(
            f"Copied {item.source} mask: {item.label}", 3000
        )

    def _mask_clipboard_for_current_video(self) -> np.ndarray | None:
        item = self._mask_clipboard
        if item is None or self.video_state is None:
            return None
        mask = (item.mask > 0).astype(np.uint8)
        target_shape = (self.video_state.height, self.video_state.width)
        if mask.shape != target_shape:
            mask = cv2.resize(
                mask,
                (self.video_state.width, self.video_state.height),
                interpolation=cv2.INTER_NEAREST,
            )
        return (mask > 0).astype(np.uint8)

    def _scaled_clipboard_geometry(self) -> dict | None:
        item = self._mask_clipboard
        if item is None or self.video_state is None:
            return None
        source_height, source_width = item.mask.shape[:2]
        scale_x = self.video_state.width / max(1.0, float(source_width))
        scale_y = self.video_state.height / max(1.0, float(source_height))
        return scale_mask_geometry(item.geometry, scale_x, scale_y)

    def _scaled_clipboard_circle_geometry(self) -> tuple[tuple[float, float], float, int, str] | None:
        item = self._mask_clipboard
        if item is None or self.video_state is None:
            return None
        source_height, source_width = item.mask.shape[:2]
        scale_x = self.video_state.width / max(1.0, float(source_width))
        scale_y = self.video_state.height / max(1.0, float(source_height))
        radius_scale = (scale_x + scale_y) / 2.0
        scaled_geometry = scale_mask_geometry(item.geometry, scale_x, scale_y)
        if isinstance(scaled_geometry, dict) and str(scaled_geometry.get("kind", "")).lower() == "circle":
            center_value = scaled_geometry.get("center")
            try:
                center = (float(center_value[0]), float(center_value[1]))
                base_radius = float(scaled_geometry.get("base_radius", scaled_geometry.get("radius")))
            except (TypeError, ValueError, IndexError):
                center = None
                base_radius = 0.0
            if center is not None and base_radius > 0:
                source = str(scaled_geometry.get("source", "exact")).lower()
                if source not in {"exact", "inferred"}:
                    source = "inferred"
                margin = int(round(float(item.circle_margin) * radius_scale)) if item.source == "Circle" else 0
                return center, max(1.0, base_radius), margin, source
        if item.circle_center is None or item.circle_base_radius is None:
            return None
        center = (
            float(item.circle_center[0]) * scale_x,
            float(item.circle_center[1]) * scale_y,
        )
        margin = int(round(float(item.circle_margin) * radius_scale))
        return center, max(1.0, float(item.circle_base_radius) * radius_scale), margin, "exact"

    def _mask_paste_action_state(self) -> tuple[str, bool]:
        if self._mask_clipboard is None:
            return "Paste Mask (clipboard empty)", False
        if self.video_state is None:
            return "Paste Mask (load a video first)", False
        tab_index = self.mode_tabs.currentIndex()
        if tab_index == self.TAB_CIRCLE:
            if self._scaled_clipboard_circle_geometry() is not None:
                return f"Paste Mask to Circle: {self._mask_clipboard.label}", True
            mask = self._mask_clipboard_for_current_video()
            if mask is not None and infer_circular_mask_geometry(mask) is not None:
                return f"Paste Mask to Circle: {self._mask_clipboard.label}", True
            return "Paste Mask to Circle (circle masks only)", False
        if tab_index == self.TAB_CHAMBER:
            target = "Chamber" if self.chamber_edit_chamber_radio.isChecked() else "Room"
            return f"Paste Mask to {target}: {self._mask_clipboard.label}", True
        if tab_index == self.TAB_OCCLUSION:
            return f"Paste Mask to Occlusion: {self._mask_clipboard.label}", True
        return "Paste Mask", False

    def paste_mask_from_clipboard(self) -> None:
        if self._mask_clipboard is None:
            self.statusBar().showMessage("Copy a mask first.", 3000)
            return
        if self.video_state is None:
            QMessageBox.information(self, "Paste Mask", "Load a video first.")
            return
        tab_index = self.mode_tabs.currentIndex()
        if tab_index == self.TAB_CHAMBER:
            self._paste_mask_to_chamber()
        elif tab_index == self.TAB_CIRCLE:
            self._paste_mask_to_circle()
        elif tab_index == self.TAB_OCCLUSION:
            self._paste_mask_to_occlusion()

    def _paste_mask_to_chamber(self) -> None:
        item = self._mask_clipboard
        mask = self._mask_clipboard_for_current_video()
        if item is None or mask is None or not np.any(mask):
            QMessageBox.information(self, "Paste Mask", "Copied mask is empty.")
            return

        self._invalidate_chamber_transform_source()
        if self.chamber_edit_chamber_radio.isChecked():
            self._push_chamber_undo("paste chamber mask")
            self.chamber_mask = mask.copy().astype(np.uint8)
            self.chamber_geometry = self._scaled_clipboard_geometry()
            self._set_chamber_boundary_mode(
                "full_frame" if self._is_full_frame_chamber_mask() else "custom"
            )
            self.frame_viewer.clear_chamber_rect_points()
            self.frame_viewer.clear_chamber_circle()
            self._refresh_chamber_viewer(refresh=True)
            self._refresh_chamber_ui()
            self.statusBar().showMessage(
                f"Pasted mask into chamber boundary: {item.label}", 3000
            )
            return

        if self.chamber_mask is None or not np.any(self.chamber_mask):
            QMessageBox.information(
                self,
                "Paste Mask",
                "Define the chamber area before pasting into a room.",
            )
            return

        current = self._selected_room()
        paste_replaces_existing_room = current is not None
        if current is None:
            name = self._next_available_room_name(item.label)
            color = QColor(item.color_name)
            if not color.isValid():
                color = MASK_PALETTE[len(self.room_records) % len(MASK_PALETTE)]
            current = RoomRecord(
                name=name,
                color=color,
                mask=np.zeros(
                    (self.video_state.height, self.video_state.width),
                    dtype=np.uint8,
                ),
            )
            self.room_records[name] = current
            self.selected_room_name = name
            self.chamber_edit_room_radio.setChecked(True)

        blocked = self._occupied_room_mask(exclude_name=current.name)
        allowed = (mask > 0) & self.chamber_mask.astype(bool)
        if blocked is not None:
            allowed &= ~blocked.astype(bool)
        if not np.any(allowed):
            QMessageBox.information(
                self,
                "Paste Mask",
                "The pasted mask has no pixels available inside the chamber.",
            )
            return
        if paste_replaces_existing_room:
            self._push_chamber_undo("paste room mask")
        current.mask = allowed.astype(np.uint8)
        current.geometry = (
            self._scaled_clipboard_geometry()
            if np.array_equal(allowed.astype(np.uint8), mask.astype(np.uint8))
            else None
        )
        self.selected_room_name = current.name
        self._rebuild_room_list()
        self.statusBar().showMessage(
            f"Pasted mask into room: {current.name}", 3000
        )

    def _paste_mask_to_occlusion(self) -> None:
        item = self._mask_clipboard
        mask = self._mask_clipboard_for_current_video()
        if item is None or mask is None or not np.any(mask):
            QMessageBox.information(self, "Paste Mask", "Copied mask is empty.")
            return

        current = self._selected_mask()
        paste_replaces_existing_mask = current is not None
        if current is None:
            name = self._next_available_mask_name(item.label)
            color = QColor(item.color_name)
            if not color.isValid():
                color = MASK_PALETTE[len(self.mask_records) % len(MASK_PALETTE)]
            current = MaskRecord(
                name=name,
                color=color,
                mask=np.zeros(
                    (self.video_state.height, self.video_state.width),
                    dtype=np.uint8,
                ),
                margin=(
                    int(item.margin)
                    if item.source == "Occlusion"
                    else self.default_mask_margin
                ),
                margin_mode=(
                    item.margin_mode
                    if item.source == "Occlusion"
                    else self.default_mask_margin_mode
                ),
            )
            self.mask_records[name] = current
            self.selected_mask_name = name

        if paste_replaces_existing_mask:
            self._push_occlusion_undo("paste occlusion mask")
        self._invalidate_mask_transform_source(current.name)
        current.mask = mask.copy().astype(np.uint8)
        current.geometry = self._scaled_clipboard_geometry()
        self._rebuild_mask_list()
        self.statusBar().showMessage(
            f"Pasted mask into occlusion: {current.name}", 3000
        )

    def _paste_mask_to_circle(self) -> None:
        item = self._mask_clipboard
        if item is None:
            return
        scaled_geometry = self._scaled_clipboard_circle_geometry()
        if scaled_geometry is None:
            mask = self._mask_clipboard_for_current_video()
            geometry = None if mask is None else infer_circular_mask_geometry(mask)
            if geometry is None:
                QMessageBox.information(
                    self,
                    "Paste Mask",
                    "Circle tab can only paste masks that are circular.",
                )
                return
            center, base_radius = geometry
            margin = 0
            geometry_source = "inferred"
        else:
            center, base_radius, margin, geometry_source = scaled_geometry

        margin = max(
            self.circle_margin_slider.minimum(),
            min(self.circle_margin_slider.maximum(), int(margin)),
        )
        self._push_circle_undo("paste circle mask")
        self._set_circle_margin_value(margin)
        self.frame_viewer.circle_start = (center[0] - base_radius, center[1])
        self.frame_viewer.circle_end = (center[0] + base_radius, center[1])
        self.frame_viewer.circle_geometry_source = geometry_source
        self.frame_viewer.update()
        self.frame_viewer.circle_changed.emit()
        self._refresh_circle_ui()
        self._save_settings()
        self.statusBar().showMessage(
            f"Pasted mask into circle: {item.label}", 3000
        )


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
        self._push_occlusion_undo("move occlusion mask")
        self._invalidate_mask_transform_source(current.name)
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
            translated_geometry = translate_mask_geometry(current.geometry, dx, dy)
            if mask_geometry_is_exact(current.geometry):
                current.mask = translated.astype(np.uint8)
                current.geometry = translated_geometry
            else:
                current.mask = smooth_binary_mask_low(translated)
                current.geometry = None

    def _invalidate_mask_transform_source(self, mask_name: str | None = None) -> None:
        if mask_name is not None and mask_name != self._mask_transform_source_name:
            return
        self._mask_transform_source_name = None
        self._mask_transform_source = None
        self._mask_transform_angle = 0.0
        self._mask_transform_scale = 1.0

    def _ensure_mask_transform_source(self, current: MaskRecord) -> bool:
        if (
            self._mask_transform_source is not None
            and self._mask_transform_source_name == current.name
            and self._mask_transform_source.mask.shape == current.mask.shape
        ):
            return True
        try:
            source = MaskTransformSource.from_mask(current.mask)
        except ValueError:
            self._invalidate_mask_transform_source()
            return False
        self._mask_transform_source_name = current.name
        self._mask_transform_source = source
        self._mask_transform_angle = 0.0
        self._mask_transform_scale = 1.0
        return True

    def _apply_selected_mask_affine_transform(
        self,
        current: MaskRecord,
        *,
        angle_delta: float = 0.0,
        scale_multiplier: float = 1.0,
    ) -> None:
        if scale_multiplier <= 0 or not self._ensure_mask_transform_source(current):
            return
        source = self._mask_transform_source
        if source is None:
            return
        next_angle = math.fmod(self._mask_transform_angle + float(angle_delta), 360.0)
        next_scale = self._mask_transform_scale * float(scale_multiplier)
        if next_scale < 0.01 or next_scale > 64.0:
            return
        transformed = source.render(next_angle, next_scale)
        if not np.any(transformed):
            return
        self._push_occlusion_undo("transform occlusion mask")
        self._mask_transform_angle = next_angle
        self._mask_transform_scale = next_scale
        current.mask = smooth_binary_mask_low(transformed)
        current.geometry = None
        self.frame_viewer.refresh_mask_record(current.name, include_margin=True)

    def scale_selected_mask(self, scale_factor: float) -> None:
        current = self._selected_mask()
        if current is None or scale_factor <= 0:
            return
        self._apply_selected_mask_affine_transform(
            current,
            scale_multiplier=float(scale_factor),
        )

    def rotate_selected_mask(self, angle_degrees: float) -> None:
        current = self._selected_mask()
        if current is None or abs(angle_degrees) < 1e-6:
            return
        self._apply_selected_mask_affine_transform(
            current,
            angle_delta=float(angle_degrees),
        )

    def _mask_transform_shortcuts_enabled(self) -> bool:
        return (
            self.mode_tabs.currentIndex() == self.TAB_OCCLUSION
            and self.mask_transform_radio.isChecked()
            and self._selected_mask() is not None
        )

    def _interpolation_transform_shortcuts_enabled(self) -> bool:
        return (
            self.mode_tabs.currentIndex() == self.TAB_INTERPOLATION
            and self.interpolation_region_transform_radio.isChecked()
            and self._has_interpolation_region()
        )

    def _chamber_transform_shortcuts_enabled(self) -> bool:
        return (
            self.mode_tabs.currentIndex() == self.TAB_CHAMBER
            and self.chamber_transform_radio.isChecked()
            and self._selected_chamber_layer() is not None
        )

    def _circle_transform_shortcuts_enabled(self) -> bool:
        return (
            self.mode_tabs.currentIndex() == self.TAB_CIRCLE
            and self.circle_transform_radio.isChecked()
            and self.frame_viewer.circle_geometry() is not None
        )

    def _scale_active_region_shortcut(self, scale_factor: float) -> None:
        if self._interpolation_transform_shortcuts_enabled():
            self.scale_interpolation_region(scale_factor)
        elif self._chamber_transform_shortcuts_enabled():
            self.scale_selected_chamber_layer(scale_factor)
        elif self._circle_transform_shortcuts_enabled():
            self.scale_circle(scale_factor)
        elif self._mask_transform_shortcuts_enabled():
            self._scale_selected_mask_shortcut(scale_factor)

    def _rotate_active_region_shortcut(self, angle_degrees: float) -> None:
        if self._chamber_transform_shortcuts_enabled():
            self.rotate_selected_chamber_layer(angle_degrees)
        elif self._mask_transform_shortcuts_enabled():
            self._rotate_selected_mask_shortcut(angle_degrees)

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

    def _write_csv_output(
        self,
        output_path: Path,
        dataframe: pd.DataFrame,
        should_cancel=None,
    ) -> Path:
        raise_if_cancelled(should_cancel)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raise_if_cancelled(should_cancel)
        dataframe.to_csv(output_path, index=False)
        return output_path

    def _current_mode_save_task(self):
        index = self.mode_tabs.currentIndex()
        if index == self.TAB_TRACKING_REPAIR:
            if self.csv_df is None or self.video_state is None:
                QMessageBox.warning(self, "Tracking Repair", "Load a video and CSV first.")
                return None
            output = self._tracking_repair_output_path()
            if output is None:
                return None
            source_df = self.csv_df.copy()
            bodyparts = list(self.bodyparts)
            width = int(self.video_state.width)
            height = int(self.video_state.height)
            config = self._tracking_repair_config()
            stats: dict[str, int] = {"input_rows": len(source_df)}

            def _export_item(*, should_cancel=None) -> Path:
                raise_if_cancelled(should_cancel)
                repair_result = self._run_tracking_repair_for(
                    source_df,
                    bodyparts,
                    width,
                    height,
                    config=config,
                )
                stats.update(
                    output_rows=len(repair_result.dataframe),
                    duplicates=repair_result.duplicate_removed,
                    z_outliers=repair_result.length_outliers_invalidated,
                )
                return self._write_csv_output(output, repair_result.dataframe, should_cancel)

            def _completed(result) -> None:
                if getattr(result, "cancelled", False):
                    self.statusBar().showMessage("Tracking postprocess save cancelled.")
                    return
                self.statusBar().showMessage(
                    f"Tracking postprocess CSV saved: {output} "
                    f"({stats.get('input_rows', 0):,} ? {stats.get('output_rows', 0):,} rows; "
                    f"duplicates removed={stats.get('duplicates', 0):,}, "
                    f"Z-score skeletons invalidated={stats.get('z_outliers', 0):,})"
                )

            return (
                "Save CSV Progress",
                "Saving tracking postprocess CSV in the background...",
                _export_item,
                _completed,
            )

        if index == self.TAB_INTERPOLATION:
            if self.csv_df is None or self.video_state is None or not self.bodyparts:
                QMessageBox.warning(self, "Interpolation", "Load a video and CSV first.")
                return None
            removal_mode = self._selected_interpolation_removal_mode()
            anchor = self._selected_interpolation_anchor()
            interpolate = self._interpolation_enabled()
            if removal_mode != "none" and not self._has_interpolation_region():
                QMessageBox.warning(self, "Interpolation", "Draw an automatic-removal region first.")
                return None
            if removal_mode != "none" and anchor is None:
                QMessageBox.warning(self, "Interpolation", "Select an anchor node for automatic removal.")
                return None
            if removal_mode == "none" and not interpolate:
                QMessageBox.warning(self, "Interpolation", "Enable automatic removal or interpolation first.")
                return None
            output = self._interpolation_output_path()
            if output is None:
                return None
            source_df = self.csv_df.copy()
            bodyparts = list(self.bodyparts)
            mask = None if self.interpolation_mask is None else self.interpolation_mask.copy().astype(np.uint8)
            width = int(self.video_state.width)
            height = int(self.video_state.height)
            extrapolate = self._interpolation_extrapolation_enabled()

            def _export_item(*, should_cancel=None) -> Path:
                raise_if_cancelled(should_cancel)
                result_df = build_interpolation_pipeline_dataframe(
                    source_df,
                    bodyparts,
                    mask,
                    width,
                    height,
                    removal_mode,
                    anchor,
                    interpolate,
                    extrapolate,
                )
                return self._write_csv_output(output, result_df, should_cancel)

            def _completed(result) -> None:
                if getattr(result, "cancelled", False):
                    self.statusBar().showMessage("Removal/interpolation save cancelled.")
                    return
                self.statusBar().showMessage(f"Removal/interpolation pipeline CSV saved: {output}")

            return (
                "Save CSV Progress",
                "Saving removal/interpolation CSV in the background...",
                _export_item,
                _completed,
            )

        if index == self.TAB_SQUARE:
            if self.csv_df is None or self.video_state is None:
                QMessageBox.warning(self, "Save", "Load a video and CSV first.")
                return None
            output = self._normalized_output_path()
            if output is None:
                return None
            source_df = self.csv_df.copy()
            bodyparts = list(self.bodyparts)
            square_points = [tuple(point) for point in self.frame_viewer.square_points]
            width = int(self.video_state.width)
            height = int(self.video_state.height)

            def _export_item(*, should_cancel=None) -> Path:
                raise_if_cancelled(should_cancel)
                result_df = build_normalized_dataframe(
                    source_df,
                    bodyparts,
                    square_points,
                    width,
                    height,
                )
                return self._write_csv_output(output, result_df, should_cancel)

            def _completed(result) -> None:
                if getattr(result, "cancelled", False):
                    self.statusBar().showMessage("Normalized CSV save cancelled.")
                    return
                self.statusBar().showMessage(f"Normalized CSV saved: {output}")

            return (
                "Save CSV Progress",
                "Saving normalized CSV in the background...",
                _export_item,
                _completed,
            )

        if index == self.TAB_CHAMBER:
            if self.video_state is None or self.csv_df is None:
                QMessageBox.warning(self, "Save", "Load a video and CSV first.")
                return None
            if self.chamber_mask is None or not np.any(self.chamber_mask):
                QMessageBox.warning(self, "Save", "Define the chamber area first.")
                return None
            if not self.room_records:
                QMessageBox.warning(self, "Save", "Add at least one room first.")
                return None
            csv_output = self._chamber_csv_output_path()
            mask_output = self._chamber_mask_output_path()
            overlay_output = self._chamber_overlay_output_path()
            manifest_output = self._chamber_manifest_output_path()
            if csv_output is None or mask_output is None or overlay_output is None or manifest_output is None:
                return None
            mask_rgb = self._chamber_mask_rgb()
            overlay_rgb = self._chamber_overlay_rgb()
            if mask_rgb is None or overlay_rgb is None:
                QMessageBox.warning(self, "Save", "A frame and chamber mask are required.")
                return None
            source_df = self.csv_df.copy()
            bodyparts = list(self.bodyparts)
            width = int(self.video_state.width)
            height = int(self.video_state.height)
            rooms = [
                RoomRecord(
                    name=room.name,
                    color=QColor(room.color),
                    mask=room.mask.copy().astype(np.uint8),
                    geometry=clone_mask_geometry(room.geometry),
                )
                for room in self._effective_room_records()
            ]
            room_metadata = [
                {
                    "name": room.name,
                    "color": room.color.name(),
                    "geometry": mask_geometry_for_export(room.geometry, room.mask),
                }
                for room in self.room_records.values()
            ]
            chamber_metadata_geometry = mask_geometry_for_export(
                getattr(self, "chamber_geometry", None),
                self.chamber_mask,
            )
            boundary_mode = self._resolved_chamber_boundary_mode()
            mask_rgb = mask_rgb.copy()
            overlay_rgb = overlay_rgb.copy()

            def _export_item(*, should_cancel=None) -> Path:
                raise_if_cancelled(should_cancel)
                result_df = build_chamber_mark_dataframe(
                    source_df,
                    bodyparts,
                    rooms,
                    width,
                    height,
                )
                raise_if_cancelled(should_cancel)
                csv_output.parent.mkdir(parents=True, exist_ok=True)
                raise_if_cancelled(should_cancel)
                result_df.to_csv(csv_output, index=False)
                if not cv2.imwrite(str(mask_output), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"Could not write {mask_output}")
                if not cv2.imwrite(str(overlay_output), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"Could not write {overlay_output}")
                metadata = {
                    "format": "happycold_chamber_mask_v1",
                    "metadata_version": 2,
                    "width": width,
                    "height": height,
                    "boundary_mode": boundary_mode,
                    "geometry": chamber_metadata_geometry,
                    "rooms": room_metadata,
                }
                manifest_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                return csv_output

            def _completed(result) -> None:
                if getattr(result, "cancelled", False):
                    self.statusBar().showMessage("Chamber output save cancelled.")
                    return
                self.statusBar().showMessage(f"Chamber outputs saved: {csv_output}")

            return (
                "Save CSV Progress",
                "Saving chamber outputs in the background...",
                _export_item,
                _completed,
            )

        if index == self.TAB_CIRCLE:
            if self.csv_df is None or self.video_state is None:
                QMessageBox.warning(self, "Save", "Load a video and CSV first.")
                return None
            geometry = self.frame_viewer.circle_geometry()
            if geometry is None:
                QMessageBox.warning(self, "Save", "Draw a circle first.")
                return None
            center, _, adjusted_radius = geometry
            output = self._circle_output_path()
            if output is None:
                return None
            source_df = self.csv_df.copy()
            bodyparts = list(self.bodyparts)
            width = int(self.video_state.width)
            height = int(self.video_state.height)

            def _export_item(*, should_cancel=None) -> Path:
                raise_if_cancelled(should_cancel)
                result_df = build_circle_detection_dataframe(
                    source_df,
                    bodyparts,
                    center,
                    adjusted_radius,
                    width,
                    height,
                )
                return self._write_csv_output(output, result_df, should_cancel)

            def _completed(result) -> None:
                if getattr(result, "cancelled", False):
                    self.statusBar().showMessage("Detection CSV save cancelled.")
                    return
                self.statusBar().showMessage(f"Detection CSV saved: {output}")

            return (
                "Save CSV Progress",
                "Saving circle detection CSV in the background...",
                _export_item,
                _completed,
            )

        if index == self.TAB_OCCLUSION:
            if self.csv_df is None or self.video_state is None or not self.mask_records:
                QMessageBox.warning(self, "Save", "Load video/CSV and prepare masks first.")
                return None
            if any(record.margin_mode == "geometric" for record in self.mask_records.values()) and len(self.frame_viewer.occ_margin_points) != 4:
                QMessageBox.warning(self, "Save", "Geometric margin mode needs four occlusion geometric points.")
                return None
            output = self._occlusion_output_path()
            if output is None:
                return None
            source_df = self.csv_df.copy()
            bodyparts = list(self.bodyparts)
            masks = [
                MaskRecord(
                    name=record.name,
                    color=QColor(record.color),
                    mask=record.mask.copy().astype(np.uint8),
                    margin=int(record.margin),
                    margin_mode=str(record.margin_mode),
                    geometry=clone_mask_geometry(record.geometry),
                )
                for record in self.mask_records.values()
            ]
            width = int(self.video_state.width)
            height = int(self.video_state.height)
            occ_margin_points = [tuple(point) for point in self.frame_viewer.occ_margin_points]

            def _export_item(*, should_cancel=None) -> Path:
                raise_if_cancelled(should_cancel)
                result_df = build_occlusion_dataframe(
                    source_df,
                    bodyparts,
                    masks,
                    width,
                    height,
                    occ_margin_points,
                )
                return self._write_csv_output(output, result_df, should_cancel)

            def _completed(result) -> None:
                if getattr(result, "cancelled", False):
                    self.statusBar().showMessage("Occlusion CSV save cancelled.")
                    return
                self.statusBar().showMessage(f"Occlusion CSV saved: {output}")

            return (
                "Save CSV Progress",
                "Saving occlusion CSV in the background...",
                _export_item,
                _completed,
            )

        return None

    def save_current_mode_output(self) -> None:
        if self._focus_active_save_progress():
            return
        task = self._current_mode_save_task()
        if task is None:
            return
        title, activity_text, export_item, on_completed = task
        self._start_single_save_export(
            title=title,
            activity_text=activity_text,
            export_item=export_item,
            on_completed=on_completed,
        )

    def closeEvent(self, event) -> None:
        active_thread = getattr(self, "_active_single_save_thread", None)
        if active_thread is not None and active_thread.isRunning():
            self._request_active_single_save_cancel()
            dialog = getattr(self, "_active_single_save_dialog", None)
            if dialog is not None:
                dialog.show()
                dialog.raise_()
                dialog.activateWindow()
            event.ignore()
            return
        active_thread = getattr(self, "_active_batch_thread", None)
        if active_thread is not None and active_thread.isRunning():
            self._request_active_batch_cancel()
            dialog = getattr(self, "_active_batch_dialog", None)
            if dialog is not None:
                dialog.show()
                dialog.raise_()
                dialog.activateWindow()
            event.ignore()
            return
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
