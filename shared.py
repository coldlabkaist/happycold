import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PyQt6.QtGui import QColor

from trajectory import bodypart_coordinate_columns, bodyparts_from_dataframe, infer_pixel_scale


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
MASK_PALETTE = [
    QColor("#ef4444"),
    QColor("#f97316"),
    QColor("#eab308"),
    QColor("#22c55e"),
    QColor("#06b6d4"),
    QColor("#3b82f6"),
    QColor("#8b5cf6"),
    QColor("#ec4899"),
]


@dataclass
class VideoState:
    path: Path
    capture: cv2.VideoCapture
    frame_count: int
    fps: float
    width: int
    height: int


@dataclass
class PinRecord:
    pin_id: str
    frame: int
    x: float
    y: float

    def normalized(self, width: int, height: int) -> tuple[float, float]:
        return self.x / max(1, width), self.y / max(1, height)


@dataclass
class MaskRecord:
    name: str
    color: QColor
    mask: np.ndarray
    margin: int = 0
    margin_mode: str = "simple"


@dataclass
class MaskTransformSource:
    """Stable source data for repeated affine transforms of a binary mask."""

    mask: np.ndarray
    signed_distance: np.ndarray
    center: tuple[float, float]

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "MaskTransformSource":
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if binary.ndim != 2 or not np.any(binary):
            raise ValueError("A non-empty 2D mask is required.")
        ys, xs = np.where(binary > 0)
        center = (float(xs.mean()), float(ys.mean()))
        inside = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_5)
        outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, cv2.DIST_MASK_5)
        return cls(
            mask=np.ascontiguousarray(binary),
            signed_distance=np.ascontiguousarray(inside - outside, dtype=np.float32),
            center=center,
        )

    def render(self, angle_degrees: float, scale_factor: float) -> np.ndarray:
        if scale_factor <= 0:
            raise ValueError("Scale factor must be positive.")
        normalized_angle = math.fmod(float(angle_degrees), 360.0)
        if abs(normalized_angle) < 1e-7 and abs(float(scale_factor) - 1.0) < 1e-7:
            return self.mask.copy()
        matrix = cv2.getRotationMatrix2D(
            self.center,
            normalized_angle,
            float(scale_factor),
        )
        height, width = self.mask.shape
        transformed_distance = cv2.warpAffine(
            self.signed_distance,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-float(max(width, height)),
        )
        return (transformed_distance > 0.0).astype(np.uint8)


@dataclass
class RoomRecord:
    name: str
    color: QColor
    mask: np.ndarray


def discover_videos(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES],
        key=lambda path: str(path).lower(),
    )




def dataframe_points_to_pixels(df: pd.DataFrame, x_col: str, y_col: str, width: int, height: int) -> np.ndarray:
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=np.float32)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float32)
    return np.column_stack(
        [
            x * infer_pixel_scale(df[x_col], width),
            y * infer_pixel_scale(df[y_col], height),
        ]
    ).astype(np.float32)


def order_quad_points(points: list[tuple[float, float]]) -> np.ndarray:
    pts = np.array(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5


def adjust_mask(mask: np.ndarray, margin: int) -> np.ndarray:
    margin = int(margin)
    if margin == 0 or not np.any(mask):
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(margin) * 2 + 1, abs(margin) * 2 + 1))
    if margin > 0:
        return cv2.dilate(mask.astype(np.uint8), kernel)
    return cv2.erode(mask.astype(np.uint8), kernel)


def build_rectified_geometry(quad_points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if len(quad_points) != 4:
        raise ValueError("Four square points are required for geometric margin.")
    ordered = order_quad_points(quad_points)
    if polygon_area(ordered) < 10:
        raise ValueError("Selected square area is too small.")

    top_width = math.dist(tuple(ordered[0]), tuple(ordered[1]))
    bottom_width = math.dist(tuple(ordered[3]), tuple(ordered[2]))
    left_height = math.dist(tuple(ordered[0]), tuple(ordered[3]))
    right_height = math.dist(tuple(ordered[1]), tuple(ordered[2]))
    rect_width = max(1, int(round(max(top_width, bottom_width))))
    rect_height = max(1, int(round(max(left_height, right_height))))
    destination = np.array(
        [[0, 0], [rect_width - 1, 0], [rect_width - 1, rect_height - 1], [0, rect_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    inverse = cv2.getPerspectiveTransform(destination, ordered)
    return matrix, inverse, (rect_width, rect_height)


def adjust_mask_by_mode(
    mask: np.ndarray,
    margin: int,
    margin_mode: str,
    quad_points: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    if margin_mode != "geometric":
        return adjust_mask(mask, margin)

    if margin == 0 or not np.any(mask):
        return mask.copy()
    if quad_points is None or len(quad_points) != 4:
        raise ValueError("Four square points are required for geometric margin.")

    matrix, inverse, (rect_width, rect_height) = build_rectified_geometry(quad_points)
    warped = cv2.warpPerspective(
        mask.astype(np.uint8),
        matrix,
        (rect_width, rect_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    adjusted = adjust_mask(warped, margin)
    restored = cv2.warpPerspective(
        adjusted.astype(np.uint8),
        inverse,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (restored > 0).astype(np.uint8)


def fill_polygon(mask: np.ndarray, points: list[tuple[float, float]], value: int) -> None:
    int_points = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)
    if len(int_points) >= 3:
        cv2.fillPoly(mask, [int_points], int(value))


def fill_circle_from_diameter(mask: np.ndarray, start: tuple[float, float], end: tuple[float, float], value: int) -> None:
    center_x = int(round((start[0] + end[0]) / 2.0))
    center_y = int(round((start[1] + end[1]) / 2.0))
    radius = int(round(math.dist(start, end) / 2.0))
    if radius > 0:
        cv2.circle(mask, (center_x, center_y), radius, int(value), -1)


def paint_brush(mask: np.ndarray, start: tuple[float, float], end: tuple[float, float], radius: int, value: int) -> None:
    radius = max(1, int(radius))
    start_point = (int(round(start[0])), int(round(start[1])))
    end_point = (int(round(end[0])), int(round(end[1])))
    brush_value = int(value)
    cv2.line(
        mask,
        start_point,
        end_point,
        brush_value,
        thickness=radius * 2,
        lineType=cv2.LINE_8,
    )
    cv2.circle(mask, start_point, radius, brush_value, -1, lineType=cv2.LINE_8)
    cv2.circle(mask, end_point, radius, brush_value, -1, lineType=cv2.LINE_8)


def build_normalized_dataframe(
    df: pd.DataFrame,
    bodyparts: list[str],
    quad_points: list[tuple[float, float]],
    width: int,
    height: int,
) -> pd.DataFrame:
    """Append perspective-normalized coordinates while preserving every source column."""
    ordered = order_quad_points(quad_points)
    if polygon_area(ordered) < 10:
        raise ValueError("Selected square area is too small.")

    destination = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    normalized_df = df.copy()

    for bodypart in bodyparts:
        x_col = f"{bodypart}.x"
        y_col = f"{bodypart}.y"
        normalized_x_col, normalized_y_col = bodypart_coordinate_columns(bodypart, normalized=True)
        points = dataframe_points_to_pixels(df, x_col, y_col, width, height)
        valid_mask = ~np.isnan(points).any(axis=1)
        transformed = np.full_like(points, np.nan, dtype=np.float32)
        if valid_mask.any():
            transformed[valid_mask] = cv2.perspectiveTransform(points[valid_mask].reshape(-1, 1, 2), matrix).reshape(-1, 2)
        normalized_df[normalized_x_col] = transformed[:, 0]
        normalized_df[normalized_y_col] = transformed[:, 1]

    return normalized_df


def build_circle_detection_dataframe(
    df: pd.DataFrame,
    bodyparts: list[str],
    center: tuple[float, float],
    radius: float,
    width: int,
    height: int,
) -> pd.DataFrame:
    if radius <= 0:
        raise ValueError("Adjusted radius must be greater than zero.")

    result = df.copy()
    center_array = np.array(center, dtype=np.float32)
    for bodypart in bodyparts:
        x_col = f"{bodypart}.x"
        y_col = f"{bodypart}.y"
        out_col = f"{bodypart}_in_out"
        points = dataframe_points_to_pixels(df, x_col, y_col, width, height)
        valid_mask = ~np.isnan(points).any(axis=1)
        distances = np.full(points.shape[0], np.nan, dtype=np.float32)
        if valid_mask.any():
            distances[valid_mask] = np.linalg.norm(points[valid_mask] - center_array, axis=1)
        values = np.full(points.shape[0], "", dtype=object)
        values[valid_mask & (distances <= radius)] = "in"
        values[valid_mask & (distances > radius)] = "out"
        result[out_col] = values
    return result


def build_occlusion_dataframe(
    df: pd.DataFrame,
    bodyparts: list[str],
    masks: list[MaskRecord],
    width: int,
    height: int,
    quad_points: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    result = df.copy()
    for mask_record in masks:
        occ_mask = adjust_mask_by_mode(mask_record.mask, mask_record.margin, mask_record.margin_mode, quad_points)
        total_occ: list[int] = []
        per_bp_occ = {bodypart: [] for bodypart in bodyparts}

        points_cache = {
            bodypart: dataframe_points_to_pixels(df, f"{bodypart}.x", f"{bodypart}.y", width, height)
            for bodypart in bodyparts
        }

        for row_index in range(len(df)):
            any_occ = 0
            for bodypart in bodyparts:
                point = points_cache[bodypart][row_index]
                occ = 0
                if not np.isnan(point).any():
                    x = int(round(point[0]))
                    y = int(round(point[1]))
                    if 0 <= x < width and 0 <= y < height and occ_mask[y, x] == 1:
                        occ = 1
                        any_occ = 1
                per_bp_occ[bodypart].append(occ)
            total_occ.append(any_occ)

        result[f"{mask_record.name}.occ"] = total_occ
        for bodypart in bodyparts:
            result[f"{mask_record.name}.{bodypart}.occ"] = per_bp_occ[bodypart]

    return result


def build_chamber_mark_dataframe(
    df: pd.DataFrame,
    bodyparts: list[str],
    rooms: list[RoomRecord],
    width: int,
    height: int,
) -> pd.DataFrame:
    result = df.copy()
    points_cache = {
        bodypart: dataframe_points_to_pixels(df, f"{bodypart}.x", f"{bodypart}.y", width, height)
        for bodypart in bodyparts
    }

    room_names = [room.name for room in rooms]
    room_masks = [room.mask.astype(bool) for room in rooms]

    for bodypart in bodyparts:
        values = np.full(len(df), "", dtype=object)
        points = points_cache[bodypart]
        for row_index, point in enumerate(points):
            if np.isnan(point).any():
                continue
            x = int(round(point[0]))
            y = int(round(point[1]))
            if not (0 <= x < width and 0 <= y < height):
                continue
            for room_name, room_mask in zip(room_names, room_masks):
                if room_mask[y, x]:
                    values[row_index] = room_name
                    break
        result[f"{bodypart}_room"] = values

    return result
