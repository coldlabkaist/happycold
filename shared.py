import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    geometry: dict[str, Any] | None = None


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
    geometry: dict[str, Any] | None = None



def clone_mask_geometry(geometry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(geometry, dict):
        return None
    return copy.deepcopy(geometry)


def _clean_point_pair(point: Any) -> list[float] | None:
    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return [x, y]


def mask_polygon_geometry(
    points: list[tuple[float, float]] | np.ndarray,
    *,
    source: str = "exact",
    shape: str = "rectangle",
) -> dict[str, Any] | None:
    cleaned = []
    for point in points:
        cleaned_point = _clean_point_pair(point)
        if cleaned_point is not None:
            cleaned.append(cleaned_point)
    if len(cleaned) < 3:
        return None
    return {
        "kind": "polygon",
        "shape": str(shape),
        "source": str(source),
        "points": cleaned,
    }


def circle_mask_geometry(
    center: tuple[float, float] | list[float],
    base_radius: float,
    adjusted_radius: float | None = None,
    *,
    source: str = "exact",
) -> dict[str, Any] | None:
    cleaned_center = _clean_point_pair(center)
    if cleaned_center is None:
        return None
    try:
        base = max(0.0, float(base_radius))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(base) or base <= 0:
        return None
    try:
        adjusted = base if adjusted_radius is None else float(adjusted_radius)
    except (TypeError, ValueError):
        adjusted = base
    if not math.isfinite(adjusted):
        adjusted = base
    return {
        "kind": "circle",
        "shape": "circle",
        "source": str(source),
        "center": cleaned_center,
        "base_radius": base,
        "adjusted_radius": max(1.0, adjusted),
    }


def full_frame_mask_geometry(width: int, height: int) -> dict[str, Any] | None:
    width = max(1, int(width))
    height = max(1, int(height))
    return mask_polygon_geometry(
        [
            (0.0, 0.0),
            (float(width - 1), 0.0),
            (float(width - 1), float(height - 1)),
            (0.0, float(height - 1)),
        ],
        source="exact",
        shape="full_frame_rectangle",
    )


def translate_mask_geometry(
    geometry: dict[str, Any] | None,
    dx: float,
    dy: float,
) -> dict[str, Any] | None:
    translated = clone_mask_geometry(geometry)
    if translated is None:
        return None
    try:
        offset_x = float(dx)
        offset_y = float(dy)
    except (TypeError, ValueError):
        return translated
    if not math.isfinite(offset_x) or not math.isfinite(offset_y):
        return translated

    def translate_point(value: Any) -> list[float] | None:
        point = _clean_point_pair(value)
        if point is None:
            return None
        return [point[0] + offset_x, point[1] + offset_y]

    if isinstance(translated.get("points"), list):
        points = []
        for point in translated["points"]:
            next_point = translate_point(point)
            if next_point is not None:
                points.append(next_point)
        translated["points"] = points
    for point_key in ("center", "centroid"):
        if point_key in translated:
            point = translate_point(translated.get(point_key))
            if point is not None:
                translated[point_key] = point
    if isinstance(translated.get("bbox"), list) and len(translated["bbox"]) == 4:
        try:
            left, top, right, bottom = [float(v) for v in translated["bbox"]]
            translated["bbox"] = [
                left + offset_x,
                top + offset_y,
                right + offset_x,
                bottom + offset_y,
            ]
        except (TypeError, ValueError):
            pass
    return translated


def mask_geometry_is_exact(geometry: dict[str, Any] | None) -> bool:
    return isinstance(geometry, dict) and str(geometry.get("source", "")).lower() == "exact"


def scale_mask_geometry(
    geometry: dict[str, Any] | None,
    scale_x: float,
    scale_y: float,
) -> dict[str, Any] | None:
    scaled = clone_mask_geometry(geometry)
    if scaled is None:
        return None
    try:
        sx = float(scale_x)
        sy = float(scale_y)
    except (TypeError, ValueError):
        return scaled
    if not math.isfinite(sx) or not math.isfinite(sy):
        return scaled
    radius_scale = (sx + sy) / 2.0

    def scale_point(value: Any) -> list[float] | None:
        point = _clean_point_pair(value)
        if point is None:
            return None
        return [point[0] * sx, point[1] * sy]

    if isinstance(scaled.get("points"), list):
        points = []
        for point in scaled["points"]:
            next_point = scale_point(point)
            if next_point is not None:
                points.append(next_point)
        scaled["points"] = points
    if "center" in scaled:
        center = scale_point(scaled.get("center"))
        if center is not None:
            scaled["center"] = center
    if isinstance(scaled.get("bbox"), list) and len(scaled["bbox"]) == 4:
        try:
            left, top, right, bottom = [float(v) for v in scaled["bbox"]]
            scaled["bbox"] = [left * sx, top * sy, right * sx, bottom * sy]
        except (TypeError, ValueError):
            pass
    for radius_key in ("base_radius", "adjusted_radius", "radius"):
        if radius_key in scaled:
            try:
                scaled[radius_key] = float(scaled[radius_key]) * radius_scale
            except (TypeError, ValueError):
                pass
    return scaled


def infer_mask_geometry(mask: np.ndarray) -> dict[str, Any] | None:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2 or not np.any(binary):
        return None
    height, width = binary.shape
    ys, xs = np.where(binary > 0)
    area = int(len(xs))
    bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    centroid = [float(xs.mean()), float(ys.mean())]
    component_count, _labels = cv2.connectedComponents(binary)
    components = max(0, int(component_count) - 1)
    points = cv2.findNonZero(binary)
    if points is None:
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    contour_area = float(cv2.contourArea(contour)) if contour is not None else float(area)
    perimeter = float(cv2.arcLength(contour, True)) if contour is not None else 0.0
    circularity = 0.0 if perimeter <= 0 else 4.0 * math.pi * float(area) / (perimeter * perimeter)
    (center_x, center_y), radius = cv2.minEnclosingCircle(points)
    radius = max(1.0, float(radius))
    ideal = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(ideal, (int(round(center_x)), int(round(center_y))), int(round(radius)), 1, -1)
    union = int(np.count_nonzero((binary > 0) | (ideal > 0)))
    intersection = int(np.count_nonzero((binary > 0) & (ideal > 0)))
    iou = 0.0 if union == 0 else intersection / float(union)
    ideal_area = max(1, int(np.count_nonzero(ideal)))
    fill_ratio = area / float(ideal_area)
    common = {
        "source": "inferred",
        "area": area,
        "bbox": bbox,
        "centroid": centroid,
        "components": components,
    }
    if circularity >= 0.82 and fill_ratio >= 0.72 and iou >= 0.72:
        geometry = circle_mask_geometry(
            (float(center_x), float(center_y)),
            radius,
            radius,
            source="inferred",
        )
        if geometry is None:
            return None
        geometry.update(common)
        geometry["circularity"] = float(circularity)
        return geometry

    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    try:
        ordered_box = order_quad_points(box.tolist()).tolist()
    except Exception:
        ordered_box = box.tolist()
    geometry = mask_polygon_geometry(ordered_box, source="inferred", shape="estimated_rectangle")
    if geometry is None:
        return None
    geometry.update(common)
    geometry["contour_area"] = contour_area
    return geometry


def mask_geometry_for_export(
    geometry: dict[str, Any] | None,
    mask: np.ndarray | None,
) -> dict[str, Any] | None:
    cloned = clone_mask_geometry(geometry)
    if cloned is not None:
        return cloned
    if mask is None:
        return None
    return infer_mask_geometry(mask)


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
    if pts.shape != (4, 2):
        raise ValueError("Four points are required.")
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    ordered = pts[np.argsort(angles)]
    start_index = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start_index, axis=0)
    return ordered.astype(np.float32)


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5


def validated_quad_points(
    points: list[tuple[float, float]],
    *,
    min_area: float = 10.0,
    min_point_distance: float = 2.0,
) -> np.ndarray | None:
    try:
        ordered = order_quad_points(points)
    except (TypeError, ValueError):
        return None
    if ordered.shape != (4, 2) or not np.all(np.isfinite(ordered)):
        return None
    for index in range(4):
        for other_index in range(index + 1, 4):
            distance = float(np.hypot(*(ordered[index] - ordered[other_index])))
            if distance < float(min_point_distance):
                return None
    if polygon_area(ordered) < float(min_area):
        return None
    contour = ordered.reshape(-1, 1, 2).astype(np.float32)
    try:
        if not cv2.isContourConvex(contour):
            return None
    except cv2.error:
        return None
    return ordered


def adjust_mask(mask: np.ndarray, margin: int) -> np.ndarray:
    margin = int(margin)
    if margin == 0 or not np.any(mask):
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(margin) * 2 + 1, abs(margin) * 2 + 1))
    if margin > 0:
        return cv2.dilate(mask.astype(np.uint8), kernel)
    return cv2.erode(mask.astype(np.uint8), kernel)


def smooth_binary_mask_low(mask: np.ndarray, *, max_area_change_ratio: float = 0.06) -> np.ndarray:
    """Return a lightly polished binary mask for committed brush/transform edits.

    The result is meant to become the actual mask used by display, export, and
    analysis. It therefore keeps the operation conservative: tiny masks are left
    alone, holes are preserved by contour hierarchy, and excessive area changes
    fall back to the safer candidate or the original binary mask.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2:
        return binary.copy()
    original_area = int(np.count_nonzero(binary))
    if original_area == 0 or min(binary.shape) < 3 or original_area < 24:
        return binary.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_candidate = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    morph_candidate = cv2.morphologyEx(morph_candidate, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(
        morph_candidate,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE,
    )
    if hierarchy is None or not contours:
        return morph_candidate.astype(np.uint8)

    approximated: list[np.ndarray] = []
    for contour in contours:
        if len(contour) < 3:
            approximated.append(contour)
            continue
        perimeter = cv2.arcLength(contour, True)
        epsilon = min(1.15, max(0.45, perimeter * 0.0025))
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        approximated.append(simplified if len(simplified) >= 3 else contour)

    hierarchy_rows = hierarchy[0]
    depths: list[int] = []
    for index, row in enumerate(hierarchy_rows):
        depth = 0
        parent = int(row[3])
        guard = 0
        while parent != -1 and guard < len(hierarchy_rows):
            depth += 1
            parent = int(hierarchy_rows[parent][3])
            guard += 1
        depths.append(depth)

    smoothed = np.zeros_like(binary)
    for contour_index in sorted(range(len(approximated)), key=lambda i: depths[i]):
        fill_value = 1 if depths[contour_index] % 2 == 0 else 0
        cv2.drawContours(smoothed, approximated, contour_index, fill_value, cv2.FILLED)
    smoothed = (smoothed > 0).astype(np.uint8)

    smoothed_area = int(np.count_nonzero(smoothed))
    allowed_delta = max(12, int(round(original_area * float(max_area_change_ratio))))
    if smoothed_area > 0 and abs(smoothed_area - original_area) <= allowed_delta:
        return smoothed

    morph_area = int(np.count_nonzero(morph_candidate))
    if morph_area > 0 and abs(morph_area - original_area) <= allowed_delta:
        return morph_candidate.astype(np.uint8)
    return binary.copy()


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
