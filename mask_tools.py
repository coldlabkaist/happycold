from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared import fill_circle_from_diameter, fill_polygon, paint_brush, validated_quad_points


@dataclass(frozen=True)
class DrawShapePayload:
    points: list[tuple[float, float]]
    add: bool
    shape_kind: str


def parse_draw_shape_payload(
    payload: object,
    *,
    default_add: bool,
    default_shape: str = "rectangle",
) -> DrawShapePayload | None:
    add = bool(default_add)
    shape_kind = default_shape
    points: list[tuple[float, float]]

    if isinstance(payload, tuple) and len(payload) >= 2 and isinstance(payload[0], list):
        points = payload[0]
        if isinstance(payload[1], bool):
            add = payload[1]
        if len(payload) >= 3:
            shape_kind = str(payload[2])
    elif isinstance(payload, list):
        points = payload
    else:
        return None
    return DrawShapePayload(points=points, add=add, shape_kind=shape_kind)


def resolve_polygon_points(
    points: list[tuple[float, float]],
    *,
    shape_kind: str,
) -> tuple[list[tuple[float, float]], str] | None:
    if shape_kind == "polygon":
        if len(points) < 3:
            return None
        return [(float(x), float(y)) for x, y in points], "polygon"

    ordered = validated_quad_points(points) if len(points) == 4 else None
    if ordered is None:
        return None
    return ordered.tolist(), "rectangle"


def build_polygon_mask(
    target_mask: np.ndarray,
    points: list[tuple[float, float]],
    *,
    shape_kind: str,
) -> tuple[np.ndarray, list[tuple[float, float]], str] | None:
    resolved = resolve_polygon_points(points, shape_kind=shape_kind)
    if resolved is None:
        return None
    ordered_points, geometry_shape = resolved
    shape_mask = np.zeros_like(target_mask, dtype=np.uint8)
    fill_polygon(shape_mask, ordered_points, 1)
    return shape_mask, ordered_points, geometry_shape


def build_circle_mask(
    target_mask: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
) -> np.ndarray:
    shape_mask = np.zeros_like(target_mask, dtype=np.uint8)
    fill_circle_from_diameter(shape_mask, start, end, 1)
    return shape_mask


def merge_mask(target_mask: np.ndarray, shape_mask: np.ndarray, *, add: bool) -> np.ndarray:
    target_bool = target_mask.astype(bool)
    shape_bool = shape_mask.astype(bool)
    if add:
        return np.logical_or(target_bool, shape_bool).astype(np.uint8)
    return np.logical_and(target_bool, ~shape_bool).astype(np.uint8)


def paint_brush_segment(
    target_mask: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    radius: int,
    add: bool,
) -> None:
    paint_brush(target_mask, start, end, max(1, int(radius)), 1 if add else 0)


def constrained_rectangle_points(start, end, *, square=False, bounds=None):
    """Image-space corners shared by drag preview and commit."""
    x, y = map(float, start)
    dx, dy = float(end[0]) - x, float(end[1]) - y
    if square:
        side = max(abs(dx), abs(dy))
        sx, sy = (-1 if dx < 0 else 1), (-1 if dy < 0 else 1)
        if bounds:
            width, height = bounds
            side = max(0.0, min(side, x if sx < 0 else width - 1 - x,
                                y if sy < 0 else height - 1 - y))
        dx, dy = sx * side, sy * side
    elif bounds:
        dx = max(0, min(bounds[0] - 1, x + dx)) - x
        dy = max(0, min(bounds[1] - 1, y + dy)) - y
    left, right = sorted((x, x + dx))
    top, bottom = sorted((y, y + dy))
    return [(left, top), (right, top), (right, bottom), (left, bottom)]
