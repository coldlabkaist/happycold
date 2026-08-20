import numpy as np

from mask_tools import (
    build_circle_mask,
    build_polygon_mask,
    merge_mask,
    paint_brush_segment,
    parse_draw_shape_payload,
)


def test_parse_draw_shape_payload_uses_defaults_for_plain_points():
    parsed = parse_draw_shape_payload([(1, 2), (3, 4)], default_add=False)

    assert parsed is not None
    assert parsed.points == [(1, 2), (3, 4)]
    assert parsed.add is False
    assert parsed.shape_kind == "rectangle"


def test_polygon_mask_can_add_and_erase_hole():
    target = np.zeros((20, 20), dtype=np.uint8)
    outer = build_polygon_mask(target, [(2, 2), (17, 2), (17, 17), (2, 17)], shape_kind="rectangle")
    inner = build_polygon_mask(target, [(7, 7), (12, 7), (12, 12), (7, 12)], shape_kind="rectangle")

    assert outer is not None
    assert inner is not None

    filled = merge_mask(target, outer[0], add=True)
    erased = merge_mask(filled, inner[0], add=False)

    assert erased[4, 4] == 1
    assert erased[9, 9] == 0


def test_circle_mask_and_brush_segment_write_pixels():
    target = np.zeros((20, 20), dtype=np.uint8)
    circle = build_circle_mask(target, (4, 10), (16, 10))

    assert circle[10, 10] == 1

    paint_brush_segment(target, (2, 2), (17, 17), radius=2, add=True)

    assert target[2, 2] == 1
    assert target[10, 10] == 1
