"""Forward rays, angular masks, read-only UI, and real direction CSV exports."""
from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cv2
import numpy as np
import pandas as pd
from PyQt6.QtCore import QEvent, QEventLoop, QTimer, Qt
from PyQt6.QtGui import QColor, QMouseEvent
from PyQt6.QtWidgets import QApplication, QMessageBox

import main
ORIGINAL_SAVE_SETTINGS = main.MainWindow._save_settings
ORIGINAL_LOAD_SETTINGS = main.MainWindow._load_settings
from batch import ExportCancelled
from direction import (
    DirectionConfig, DirectionMask, DirectionRanges, DirectionTarget, DirectionZones, build_direction_dataframe, direction_mask_from_object,
    direction_label, export_direction_masks, heading_points, load_direction_masks, prepare_direction_pose,
)
from mask_io import read_mask_png, write_mask_png
from occlusion_3d import LengthReference, ObjectFace, ObjectWorkspace, OcclusionObject, apply_margin, build_envelope, save_bundle
from shared import MaskRecord, adjust_mask_by_mode


def rect(x0, y0, x1, y1, shape=(120, 200)):
    result = np.zeros(shape, np.uint8)
    result[y0:y1 + 1, x0:x1 + 1] = 1
    return result


class DirectionGeometryTests(unittest.TestCase):
    def classify(self, mask, nose=(20, 30), origin=(10, 30), angle=0):
        return DirectionTarget(mask).classify(np.array([origin], float), np.array([nose], float), np.array([True]), angle)[0]

    def test_zero_is_forward_ray_and_does_not_include_heading_baseline_or_behind(self):
        self.assertEqual(self.classify(rect(40, 29, 45, 31)), 1)
        self.assertEqual(self.classify(rect(0, 29, 5, 31)), 0)
        self.assertEqual(self.classify(rect(14, 29, 16, 31)), 0)
        self.assertEqual(self.classify(rect(40, 40, 45, 42)), 0)

    def test_zero_hits_single_pixel_cells_and_inclusive_boundary(self):
        pixel = rect(40, 30, 40, 30)
        self.assertEqual(self.classify(pixel, nose=(20, 30.4), origin=(10, 30.4)), 1)
        self.assertEqual(self.classify(pixel, nose=(20, 30.5), origin=(10, 30.5)), 1)
        self.assertEqual(self.classify(pixel, nose=(20, 30.6), origin=(10, 30.6)), 0)
        self.assertEqual(self.classify(pixel, nose=(20, 10), origin=(10, 0)), 1)

    def test_angle_boundary_and_objects_without_a_center_in_cone(self):
        pixel = rect(40, 40, 40, 40)
        boundary = math.degrees(math.atan2(9.5, 20.5))
        self.assertEqual(self.classify(pixel, angle=boundary - 0.001), 0)
        self.assertEqual(self.classify(pixel, angle=boundary), 1)
        self.assertEqual(self.classify(rect(50, 31, 90, 90), angle=15), 1)
        # A cone can cross a rectangle even when none of its corners is inside it.
        self.assertEqual(self.classify(rect(40, 0, 50, 70), angle=5), 1)

    def test_vertical_and_ninety_degree_half_plane(self):
        self.assertEqual(self.classify(rect(19, 70, 21, 75), origin=(20, 10)), 1)
        self.assertEqual(self.classify(rect(20, 80, 21, 85), angle=90), 1)
        self.assertEqual(self.classify(rect(0, 80, 10, 85), angle=90), 0)

    def test_holes_and_disconnected_regions_are_not_convexified(self):
        ring = rect(50, 50, 90, 90)
        ring[60:81, 60:81] = 0
        self.assertEqual(self.classify(ring, nose=(70, 70), origin=(65, 70)), 1)
        separate = rect(40, 20, 40, 20) | rect(40, 40, 40, 40)
        self.assertEqual(self.classify(separate), 0)
        self.assertEqual(self.classify(separate, angle=30), 1)

    def test_random_masks_match_independent_pixel_polygon_clipping(self):
        rng = np.random.default_rng(721)

        def clip(polygon, normal):
            if not polygon:
                return []
            result = []
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                da, db = np.dot(a, normal), np.dot(b, normal)
                if da >= -1e-9:
                    result.append(a)
                if (da >= -1e-9) != (db >= -1e-9):
                    result.append(a + (b - a) * da / (da - db))
            return result

        for _ in range(100):
            mask = (rng.random((10, 12)) < 0.1).astype(np.uint8)
            mask[0, 0] = 1
            nose = rng.uniform((1, 1), (10, 8))
            heading = rng.uniform(-math.pi, math.pi)
            direction = np.array([math.cos(heading), math.sin(heading)])
            angle = float(rng.choice([0, 1, 15, 45, 90]))
            expected, inside = False, False
            cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            for y, x in np.argwhere(mask):
                inside |= x - .5 <= nose[0] <= x + .5 and y - .5 <= nose[1] <= y + .5
                corners = np.array([[x - .5, y - .5], [x + .5, y - .5], [x + .5, y + .5], [x - .5, y + .5]]) - nose
                local = [np.array([p @ direction, p[1] * direction[0] - p[0] * direction[1]]) for p in corners]
                for normal in (np.array([1, 0]), np.array([sine, cosine]), np.array([sine, -cosine])):
                    local = clip(local, normal)
                expected |= bool(local)
            actual = self.classify(mask, nose, nose - direction, angle)
            if inside:
                self.assertTrue(np.isnan(actual))
            else:
                self.assertEqual(actual, int(expected))

    def test_inside_ray_origin_is_reserved_for_occlusion_status_layer(self):
        self.assertTrue(np.isnan(self.classify(rect(15, 25, 25, 35))))

    def test_selected_method_and_invalid_points_preserve_original_rows(self):
        df = pd.DataFrame({"frame": [1, 1, 3, 4], "instance.id": ["a", "b", "a", "a"],
            "nose.x": [60, 60, np.nan, 60], "nose.y": [60] * 4,
            "neck.x": [40, 80, 40, 60], "neck.y": [60] * 4,
            "body_c.x": [80] * 4, "body_c.y": [60] * 4,
            "ear_l.x": [40] * 4, "ear_l.y": [50] * 4,
            "ear_r.x": [40] * 4, "ear_r.y": [70] * 4})
        config = DirectionConfig(method="neck", anchor="neck", angle=0, coordinates="pixels")
        masks = [DirectionMask("A", rect(90, 55, 100, 65)), DirectionMask("Behind", rect(10, 55, 20, 65))]
        for item in masks:
            item.ranges.near_margin = 100
        result = build_direction_dataframe(df, config, masks, 200, 120)
        pd.testing.assert_frame_equal(result[df.columns], df)
        self.assertEqual(result["A.direction"].tolist(), ["NearFacing", "NearNotFacing", "Unknown", "Unknown"])
        self.assertEqual(result["A.direction_reason"].iloc[2], "missing_nose")
        self.assertIn("zero_length_heading", result["A.direction_reason"].iloc[3])
        self.assertEqual(result["Behind.direction"].iloc[:2].tolist(), ["NearNotFacing", "NearFacing"])
        body = build_direction_dataframe(df.iloc[:1], replace(config, method="body", anchor="body_c"), masks, 200, 120)
        self.assertEqual(body["A.direction"].iloc[0], "NearNotFacing")

    def test_normalized_coordinates_are_converted_before_angles(self):
        df = pd.DataFrame({"nose.x": [.2], "nose.y": [.3], "neck.x": [.1], "neck.y": [.1]})
        config = DirectionConfig(method="neck", angle=0, coordinates="normalized")
        mask = DirectionMask("Diagonal", rect(70, 60, 71, 61, shape=(100, 200)))
        mask.ranges.near_margin = 60
        self.assertEqual(build_direction_dataframe(df, config, [mask], 200, 100)["Diagonal.direction"].iloc[0], "NearFacing")
        self.assertEqual(build_direction_dataframe(df, replace(config, coordinates="auto"), [mask], 200, 100)["Diagonal.direction"].iloc[0], "NearFacing")

    def test_distance_node_does_not_change_when_heading_method_changes(self):
        df = pd.DataFrame({"nose.x": [60], "nose.y": [60], "neck.x": [40], "neck.y": [60],
            "body_c.x": [20], "body_c.y": [60], "ear_l.x": [40], "ear_l.y": [50],
            "ear_r.x": [40], "ear_r.y": [70]})
        mask = DirectionMask("Object", rect(19, 59, 21, 61))
        for method, anchor in (("ears", "neck"), ("neck", "neck"), ("body", "body_c")):
            config = DirectionConfig(method=method, anchor=anchor, distance_node="body_c", coordinates="pixels")
            result = build_direction_dataframe(df, config, [mask], 200, 120)
            self.assertEqual(result["Object.direction"].iloc[0], "Overlap")
            self.assertEqual(result["Object.direction_reason"].iloc[0], "body_c_inside_object_zone")
        # A heading node inside the object is not the chosen distance point.
        result = build_direction_dataframe(df, replace(config, distance_node="nose"), [mask], 200, 120)
        self.assertEqual(result["Object.direction"].iloc[0], "Far")

    def test_distance_priority_and_unknown_reasons_preserve_input(self):
        df = pd.DataFrame({"nose.x": [60, np.nan, 210, 60, np.inf], "nose.y": [60] * 5,
                           "neck.x": [40, 40, 40, 60, 40], "neck.y": [60] * 5})
        original = df.copy()
        config = DirectionConfig(method="neck", angle=0, coordinates="pixels")
        masks = [DirectionMask("Object", rect(55, 55, 65, 65)), DirectionMask("Ahead", rect(90, 55, 100, 65))]
        masks[1].ranges.near_margin = 40
        result = build_direction_dataframe(df, config, masks, 200, 120)
        self.assertEqual(result["Object.direction"].tolist(), ["Overlap", "Unknown", "Unknown", "Overlap", "Unknown"])
        self.assertEqual(result["Object.direction_reason"].tolist(), ["nose_inside_object_zone", "missing_nose", "nose_outside_frame", "nose_inside_object_zone", "invalid_nose"])
        self.assertEqual(result["Ahead.direction"].tolist(), ["NearFacing", "Unknown", "Unknown", "Unknown", "Unknown"])
        self.assertIn("zero_length_heading", result["Ahead.direction_reason"].iloc[3])
        pd.testing.assert_frame_equal(result[original.columns], original)
        pd.testing.assert_frame_equal(df, original)

    def test_missing_coordinates_take_priority_over_known_node_overlap(self):
        df = pd.DataFrame({"nose.x": [np.nan], "nose.y": [60], "neck.x": [40], "neck.y": [60]})
        mask = DirectionMask("Object", rect(39, 59, 41, 61))
        result = build_direction_dataframe(df, DirectionConfig(method="neck", coordinates="pixels"), [mask], 200, 120)
        self.assertEqual(result["Object.direction"].iloc[0], "Unknown")
        self.assertEqual(result["Object.direction_reason"].iloc[0], "missing_nose")

    def test_readable_labels_and_normalized_overlap(self):
        self.assertEqual(direction_label("Overlap", "nose_inside_object_zone"), "Overlap (nose)")
        self.assertEqual(direction_label("NearFacing", "nose_inside_near_zone;heading_intersects_object_zone"), "Near / Facing")
        self.assertEqual(direction_label("Unknown", "missing_nose"), "Unknown (nose missing)")
        df = pd.DataFrame({"nose.x": [.3], "nose.y": [.5], "neck.x": [.2], "neck.y": [.5]})
        mask = DirectionMask("Object", rect(55, 55, 65, 65))
        result = build_direction_dataframe(df, DirectionConfig(method="neck", coordinates="normalized"), [mask], 200, 120)
        self.assertEqual(result["Object.direction"].iloc[0], "Overlap")
        self.assertEqual(result["Object.direction_reason"].iloc[0], "nose_inside_object_zone")

    def test_validation_and_cancellation(self):
        for angle in (-1, 91, np.nan):
            with self.assertRaises(ValueError):
                DirectionConfig(angle=angle).required_bodyparts()
        with self.assertRaises(ValueError):
            DirectionConfig(method="neck", nose="nose", anchor="nose").required_bodyparts()
        with self.assertRaises(ExportCancelled):
            DirectionTarget(rect(40, 20, 50, 40)).classify(np.array([[10, 30.]]), np.array([[20, 30.]]), np.array([True]), 0, lambda: True)

    def test_folder_source_margin_is_ignored_and_direction_ranges_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            raw = rect(40, 40, 50, 50)
            write_mask_png(folder / "Box.png", raw * 255)
            (folder / "masks_manifest.json").write_text(json.dumps({"masks": {"Box": {"margin": 15}}}))
            loaded = load_direction_masks(folder, 200, 120)
            np.testing.assert_array_equal(loaded[0].mask, raw)
            loaded[0].ranges.object_margin, loaded[0].ranges.near_margin = 2, 5
            expected = loaded[0].regions()
            export_direction_masks(loaded, folder, DirectionConfig(distance_node="neck"))
            np.testing.assert_array_equal(read_mask_png(folder / "Box.png") > 0, raw > 0)
            restored = load_direction_masks(folder, 200, 120)
            np.testing.assert_array_equal(restored[0].regions()[0], expected[0])
            np.testing.assert_array_equal(restored[0].regions()[1], expected[1])

    def test_source_geometric_margin_is_ignored_but_direction_can_use_its_points(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            raw = rect(40, 40, 50, 50)
            write_mask_png(folder / "Box.png", raw * 255)
            metadata = {"masks": {"Box": {"margin": 30, "margin_mode": "geometric"}}}
            path = folder / "masks_manifest.json"
            path.write_text(json.dumps(metadata))
            item = load_direction_masks(folder, 200, 120)[0]
            np.testing.assert_array_equal(item.mask, raw)
            item.ranges = DirectionRanges(2, 4, "geometric")
            with self.assertRaises(ValueError):
                item.regions()
            quad = [[10, 10], [190, 10], [190, 110], [10, 110]]
            item.ranges.quad_points = quad
            np.testing.assert_array_equal(item.regions()[0], adjust_mask_by_mode(raw, 2, "geometric", quad))

    def test_unicode_and_video_scoped_masks_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "한글"
            mask = DirectionMask("물체", rect(40, 40, 50, 50), video_key="video-a")
            export_direction_masks([mask], folder, DirectionConfig(angle=0))
            loaded = load_direction_masks(folder, 200, 120)
            self.assertEqual(loaded[0].name, mask.name)
            self.assertEqual(loaded[0].video_key, "video-a")
            np.testing.assert_array_equal(loaded[0].mask, mask.mask)
            resized = load_direction_masks(folder, 400, 240, "video-b")[0]
            self.assertEqual(resized.mask.shape, (240, 400))
            self.assertEqual(resized.video_key, "video-b")


class DirectionRangeTests(unittest.TestCase):
    def test_distance_priority_missing_heading_and_all_four_states(self):
        df = pd.DataFrame({"nose.x": [np.nan, np.nan, 60, 60, np.nan, 60, 95, 60],
            "nose.y": [60] * 8, "neck.x": [40, 40, 40, 80, 40, 40, 80, 60],
            "neck.y": [60] * 8, "body_c.x": [95, 20, 70, 70, 70, np.nan, 70, 70],
            "body_c.y": [60] * 8})
        original = df.copy()
        mask = DirectionMask("Object", rect(90, 55, 100, 65), ranges=DirectionRanges(0, 40))
        cfg = DirectionConfig(method="neck", angle=0, coordinates="pixels", distance_node="body_c")
        result = build_direction_dataframe(df, cfg, [mask], 200, 120)
        self.assertEqual(result["Object.direction"].tolist(),
            ["Overlap", "Far", "NearFacing", "NearNotFacing", "Unknown", "Unknown", "Unknown", "Unknown"])
        self.assertEqual(result["Object.direction_reason"].tolist(), [
            "body_c_inside_object_zone", "body_c_outside_near_zone",
            "body_c_inside_near_zone;heading_intersects_object_zone",
            "body_c_inside_near_zone;heading_misses_object_zone",
            "body_c_inside_near_zone;missing_nose", "missing_body_c",
            "body_c_inside_near_zone;nose_inside_object_zone", "body_c_inside_near_zone;zero_length_heading"])
        pd.testing.assert_frame_equal(result[original.columns], original)
        pd.testing.assert_frame_equal(df, original)

        renamed = df.rename(columns={"nose.x": "snout.x", "nose.y": "snout.y"})
        result = build_direction_dataframe(renamed, replace(cfg, nose="snout"), [mask], 200, 120)
        self.assertEqual(result["Object.direction_reason"].iloc[6], "body_c_inside_near_zone;snout_inside_object_zone")

    def test_heading_tests_object_not_the_near_ring_and_zero_is_supported(self):
        df = pd.DataFrame({"nose.x": [60], "nose.y": [30], "neck.x": [40], "neck.y": [30],
                           "body_c.x": [70], "body_c.y": [60]})
        mask = DirectionMask("Object", rect(90, 55, 100, 65), ranges=DirectionRanges(0, 40))
        cfg = DirectionConfig(method="neck", angle=0, coordinates="pixels", distance_node="body_c")
        self.assertEqual(build_direction_dataframe(df, cfg, [mask], 200, 120)["Object.direction"].iloc[0], "NearNotFacing")
        self.assertEqual(build_direction_dataframe(df, replace(cfg, angle=45), [mask], 200, 120)["Object.direction"].iloc[0], "NearFacing")

    def test_both_margins_start_from_raw_shape_and_can_be_negative(self):
        raw = rect(40, 40, 60, 60)
        original = raw.copy()
        mask = DirectionMask("Object", raw, ranges=DirectionRanges(2, 5))
        inner, outer = mask.regions()
        self.assertEqual(np.where(inner)[1].min(), 38)
        self.assertEqual(np.where(inner)[1].max(), 62)
        self.assertEqual(np.where(outer)[1].min(), 35)
        self.assertEqual(np.where(outer)[1].max(), 65)
        mask.ranges = DirectionRanges(-2, -1)
        inner, outer = mask.regions()
        self.assertEqual(np.where(inner)[1].min(), 42)
        self.assertEqual(np.where(outer)[1].min(), 41)
        np.testing.assert_array_equal(raw, original)
        mask.ranges = DirectionRanges(2, 1)
        with self.assertRaisesRegex(ValueError, "at least"):
            mask.regions()
        mask.ranges = DirectionRanges(-30, 0)
        with self.assertRaisesRegex(ValueError, "entire object"):
            mask.regions()

    def test_equal_margins_and_pixel_cell_boundary_priority(self):
        df = pd.DataFrame({"nose.x": [39.5, 39.49, 60.5, 60.51], "nose.y": [50] * 4,
                           "neck.x": [30] * 4, "neck.y": [50] * 4})
        mask = DirectionMask("Object", rect(40, 40, 60, 60), ranges=DirectionRanges(0, 0))
        result = build_direction_dataframe(df, DirectionConfig(method="neck", coordinates="pixels"), [mask], 200, 120)
        self.assertEqual(result["Object.direction"].tolist(), ["Overlap", "Far", "Overlap", "Far"])

    def make_prism(self):
        top_ref = LengthReference("Top ruler", .5, "m", [(10, 10), (60, 10)])
        bottom_ref = LengthReference("Bottom ruler", 50, "cm", [(10, 105), (110, 105)])
        obj = OcclusionObject("Prism", ObjectFace(rect(40, 20, 60, 40), 50, "cm", top_ref.id),
            ObjectFace(rect(80, 70, 100, 90), -50, "cm", bottom_ref.id))
        return obj, [top_ref, bottom_ref]

    def test_3d_raw_faces_ignore_source_margins_and_use_distinct_references(self):
        obj, refs = self.make_prism()
        item = direction_mask_from_object(obj, refs, "video-a")
        self.assertEqual(np.where(item.mask)[1].min(), 40)
        self.assertEqual(np.where(item.mask)[1].max(), 100)
        item.ranges.mode, item.ranges.object_margin, item.ranges.near_margin = "physical", 1, 3
        inner, outer = item.regions()
        # 1 cm = 1 px at the top and 2 px at the bottom.
        self.assertEqual((np.where(inner)[1].min(), np.where(inner)[1].max()), (39, 102))
        self.assertEqual((np.where(outer)[1].min(), np.where(outer)[1].max()), (37, 106))
        self.assertEqual((np.where(inner)[0].min(), np.where(inner)[0].max()), (19, 92))
        np.testing.assert_array_equal(item.top, obj.top.mask)
        np.testing.assert_array_equal(item.bottom, obj.bottom.mask)
        item.references[0].length = 1
        self.assertEqual(refs[0].length, .5)

    def test_3d_folder_and_direction_export_round_trip_raw_faces_and_ranges(self):
        obj, refs = self.make_prism()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            save_bundle(ObjectWorkspace(200, 120, refs, [obj], "video-a"), folder)
            # Source negative margin erases its bottom, so no final PNG exists.
            self.assertFalse((folder / "Prism.png").exists())
            item = load_direction_masks(folder, 200, 120, "video-a")[0]
            np.testing.assert_array_equal(item.top, obj.top.mask)
            self.assertEqual(item.ranges.object_margin, 0)
            item.ranges.mode, item.ranges.object_margin, item.ranges.near_margin = "physical", 1, 3
            expected = item.regions()
            export_direction_masks([item], folder, DirectionConfig(distance_node="body_c"))
            restored = load_direction_masks(folder / "Prism.png", 200, 120, "video-a")[0]
            self.assertEqual(restored.ranges.bottom_reference_id, refs[1].id)
            np.testing.assert_array_equal(restored.bottom, obj.bottom.mask)
            np.testing.assert_array_equal(restored.regions()[0], expected[0])
            np.testing.assert_array_equal(restored.regions()[1], expected[1])
            np.testing.assert_array_equal(read_mask_png(folder / "Prism.png") > 0, item.mask > 0)

    def test_single_final_3d_png_recovers_raw_recipe(self):
        obj, refs = self.make_prism()
        obj.top.margin = obj.bottom.margin = 2
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            save_bundle(ObjectWorkspace(200, 120, refs, [obj], "video-a"), folder)
            loaded = load_direction_masks(folder / "Prism.png", 200, 120, "video-a")[0]
            np.testing.assert_array_equal(loaded.mask, build_envelope(obj.top.mask, obj.bottom.mask)[0])
            self.assertFalse(np.array_equal(loaded.mask > 0, read_mask_png(folder / "Prism.png") > 0))

    def test_legacy_baked_export_is_rejected_and_raw_face_paths_stay_in_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_mask_png(folder / "Object.png", rect(40, 40, 60, 60) * 255)
            path = folder / "masks_manifest.json"
            path.write_text(json.dumps({"masks": {"Object": {}}, "direction_settings": {"angle": 0}}))
            with self.assertRaisesRegex(ValueError, "baked margins"):
                load_direction_masks(folder, 200, 120)
            obj, refs = self.make_prism()
            item = direction_mask_from_object(obj, refs, "video-a")
            export_direction_masks([item], folder, DirectionConfig())
            data = json.loads(path.read_text())
            data["masks"]["Prism"]["direction_recipe"]["top"] = "../outside.png"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "inside the mask folder"):
                load_direction_masks(folder / "Prism.png", 200, 120)

    def test_import_never_reuses_other_video_or_resolution_measurements(self):
        ref = LengthReference("Ruler", 1, "m", [(10, 10), (110, 10)])
        item = DirectionMask("Object", rect(40, 40, 60, 60), references=[ref])
        item.ranges.reference_id = ref.id
        item.ranges.calibration_video_key = "video-a"
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            export_direction_masks([item], folder, DirectionConfig())
            same = load_direction_masks(folder, 200, 120, "video-a")[0]
            self.assertEqual(same.references[0].points, ref.points)
            # Invalidate even in Pixels mode, before a later switch to Physical.
            for width, height, key in ((200, 120, "video-b"), (400, 240, "video-a")):
                loaded = load_direction_masks(folder, width, height, key)[0]
                self.assertEqual(loaded.references[0].points, [])
                loaded.ranges.mode = "physical"
                with self.assertRaisesRegex(ValueError, "measure"):
                    loaded.regions()

    def test_3d_direction_recipe_import_resizes_faces_and_keeps_settings_for_new_video(self):
        obj, refs = self.make_prism()
        item = direction_mask_from_object(obj, refs, "video-a")
        item.ranges.mode, item.ranges.object_margin, item.ranges.near_margin = "physical", 1, 3
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            export_direction_masks([item], folder, DirectionConfig())
            loaded = load_direction_masks(folder, 400, 240, "video-b")[0]
            self.assertEqual(loaded.video_key, "video-b")
            self.assertEqual(loaded.top.shape, (240, 400))
            self.assertEqual(loaded.bottom.shape, (240, 400))
            self.assertEqual((loaded.ranges.mode, loaded.ranges.object_margin, loaded.ranges.near_margin), ("physical", 1, 3))
            self.assertEqual(loaded.ranges.top_reference_id, refs[0].id)
            self.assertEqual(loaded.ranges.bottom_reference_id, refs[1].id)
            self.assertTrue(all(not r.points for r in loaded.references))
            self.assertIsNone(loaded.ranges.calibration_video_key)
            with self.assertRaisesRegex(ValueError, "measure"):
                loaded.regions()


class DirectionQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for patcher in (patch.object(main.MainWindow, "_load_settings", lambda _: {}),
                        patch.object(main.MainWindow, "_save_settings", lambda _: None),
                        patch.object(main, "DEFAULT_SAVE_DIR", self.root / "output"),
                        patch.object(QMessageBox, "warning"), patch.object(QMessageBox, "information")):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.w = main.MainWindow()
        self.addCleanup(self.close)
        self.video = self.make_video("a")
        self.w.load_video(self.video)
        self.w._open_pipeline_settings(self.w.TAB_DIRECTION)
        self.w.direction_coordinates.setCurrentIndex(self.w.direction_coordinates.findData("pixels"))
        self.w.mask_records = {"Object": MaskRecord("Object", QColor("#06b6d4"), rect(90, 55, 100, 65))}
        self.w._direction_load_masks()
        self.w.direction_near_margin.setValue(40)
        self.w.direction_angle.setValue(0)

    def make_video(self, name, width=200, height=120):
        path = self.root / f"{name}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (width, height))
        self.assertTrue(writer.isOpened())
        for _ in range(3):
            writer.write(np.full((height, width, 3), 50, np.uint8))
        writer.release()
        scale_x, scale_y = width / 200, height / 120
        df = pd.DataFrame({"frame": [1, 2, 3], "nose.x": [60, 60, np.nan], "nose.y": [60] * 3,
            "ear_l.x": [40, 80, 40], "ear_l.y": [50] * 3,
            "ear_r.x": [40, 80, 40], "ear_r.y": [70] * 3,
            "neck.x": [40, 80, 40], "neck.y": [60] * 3,
            "body_c.x": [20, 100, 20], "body_c.y": [60] * 3, "confidence": [.9] * 3})
        for col in df:
            if col.endswith(".x"):
                df[col] *= scale_x
            elif col.endswith(".y"):
                df[col] *= scale_y
        df.to_csv(path.with_suffix(".csv"), index=False)
        return path

    def wait_save(self):
        if not self.w._save_job_is_running():
            return
        loop = QEventLoop()
        timer = QTimer()
        timer.setInterval(5)
        timer.timeout.connect(lambda: None if self.w._save_job_is_running() else loop.quit())
        timer.start()
        QTimer.singleShot(15000, loop.quit)
        loop.exec()
        timer.stop()
        self.assertFalse(self.w._save_job_is_running())

    def close(self):
        self.wait_save()
        for name in ("_active_single_save_dialog", "_active_batch_dialog"):
            dialog = getattr(self.w, name, None)
            if dialog:
                dialog.close()
        self.w.close()
        self.w.deleteLater()
        self.app.processEvents()

    def test_real_csv_zero_angle_and_output_naming(self):
        w = self.w
        self.assertEqual(w._direction_output_path().name, "a_direction.csv")
        self.assertTrue(w.save_current_button.isEnabled())
        self.assertEqual(w.direction_angle_label.text(), "Forward ray from nose")
        w.current_output_prefix_edit.setText("test")
        w.current_output_suffix_edit.setText("checked")
        output = w._direction_output_path()
        self.assertEqual(output.name, "test_a_direction_checked.csv")
        original = w.csv_df.copy()
        w.save_current_mode_output()
        self.wait_save()
        result = pd.read_csv(output)
        pd.testing.assert_frame_equal(result[original.columns], original)
        self.assertEqual(result["Object.direction"].iloc[:2].tolist(), ["NearFacing", "NearNotFacing"])
        self.assertEqual(result["Object.direction"].iloc[2], "Unknown")
        self.assertEqual(result["Object.direction_reason"].iloc[2], "missing_nose")

    def test_method_specific_controls_and_frame_preview(self):
        w = self.w
        self.assertEqual(w.direction_result.text(), "Object: Near / Facing")
        self.assertFalse(w.direction_point_controls["left_ear"][0].isHidden())
        self.assertTrue(w.direction_point_controls["anchor"][0].isHidden())
        w.direction_method.setCurrentIndex(w.direction_method.findData("neck"))
        self.assertTrue(w.direction_point_controls["left_ear"][0].isHidden())
        self.assertFalse(w.direction_point_controls["anchor"][0].isHidden())
        w.frame_slider.setValue(2)
        self.assertEqual(w.direction_result.text(), "Object: Near / Not Facing")
        w.frame_slider.setValue(3)
        self.assertEqual(w.direction_result.text(), "Object: Unknown (nose missing)")
        self.assertIsNone(w.frame_viewer.direction_preview)

    def test_occluded_preview_and_saved_reasons_agree(self):
        w = self.w
        w.mask_records["Object"].mask = rect(55, 55, 65, 65)
        w._direction_load_masks()
        self.assertEqual(w.direction_result.text(), "Object: Overlap (nose)")
        self.assertEqual(w.direction_objects.item(0).text(), "Object: Overlap (nose)")
        self.assertEqual(w.frame_viewer.mask_records["Object"].color.name(), "#f59e0b")
        w.save_current_mode_output()
        self.wait_save()
        result = pd.read_csv(w._direction_output_path())
        self.assertEqual(result["Object.direction"].tolist(), ["Overlap", "Overlap", "Unknown"])
        self.assertEqual(result["Object.direction_reason"].tolist(), ["nose_inside_object_zone", "nose_inside_object_zone", "missing_nose"])

    def test_distance_selection_stays_independent_of_heading_method(self):
        w = self.w
        w.direction_distance_node.setCurrentIndex(w.direction_distance_node.findData("body_c"))
        self.assertEqual(w.direction_result.text(), "Object: Far")
        w.direction_method.setCurrentIndex(w.direction_method.findData("neck"))
        self.assertEqual(w.direction_distance_node.currentData(), "body_c")
        self.assertEqual(w.direction_result.text(), "Object: Far")

    def test_frame_without_csv_row_has_explicit_unknown_reason(self):
        w = self.w
        w.csv_df.iloc[:1].to_csv(w.csv_path, index=False)
        w.load_csv(w.csv_path, False)
        w.frame_slider.setValue(2)
        self.assertEqual(w.direction_result.text(), "Object: Unknown (no frame data)")

    def test_source_margin_is_not_imported_and_direction_ranges_are_independent(self):
        w = self.w
        source = w.mask_records["Object"]
        source.margin = 25
        w._direction_load_masks()
        np.testing.assert_array_equal(w.direction_masks[0].mask, source.mask)
        self.assertEqual(w.direction_object_margin.value(), 0)
        w.direction_object_margin.setValue(5)
        np.testing.assert_array_equal(w._direction_targets["Object"].object_mask, adjust_mask_by_mode(source.mask, 5, "simple"))
        before = w.direction_masks[0].mask.copy()
        source.mask[:] = 0
        np.testing.assert_array_equal(w.direction_masks[0].mask, before)

    def click_image(self, point):
        viewer = self.w.frame_viewer
        position = viewer._image_to_widget(point)
        for kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
            event = QMouseEvent(kind, position, position, Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton if kind == QEvent.Type.MouseButtonPress else Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier)
            QApplication.sendEvent(viewer, event)

    def test_per_object_controls_and_apply_all_preserve_each_calibration(self):
        w = self.w
        first = DirectionMask("First", rect(90, 55, 100, 65), ranges=DirectionRanges(2, 5))
        second = DirectionMask("Second", rect(120, 55, 130, 65), ranges=DirectionRanges(1, 10))
        first.ranges.quad_points = [(5, 5), (195, 5), (195, 115), (5, 115)]
        second.ranges.quad_points = [(10, 10), (190, 10), (190, 110), (10, 110)]
        first.ranges.reference_id, second.ranges.reference_id = "first-ref", "second-ref"
        w._direction_set_masks([first, second])
        w.direction_objects.setCurrentRow(1)
        self.assertEqual(w.direction_object_margin.value(), 1)
        self.assertEqual(w.direction_near_margin.value(), 10)
        w.direction_objects.setCurrentRow(0)
        w.direction_apply_ranges.click()
        self.assertEqual((second.ranges.object_margin, second.ranges.near_margin), (2, 5))
        self.assertEqual(second.ranges.reference_id, "second-ref")
        self.assertEqual(second.ranges.quad_points[0], (10, 10))
        w.direction_object_margin.setValue(6)
        self.assertEqual(w.direction_ready.text(), "Not ready")
        self.assertIn("at least", w.direction_ready.toolTip())
        self.assertFalse(w.save_current_button.isEnabled())
        w.direction_near_margin.setValue(8)
        self.assertEqual(w.direction_ready.text(), "Ready")
        self.assertIsNotNone(w.frame_viewer.direction_range_overlay)
        w._direction_load_masks()
        self.assertTrue(w.save_current_button.isEnabled())

    def test_physical_measurement_unit_conversion_cancel_and_video_scope(self):
        w = self.w
        original = w.mask_records["Object"].mask.copy()
        w.resize(1320, 900)
        w.show()
        self.app.processEvents()
        w.direction_range_mode.setCurrentIndex(w.direction_range_mode.findData("physical"))
        w.direction_calibration_toggle.setChecked(True)
        self.app.processEvents()
        self.assertGreaterEqual(w.direction_range_unit.width(), 80)
        self.assertGreaterEqual(w.direction_reference_unit.width(), 80)
        self.assertEqual(w.direction_ready.text(), "Not ready")
        self.assertEqual(len(w.direction_masks[0].references), 1)
        self.assertTrue(w.direction_reference_target_buttons["line"].isChecked())
        self.assertEqual(w.frame_viewer.mode, "reference_line")
        self.assertFalse(w.save_current_button.isEnabled())
        self.click_image((20, 20))
        self.click_image((120, 20))
        self.assertEqual(w.frame_viewer.mode, "reference_line")
        self.assertTrue(w.direction_reference_target_buttons["line"].isChecked())
        self.assertEqual(w.direction_ready.text(), "Ready")
        w.direction_reference_length.setValue(50)
        w.direction_object_margin.setValue(1)
        item = w.direction_masks[0]
        self.assertEqual(np.where(w._direction_targets["Object"].object_mask)[1].min(), 88)
        self.assertEqual(np.where(w._direction_targets["Object"].near_mask)[1].min(), 86)
        w.direction_range_unit.setCurrentText("m")
        self.assertAlmostEqual(item.ranges.object_margin, .01)
        self.assertAlmostEqual(item.ranges.near_margin, .02)
        w.direction_reference_unit.setCurrentText("m")
        self.assertAlmostEqual(item.references[0].length, .5)
        before = list(item.references[0].points)
        w.direction_redraw_line.click()
        self.click_image((30, 30))
        w.direction_reference_target_buttons["object"].click()
        self.assertEqual(item.references[0].points, before)
        self.assertTrue(w.save_current_button.isEnabled())
        self.assertFalse(w.save_multiple_button.isEnabled())
        with patch.object(w, "_select_videos_for_batch") as choose:
            w.save_multiple_mode_outputs()
            choose.assert_not_called()
        self.assertEqual(w.object3d_workspace.references, [])
        np.testing.assert_array_equal(w.mask_records["Object"].mask, original)
        w.load_video(self.make_video("other-physical"))
        self.assertEqual(w.direction_masks[0].references[0].points, [])
        self.assertEqual(w.direction_ready.text(), "Not ready")
        w.load_video(self.video)
        self.assertEqual(w.direction_masks[0].references[0].points, before)
        self.assertEqual(w.direction_ready.text(), "Ready")

    def test_real_settings_save_and_restart_preserve_independent_distance_mapping(self):
        w = self.w
        w.direction_distance_node.setCurrentIndex(w.direction_distance_node.findData("body_c"))
        w.direction_method.setCurrentIndex(w.direction_method.findData("neck"))
        with patch.object(main, "SETTINGS_PATH", self.root / "settings.json"):
            ORIGINAL_SAVE_SETTINGS(w)
            saved = json.loads(main.SETTINGS_PATH.read_text(encoding="utf-8"))
            self.assertEqual(saved["direction"]["points"]["distance"], "body_c")
            with patch.object(main.MainWindow, "_load_settings", ORIGINAL_LOAD_SETTINGS):
                restored = main.MainWindow()
            try:
                restored.load_video(self.video)
                restored._open_pipeline_settings(restored.TAB_DIRECTION)
                self.assertEqual(restored.direction_distance_node.currentData(), "body_c")
                self.assertEqual(restored.direction_method.currentData(), "neck")
                self.assertEqual(restored.direction_angle.value(), 0)
            finally:
                restored.close()
                restored.deleteLater()
                self.app.processEvents()

    def test_reference_targets_group_line_editor_and_independent_face_bindings(self):
        w = self.w
        obj, refs = DirectionRangeTests().make_prism()
        w._direction_set_masks([direction_mask_from_object(obj, refs, w._direction_video_key)])
        w.direction_range_mode.setCurrentIndex(w.direction_range_mode.findData("physical"))
        w.direction_calibration_toggle.setChecked(True)
        self.assertFalse(w.direction_reference_editor.isHidden())
        self.assertTrue(w.direction_reference_assignment.isHidden())
        self.assertTrue(w.direction_reference_target_buttons["object"].isHidden())
        w.direction_reference_target_buttons["top"].click()
        self.assertTrue(w.direction_reference_editor.isHidden())
        self.assertFalse(w.direction_reference_assignment.isHidden())
        top = w.direction_reference_controls["top_reference_id"][0]
        bottom = w.direction_reference_controls["bottom_reference_id"][0]
        self.assertFalse(top.isHidden())
        self.assertTrue(bottom.isHidden())
        top.setCurrentIndex(top.findData(refs[1].id))
        w.direction_reference_target_buttons["bottom"].click()
        bottom.setCurrentIndex(bottom.findData(refs[0].id))
        item = w.direction_masks[0]
        self.assertEqual((item.ranges.top_reference_id, item.ranges.bottom_reference_id), (refs[1].id, refs[0].id))
        w.direction_reference_target_buttons["line"].click()
        editor = w.direction_reference_controls["edit"][0]
        editor.setCurrentIndex(editor.findData(refs[1].id))
        self.assertEqual((item.ranges.top_reference_id, item.ranges.bottom_reference_id), (refs[1].id, refs[0].id))
        self.assertTrue(w.save_current_button.isEnabled())
        with patch("tab_mixins.direction_ranges.QInputDialog.getText", return_value=("Extra ruler", True)):
            w._direction_new_reference()
        self.assertEqual(len(item.references), 3)
        self.assertEqual((item.ranges.top_reference_id, item.ranges.bottom_reference_id), (refs[1].id, refs[0].id))
        self.assertTrue(w.frame_viewer.reference_pick_active)
        self.assertFalse(w.save_current_button.isEnabled())
        self.assertEqual(obj.top.reference_id, refs[0].id)
        self.assertEqual(obj.bottom.reference_id, refs[1].id)

    def test_current_video_3d_lines_are_reused_and_can_be_adjusted_without_changing_source(self):
        w = self.w
        ref = LengthReference("Shared ruler", 50, "cm", [(20, 20), (120, 20)])
        w.object3d_workspace.references.append(ref)
        w._open_pipeline_settings(w.TAB_OCCLUSION_3D)
        w._open_pipeline_settings(w.TAB_DIRECTION)
        w.direction_range_mode.setCurrentIndex(w.direction_range_mode.findData("physical"))
        w.direction_calibration_toggle.setChecked(True)
        item = w.direction_masks[0]
        self.assertEqual(len(item.references), 1)
        self.assertEqual(item.ranges.reference_id, ref.id)
        self.assertEqual(w.frame_viewer.reference_pick_points, ref.points)
        self.assertTrue(w.save_current_button.isEnabled())
        w.resize(1320, 900)
        w.show()
        self.app.processEvents()
        viewer = w.frame_viewer
        for kind, point in ((QEvent.Type.MouseButtonPress, (20, 20)), (QEvent.Type.MouseMove, (30, 20)),
                            (QEvent.Type.MouseButtonRelease, (30, 20))):
            position = viewer._image_to_widget(point)
            event = QMouseEvent(kind, position, position,
                Qt.MouseButton.NoButton if kind == QEvent.Type.MouseMove else Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier)
            QApplication.sendEvent(viewer, event)
        self.assertAlmostEqual(item.references[0].points[0][0], 30)
        self.assertEqual(ref.points, [(20, 20), (120, 20)])
        self.assertEqual(viewer.mode, "reference_line")
        self.assertTrue(w.save_current_button.isEnabled())

    def test_video_switch_keeps_latest_settings_selection_and_mappings_but_only_restores_local_lines(self):
        w = self.w
        ref = LengthReference("Ruler", 50, "cm", [(20, 20), (120, 20)])
        first = DirectionMask("First", rect(90, 55, 100, 65), references=[ref])
        second = DirectionMask("Second", rect(120, 55, 130, 65), references=[ref])
        for item in (first, second):
            item.ranges = DirectionRanges(1, 3, "physical", "cm", ref.id, calibration_video_key=w._direction_video_key)
        w._direction_set_masks([first, second], {"Second"}, "Second")
        w.direction_distance_node.setCurrentIndex(w.direction_distance_node.findData("body_c"))
        w.direction_method.setCurrentIndex(w.direction_method.findData("neck"))
        w.direction_angle.setValue(25)
        other = self.make_video("settings-carried", 400, 240)
        df = pd.read_csv(other.with_suffix(".csv")).drop(columns=["body_c.x", "body_c.y"])
        df.to_csv(other.with_suffix(".csv"), index=False)
        w.load_video(other)
        self.assertEqual(w._direction_point_names["distance"], "body_c")
        self.assertIsNone(w.direction_distance_node.currentData())
        self.assertEqual(w.direction_method.currentData(), "neck")
        self.assertEqual(w.direction_angle.value(), 25)
        self.assertEqual(w._direction_range_object().name, "Second")
        self.assertEqual([m.name for m in w._direction_selected_masks()], ["Second"])
        self.assertEqual(w.direction_masks[1].mask.shape, (240, 400))
        self.assertEqual(w.direction_masks[1].references[0].points, [])
        self.assertEqual(w.direction_range_mode.currentData(), "physical")
        self.assertEqual(w.direction_range_unit.currentText(), "cm")
        w.direction_object_margin.setValue(2)
        w.direction_near_margin.setValue(5)
        w.load_video(self.video)
        self.assertEqual(w.direction_distance_node.currentData(), "body_c")
        self.assertEqual(w.direction_coordinates.currentData(), "pixels")
        self.assertEqual(w.direction_angle.value(), 25)
        self.assertEqual(w._direction_range_object().name, "Second")
        self.assertEqual((w.direction_object_margin.value(), w.direction_near_margin.value()), (2, 5))
        self.assertEqual(w.direction_masks[1].references[0].points, ref.points)
        self.assertTrue(w.save_current_button.isEnabled())

    def test_import_buttons_load_png_or_direction_recipe_and_cancel_keeps_workspace(self):
        w = self.w
        png = self.root / "Imported.png"
        write_mask_png(png, rect(80, 40, 100, 70) * 255)
        with patch("tab_mixins.direction_tab.QFileDialog.getOpenFileName", return_value=(str(png), "PNG Files (*.png)")):
            w.direction_import_mask.click()
        self.assertEqual(w.direction_masks[0].name, "Imported")
        before = w.direction_masks[0].mask.copy()
        with patch("tab_mixins.direction_tab.QFileDialog.getExistingDirectory", return_value=""):
            w.direction_import_folder.click()
        np.testing.assert_array_equal(w.direction_masks[0].mask, before)
        obj, refs = DirectionRangeTests().make_prism()
        item = direction_mask_from_object(obj, refs, "another-video")
        item.ranges.mode, item.ranges.object_margin, item.ranges.near_margin = "physical", 1, 3
        folder = self.root / "exported-direction"
        export_direction_masks([item], folder, DirectionConfig())
        with patch("tab_mixins.direction_tab.QFileDialog.getExistingDirectory", return_value=str(folder)):
            w.direction_import_folder.click()
        self.assertEqual(w.direction_masks[0].name, "Prism")
        self.assertEqual(w.direction_masks[0].video_key, w._direction_video_key)
        self.assertEqual((w.direction_object_margin.value(), w.direction_near_margin.value()), (1, 3))
        self.assertTrue(all(not r.points for r in w.direction_masks[0].references))
        self.assertFalse(w.save_current_button.isEnabled())

    def test_perspective_clicks_and_cancel_do_not_modify_occlusion_calibration(self):
        w = self.w
        original = [(4, 4), (196, 4), (196, 116), (4, 116)]
        w.frame_viewer.occ_margin_points = list(original)
        w.resize(1320, 900)
        w.show()
        self.app.processEvents()
        w.direction_range_mode.setCurrentIndex(w.direction_range_mode.findData("geometric"))
        w.direction_calibration_toggle.setChecked(True)
        w.direction_quad_button.click()
        self.assertEqual(w.frame_viewer.mode, "direction_quad")
        self.assertFalse(w.save_current_button.isEnabled())
        quad = [(10, 10), (190, 10), (190, 110), (10, 110)]
        for point in quad:
            self.click_image(point)
        self.assertEqual(len(w.direction_masks[0].ranges.quad_points), 4)
        self.assertEqual(w.frame_viewer.mode, "direction")
        self.assertEqual(w.direction_ready.text(), "Ready")
        self.assertEqual(w.frame_viewer.occ_margin_points, original)
        committed = list(w.direction_masks[0].ranges.quad_points)
        w.direction_quad_button.click()
        self.click_image((20, 20))
        w.direction_quad_button.click()
        self.assertEqual(w.direction_masks[0].ranges.quad_points, committed)
        self.assertEqual(w.frame_viewer.occ_margin_points, original)
        self.assertTrue(w.save_current_button.isEnabled())
        # Changing tabs must not leave the shared line/point editor enabled.
        w.direction_quad_button.click()
        w._open_pipeline_settings(w.TAB_OCCLUSION)
        self.assertIsNone(w._direction_calibration)
        self.assertFalse(w.frame_viewer.reference_pick_active)

    def test_invalid_reload_keeps_previous_direction_masks_usable(self):
        w = self.w
        before = w.direction_masks[0].mask.copy()
        w.mask_records["Object"].mask[:] = 0
        w._direction_load_masks()
        np.testing.assert_array_equal(w.direction_masks[0].mask, before)
        self.assertTrue(w.save_current_button.isEnabled())
        self.assertEqual(w.direction_result.text(), "Object: Near / Facing")

    def test_two_dimensional_video_switch_resizes_and_restores_snapshots(self):
        w = self.w
        before = w.direction_masks[0].mask.copy()
        other = self.make_video("resized", 400, 240)
        w.load_video(other)
        self.assertEqual(w.direction_masks[0].mask.shape, (240, 400))
        self.assertEqual(w.direction_result.text(), "Object: Far")
        w.load_video(self.video)
        np.testing.assert_array_equal(w.direction_masks[0].mask, before)
        self.assertEqual(w.direction_result.text(), "Object: Near / Facing")

    def test_direction_is_added_to_existing_visibility_settings_and_has_f1(self):
        w = self.w
        self.assertTrue(w.tab_visibility_actions[w.TAB_DIRECTION].isChecked())
        self.assertEqual(w._mode_tab_category(w.TAB_DIRECTION), w.CATEGORY_ANNOTATE)
        with patch.object(w, "show_annotate_controls_help") as show:
            w._open_context_help()
            show.assert_called_once_with()

    def test_direction_canvas_is_read_only_and_old_occlusion_restores(self):
        w = self.w
        original = w.direction_masks[0].mask.copy()
        viewer = w.frame_viewer
        for kind, point in ((QEvent.Type.MouseButtonPress, (10, 10)), (QEvent.Type.MouseMove, (50, 50)),
                            (QEvent.Type.MouseButtonRelease, (50, 50))):
            position = viewer._image_to_widget(point)
            event = QMouseEvent(kind, position, position,
                Qt.MouseButton.NoButton if kind == QEvent.Type.MouseMove else Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier)
            QApplication.sendEvent(viewer, event)
        w._scale_active_region_shortcut(1.5)
        np.testing.assert_array_equal(w.direction_masks[0].mask, original)
        self.assertEqual(viewer.mode, "direction")
        w._open_pipeline_settings(w.TAB_OCCLUSION)
        self.assertFalse(viewer.direction_active)
        self.assertIsNone(viewer.direction_preview)
        self.assertIs(viewer.mask_records, w.mask_records)

    def test_multiple_animals_preview_and_export_remain_row_local(self):
        w = self.w
        df = w.csv_df.iloc[:2].copy()
        df["frame"] = 1
        df["instance.id"] = ["first", "second"]
        df.to_csv(w.csv_path, index=False)
        w.load_csv(w.csv_path, False)
        w._direction_refresh_preview()
        self.assertEqual(w.direction_instance.count(), 2)
        self.assertEqual(w.direction_result.text(), "Object: Near / Facing")
        w.direction_instance.setCurrentIndex(1)
        self.assertEqual(w.direction_result.text(), "Object: Near / Not Facing")
        w.save_current_mode_output()
        self.wait_save()
        result = pd.read_csv(w._direction_output_path())
        self.assertEqual(result["instance.id"].tolist(), ["first", "second"])
        self.assertEqual(result["Object.direction"].tolist(), ["NearFacing", "NearNotFacing"])

    def test_3d_source_keeps_raw_faces_and_ranges_but_remeasures_other_video(self):
        w = self.w
        reference = LengthReference("Ruler", 1, "m", [(0, 0), (100, 0)])
        obj = OcclusionObject("Prism", ObjectFace(rect(90, 20, 100, 30), 2, "cm", reference.id),
                              ObjectFace(rect(90, 90, 100, 100), 2, "cm", reference.id))
        w.object3d_workspace = ObjectWorkspace(200, 120, [reference], [obj], w._object3d_video_key)
        w._object3d_changed()
        w.direction_source.setCurrentIndex(w.direction_source.findData("3d"))
        w._direction_load_masks()
        np.testing.assert_array_equal(w.direction_masks[0].mask, build_envelope(obj.top.mask, obj.bottom.mask)[0])
        self.assertFalse(w.save_multiple_button.isEnabled())
        with patch.object(w, "_select_videos_for_batch") as choose:
            w.save_multiple_mode_outputs()
            choose.assert_not_called()
        w.direction_range_mode.setCurrentIndex(w.direction_range_mode.findData("physical"))
        w.direction_object_margin.setValue(1)
        w.direction_near_margin.setValue(3)
        w.load_video(self.make_video("other"))
        self.assertEqual(w.direction_masks[0].name, "Prism")
        self.assertEqual(w.direction_masks[0].video_key, w._direction_video_key)
        self.assertEqual(w.direction_masks[0].references[0].points, [])
        self.assertEqual((w.direction_object_margin.value(), w.direction_near_margin.value()), (1, 3))
        self.assertFalse(w.save_current_button.isEnabled())
        w.load_video(self.video)
        self.assertEqual(w.direction_masks[0].name, "Prism")
        self.assertTrue(w.save_current_button.isEnabled())

    def test_two_dimensional_batch_uses_each_video_dimensions_and_existing_io(self):
        w = self.w
        other = self.make_video("large", 400, 240)
        w.direction_near_margin.setValue(100)
        with patch.object(w, "_select_videos_for_batch", return_value=[self.video, other]):
            w.save_multiple_mode_outputs()
            self.wait_save()
        for name in ("a", "large"):
            result = pd.read_csv(w.save_folder / f"{name}_direction.csv")
            self.assertEqual(result["Object.direction"].iloc[:2].tolist(), ["NearFacing", "NearNotFacing"])
            self.assertEqual(result["Object.direction"].iloc[2], "Unknown")
            self.assertEqual(result["Object.direction_reason"].iloc[2], "missing_nose")

    def test_folder_import_ignores_source_margin_and_exports_base_with_own_ranges(self):
        w = self.w
        w.mask_records["Object"].margin_mode = "geometric"
        w.mask_records["Object"].margin = 20
        quad = [(10, 10), (190, 10), (190, 110), (10, 110)]
        w.frame_viewer.occ_margin_points = quad
        w.export_masks()
        folder = w._mask_export_folder()
        with patch("tab_mixins.direction_tab.QFileDialog.getExistingDirectory", return_value=str(folder)):
            w.direction_import_folder.click()
        np.testing.assert_array_equal(w.direction_masks[0].mask, w.mask_records["Object"].mask)
        w.direction_object_margin.setValue(3)
        w._direction_export()
        np.testing.assert_array_equal(read_mask_png(folder / "Object.png") > 0, w.mask_records["Object"].mask > 0)
        data = json.loads((folder / "masks_manifest.json").read_text())
        self.assertEqual(data["masks"]["Object"]["direction_recipe"]["ranges"]["object_margin"], 3)
        self.assertTrue(data["direction_base_masks"])

    def test_zero_preview_is_visible_and_does_not_mutate_masks(self):
        w = self.w
        w.resize(1320, 900)
        w.show()
        self.app.processEvents()
        viewer = w.frame_viewer
        before = w.direction_masks[0].mask.copy()
        image = viewer.grab().toImage()
        point = viewer._image_to_widget((75, 60)).toPoint()
        colors = [image.pixelColor(point.x() + x, point.y() + y) for x in range(-6, 7) for y in range(-2, 3)]
        self.assertTrue(any(c.red() > 150 and c.green() > 100 and c.blue() < 100 for c in colors))
        self.assertEqual(viewer.direction_preview[2], 0)
        np.testing.assert_array_equal(w.direction_masks[0].mask, before)


if __name__ == "__main__":
    unittest.main()
