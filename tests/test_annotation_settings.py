"""Real settings-file round trips for annotation preferences."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cv2
import numpy as np
import pandas as pd
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QMessageBox

import main
from annotation_settings import restore_ranges, restore_references
from direction import DirectionConfig, DirectionMask, DirectionRanges, export_direction_masks
from occlusion_3d import LengthReference, ObjectFace, ObjectWorkspace, OcclusionObject
from shared import MaskRecord


def rectangle(x, y):
    result = np.zeros((120, 200), np.uint8)
    result[y:y + 12, x:x + 12] = 1
    return result


class AnnotationPreferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for patcher in (patch.object(main, "SETTINGS_PATH", self.root / "settings.json"),
                        patch.object(main, "DEFAULT_SAVE_DIR", self.root / "output"),
                        patch.object(QMessageBox, "warning"), patch.object(QMessageBox, "information")):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.windows = []
        self.addCleanup(self.close_all)
        self.video = self.root / "video.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (200, 120))
        self.assertTrue(writer.isOpened())
        for _ in range(3):
            writer.write(np.full((120, 200, 3), 50, np.uint8))
        writer.release()
        pd.DataFrame({"frame": [1, 2, 3], "nose.x": [60] * 3, "nose.y": [60] * 3,
            "neck.x": [40] * 3, "neck.y": [60] * 3,
            "body_c.x": [20] * 3, "body_c.y": [60] * 3,
            "ear_l.x": [40] * 3, "ear_l.y": [50] * 3,
            "ear_r.x": [40] * 3, "ear_r.y": [70] * 3}).to_csv(self.video.with_suffix(".csv"), index=False)

    def window(self):
        w = main.MainWindow()
        self.windows.append(w)
        return w

    def close(self, w):
        w.close()
        self.windows.remove(w)
        w.deleteLater()
        self.app.processEvents()

    def close_all(self):
        for w in self.windows[:]:
            self.close(w)

    def settings(self):
        return json.loads(main.SETTINGS_PATH.read_text(encoding="utf-8"))

    def test_direction_restores_ranges_profiles_references_and_bodyparts_after_restart(self):
        w = self.window()
        w.load_video(self.video)
        w._open_pipeline_settings(w.TAB_DIRECTION)
        ref = LengthReference("Desk ruler", 60, "cm", [(10, 10), (160, 10)])
        first = DirectionMask("First", rectangle(90, 55), ranges=DirectionRanges(2, 35), references=[ref])
        second = DirectionMask("Second", rectangle(130, 55), references=[copy.deepcopy(ref)])
        second.ranges = DirectionRanges(.02, .06, "physical", "m", ref.id)
        w._direction_set_masks([first, second], {"First"}, "Second")
        w.direction_distance_node.setCurrentIndex(w.direction_distance_node.findData("body_c"))
        w.direction_method.setCurrentIndex(w.direction_method.findData("body"))
        w.direction_coordinates.setCurrentIndex(w.direction_coordinates.findData("pixels"))
        w.direction_angle.setValue(0)
        w.direction_calibration_toggle.setChecked(True)
        w.direction_reference_target_buttons["object"].click()
        w.direction_source.setCurrentIndex(w.direction_source.findData("3d"))
        self.close(w)
        saved = self.settings()["direction"]
        self.assertNotIn("points", saved["references"][0])
        self.assertNotIn("quad_points", saved["objects"]["Second"]["ranges"])
        self.assertNotIn("calibration_video_key", saved["objects"]["Second"]["ranges"])
        # Starting and closing before loading data must not replace saved profiles.
        empty = self.window()
        self.assertEqual(empty.direction_source.currentData(), "3d")
        self.assertEqual(empty.direction_range_mode.currentData(), "physical")
        self.assertAlmostEqual(empty.direction_object_margin.value(), .02)
        self.close(empty)
        self.assertEqual(self.settings()["direction"]["objects"], saved["objects"])
        restored = self.window()
        restored.load_video(self.video)
        restored.mask_records = {name: MaskRecord(name, QColor("cyan"), rectangle(x, 55)) for name, x in (("First", 90), ("Second", 130))}
        restored._open_pipeline_settings(restored.TAB_DIRECTION)
        restored._direction_load_masks(source="occlusion")
        self.assertEqual(restored._direction_range_object().name, "Second")
        self.assertEqual([m.name for m in restored._direction_selected_masks()], ["First"])
        self.assertEqual(restored.direction_distance_node.currentData(), "body_c")
        self.assertEqual(restored.direction_method.currentData(), "body")
        self.assertEqual(restored.direction_coordinates.currentData(), "pixels")
        self.assertEqual(restored.direction_angle.value(), 0)
        self.assertEqual(restored._direction_reference_target, "object")
        self.assertTrue(restored.direction_calibration_toggle.isChecked())
        self.assertEqual(restored.direction_range_unit.currentText(), "m")
        self.assertAlmostEqual(restored.direction_object_margin.value(), .02)
        self.assertAlmostEqual(restored.direction_near_margin.value(), .06)
        ruler = restored.direction_masks[1].references[0]
        self.assertEqual((ruler.name, ruler.length, ruler.unit), ("Desk ruler", 60, "cm"))
        self.assertEqual(ruler.points, [])
        self.assertEqual(restored.direction_masks[0].ranges.mode, "pixels")
        self.assertEqual(restored.direction_masks[0].ranges.near_margin, 35)

    def test_occlusion_restores_tools_modes_and_existing_numeric_defaults(self):
        w = self.window()
        w._open_pipeline_settings(w.TAB_OCCLUSION)
        w.mask_shape_combo.setCurrentText("Square Drag")
        w.mask_transform_radio.setChecked(True)
        w.mask_erase_radio.setChecked(True)
        w.mask_brush_spinbox.setValue(37)
        w.mask_margin_spinbox.setValue(-12)
        w.mask_margin_geometric_radio.setChecked(True)
        self.close(w)
        restored = self.window()
        self.assertEqual(restored.mask_shape_combo.currentText(), "Square Drag")
        self.assertTrue(restored.mask_transform_radio.isChecked())
        self.assertTrue(restored.mask_erase_radio.isChecked())
        self.assertEqual(restored.mask_brush_spinbox.value(), 37)
        self.assertEqual(restored.mask_margin_spinbox.value(), -12)
        self.assertTrue(restored.mask_margin_geometric_radio.isChecked())
        restored.load_video(self.video)
        with patch("tab_mixins.occlusion_tab.QInputDialog.getText", return_value=("New", True)):
            restored.add_mask()
        self.assertEqual(restored.mask_records["New"].margin, -12)
        self.assertEqual(restored.mask_records["New"].margin_mode, "geometric")

    def test_3d_restores_face_tools_margins_units_and_reference_definitions(self):
        w = self.window()
        w.load_video(self.video)
        a = LengthReference("Top ruler", 25, "cm", [(10, 10), (110, 10)])
        b = LengthReference("Bottom ruler", 60, "cm", [(10, 110), (160, 110)])
        obj = OcclusionObject("Prism", ObjectFace(rectangle(80, 20), reference_id=a.id), ObjectFace(rectangle(100, 80), reference_id=b.id))
        w.object3d_workspace = ObjectWorkspace(200, 120, [a, b], [obj], w._object3d_video_key)
        w._object3d_changed()
        w._open_pipeline_settings(w.TAB_OCCLUSION_3D)
        top, bottom = w.object3d_face_controls["top"], w.object3d_face_controls["bottom"]
        top["tool"].setCurrentText("Square Drag")
        top["transform"].setChecked(True)
        top["erase"].setChecked(True)
        top["brush"].setValue(43)
        top["margin"].setValue(1.5)
        bottom["tool"].setCurrentText("Brush")
        bottom["brush"].setValue(27)
        bottom["unit"].setCurrentText("mm")
        bottom["margin"].setValue(-2)
        w.object3d_reference_combo.setCurrentIndex(w.object3d_reference_combo.findData(b.id))
        w.object3d_include.setChecked(False)
        w._object3d_set_edit_target("bottom")
        self.close(w)
        saved = self.settings()["occlusion_3d"]
        self.assertNotIn("points", saved["references"][0])
        self.assertNotIn("objects", saved)
        restored = self.window()
        top, bottom = restored.object3d_face_controls["top"], restored.object3d_face_controls["bottom"]
        self.assertEqual(restored._object3d_edit_target, "bottom")
        self.assertEqual(top["tool"].currentText(), "Square Drag")
        self.assertTrue(top["transform"].isChecked())
        self.assertTrue(top["erase"].isChecked())
        self.assertEqual(top["brush"].value(), 43)
        self.assertEqual((top["margin"].value(), top["unit"].currentText()), (1.5, "cm"))
        self.assertEqual(bottom["tool"].currentText(), "Brush")
        self.assertEqual(bottom["brush"].value(), 27)
        self.assertEqual((bottom["margin"].value(), bottom["unit"].currentText()), (-2, "mm"))
        restored.load_video(self.video)
        self.assertEqual(len(restored.object3d_workspace.references), 2)
        self.assertTrue(all(not r.points for r in restored.object3d_workspace.references))
        self.assertEqual(restored.object3d_reference_combo.currentData(), b.id)
        with patch("tab_mixins.occlusion_3d_tab.QInputDialog.getText", return_value=("Next", True)):
            restored._object3d_add_object()
        new = restored.object3d_workspace.objects[0]
        self.assertEqual((new.top.margin, new.top.unit, new.top.reference_id), (1.5, "cm", a.id))
        self.assertEqual((new.bottom.margin, new.bottom.unit, new.bottom.reference_id), (-2, "mm", b.id))
        self.assertFalse(new.included)

    def test_explicit_direction_recipe_overrides_saved_object_preferences(self):
        w = self.window()
        w.load_video(self.video)
        w._open_pipeline_settings(w.TAB_DIRECTION)
        w._direction_set_masks([DirectionMask("Object", rectangle(90, 55), ranges=DirectionRanges(2, 35))])
        self.close(w)
        folder = self.root / "import"
        ref = LengthReference("Bundle ruler", 80, "cm")
        mask = DirectionMask("Object", rectangle(90, 55), ranges=DirectionRanges(4, 8, "physical", "cm", ref.id), references=[ref])
        export_direction_masks([mask], folder, DirectionConfig())
        restored = self.window()
        restored.load_video(self.video)
        restored._open_pipeline_settings(restored.TAB_DIRECTION)
        with patch("tab_mixins.direction_tab.QFileDialog.getExistingDirectory", return_value=str(folder)):
            restored.direction_import_folder.click()
        self.assertEqual((restored.direction_object_margin.value(), restored.direction_near_margin.value()), (4, 8))
        self.assertEqual(restored.direction_range_mode.currentData(), "physical")
        self.assertEqual(restored.direction_masks[0].references[0].length, 80)

    def test_direction_keeps_source_face_bindings_and_remaps_restored_rulers_by_name(self):
        w = self.window()
        w.load_video(self.video)
        a = LengthReference("Top ruler", 60, "cm", [(10, 10), (110, 10)])
        b = LengthReference("Bottom ruler", 35, "cm", [(10, 110), (160, 110)])
        obj = OcclusionObject("Prism", ObjectFace(rectangle(80, 20), reference_id=a.id), ObjectFace(rectangle(100, 80), reference_id=b.id))
        w.object3d_workspace = ObjectWorkspace(200, 120, [a, b], [obj], w._object3d_video_key)
        w._object3d_changed()
        w._open_pipeline_settings(w.TAB_DIRECTION)
        w.direction_source.setCurrentIndex(w.direction_source.findData("3d"))
        w._direction_load_masks()
        spec = w.direction_masks[0].ranges
        self.assertEqual((spec.top_reference_id, spec.bottom_reference_id), (a.id, b.id))
        w.direction_range_mode.setCurrentIndex(w.direction_range_mode.findData("physical"))
        w.direction_calibration_toggle.setChecked(True)
        w.direction_reference_length.setValue(65)
        self.assertEqual(a.length, 60)
        self.close(w)
        restored = self.window()
        restored.load_video(self.video)
        # Importing 3D objects gives their references fresh IDs.
        new_a = LengthReference("Top ruler", 55, "cm", [(10, 10), (110, 10)])
        new_b = LengthReference("Bottom ruler", 80, "cm", [(10, 110), (160, 110)])
        new_obj = OcclusionObject("Prism", ObjectFace(rectangle(80, 20), reference_id=new_a.id), ObjectFace(rectangle(100, 80), reference_id=new_b.id))
        restored.object3d_workspace = ObjectWorkspace(200, 120, [new_a, new_b], [new_obj], restored._object3d_video_key)
        restored._object3d_changed()
        restored._open_pipeline_settings(restored.TAB_DIRECTION)
        restored._direction_load_masks()
        spec = restored.direction_masks[0].ranges
        self.assertEqual((spec.top_reference_id, spec.bottom_reference_id), (new_a.id, new_b.id))
        top_ref = next(r for r in restored.direction_masks[0].references if r.id == new_a.id)
        self.assertEqual(top_ref.length, 65)
        self.assertEqual(top_ref.points, new_a.points)
        self.assertEqual(new_a.length, 55)
        self.assertEqual(new_b.length, 80)

    def test_old_preferences_migrate_without_startup_overwriting_restored_values(self):
        main.SETTINGS_PATH.write_text(json.dumps({"occlusion_mask_margin": 23, "occlusion_mask_brush": 18,
            "occlusion_mask_margin_mode": "geometric", "direction": {"method": "neck", "angle": 0,
            "points": {"distance": "body_c"}, "coordinates": "pixels"}}), encoding="utf-8")
        w = self.window()
        self.assertEqual(w.mask_margin_spinbox.value(), 23)
        self.assertEqual(w.mask_brush_spinbox.value(), 18)
        self.assertEqual(self.settings()["occlusion_mask_margin"], 23)
        self.assertEqual(self.settings()["occlusion_mask_brush"], 18)
        self.assertEqual(self.settings()["direction"]["points"]["distance"], "body_c")
        self.close(w)
        restored = self.window()
        restored.load_video(self.video)
        self.assertEqual(restored.direction_distance_node.currentData(), "body_c")
        self.assertEqual(restored.direction_method.currentData(), "neck")
        self.assertEqual(restored.direction_angle.value(), 0)

    def test_invalid_new_preferences_fall_back_without_crashing(self):
        main.SETTINGS_PATH.write_text(json.dumps({"direction": {"reference_target": [], "source": [],
            "range_defaults": {"mode": [], "unit": "yards", "object_margin": "bad", "near_margin": "NaN"},
            "references": [{"id": "../bad", "name": "Ruler"}]}, "occlusion": {"tool": []},
            "occlusion_3d": {"target": [], "face": [], "faces": {"top": {"tool": [], "unit": "yards", "brush": None, "margin": "NaN"}}}}), encoding="utf-8")
        w = self.window()
        self.assertEqual(w.direction_range_mode.currentData(), "pixels")
        self.assertEqual(w.direction_near_margin.value(), 20)
        self.assertEqual(w._object3d_edit_target, "line")
        self.assertEqual(w.object3d_face_controls["top"]["margin"].value(), 2)
        self.assertEqual(w.mask_shape_combo.currentText(), "Rectangle Drag")
        self.assertEqual(restore_references(None), [])
        self.assertEqual(restore_ranges({"object_margin": float("inf")}).object_margin, 0)


if __name__ == "__main__":
    unittest.main()
