"""Geometry, Qt editing, video-local calibration, and real CSV export regressions."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pandas as pd
from PyQt6.QtCore import QEvent, QEventLoop, QPointF, QRectF, QTimer, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication, QCheckBox, QMessageBox

import main
from mask_io import read_mask_png, write_mask_png
from mask_tools import constrained_rectangle_points
from occlusion_3d import (
    LengthReference, ObjectFace, ObjectWorkspace, OcclusionObject,
    apply_margin, connect_faces, convert_length, load_bundle, save_bundle,
)


def rectangle(shape, corners):
    mask = np.zeros(shape, np.uint8)
    cv2.rectangle(mask, corners[:2], corners[2:], 1, -1)
    return mask


def workspace():
    ws = ObjectWorkspace(200, 200, video_key="video-a")
    ref = LengthReference("Ruler", 1, "m", [(10, 10), (110, 10)])
    ws.references.append(ref)
    ws.objects.append(OcclusionObject("Box", ObjectFace(rectangle((200, 200), (40, 40, 60, 60)), 0, "cm", ref.id),
                                     ObjectFace(rectangle((200, 200), (40, 100, 60, 120)), 0, "cm", ref.id)))
    return ws


class ObjectGeometryTests(unittest.TestCase):
    def test_reference_units_and_relative_margin(self):
        ws = workspace()
        ws.references[0].points = [(0, 0), (350, 0)]
        ws.references[0].length, ws.references[0].unit = 0.35, "m"
        ws.objects[0].top.margin, ws.objects[0].top.unit = 0.02, "m"
        self.assertAlmostEqual(ws.margin_pixels(ws.objects[0], "top"), 20)
        ws.objects[0].top.margin, ws.objects[0].top.unit = 2, "cm"
        self.assertAlmostEqual(ws.margin_pixels(ws.objects[0], "top"), 20)

    def test_cm_half_pixel_rounding_is_consistent_across_metric_units_and_signs(self):
        ws = workspace()
        ref = ws.references[0]
        ref.length, ref.unit, ref.points = 35, "cm", [(0, 0), (350, 0)]
        obj = ws.objects[0]
        obj.top.mask = rectangle((200, 200), (50, 50, 150, 150))
        for index in (1, 3, 5, 7, 15, 19, 29, 39):
            centimetres = index / 20
            radius = (index + 1) // 2
            for sign in (1, -1):
                for unit in ("m", "cm", "mm"):
                    with self.subTest(cm=sign * centimetres, unit=unit):
                        obj.top.margin = convert_length(sign * centimetres, "cm", unit)
                        obj.top.unit = unit
                        adjusted = ws.adjusted_face(obj, "top")
                        ys, xs = np.where(adjusted)
                        self.assertEqual(int(xs.min()), 50 - sign * radius)
                        self.assertEqual(int(xs.max()), 150 + sign * radius)
                        self.assertEqual(int(ys.min()), 50 - sign * radius)
        unchanged = obj.top.mask
        np.testing.assert_array_equal(apply_margin(unchanged, 0.499999), unchanged)
        np.testing.assert_array_equal(apply_margin(unchanged, -0.499999), unchanged)

    def test_two_cm_expands_each_side_twenty_pixels_and_negative_margin_reverses_it(self):
        ws = workspace()
        ref = ws.references[0]
        ref.length, ref.unit, ref.points = 35, "cm", [(0, 0), (350, 0)]
        obj = ws.objects[0]
        obj.top.mask = rectangle((200, 200), (60, 60, 140, 140))
        obj.top.margin = 2
        positive = ws.adjusted_face(obj, "top")
        self.assertEqual(positive[100, 40], 1)
        self.assertEqual(positive[100, 39], 0)
        self.assertEqual(positive[100, 160], 1)
        self.assertEqual(positive[100, 161], 0)
        obj.top.margin = -2
        negative = ws.adjusted_face(obj, "top")
        ys, xs = np.where(negative)
        self.assertEqual((int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())), (80, 120, 80, 120))

    def test_diagonal_reference_uses_full_pixel_length(self):
        ws = workspace()
        ws.references[0].length, ws.references[0].unit = 5, "cm"
        ws.references[0].points = [(0, 0), (30, 40)]
        ws.objects[0].top.margin = 2
        self.assertAlmostEqual(ws.margin_pixels(ws.objects[0], "top"), 20)

    def test_metric_conversion_keeps_exact_decimal_values_after_repeated_unit_changes(self):
        value, unit = 35.0, "cm"
        for _ in range(20):
            for target in ("m", "mm", "cm"):
                value, unit = convert_length(value, unit, target), target
        self.assertEqual(value, 35.0)
        self.assertEqual(convert_length(2, "cm", "m"), 0.02)

    def test_unicode_bundle_names_and_pngs_round_trip(self):
        ws = workspace()
        ws.objects[0].name = "\uc6d0\uae30\ub465"
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "\ub9c8\uc2a4\ud06c"
            save_bundle(ws, folder)
            self.assertTrue((folder / (ws.objects[0].name + ".png")).exists())
            np.testing.assert_array_equal(read_mask_png(folder / (ws.objects[0].name + ".png")), ws.render(ws.objects[0]) * 255)
            restored = load_bundle(folder, 200, 200, video_key="video-a")
            self.assertEqual(restored.objects[0].name, ws.objects[0].name)
            np.testing.assert_array_equal(restored.render(restored.objects[0]), ws.render(ws.objects[0]))

    def test_envelope_fills_gap_without_destroying_faces(self):
        ws = workspace()
        obj = ws.objects[0]
        original = obj.top.mask.copy()
        result = ws.render(obj)
        self.assertEqual(result[80, 50], 1)
        self.assertEqual(result[80, 30], 0)
        np.testing.assert_array_equal(original, obj.top.mask)

    def test_margin_is_applied_to_each_face_before_connecting(self):
        ws = workspace()
        obj = ws.objects[0]
        obj.top.margin = 5  # 5 cm => 5 pixels
        result = ws.render(obj)
        self.assertEqual(result[50, 35], 1)
        self.assertEqual(result[120, 35], 0)

    def test_missing_and_degenerate_reference_never_use_previous_scale(self):
        ws = workspace()
        ws.references[0].points = [(3, 3), (3, 3)]
        with self.assertRaisesRegex(ValueError, "at least 1 px"):
            ws.render(ws.objects[0])
        ws.references[0].points = []
        self.assertTrue(ws.errors(ws.objects[0]))

    def test_negative_margin_can_invalidate_a_face(self):
        ws = workspace()
        ws.objects[0].top.margin = -30
        self.assertIn("removes the entire face", ws.errors(ws.objects[0])[0])

    def test_nested_circles_and_large_margins(self):
        small, large = np.zeros((80, 80), np.uint8), np.zeros((80, 80), np.uint8)
        cv2.circle(small, (40, 40), 10, 1, -1)
        cv2.circle(large, (40, 40), 20, 1, -1)
        result = connect_faces(small, large)
        self.assertTrue(np.all(result[large > 0] == 1))
        self.assertTrue(np.all(apply_margin(small, 1000000) == 1))
        self.assertFalse(np.any(apply_margin(large, -1000000)))

    def test_new_video_copies_faces_but_clears_measurements(self):
        ws = workspace()
        other = ws.for_new_video(400, 400)
        self.assertEqual(other.objects[0].top.mask.shape, (400, 400))
        self.assertFalse(other.references[0].points)
        self.assertTrue(ws.references[0].points)
        other.objects[0].top.mask[:] = 0
        self.assertTrue(np.any(ws.objects[0].top.mask))

    def test_bundle_restores_editable_faces_and_only_same_video_measurements(self):
        ws = workspace()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_bundle(ws, folder)
            same = load_bundle(folder, 200, 200, video_key="video-a")
            np.testing.assert_array_equal(same.objects[0].top.mask, ws.objects[0].top.mask)
            np.testing.assert_array_equal(same.render(same.objects[0]), ws.render(ws.objects[0]))
            other = load_bundle(folder, 200, 200, video_key="video-b")
            self.assertFalse(other.references[0].points)
            self.assertEqual(len(list(folder.glob("*.png"))), 1)
            manifest = json.loads((folder / "masks_manifest.json").read_text())
            self.assertEqual(manifest["masks"]["Box"]["margin"], 0)

    def test_legacy_linked_bundle_keeps_effective_reference_but_imports_independent_faces(self):
        ws = workspace()
        other = LengthReference("Other ruler", 0.5, "m", [(10, 10), (110, 10)])
        ws.references.append(other)
        obj = ws.objects[0]
        obj.bottom.reference_id = other.id
        obj.same_reference = True
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_bundle(ws, folder)
            loaded = load_bundle(folder, 200, 200, video_key="video-a")
            restored = loaded.objects[0]
            self.assertFalse(restored.same_reference)
            self.assertEqual(restored.bottom.reference_id, restored.top.reference_id)
            np.testing.assert_array_equal(loaded.render(restored), ws.render(obj))

    def test_bundle_rejects_source_path_outside_folder(self):
        ws = workspace()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_bundle(ws, folder)
            path = folder / "masks_manifest.json"
            data = json.loads(path.read_text())
            data["objects_3d"][0]["top"]["file"] = "../outside.png"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "inside"):
                load_bundle(folder, 200, 200)

    def test_reexport_drops_stale_owned_final_mask_but_preserves_other_files(self):
        ws = workspace()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_bundle(ws, folder)
            cv2.imwrite(str(folder / "unrelated.png"), ws.objects[0].top.mask)
            ws.references[0].points = []
            save_bundle(ws, folder)
            self.assertFalse((folder / "Box.png").exists())
            self.assertTrue((folder / "unrelated.png").exists())
            self.assertEqual(len(load_bundle(folder, 200, 200).objects), 1)

    def test_square_all_directions_and_frame_edges(self):
        for end in ((90, 70), (10, 70), (90, 20), (10, 20), (200, 70)):
            points = constrained_rectangle_points((50, 50), end, square=True, bounds=(100, 80))
            self.assertAlmostEqual(points[1][0] - points[0][0], points[2][1] - points[1][1])
            self.assertTrue(all(0 <= x <= 99 and 0 <= y <= 79 for x, y in points))


class ObjectQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name, replacement in (("_load_settings", lambda _: {}), ("_save_settings", lambda _: None)):
            patcher = patch.object(main.MainWindow, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(main, "DEFAULT_SAVE_DIR", self.root / "output")
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("warning", "information"):
            patcher = patch.object(QMessageBox, name)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.window = main.MainWindow()
        self.addCleanup(self.close_window)
        self.video = self.make_video("a")
        self.window.load_video(self.video)
        self.window._open_pipeline_settings(self.window.TAB_OCCLUSION_3D)
        ws = workspace()
        ws.video_key = str(self.video.resolve())
        self.window.object3d_workspace = ws
        self.window._object3d_changed()
        self.window._object3d_activate_face("top")

    def close_window(self):
        self.wait_save()
        dialog = getattr(self.window, "_active_single_save_dialog", None)
        if dialog is not None:
            dialog.close()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def make_video(self, name, size=(200, 200)):
        path = self.root / f"{name}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, size)
        self.assertTrue(writer.isOpened())
        for _ in range(3):
            writer.write(np.full((size[1], size[0], 3), 50, np.uint8))
        writer.release()
        pd.DataFrame({"frame": [1, 2, 3], "nose.x": [50, 140, 190], "nose.y": [80, 80, 190]}).to_csv(path.with_suffix(".csv"), index=False)
        return path

    def wait_save(self):
        if not self.window._save_job_is_running():
            return
        loop = QEventLoop()
        timer = QTimer()
        timer.setInterval(5)
        timer.timeout.connect(lambda: None if self.window._save_job_is_running() else loop.quit())
        timer.start()
        QTimer.singleShot(10000, loop.quit)
        loop.exec()
        timer.stop()
        self.assertFalse(self.window._save_job_is_running())

    def mouse(self, viewer, kind, point, *, modifiers=Qt.KeyboardModifier.NoModifier):
        position = viewer._image_to_widget(point)
        button = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseMove else Qt.MouseButton.LeftButton
        buttons = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else Qt.MouseButton.LeftButton
        event = QMouseEvent(kind, position, position, button, buttons, modifiers)
        QApplication.sendEvent(viewer, event)

    def test_multiple_objects_export_real_csv_with_existing_column_format(self):
        ws = self.window.object3d_workspace
        obj = copy.deepcopy(ws.objects[0])
        from occlusion_3d import new_id
        obj.id, obj.name = new_id(), "Cylinder"
        obj.top.mask = rectangle((200, 200), (130, 40, 150, 60))
        obj.bottom.mask = rectangle((200, 200), (130, 100, 150, 120))
        ws.objects.append(obj)
        self.window._object3d_changed()
        self.assertTrue(self.window.save_current_button.isEnabled())
        self.assertFalse(self.window.save_multiple_button.isEnabled())
        output = self.window._occlusion_output_path()
        self.assertIn("_occlusion", output.stem)
        self.window.save_current_mode_output()
        self.wait_save()
        result = pd.read_csv(output)
        self.assertEqual(result["Box.occ"].tolist(), [1, 0, 0])
        self.assertEqual(result["Cylinder.nose.occ"].tolist(), [0, 1, 0])
        self.assertEqual(result["nose.x"].tolist(), [50, 140, 190])

    def test_unit_selectors_preserve_physical_reference_and_face_margins(self):
        w = self.window
        ws, obj = w.object3d_workspace, w._object3d_object()
        obj.top.margin, obj.bottom.margin = 2, -2
        w._object3d_changed()
        before = ws.render(obj)
        w.object3d_reference_unit.setCurrentText("cm")
        self.assertEqual(ws.references[0].length, 100)
        self.assertEqual(w.object3d_reference_length.value(), 100)
        for key, expected in (("top", 0.02), ("bottom", -0.02)):
            controls = w.object3d_face_controls[key]
            controls["unit"].setCurrentText("m")
            self.assertEqual(getattr(obj, key).margin, expected)
            self.assertEqual(controls["margin"].value(), expected)
            controls["unit"].setCurrentText("mm")
            self.assertEqual(getattr(obj, key).margin, expected * 1000)
        np.testing.assert_array_equal(ws.render(obj), before)
        # The largest original m input must not clamp when converted to cm or mm.
        w.object3d_reference_unit.setCurrentText("m")
        w.object3d_reference_length.setValue(100000)
        w.object3d_reference_unit.setCurrentText("mm")
        self.assertEqual(ws.references[0].length, 100000000)
        self.assertEqual(w.object3d_reference_length.value(), 100000000)

    def test_finished_save_stays_busy_until_queued_gui_cleanup(self):
        from PyQt6.QtCore import QThread

        class FinishedThread(QThread):
            def run(self):
                pass

        w = self.window
        for kind in ("single_save", "batch"):
            with self.subTest(kind=kind):
                thread = FinishedThread(w)
                setattr(w, f"_active_{kind}_thread", thread)
                cleanup = (w._on_single_save_thread_finished if kind == "single_save"
                           else w._on_batch_thread_finished)
                thread.finished.connect(cleanup)
                thread.start()
                self.assertTrue(thread.wait(2000))
                self.assertFalse(thread.isRunning())
                self.assertTrue(w._save_job_is_running())
                with patch.object(w, "_current_mode_save_task") as make_task:
                    w.save_current_mode_output()
                    make_task.assert_not_called()
                self.app.processEvents()
                self.assertIsNone(getattr(w, f"_active_{kind}_thread"))
                self.assertFalse(w._save_job_is_running())
                thread.deleteLater()

    def test_cm_csv_and_exported_png_match_legacy_occlusion_io(self):
        w = self.window
        source = pd.DataFrame({
            "frame": range(1, 7), "nose.x": [50] * 6,
            "nose.y": [19, 20, 21, 129, 130, 131], "tail.x": [199] * 6,
            "tail.y": [199] * 6, "confidence": [0.9] * 6,
        })
        source.to_csv(w.csv_path, index=False)
        w.load_csv(w.csv_path, auto_matched=False)
        ws, obj = w.object3d_workspace, w._object3d_object()
        ws.references[0].length, ws.references[0].unit = 10, "cm"
        bottom_ref = LengthReference("Bottom scale", 20, "cm", [(10, 10), (110, 10)])
        ws.references.append(bottom_ref)
        obj.top.margin = obj.bottom.margin = 2
        obj.bottom.reference_id = bottom_ref.id
        w._object3d_changed()
        w.current_output_prefix_edit.setText("three")
        w.current_output_suffix_edit.setText("audit")
        three_path = w._occlusion_output_path()
        self.assertEqual(three_path.name, "three_a_occlusion_audit.csv")
        w.save_current_mode_output()
        self.wait_save()
        three = pd.read_csv(three_path)
        self.assertEqual(three["Box.nose.occ"].tolist(), [0, 1, 1, 1, 1, 0])
        self.assertEqual(three["Box.tail.occ"].tolist(), [0] * 6)
        pd.testing.assert_frame_equal(three[source.columns], source)
        with patch("tab_mixins.occlusion_3d_tab.QFileDialog.getExistingDirectory") as choose:
            w._object3d_export()
            choose.assert_not_called()
        folder = w.save_folder / "masks"
        self.assertTrue((folder / "masks_manifest.json").exists())
        png = read_mask_png(folder / "Box.png")
        np.testing.assert_array_equal(png > 0, ws.render(obj) > 0)
        w._open_pipeline_settings(w.TAB_OCCLUSION)
        with patch("tab_mixins.occlusion_tab.QFileDialog.getExistingDirectory", return_value=str(folder)):
            w.import_mask_folder()
        self.assertEqual(w.mask_records["Box"].margin, 0)
        w.current_output_prefix_edit.setText("legacy")
        legacy_path = w._occlusion_output_path()
        self.assertEqual(legacy_path.name, "legacy_a_occlusion_audit.csv")
        w.save_current_mode_output()
        self.wait_save()
        pd.testing.assert_frame_equal(pd.read_csv(legacy_path), three)

    def test_legacy_export_does_not_lose_editable_3d_sources_in_shared_folder(self):
        w = self.window
        w._object3d_export()
        folder = w._mask_export_folder()
        original = w.object3d_workspace.render(w._object3d_object())
        w._open_pipeline_settings(w.TAB_OCCLUSION)
        with patch("tab_mixins.occlusion_tab.QFileDialog.getExistingDirectory", return_value=str(folder)):
            w.import_mask_folder()
        w.export_masks()
        manifest = json.loads((folder / "masks_manifest.json").read_text())
        self.assertNotIn("objects_3d", manifest)
        restored = load_bundle(folder, 200, 200, video_key=w._object3d_video_key)
        np.testing.assert_array_equal(restored.render(restored.objects[0]), original)
        self.assertEqual(restored.objects[0].top.mask[80, 50], 0)
        self.assertEqual(original[80, 50], 1)

    def test_import_into_empty_line_editor_activates_restored_reference(self):
        w = self.window
        folder = self.root / "restored"
        save_bundle(w.object3d_workspace, folder)
        w.object3d_workspace = ObjectWorkspace(200, 200, video_key=w._object3d_video_key)
        w._object3d_selected_id = None
        w._object3d_changed()
        w._object3d_set_edit_target("line")
        self.assertFalse(w.frame_viewer.reference_pick_active)
        with patch("tab_mixins.occlusion_3d_tab.QFileDialog.getExistingDirectory", return_value=str(folder)):
            w._object3d_import_folder()
        self.assertTrue(w.frame_viewer.reference_pick_active)
        self.assertEqual(w.frame_viewer.reference_pick_points, [(10, 10), (110, 10)])

    def test_unicode_png_can_import_into_active_face_and_legacy_metadata_keeps_name(self):
        w = self.window
        path = self.root / "\uc0c8\ub85c\uc6b4\uba74.png"
        mask = rectangle((200, 200), (80, 80, 100, 100))
        self.assertTrue(write_mask_png(path, mask * 255))
        with patch("tab_mixins.occlusion_3d_tab.QFileDialog.getOpenFileName", return_value=(str(path), "PNG")):
            w._object3d_import_png()
        np.testing.assert_array_equal(w._object3d_object().top.mask, mask)
        w._import_mask_file(path, {"name": "Original_name_", "margin": 0})
        self.assertIn("Original_name_", w.mask_records)

    def test_top_only_transform_and_undo_preserve_bottom(self):
        w = self.window
        obj = w._object3d_object()
        bottom = obj.bottom.mask.copy()
        top = obj.top.mask.copy()
        w._dispatch_occlusion_edit("translate_selected_mask", (20, 0))
        np.testing.assert_array_equal(obj.bottom.mask, bottom)
        self.assertFalse(np.array_equal(obj.top.mask, top))
        w.undo_active_mask_edit()
        np.testing.assert_array_equal(w._object3d_object().top.mask, top)

    def test_reference_updates_every_dependent_face_and_rename_keeps_binding(self):
        w = self.window
        obj = w._object3d_object()
        obj.top.margin = obj.bottom.margin = 2
        w._object3d_changed()
        before = w.object3d_workspace.margin_pixels(obj, "top")
        reference = w.object3d_workspace.references[0]
        reference.name = "Renamed ruler"
        w.object3d_reference_length.setValue(2)
        self.assertAlmostEqual(w.object3d_workspace.margin_pixels(obj, "top"), before / 2)
        self.assertAlmostEqual(w.object3d_workspace.margin_pixels(obj, "bottom"), before / 2)
        self.assertEqual(w.object3d_face_controls["bottom"]["reference"].currentData(), reference.id)

    def test_video_switch_requires_remeasurement_then_restores_video_specific_state(self):
        w = self.window
        original = w._object3d_object().top.mask.copy()
        other = self.make_video("b", (400, 400))
        w.load_video(other)
        self.assertFalse(w.object3d_workspace.references[0].points)
        self.assertFalse(w.save_current_button.isEnabled())
        self.assertEqual(w._object3d_object().top.mask.shape, (400, 400))
        w.object3d_workspace.references[0].points = [(20, 20), (220, 20)]
        w._object3d_changed()
        self.assertTrue(w.save_current_button.isEnabled())
        w._object3d_object().top.mask[:] = 0
        w._object3d_changed()
        w.load_video(self.video)
        self.assertTrue(w.save_current_button.isEnabled())
        np.testing.assert_array_equal(w._object3d_object().top.mask, original)
        self.assertEqual(w.object3d_workspace.references[0].points, [(10, 10), (110, 10)])

    def test_pipeline_placeholder_is_disabled_and_batch_entry_is_blocked(self):
        w = self.window
        placeholders = [widget for widget in w.pipeline_tab.findChildren(QCheckBox) if "3D Object" in widget.text()]
        self.assertEqual(len(placeholders), 1)
        self.assertFalse(placeholders[0].isEnabled())
        self.assertNotIn("occlusion_3d", w.pipeline_stage_controls)
        with patch.object(w, "_select_videos_for_batch") as select:
            w.save_multiple_mode_outputs()
            select.assert_not_called()

    def test_incomplete_included_object_blocks_save_and_excluded_one_does_not(self):
        w = self.window
        obj = copy.deepcopy(w._object3d_object())
        from occlusion_3d import new_id
        obj.id, obj.name = new_id(), "Incomplete"
        obj.top.mask[:] = 0
        w.object3d_workspace.objects.append(obj)
        w._object3d_changed()
        self.assertFalse(w.save_current_button.isEnabled())
        self.assertEqual(w.object3d_summary.text(), "Not ready")
        obj.included = False
        w._object3d_changed()
        self.assertTrue(w.save_current_button.isEnabled())
        self.assertEqual(w.object3d_summary.text(), "Ready")

    def test_reference_clicks_and_endpoint_drag(self):
        w = self.window
        w._object3d_measure(replace=True)
        viewer = w.frame_viewer
        for point in ((20, 20), (120, 20)):
            self.mouse(viewer, QEvent.Type.MouseButtonPress, point)
            self.mouse(viewer, QEvent.Type.MouseButtonRelease, point)
        reference = w.object3d_workspace.references[0]
        np.testing.assert_allclose(reference.points, [(20, 20), (120, 20)])
        self.mouse(viewer, QEvent.Type.MouseButtonPress, (120, 20))
        self.mouse(viewer, QEvent.Type.MouseMove, (160, 20))
        self.mouse(viewer, QEvent.Type.MouseButtonRelease, (160, 20))
        np.testing.assert_allclose(reference.points, [(20, 20), (160, 20)])

    def test_replacing_reference_invalidates_save_immediately_and_undo_restores_it(self):
        w = self.window
        w._object3d_measure(replace=True)
        self.assertFalse(w.save_current_button.isEnabled())
        w.undo_active_mask_edit()
        self.assertTrue(w.save_current_button.isEnabled())
        self.assertTrue(w.frame_viewer.reference_pick_active)
        self.assertEqual(w._object3d_edit_target, "line")

    def test_circle_gesture_commits_face_and_clears_draft(self):
        w = self.window
        obj = w._object3d_object()
        obj.top.mask[:] = 0
        w._object3d_changed()
        w.object3d_face_controls["top"]["tool"].setCurrentIndex(4)
        viewer = w.frame_viewer
        self.mouse(viewer, QEvent.Type.MouseButtonPress, (40, 50))
        self.mouse(viewer, QEvent.Type.MouseMove, (80, 50))
        self.mouse(viewer, QEvent.Type.MouseButtonRelease, (80, 50))
        self.assertEqual(obj.top.mask[50, 60], 1)
        self.assertEqual(obj.top.mask[30, 40], 0)
        self.assertIsNone(viewer.occ_circle_geometry())

    def test_ctrl_shift_rectangle_erases_square_and_brush_undo_is_one_gesture(self):
        w = self.window
        obj = w._object3d_object()
        obj.top.mask = rectangle((200, 200), (20, 20, 100, 100))
        original = obj.top.mask.copy()
        w._object3d_changed()
        modifiers = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        for kind, point in ((QEvent.Type.MouseButtonPress, (40, 40)),
                            (QEvent.Type.MouseMove, (60, 50)),
                            (QEvent.Type.MouseButtonRelease, (60, 50))):
            self.mouse(w.frame_viewer, kind, point, modifiers=modifiers)
        self.assertEqual(obj.top.mask[59, 59], 0)
        self.assertEqual(obj.top.mask[65, 65], 1)
        w.undo_active_mask_edit()
        np.testing.assert_array_equal(w._object3d_object().top.mask, original)
        controls = w.object3d_face_controls["top"]
        controls["tool"].setCurrentIndex(5)
        depth = len(w._object3d_undo)
        for kind, point in ((QEvent.Type.MouseButtonPress, (130, 130)),
                            (QEvent.Type.MouseMove, (140, 140)),
                            (QEvent.Type.MouseButtonRelease, (140, 140))):
            self.mouse(w.frame_viewer, kind, point)
        self.assertEqual(len(w._object3d_undo), depth + 1)
        w.undo_active_mask_edit()
        np.testing.assert_array_equal(w._object3d_object().top.mask, original)

    def test_folder_import_restores_recipes_and_can_edit_after_import(self):
        w = self.window
        folder = self.root / "bundle"
        folder.mkdir()
        with patch.object(w, "_mask_export_folder", return_value=folder):
            w._object3d_export()
        with patch("tab_mixins.occlusion_3d_tab.QFileDialog.getExistingDirectory", return_value=str(folder)):
            w._object3d_import_folder()
        self.assertEqual(len(w.object3d_workspace.objects), 2)
        self.assertTrue(w.save_current_button.isEnabled())
        w.object3d_combo.setCurrentIndex(1)
        imported = w._object3d_object()
        self.assertNotEqual(imported.top.reference_id, w.object3d_workspace.objects[0].top.reference_id)
        self.assertEqual(imported.name, "Box imported")
        before = w.object3d_workspace.objects[0].top.mask.copy()
        w._dispatch_occlusion_edit("translate_selected_mask", (20, 0))
        np.testing.assert_array_equal(w.object3d_workspace.objects[0].top.mask, before)

    def test_square_room_still_respects_chamber_and_room_overlap(self):
        from shared import RoomRecord
        from PyQt6.QtGui import QColor
        w = self.window
        w._open_pipeline_settings(w.TAB_CHAMBER)
        w.chamber_mask = rectangle((200, 200), (0, 0, 80, 199))
        w.chamber_boundary_mode = "custom"
        w.room_records = {
            "Existing": RoomRecord("Existing", QColor("red"), rectangle((200, 200), (50, 50, 55, 55))),
            "Square": RoomRecord("Square", QColor("blue"), np.zeros((200, 200), np.uint8)),
        }
        w.selected_room_name = "Square"
        w._rebuild_room_list()
        w.chamber_edit_room_radio.setChecked(True)
        w.chamber_shape_combo.setCurrentIndex(5)
        for kind, point in ((QEvent.Type.MouseButtonPress, (40, 40)),
                            (QEvent.Type.MouseMove, (100, 70)),
                            (QEvent.Type.MouseButtonRelease, (100, 70))):
            self.mouse(w.frame_viewer, kind, point)
        mask = w.room_records["Square"].mask
        self.assertEqual(mask[90, 70], 1)
        self.assertEqual(mask[90, 90], 0)
        self.assertEqual(mask[52, 52], 0)
        self.assertIn("clipped", w.statusBar().currentMessage())

    def test_faces_can_select_independent_references(self):
        w = self.window
        ws = w.object3d_workspace
        other = LengthReference("Top ruler", 0.5, "m", [(10, 10), (110, 10)])
        ws.references.append(other)
        obj = w._object3d_object()
        obj.top.margin = obj.bottom.margin = 2
        w._object3d_changed()
        controls = w.object3d_face_controls["bottom"]
        self.assertTrue(controls["reference"].isEnabled())
        controls["reference"].setCurrentIndex(controls["reference"].findData(other.id))
        self.assertAlmostEqual(ws.margin_pixels(obj, "top"), 2)
        self.assertAlmostEqual(ws.margin_pixels(obj, "bottom"), 4)
        top_reference = w.object3d_face_controls["top"]["reference"]
        top_reference.setCurrentIndex(top_reference.findData(other.id))
        self.assertAlmostEqual(ws.margin_pixels(obj, "top"), 4)
        controls["reference"].setCurrentIndex(controls["reference"].findData(ws.references[0].id))
        self.assertAlmostEqual(ws.margin_pixels(obj, "bottom"), 2)
        self.assertFalse(obj.same_reference)

    def test_current_edit_buttons_control_only_the_selected_target(self):
        w = self.window
        obj = w._object3d_object()
        original = obj.top.mask.copy()
        for target in ("line", "bottom", "top", "line"):
            w.object3d_target_buttons[target].click()
            self.assertEqual(w._object3d_edit_target, target)
            self.assertEqual([key for key, button in w.object3d_target_buttons.items() if button.isChecked()], [target])
            self.assertEqual(w.frame_viewer.reference_pick_active, target == "line")
            self.assertEqual(w.frame_viewer.mode == "reference_line", target == "line")
            self.assertEqual(not w.object3d_references_group.isHidden(), target == "line")
            for face, controls in w.object3d_face_controls.items():
                self.assertEqual(not controls["group"].isHidden(), target == face)
        # Clicking Line again keeps it selected, rather than silently returning to a face.
        w.object3d_target_buttons["line"].click()
        self.assertTrue(w.frame_viewer.reference_pick_active)
        w._switch_region_edit_mode_shortcut(True)
        w._object3d_affine(scale=1.5)
        w._dispatch_occlusion_edit("translate_selected_mask", (20, 0))
        np.testing.assert_array_equal(obj.top.mask, original)
        self.assertIsNone(w._active_mask_clipboard_item())

    def test_object_first_workflow_can_open_line_panel_to_add_first_reference(self):
        w = self.window
        w.object3d_workspace = ObjectWorkspace(200, 200, video_key=w._object3d_video_key)
        w._object3d_changed()
        with patch("tab_mixins.occlusion_3d_tab.QInputDialog.getText", return_value=("First object", True)):
            w._object3d_add_object()
        self.assertEqual(w._object3d_edit_target, "top")
        self.assertTrue(w.object3d_target_buttons["line"].isEnabled())
        w.object3d_target_buttons["line"].click()
        self.assertFalse(w.object3d_references_group.isHidden())
        with patch("tab_mixins.occlusion_3d_tab.QInputDialog.getText", return_value=("First ruler", True)):
            w._object3d_add_reference()
        self.assertTrue(w.frame_viewer.reference_pick_active)
        self.assertEqual(w._object3d_reference().name, "First ruler")
        outer = w.mode_tabs.widget(w.TAB_OCCLUSION_3D).layout()
        self.assertIs(outer.itemAt(0).widget(), w.object3d_objects_group)
        self.assertIs(outer.itemAt(1).layout().itemAt(0).widget(), w.object3d_target_buttons["line"])
        self.assertIs(outer.itemAt(2).widget(), w.object3d_scroll)

    def test_hover_shows_both_faces_before_envelope_in_every_edit_target(self):
        w = self.window
        viewer = w.frame_viewer
        for target in ("top", "bottom", "line"):
            w._object3d_set_edit_target(target)
            for face, point, bbox in (("Top", (50, 50), "bbox=(40, 40)-(60, 60)"),
                                      ("Bottom", (50, 110), "bbox=(40, 100)-(60, 120)")):
                with self.subTest(target=target, face=face):
                    info = viewer._build_mask_hover_info(point)
                    self.assertIsNotNone(info)
                    self.assertEqual(info.title, f"Mask: Box / {face} Face")
                    self.assertIn("area=441 px", info.lines)
                    self.assertIn("width=21 px | height=21 px", info.lines)
                    self.assertIn(bbox, info.lines)
            envelope = viewer._build_mask_hover_info((50, 80))
            self.assertEqual(envelope.title, "3D Object: Box")
            self.assertIn("area=1701 px", envelope.lines)
            self.assertIsNone(viewer._build_mask_hover_info((150, 80)))

    def test_face_outline_hit_has_priority_over_envelope_interior(self):
        w = self.window
        w._object3d_set_edit_target("bottom")
        viewer = w.frame_viewer
        # Five pixels below the top outline is also inside the full envelope.
        self.assertEqual(viewer._mask_name_at_point((50, 65)), "face:top")
        self.assertEqual(viewer._build_mask_hover_info((50, 65)).title, "Mask: Box / Top Face")
        # Beyond the outline tolerance, the envelope still provides its own info.
        self.assertEqual(viewer._build_mask_hover_info((50, 80)).title, "3D Object: Box")

    def test_overlapping_face_hover_prefers_the_current_face(self):
        w = self.window
        obj = w._object3d_object()
        obj.bottom.mask = obj.top.mask.copy()
        w._object3d_changed()
        for face in ("bottom", "top"):
            w._object3d_set_edit_target(face)
            info = w.frame_viewer._build_mask_hover_info((50, 50))
            self.assertEqual(info.title, f"Mask: Box / {face.title()} Face")

    def test_face_hover_events_and_double_click_can_reach_unselected_face(self):
        w = self.window
        viewer = w.frame_viewer
        for target in ("bottom", "line"):
            w._object3d_set_edit_target(target)
            position = viewer._image_to_widget((50, 50))
            move = QMouseEvent(QEvent.Type.MouseMove, position, position,
                               Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
                               Qt.KeyboardModifier.NoModifier)
            QApplication.sendEvent(viewer, move)
            self.assertIsNotNone(viewer._mask_hover_info)
            self.assertEqual(viewer._mask_hover_info.title, "Mask: Box / Top Face")
        # Double-click selection follows the same face priority as inspection.
        w._object3d_set_edit_target("bottom")
        position = viewer._image_to_widget((50, 50))
        double_click = QMouseEvent(QEvent.Type.MouseButtonDblClick, position, position,
                                   Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                                   Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(viewer, double_click)
        self.assertEqual(w._object3d_edit_target, "top")

    def test_legacy_mask_hit_and_hover_keep_existing_selection_order(self):
        from PyQt6.QtGui import QColor
        from shared import MaskRecord
        w = self.window
        w._open_pipeline_settings(w.TAB_OCCLUSION)
        viewer = w.frame_viewer
        masks = {
            "First": MaskRecord("First", QColor("red"), rectangle((200, 200), (20, 20, 180, 180))),
            "Selected": MaskRecord("Selected", QColor("blue"), rectangle((200, 200), (40, 40, 60, 60))),
        }
        viewer.set_mask_records(masks, "Selected")
        self.assertEqual(viewer._build_mask_hover_info((50, 50)).title, "Mask: Selected")
        self.assertEqual(viewer._build_mask_hover_info((50, 65)).title, "Mask: First")
        viewer.set_mask_records(masks, None)
        self.assertEqual(viewer._build_mask_hover_info((50, 50)).title, "Mask: First")

    def test_active_reference_is_drawn_once_and_follows_reference_selection(self):
        w = self.window
        reference = w.object3d_workspace.references[0]
        other = LengthReference("Second ruler", 1, "m", [(20, 30), (120, 30)])
        w.object3d_workspace.references.append(other)
        w._object3d_changed()
        w.object3d_target_buttons["line"].click()
        w.frame_viewer.reference_pick_points = [(30, 40), (130, 40)]
        rows = w.frame_viewer._reference_lines_for_display()
        self.assertEqual(len(rows), 2)
        self.assertEqual([name for name, points, active in rows].count(reference.name), 1)
        self.assertEqual([points for name, points, active in rows if active], [[(30, 40), (130, 40)]])
        combo = w.object3d_reference_combo
        combo.setCurrentIndex(combo.findData(other.id))
        self.assertEqual(w._object3d_edit_target, "line")
        self.assertTrue(w.frame_viewer.reference_pick_active)
        rows = w.frame_viewer._reference_lines_for_display()
        self.assertEqual([name for name, points, active in rows if active], [other.name])
        self.assertEqual(w.frame_viewer.reference_pick_points, other.points)

    def test_reference_name_badges_do_not_overlap_or_escape_viewer(self):
        bounds = QRectF(0, 0, 320, 180)
        occupied = []
        for anchor in (QPointF(3, 2), QPointF(3, 2), QPointF(3, 2), QPointF(315, 175)):
            label = main.FrameViewer._reference_label_rect(anchor, 110, 22, bounds, occupied)
            self.assertIsNotNone(label)
            self.assertTrue(bounds.contains(label))
            self.assertFalse(any(label.intersects(other) for other in occupied))
            occupied.append(label)

    def test_adding_reference_enters_line_target_without_a_second_mode_button(self):
        w = self.window
        with patch("tab_mixins.occlusion_3d_tab.QInputDialog.getText", return_value=("New ruler", True)):
            w._object3d_add_reference()
        self.assertEqual(w._object3d_edit_target, "line")
        self.assertTrue(w.object3d_target_buttons["line"].isChecked())
        self.assertTrue(w.frame_viewer.reference_pick_active)
        self.assertEqual(w._object3d_reference().name, "New ruler")
        self.assertEqual(w.frame_viewer.reference_pick_points, [])

    def test_repeated_rotation_preserves_source_and_clipboard_targets_active_face(self):
        w = self.window
        obj = w._object3d_object()
        original = obj.top.mask.copy()
        bottom = obj.bottom.mask.copy()
        w.object3d_face_controls["top"]["transform"].setChecked(True)
        for _ in range(4):
            w._object3d_affine(angle=90)
        np.testing.assert_array_equal(obj.top.mask, original)
        np.testing.assert_array_equal(obj.bottom.mask, bottom)
        w.copy_active_mask_to_clipboard()
        w._object3d_activate_face("bottom")
        self.assertTrue(w._mask_paste_action_state()[1])
        w.paste_mask_from_clipboard()
        np.testing.assert_array_equal(obj.bottom.mask, original)
        np.testing.assert_array_equal(obj.top.mask, original)

    def test_square_preview_commit_and_legacy_occlusion_isolation(self):
        w = self.window
        viewer = w.frame_viewer
        obj = w._object3d_object()
        obj.top.mask[:] = 0
        w._object3d_changed()
        shift = Qt.KeyboardModifier.ShiftModifier
        self.mouse(viewer, QEvent.Type.MouseButtonPress, (40, 40), modifiers=shift)
        self.mouse(viewer, QEvent.Type.MouseMove, (100, 70), modifiers=shift)
        points = viewer.occ_rect_points
        self.assertAlmostEqual(points[1][0] - points[0][0], points[2][1] - points[1][1])
        self.mouse(viewer, QEvent.Type.MouseButtonRelease, (100, 70), modifiers=shift)
        ys, xs = np.where(obj.top.mask)
        self.assertEqual(xs.max() - xs.min(), ys.max() - ys.min())
        self.assertFalse(w.mask_records)
        w._open_pipeline_settings(w.TAB_OCCLUSION)
        self.assertFalse(viewer.object3d_active)
        self.assertFalse(viewer.mask_records)


if __name__ == "__main__":
    unittest.main()
