"""Shared reference gestures, keyboard routing, constraints, and persistence."""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from reference_line import reference_endpoint
from occlusion_3d import LengthReference, reference_pixel_length
import test_annotation_settings as settings_tests
import test_direction as direction_tests
import test_occlusion_3d as object_tests


class ReferenceGeometryTests(unittest.TestCase):
    def test_exact_one_pixel_survives_coordinate_roundoff_but_shorter_lines_do_not(self):
        anchor = (100, 100)
        end = reference_endpoint(anchor, (103, 104), 1)
        self.assertEqual(reference_pixel_length([anchor, end]), 1)
        self.assertEqual(LengthReference("Ruler", 1, "m", [anchor, end]).pixels_per_metre, 1)
        with self.assertRaises(ValueError):
            reference_pixel_length([anchor, (100.6, 100.799)])

    def test_exact_subpixel_lengths_all_quadrants_and_axis_snapping(self):
        for pointer in ((80, 70), (20, 70), (20, 30), (80, 30)):
            point = reference_endpoint((50, 50), pointer, 37.25)
            self.assertAlmostEqual(math.dist((50, 50), point), 37.25, places=12)
        self.assertEqual(reference_endpoint((50, 50), (80, 70), 25, True), (75, 50))
        self.assertEqual(reference_endpoint((50, 50), (20, 70), 25, True), (25, 50))
        self.assertEqual(reference_endpoint((50, 50), (60, 90), 25, True), (50, 75))
        self.assertEqual(reference_endpoint((50, 50), (60, 10), 25, True), (50, 25))
        self.assertEqual(reference_endpoint((50, 50), (80, 70), None, True), (80, 50))
        self.assertIsNone(reference_endpoint((50, 50), (50, 50), 25, True))


class ReferenceLineQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @contextmanager
    def editing(self, kind):
        if kind == "direction":
            direction_tests.DirectionQtTests.setUpClass()
            fixture = direction_tests.DirectionQtTests("test_real_csv_zero_angle_and_output_naming")
        else:
            object_tests.ObjectQtTests.setUpClass()
            fixture = object_tests.ObjectQtTests("test_adding_reference_enters_line_target_without_a_second_mode_button")
        fixture.setUp()
        try:
            w = fixture.w if kind == "direction" else fixture.window
            w.resize(1320, 900)
            w.show()
            w.activateWindow()
            self.app.processEvents()
            if kind == "direction":
                w.direction_range_mode.setCurrentIndex(w.direction_range_mode.findData("physical"))
                w.direction_calibration_toggle.setChecked(True)
                controls = w.direction_line_input
                ref = w._direction_edit_reference
                redraw = w.direction_redraw_line.click
            else:
                w.object3d_target_buttons["line"].click()
                controls = w.object3d_line_input
                ref = w._object3d_reference
                redraw = w.object3d_redraw_button.click
            redraw()
            self.click(w, (20, 20))
            self.click(w, (120, 20))
            self.assertEqual(ref().points, [(20, 20), (120, 20)])
            yield w, controls, ref, redraw
        finally:
            fixture.doCleanups()

    def mouse(self, w, kind, point, *, modifiers=Qt.KeyboardModifier.NoModifier, dragging=False):
        viewer = w.frame_viewer
        position = viewer._image_to_widget(point)
        event = QMouseEvent(kind, position, position,
            Qt.MouseButton.NoButton if kind == QEvent.Type.MouseMove else Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton if kind == QEvent.Type.MouseMove and dragging else Qt.MouseButton.NoButton if kind in (QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease) else Qt.MouseButton.LeftButton,
            modifiers)
        QApplication.sendEvent(viewer, event)

    def click(self, w, point, *, modifiers=Qt.KeyboardModifier.NoModifier):
        self.mouse(w, QEvent.Type.MouseButtonPress, point, modifiers=modifiers)
        self.mouse(w, QEvent.Type.MouseButtonRelease, point, modifiers=modifiers)

    def assert_points_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for point, target in zip(actual, expected):
            for value, wanted in zip(point, target):
                self.assertAlmostEqual(value, wanted, places=9)

    def test_rubber_band_and_shift_press_release_update_without_moving(self):
        for kind in ("direction", "object3d"):
            with self.subTest(kind=kind), self.editing(kind) as (w, controls, ref, redraw):
                original = list(ref().points)
                redraw()
                self.assertEqual(ref().points, original)
                self.assertFalse(w.save_current_button.isEnabled())
                self.click(w, (20, 20))
                self.mouse(w, QEvent.Type.MouseMove, (70, 45))
                viewer = w.frame_viewer
                self.assert_points_close(viewer.reference_display_points(), [(20, 20), (70, 45)])
                QTest.keyPress(viewer, Qt.Key.Key_Shift)
                self.assert_points_close(viewer.reference_display_points(), [(20, 20), (70, 20)])
                QTest.keyRelease(viewer, Qt.Key.Key_Shift)
                self.assert_points_close(viewer.reference_display_points(), [(20, 20), (70, 45)])
                self.assertEqual(ref().points, original)
                self.click(w, (70, 45), modifiers=Qt.KeyboardModifier.ShiftModifier)
                self.assert_points_close(ref().points, [(20, 20), (70, 20)])
                self.assertTrue(w.save_current_button.isEnabled())

    def test_canvas_numeric_input_uses_exact_length_and_does_not_select_objects(self):
        for kind in ("direction", "object3d"):
            with self.subTest(kind=kind), self.editing(kind) as (w, controls, ref, redraw):
                redraw()
                self.click(w, (20, 20))
                self.mouse(w, QEvent.Type.MouseMove, (60, 50))
                with patch.object(w, "_select_mask_by_slot") as select:
                    QTest.keyClicks(w.frame_viewer, "75.25")
                    QTest.keyClick(w.frame_viewer, Qt.Key.Key_Return)
                    select.assert_not_called()
                self.assertAlmostEqual(math.dist(*ref().points), 75.25, places=10)
                self.assertAlmostEqual(ref().points[1][0], 80.2, places=10)
                self.assertAlmostEqual(ref().points[1][1], 65.15, places=10)
                self.assertTrue(controls.fixed.isChecked())
                self.assertFalse(w.frame_viewer.reference_line_pending)
                if kind == "object3d":
                    w.object3d_target_buttons["top"].click()
                else:
                    w._open_pipeline_settings(w.TAB_OCCLUSION)
                w.frame_viewer.setFocus()
                with patch.object(w, "_select_mask_by_slot") as select:
                    QTest.keyClick(w.frame_viewer, Qt.Key.Key_1)
                    select.assert_called_once_with(0)

    def test_numeric_panel_resize_is_transactional_and_undo_is_one_action(self):
        for kind in ("direction", "object3d"):
            with self.subTest(kind=kind), self.editing(kind) as (w, controls, ref, redraw):
                original = list(ref().points)
                real_length = ref().length
                controls.length.setFocus()
                controls.length.lineEdit().selectAll()
                with patch.object(w, "_select_mask_by_slot") as select:
                    QTest.keyClicks(controls.length.lineEdit(), "60.5")
                    select.assert_not_called()
                self.assertEqual(ref().points, original)
                QTest.keyClick(controls.length, Qt.Key.Key_Return)
                self.assertTrue(controls.fixed.isChecked())
                self.assertAlmostEqual(math.dist(*ref().points), 60.5)
                self.assertEqual(ref().points[0], (20, 20))
                self.assertEqual(ref().length, real_length)
                w.undo_active_mask_edit()
                self.assertEqual(ref().points, original)
                self.assertTrue(w.save_current_button.isEnabled())

    def test_outside_frame_is_not_clipped_and_escape_restores_original(self):
        for kind in ("direction", "object3d"):
            with self.subTest(kind=kind), self.editing(kind) as (w, controls, ref, redraw):
                original = list(ref().points)
                redraw()
                self.click(w, (170, 60))
                self.mouse(w, QEvent.Type.MouseMove, (190, 60))
                QTest.keyClicks(w.frame_viewer, "60")
                self.assertAlmostEqual(w.frame_viewer.reference_display_points()[1][0], 230)
                self.assertAlmostEqual(w.frame_viewer.reference_display_points()[1][1], 60)
                QTest.keyClick(w.frame_viewer, Qt.Key.Key_Return)
                self.assertEqual(ref().points, original)
                self.assertTrue(w.frame_viewer.reference_line_pending)
                self.assertIn("outside", w.frame_viewer.reference_instruction())
                self.assertFalse(w.save_current_button.isEnabled())
                QTest.keyClick(controls.fixed, Qt.Key.Key_Escape)
                self.assertEqual(w.frame_viewer.reference_pick_points, original)
                self.assertEqual(ref().points, original)
                self.assertTrue(w.save_current_button.isEnabled())

    def test_blank_negative_and_subpixel_input_cannot_commit(self):
        for kind in ("direction", "object3d"):
            with self.subTest(kind=kind), self.editing(kind) as (w, controls, ref, redraw):
                original = list(ref().points)
                for text in ("0", "-5", "0.5"):
                    redraw()
                    self.click(w, (20, 20))
                    self.mouse(w, QEvent.Type.MouseMove, (60, 40))
                    QTest.keyClicks(w.frame_viewer, text)
                    QTest.keyClick(w.frame_viewer, Qt.Key.Key_Return)
                    self.assertEqual(ref().points, original)
                    self.assertTrue(w.frame_viewer.reference_line_pending)
                QTest.keyClick(w.frame_viewer, Qt.Key.Key_Backspace)
                QTest.keyClick(w.frame_viewer, Qt.Key.Key_Backspace)
                QTest.keyClick(w.frame_viewer, Qt.Key.Key_Backspace)
                QTest.keyClick(w.frame_viewer, Qt.Key.Key_Return)
                self.assertEqual(ref().points, original)
                self.assertTrue(w.frame_viewer.reference_line_pending)
                QTest.keyClick(w.frame_viewer, Qt.Key.Key_Escape)

    def test_fixed_length_is_in_video_pixels_at_zoom_and_shift_constrains_it(self):
        for kind in ("direction", "object3d"):
            with self.subTest(kind=kind), self.editing(kind) as (w, controls, ref, redraw):
                viewer = w.frame_viewer
                viewer._apply_zoom(2)
                redraw()
                self.click(w, (80, 55))
                controls.length.setValue(27.25)
                self.mouse(w, QEvent.Type.MouseMove, (100, 85), modifiers=Qt.KeyboardModifier.ShiftModifier)
                self.click(w, (100, 85), modifiers=Qt.KeyboardModifier.ShiftModifier)
                self.assertAlmostEqual(math.dist(*ref().points), 27.25)
                self.assertEqual(ref().points[0][0], ref().points[1][0])
                self.assertAlmostEqual(ref().points[1][1], 82.25)

    def test_pixel_input_preferences_survive_restart_independently_per_tab(self):
        settings_tests.AnnotationPreferenceTests.setUpClass()
        fixture = settings_tests.AnnotationPreferenceTests("test_invalid_new_preferences_fall_back_without_crashing")
        fixture.setUp()
        try:
            w = fixture.window()
            w.direction_line_input.length.setValue(123.25)
            w.object3d_line_input.length.setValue(67.5)
            w.object3d_line_input.fixed.setChecked(False)
            fixture.close(w)
            restored = fixture.window()
            self.assertEqual(restored.direction_line_input.length.value(), 123.25)
            self.assertTrue(restored.direction_line_input.fixed.isChecked())
            self.assertEqual(restored.object3d_line_input.length.value(), 67.5)
            self.assertFalse(restored.object3d_line_input.fixed.isChecked())
        finally:
            fixture.doCleanups()

    def test_fixed_endpoint_drag_preserves_length_and_free_drag_can_change_it(self):
        for kind in ("direction", "object3d"):
            with self.subTest(kind=kind), self.editing(kind) as (w, controls, ref, redraw):
                controls.fixed.setChecked(True)
                self.mouse(w, QEvent.Type.MouseButtonPress, (120, 20))
                self.mouse(w, QEvent.Type.MouseMove, (140, 80), dragging=True)
                self.mouse(w, QEvent.Type.MouseButtonRelease, (140, 80))
                self.assertAlmostEqual(math.dist(*ref().points), 100, places=10)
                self.assertEqual(ref().points[0], (20, 20))
                opposite = ref().points[1]
                self.mouse(w, QEvent.Type.MouseButtonPress, (20, 20))
                self.mouse(w, QEvent.Type.MouseMove, (50, 80), modifiers=Qt.KeyboardModifier.ShiftModifier, dragging=True)
                self.mouse(w, QEvent.Type.MouseButtonRelease, (50, 80), modifiers=Qt.KeyboardModifier.ShiftModifier)
                self.assertEqual(ref().points[1], opposite)
                self.assertAlmostEqual(ref().points[0][1], opposite[1])
                self.assertAlmostEqual(math.dist(*ref().points), 100, places=10)
                controls.fixed.setChecked(False)
                self.mouse(w, QEvent.Type.MouseButtonPress, opposite)
                self.mouse(w, QEvent.Type.MouseMove, (130, 90), dragging=True)
                self.mouse(w, QEvent.Type.MouseButtonRelease, (130, 90))
                self.assertAlmostEqual(ref().points[1][0], 130)
                self.assertAlmostEqual(ref().points[1][1], 90)
                self.assertNotAlmostEqual(math.dist(*ref().points), 100)


if __name__ == "__main__":
    unittest.main()
