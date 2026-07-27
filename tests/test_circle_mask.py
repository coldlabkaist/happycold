import tempfile
import unittest
from pathlib import Path

import cv2

from tab_mixins.circle_tab import (
    build_circle_mask,
    infer_circle_geometry_from_mask,
    infer_circular_mask_geometry,
    load_circle_mask_bundle,
    write_circle_mask_bundle,
)


class CircleMaskTests(unittest.TestCase):
    def test_circle_mask_geometry_can_be_inferred(self) -> None:
        mask = build_circle_mask(100, 80, (40.0, 30.0), 12.0)
        geometry = infer_circle_geometry_from_mask(mask)
        self.assertIsNotNone(geometry)
        center, radius = geometry
        self.assertAlmostEqual(center[0], 40.0, delta=0.5)
        self.assertAlmostEqual(center[1], 30.0, delta=0.5)
        self.assertAlmostEqual(radius, 12.0, delta=0.6)

    def test_circle_mask_bundle_round_trip_preserves_editable_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mask_path = Path(directory) / "circle.png"
            manifest_path = mask_path.with_suffix(".json")
            write_circle_mask_bundle(
                mask_path,
                manifest_path,
                width=120,
                height=90,
                center=(48.5, 36.5),
                base_radius=14.25,
                margin=5,
            )

            center, base_radius, margin = load_circle_mask_bundle(
                mask_path,
                width=120,
                height=90,
            )
            self.assertEqual(center, (48.5, 36.5))
            self.assertAlmostEqual(base_radius, 14.25)
            self.assertEqual(margin, 5)
            image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(image)
            self.assertGreater(int(image[36, 48]), 0)

    def test_png_without_manifest_imports_as_zero_margin_circle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mask_path = Path(directory) / "plain.png"
            mask = build_circle_mask(80, 60, (30.0, 25.0), 10.0) * 255
            self.assertTrue(cv2.imwrite(str(mask_path), mask))

            center, base_radius, margin = load_circle_mask_bundle(
                mask_path,
                width=80,
                height=60,
            )
            self.assertAlmostEqual(center[0], 30.0, delta=0.5)
            self.assertAlmostEqual(center[1], 25.0, delta=0.5)
            self.assertAlmostEqual(base_radius, 10.0, delta=0.6)
            self.assertEqual(margin, 0)


    def test_strict_circle_paste_accepts_circle_mask(self) -> None:
        mask = build_circle_mask(100, 80, (40.0, 30.0), 12.0)

        geometry = infer_circular_mask_geometry(mask)

        self.assertIsNotNone(geometry)
        center, radius = geometry
        self.assertAlmostEqual(center[0], 40.0, delta=0.5)
        self.assertAlmostEqual(center[1], 30.0, delta=0.5)
        self.assertAlmostEqual(radius, 12.0, delta=0.8)

    def test_strict_circle_paste_rejects_rectangular_mask(self) -> None:
        import numpy as np

        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[20:55, 25:75] = 1

        self.assertIsNone(infer_circular_mask_geometry(mask))


if __name__ == "__main__":
    unittest.main()
