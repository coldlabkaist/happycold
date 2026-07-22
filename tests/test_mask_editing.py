import unittest

import cv2
import numpy as np

from shared import MaskTransformSource, paint_brush


class MaskEditingTests(unittest.TestCase):
    def test_transform_source_returns_exact_mask_for_full_rotation(self) -> None:
        mask = np.zeros((96, 128), dtype=np.uint8)
        cv2.rectangle(mask, (24, 20), (92, 70), 1, -1)
        cv2.circle(mask, (84, 36), 13, 0, -1)

        source = MaskTransformSource.from_mask(mask)

        np.testing.assert_array_equal(source.render(360.0, 1.0), mask)

    def test_signed_distance_rotation_keeps_solid_mask_connected(self) -> None:
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.ellipse(mask, (64, 64), (34, 18), 0, 0, 360, 1, -1)
        source = MaskTransformSource.from_mask(mask)

        transformed = source.render(37.0, 1.25)

        self.assertEqual(set(np.unique(transformed)), {0, 1})
        component_count, _ = cv2.connectedComponents(transformed)
        self.assertEqual(component_count, 2)
        self.assertGreater(int(np.count_nonzero(transformed)), int(np.count_nonzero(mask)))

    def test_signed_distance_rotation_preserves_mask_hole(self) -> None:
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.circle(mask, (64, 64), 34, 1, -1)
        cv2.circle(mask, (64, 64), 13, 0, -1)
        source = MaskTransformSource.from_mask(mask)

        transformed = source.render(31.0, 1.15)

        self.assertEqual(int(transformed[64, 64]), 0)
        self.assertGreater(int(np.count_nonzero(transformed[40:88, 40:88])), 0)

    def test_brush_segment_is_continuous_and_erasable(self) -> None:
        mask = np.zeros((96, 96), dtype=np.uint8)
        start = (8.0, 12.0)
        end = (82.0, 75.0)

        paint_brush(mask, start, end, radius=4, value=1)

        for t in np.linspace(0.0, 1.0, 80):
            x = int(round(start[0] + (end[0] - start[0]) * float(t)))
            y = int(round(start[1] + (end[1] - start[1]) * float(t)))
            self.assertEqual(int(mask[y, x]), 1)

        paint_brush(mask, start, end, radius=4, value=0)
        self.assertFalse(np.any(mask))


if __name__ == "__main__":
    unittest.main()
