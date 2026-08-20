from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from heatmap import calculate_body_occupancy_heatmap


class HeatmapCalculationTests(unittest.TestCase):
    def test_pixel_heatmap_uses_vectorized_coordinates_and_skips_invalid_rows(self) -> None:
        df = pd.DataFrame(
            {
                "a.x": [10.0, 20.0, np.nan],
                "a.y": [10.0, 10.0, np.nan],
                "b.x": [40.0, 50.0, 10.0],
                "b.y": [10.0, 10.0, 10.0],
                "c.x": [25.0, 35.0, 20.0],
                "c.y": [35.0, 35.0, 20.0],
            }
        )

        result = calculate_body_occupancy_heatmap(
            df=df,
            bodyparts=["a", "b", "c"],
            frame_size=(100, 50),
            max_bins=80,
            blur_sigma=0.0,
        )

        self.assertEqual(result.valid_frame_count, 2)
        self.assertEqual(result.x_limit, 100.0)
        self.assertEqual(result.y_limit, 50.0)
        self.assertEqual(result.values.shape, (64, 80))
        self.assertGreater(float(result.values.max()), 0.0)
        self.assertLessEqual(float(result.values.max()), 1.0)

    def test_raw_heatmap_display_bounds_crop_pixel_coordinates(self) -> None:
        df = pd.DataFrame(
            {
                "a.x": [30.0],
                "a.y": [20.0],
                "b.x": [70.0],
                "b.y": [20.0],
                "c.x": [50.0],
                "c.y": [35.0],
            }
        )

        result = calculate_body_occupancy_heatmap(
            df=df,
            bodyparts=["a", "b", "c"],
            frame_size=(100, 50),
            max_bins=80,
            blur_sigma=0.0,
            display_bounds=(20.0, 10.0, 80.0, 40.0),
        )

        self.assertEqual(result.valid_frame_count, 1)
        self.assertEqual(result.x_limit, 60.0)
        self.assertEqual(result.y_limit, 30.0)
        self.assertEqual(result.values.shape, (64, 80))

    def test_raw_heatmap_display_bounds_scale_normalized_source_coordinates(self) -> None:
        df = pd.DataFrame(
            {
                "a.x": [0.3],
                "a.y": [0.4],
                "b.x": [0.7],
                "b.y": [0.4],
                "c.x": [0.5],
                "c.y": [0.7],
            }
        )

        result = calculate_body_occupancy_heatmap(
            df=df,
            bodyparts=["a", "b", "c"],
            frame_size=(100, 50),
            max_bins=80,
            blur_sigma=0.0,
            display_bounds=(20.0, 10.0, 80.0, 40.0),
        )

        self.assertEqual(result.valid_frame_count, 1)
        self.assertEqual(result.x_limit, 60.0)
        self.assertEqual(result.y_limit, 30.0)
        self.assertEqual(result.values.shape, (64, 80))

    def test_normalized_heatmap_scales_to_rectified_display_size(self) -> None:
        df = pd.DataFrame(
            {
                "a.x": [0.1],
                "a.y": [0.1],
                "b.x": [0.8],
                "b.y": [0.1],
                "c.x": [0.4],
                "c.y": [0.9],
            }
        )

        result = calculate_body_occupancy_heatmap(
            df=df,
            bodyparts=["a", "b", "c"],
            frame_size=(640, 480),
            normalized=True,
            rectified_size=(200, 100),
            max_bins=100,
            blur_sigma=0.0,
        )

        self.assertEqual(result.valid_frame_count, 1)
        self.assertEqual(result.x_limit, 200.0)
        self.assertEqual(result.y_limit, 100.0)
        self.assertEqual(result.values.shape, (64, 100))

    def test_normalized_heatmap_display_bounds_crop_rectified_coordinates(self) -> None:
        df = pd.DataFrame(
            {
                "a.x": [0.2],
                "a.y": [0.2],
                "b.x": [0.8],
                "b.y": [0.2],
                "c.x": [0.5],
                "c.y": [0.7],
            }
        )

        result = calculate_body_occupancy_heatmap(
            df=df,
            bodyparts=["a", "b", "c"],
            frame_size=(640, 480),
            normalized=True,
            rectified_size=(200, 100),
            max_bins=100,
            blur_sigma=0.0,
            display_bounds=(20.0, 10.0, 180.0, 70.0),
        )

        self.assertEqual(result.valid_frame_count, 1)
        self.assertEqual(result.x_limit, 160.0)
        self.assertEqual(result.y_limit, 60.0)
        self.assertEqual(result.values.shape, (64, 100))

    def test_normalized_heatmap_prefers_appended_normalized_columns(self) -> None:
        df = pd.DataFrame(
            {
                "a.x": [np.nan],
                "a.y": [np.nan],
                "b.x": [np.nan],
                "b.y": [np.nan],
                "c.x": [np.nan],
                "c.y": [np.nan],
                "a.x_normalized": [0.1],
                "a.y_normalized": [0.1],
                "b.x_normalized": [0.8],
                "b.y_normalized": [0.1],
                "c.x_normalized": [0.4],
                "c.y_normalized": [0.9],
            }
        )

        result = calculate_body_occupancy_heatmap(
            df=df,
            bodyparts=["a", "b", "c"],
            frame_size=(640, 480),
            normalized=True,
            rectified_size=(200, 100),
            max_bins=100,
            blur_sigma=0.0,
        )

        self.assertEqual(result.valid_frame_count, 1)
        self.assertEqual(result.x_limit, 200.0)
        self.assertEqual(result.y_limit, 100.0)

    def test_heatmap_rejects_rows_without_three_valid_points(self) -> None:
        df = pd.DataFrame(
            {
                "a.x": [0.1],
                "a.y": [0.1],
                "b.x": [0.8],
                "b.y": [0.8],
                "c.x": [np.nan],
                "c.y": [np.nan],
            }
        )

        with self.assertRaisesRegex(ValueError, "No valid body polygon"):
            calculate_body_occupancy_heatmap(
                df=df,
                bodyparts=["a", "b", "c"],
                frame_size=(640, 480),
            )


if __name__ == "__main__":
    unittest.main()
