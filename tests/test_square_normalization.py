import unittest

import numpy as np
import pandas as pd

from shared import build_normalized_dataframe


class SquareNormalizationTests(unittest.TestCase):
    def test_normalization_appends_coordinates_without_overwriting_source_columns(self) -> None:
        source = pd.DataFrame(
            {
                "frame": [1, 2, 3],
                "nose.x": [10.0, 60.0, np.nan],
                "nose.y": [20.0, 120.0, np.nan],
                "nose_in_out": ["in", "out", "out"],
            }
        )

        result = build_normalized_dataframe(
            source,
            ["nose"],
            [(10.0, 20.0), (110.0, 20.0), (110.0, 220.0), (10.0, 220.0)],
            width=200,
            height=300,
        )

        pd.testing.assert_series_equal(result["nose.x"], source["nose.x"])
        pd.testing.assert_series_equal(result["nose.y"], source["nose.y"])
        pd.testing.assert_series_equal(result["nose_in_out"], source["nose_in_out"])
        self.assertNotIn("nose.x_normalized", source.columns)
        self.assertNotIn("nose.y_normalized", source.columns)
        np.testing.assert_allclose(result["nose.x_normalized"].iloc[:2], [0.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(result["nose.y_normalized"].iloc[:2], [0.0, 0.5], atol=1e-6)
        self.assertTrue(np.isnan(result["nose.x_normalized"].iloc[2]))
        self.assertTrue(np.isnan(result["nose.y_normalized"].iloc[2]))

    def test_existing_normalized_columns_are_recalculated_in_place(self) -> None:
        source = pd.DataFrame(
            {
                "nose.x": [25.0],
                "nose.y": [75.0],
                "nose.x_normalized": [-1.0],
                "nose.y_normalized": [-1.0],
            }
        )

        result = build_normalized_dataframe(
            source,
            ["nose"],
            [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
            width=100,
            height=100,
        )

        self.assertAlmostEqual(float(result["nose.x_normalized"].iloc[0]), 0.25, places=6)
        self.assertAlmostEqual(float(result["nose.y_normalized"].iloc[0]), 0.75, places=6)
        self.assertEqual(result.columns.tolist().count("nose.x_normalized"), 1)
        self.assertEqual(result.columns.tolist().count("nose.y_normalized"), 1)


if __name__ == "__main__":
    unittest.main()
