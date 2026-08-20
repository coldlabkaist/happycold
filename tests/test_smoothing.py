import unittest

import pandas as pd

from smoothing import anchor_median_smoothing


class AnchorMedianSmoothingTests(unittest.TestCase):
    def test_even_window_keeps_current_anchor_when_it_is_a_median_candidate(self) -> None:
        df = pd.DataFrame(
            {
                "frame": [1, 2, 4],
                "anchor.x": [0.0, 10.0, 20.0],
                "anchor.y": [0.0, 5.0, 10.0],
                "nose.x": [1.0, 11.0, 21.0],
                "nose.y": [2.0, 7.0, 12.0],
                "nose.score": [0.7, 0.8, 0.9],
            }
        )

        result = anchor_median_smoothing(df, ["anchor", "nose"], "anchor", 5)

        self.assertEqual(result.loc[1, "anchor.x"], 10.0)
        self.assertEqual(result.loc[1, "anchor.y"], 5.0)
        self.assertEqual(result.loc[1, "nose.x"], 11.0)
        self.assertEqual(result.loc[1, "nose.score"], 0.8)

    def test_missing_frames_are_not_replaced_by_jump_frames(self) -> None:
        df = pd.DataFrame(
            {
                "frame": [1, 2, 5],
                "anchor.x": [0.0, 100.0, 10.0],
                "anchor.y": [0.0, 100.0, 10.0],
                "nose.x": [2.0, 102.0, 12.0],
                "nose.y": [4.0, 104.0, 14.0],
            }
        )

        result = anchor_median_smoothing(df, ["anchor", "nose"], "anchor", 3)

        self.assertEqual(result.loc[2, "anchor.x"], 10.0)
        self.assertEqual(result.loc[2, "nose.x"], 12.0)

    def test_even_window_averages_when_current_anchor_is_not_a_median_candidate(self) -> None:
        df = pd.DataFrame(
            {
                "frame": [1, 2, 3, 4],
                "anchor.x": [0.0, 100.0, 20.0, 30.0],
                "anchor.y": [0.0, 100.0, 20.0, 30.0],
                "nose.x": [1.0, 101.0, 21.0, 31.0],
                "nose.y": [1.0, 101.0, 21.0, 31.0],
            }
        )

        result = anchor_median_smoothing(df, ["anchor", "nose"], "anchor", 5)

        self.assertEqual(result.loc[1, "anchor.x"], 25.0)
        self.assertEqual(result.loc[1, "anchor.y"], 25.0)
        self.assertEqual(result.loc[1, "nose.x"], 26.0)
        self.assertEqual(result.loc[1, "nose.y"], 26.0)

    def test_tracks_are_smoothed_independently(self) -> None:
        df = pd.DataFrame(
            {
                "track": ["a", "a", "a", "b", "b", "b"],
                "frame": [1, 2, 3, 1, 2, 3],
                "anchor.x": [0.0, 50.0, 10.0, 1000.0, 1050.0, 1010.0],
                "anchor.y": [0.0, 50.0, 10.0, 1000.0, 1050.0, 1010.0],
                "tail.x": [3.0, 53.0, 13.0, 1003.0, 1053.0, 1013.0],
                "tail.y": [4.0, 54.0, 14.0, 1004.0, 1054.0, 1014.0],
            }
        )

        result = anchor_median_smoothing(df, ["anchor", "tail"], "anchor", 3)

        self.assertEqual(result.loc[1, "anchor.x"], 10.0)
        self.assertEqual(result.loc[1, "tail.x"], 13.0)
        self.assertEqual(result.loc[4, "anchor.x"], 1010.0)
        self.assertEqual(result.loc[4, "tail.x"], 1013.0)


if __name__ == "__main__":
    unittest.main()
