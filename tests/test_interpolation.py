from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from interpolation import (
    build_automatic_removal_dataframe,
    build_interpolated_dataframe,
    build_interpolation_pipeline_dataframe,
)
from tab_mixins.interpolation_tab import InterpolationTabMixin


class _InterpolationRectangleHarness(InterpolationTabMixin):
    def __init__(self) -> None:
        self.interpolation_mask = np.zeros((50, 50), dtype=np.uint8)

    def _ensure_interpolation_mask(self) -> np.ndarray:
        return self.interpolation_mask

    def _set_interpolation_mask(self, mask: np.ndarray | None, refresh: bool = True) -> None:
        self.interpolation_mask = mask


class InterpolationCalculationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = np.zeros((50, 50), dtype=np.uint8)
        self.mask[5:20, 15:26] = 1
        self.df = pd.DataFrame(
            {
                "frame": [0, 1, 2],
                "nose.x": [10.0, np.nan, 30.0],
                "nose.y": [10.0, np.nan, 10.0],
            }
        )

    def test_inside_mode_fills_gap_whose_result_is_inside_region(self) -> None:
        result = build_interpolated_dataframe(
            self.df,
            ["nose"],
            self.mask,
            width=50,
            height=50,
            region_mode="inside",
        )

        self.assertEqual(result.loc[1, "nose.x"], 20.0)
        self.assertEqual(result.loc[1, "nose.y"], 10.0)
        self.assertEqual(result.loc[0, "nose.x"], 10.0)
        self.assertEqual(result.loc[2, "nose.x"], 30.0)

    def test_outside_mode_rejects_gap_whose_result_is_inside_region(self) -> None:
        result = build_interpolated_dataframe(
            self.df,
            ["nose"],
            self.mask,
            width=50,
            height=50,
            region_mode="outside",
        )

        self.assertTrue(pd.isna(result.loc[1, "nose.x"]))
        self.assertTrue(pd.isna(result.loc[1, "nose.y"]))

    def test_all_mode_fills_gap_without_a_region_mask(self) -> None:
        result = build_interpolated_dataframe(
            self.df,
            ["nose"],
            None,
            width=50,
            height=50,
            region_mode="all",
        )

        self.assertEqual(result.loc[1, "nose.x"], 20.0)
        self.assertEqual(result.loc[1, "nose.y"], 10.0)

    def test_automatic_removal_deletes_complete_skeleton_by_anchor(self) -> None:
        df = pd.DataFrame(
            {
                "frame_idx": [1, 1],
                "instance.id": [1, 2],
                "instance.score": [0.9, 0.8],
                "anchor.x": [10.0, 40.0],
                "anchor.y": [10.0, 40.0],
                "anchor.score": [0.95, 0.85],
                "nose.x": [12.0, 42.0],
                "nose.y": [12.0, 42.0],
                "nose.score": [0.9, 0.8],
            }
        )
        region = np.zeros((50, 50), dtype=np.uint8)
        region[:25, :25] = 1

        result = build_automatic_removal_dataframe(
            df,
            ["anchor", "nose"],
            region,
            width=50,
            height=50,
            anchor_bodypart="anchor",
            removal_mode="inside",
        )

        self.assertTrue(pd.isna(result.loc[0, "anchor.x"]))
        self.assertTrue(pd.isna(result.loc[0, "nose.y"]))
        self.assertTrue(pd.isna(result.loc[0, "nose.score"]))
        self.assertTrue(pd.isna(result.loc[0, "instance.score"]))
        self.assertEqual(result.loc[0, "instance.id"], 1)
        self.assertEqual(result.loc[1, "nose.x"], 42.0)

    def test_removal_pipeline_interpolates_only_when_enabled(self) -> None:
        df = pd.DataFrame(
            {
                "track": ["mouse", "mouse", "mouse"],
                "instance.id": [1, 1, 1],
                "frame_idx": [1, 2, 3],
                "anchor.x": [10.0, 15.0, 20.0],
                "anchor.y": [10.0, 10.0, 10.0],
                "nose.x": [20.0, 25.0, 30.0],
                "nose.y": [20.0, 20.0, 20.0],
            }
        )
        region = np.zeros((50, 50), dtype=np.uint8)
        region[5:15, 14:17] = 1

        removed = build_interpolation_pipeline_dataframe(
            df,
            ["anchor", "nose"],
            region,
            width=50,
            height=50,
            removal_mode="inside",
            anchor_bodypart="anchor",
            interpolate=False,
        )
        interpolated = build_interpolation_pipeline_dataframe(
            df,
            ["anchor", "nose"],
            region,
            width=50,
            height=50,
            removal_mode="inside",
            anchor_bodypart="anchor",
            interpolate=True,
        )

        self.assertTrue(pd.isna(removed.loc[1, "anchor.x"]))
        self.assertTrue(pd.isna(removed.loc[1, "nose.x"]))
        self.assertEqual(interpolated.loc[1, "anchor.x"], 15.0)
        self.assertEqual(interpolated.loc[1, "nose.x"], 25.0)

    def test_instance_id_groups_are_interpolated_independently(self) -> None:
        df = pd.DataFrame(
            {
                "track": ["mouse"] * 6,
                "instance.id": [1, 2, 1, 2, 1, 2],
                "frame_idx": [1, 1, 2, 2, 3, 3],
                "nose.x": [10.0, 30.0, np.nan, np.nan, 20.0, 40.0],
                "nose.y": [10.0, 30.0, np.nan, np.nan, 10.0, 30.0],
            }
        )

        result = build_interpolated_dataframe(
            df,
            ["nose"],
            None,
            width=50,
            height=50,
            region_mode="all",
        )

        self.assertEqual(result.loc[2, "nose.x"], 15.0)
        self.assertEqual(result.loc[3, "nose.x"], 35.0)

    def test_tracks_are_interpolated_independently(self) -> None:
        df = pd.DataFrame(
            {
                "track": ["A", "B", "A", "B", "A", "B"],
                "frame": [0, 0, 1, 1, 2, 2],
                "nose.x": [10.0, 30.0, np.nan, np.nan, 20.0, 40.0],
                "nose.y": [10.0, 30.0, np.nan, np.nan, 10.0, 30.0],
            }
        )
        full_mask = np.ones((50, 50), dtype=np.uint8)

        result = build_interpolated_dataframe(
            df,
            ["nose"],
            full_mask,
            width=50,
            height=50,
            region_mode="inside",
        )

        self.assertEqual(result.loc[2, "nose.x"], 15.0)
        self.assertEqual(result.loc[3, "nose.x"], 35.0)

    def test_normalized_coordinates_are_tested_in_pixel_space(self) -> None:
        df = pd.DataFrame(
            {
                "frame": [0, 1, 2],
                "nose.x": [0.2, np.nan, 0.6],
                "nose.y": [0.2, np.nan, 0.2],
            }
        )
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[15:25, 35:45] = 1

        result = build_interpolated_dataframe(
            df,
            ["nose"],
            mask,
            width=100,
            height=100,
            region_mode="inside",
        )

        self.assertAlmostEqual(result.loc[1, "nose.x"], 0.4)
        self.assertAlmostEqual(result.loc[1, "nose.y"], 0.2)

    def test_edge_gaps_are_not_extrapolated(self) -> None:
        df = pd.DataFrame(
            {
                "frame": [0, 1, 2],
                "nose.x": [np.nan, 20.0, 30.0],
                "nose.y": [np.nan, 10.0, 10.0],
            }
        )
        full_mask = np.ones((50, 50), dtype=np.uint8)

        result = build_interpolated_dataframe(
            df,
            ["nose"],
            full_mask,
            width=50,
            height=50,
            region_mode="inside",
        )

        self.assertTrue(pd.isna(result.loc[0, "nose.x"]))
        self.assertTrue(pd.isna(result.loc[0, "nose.y"]))

    def test_rectangle_regions_accumulate_into_a_union(self) -> None:
        harness = _InterpolationRectangleHarness()

        harness.apply_interpolation_rect([(5, 5), (15, 5), (15, 15), (5, 15)])
        harness.apply_interpolation_rect([(30, 30), (40, 30), (40, 40), (30, 40)])

        self.assertEqual(harness.interpolation_mask[10, 10], 1)
        self.assertEqual(harness.interpolation_mask[35, 35], 1)
        self.assertEqual(harness.interpolation_mask[22, 22], 0)

    def test_start_end_extrapolation_is_user_selectable(self) -> None:
        source = pd.DataFrame(
            {
                "frame_idx": list(range(1, 8)),
                "track": ["mouse"] * 7,
                "nose.x": [None, None, 2.0, None, 4.0, None, None],
                "nose.y": [None, None, 12.0, None, 14.0, None, None],
            }
        )

        internal_only = build_interpolation_pipeline_dataframe(
            source,
            bodyparts=["nose"],
            region_mask=None,
            width=100,
            height=100,
            removal_mode="none",
            anchor_bodypart=None,
            interpolate=True,
            extrapolate=False,
        )
        extrapolated = build_interpolation_pipeline_dataframe(
            source,
            bodyparts=["nose"],
            region_mask=None,
            width=100,
            height=100,
            removal_mode="none",
            anchor_bodypart=None,
            interpolate=True,
            extrapolate=True,
        )

        self.assertTrue(pd.isna(internal_only.loc[0, "nose.x"]))
        self.assertEqual(float(internal_only.loc[3, "nose.x"]), 3.0)
        self.assertTrue(pd.isna(internal_only.loc[6, "nose.x"]))
        self.assertEqual(
            extrapolated["nose.x"].tolist(),
            [2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0],
        )
        self.assertEqual(
            extrapolated["nose.y"].tolist(),
            [12.0, 12.0, 12.0, 13.0, 14.0, 14.0, 14.0],
        )


if __name__ == "__main__":
    unittest.main()
