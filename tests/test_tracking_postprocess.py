import unittest

import pandas as pd

from interpolation import build_interpolation_pipeline_dataframe
from pipeline import DataFramePipelineStage, run_dataframe_pipeline
from tracking_postprocess import (
    TrackingPostprocessConfig,
    invalidate_length_outlier_skeletons,
    remove_duplicate_skeletons,
    remove_length_outlier_skeletons,
    run_tracking_postprocess,
)


def _skeleton_row(frame: int, track: str, score: float, center_x: float, nose_x: float) -> dict:
    return {
        "frame_idx": frame,
        "track": track,
        "instance.score": score,
        "body_c.x": center_x,
        "body_c.y": 20.0,
        "nose.x": nose_x,
        "nose.y": 20.0,
    }


class TrackingPostprocessTests(unittest.TestCase):
    def test_duplicate_removal_keeps_higher_confidence_row(self) -> None:
        source = pd.DataFrame(
            [
                _skeleton_row(1, "low", 0.2, 10.0, 20.0),
                _skeleton_row(1, "high", 0.9, 10.5, 20.5),
                _skeleton_row(2, "other", 0.5, 100.0, 110.0),
            ]
        )

        result = remove_duplicate_skeletons(
            source,
            bodyparts=["body_c", "nose"],
            criteria=2.0,
            width=200,
            height=100,
        )

        self.assertEqual(result.candidate_count, 1)
        self.assertEqual(result.removed_count, 1)
        self.assertEqual(result.dataframe["track"].tolist(), ["high", "other"])

    def test_robust_zscore_keeps_row_and_invalidates_skeleton_payload(self) -> None:
        lengths = [10.0, 11.0, 9.0, 10.0, 100.0]
        source = pd.DataFrame(
            [
                _skeleton_row(frame, f"t{frame}", 0.5, 0.0, length)
                for frame, length in enumerate(lengths, start=1)
            ]
        )

        result = remove_length_outlier_skeletons(
            source,
            bodyparts=["body_c", "nose"],
            high_z_threshold=3.5,
            low_z_threshold=3.5,
            deviation_mode="high_only",
            width=200,
            height=100,
        )

        self.assertEqual(result.candidate_count, 1)
        self.assertEqual(result.affected_count, 1)
        self.assertEqual(len(result.dataframe), len(source))
        outlier = result.dataframe.loc[result.dataframe["track"] == "t5"].iloc[0]
        self.assertEqual(int(outlier["frame_idx"]), 5)
        self.assertTrue(outlier[["body_c.x", "body_c.y", "nose.x", "nose.y", "instance.score"]].isna().all())

    def test_combined_repair_runs_duplicate_before_zscore(self) -> None:
        source = pd.DataFrame(
            [
                _skeleton_row(1, "duplicate-low", 0.1, 0.0, 10.0),
                _skeleton_row(1, "duplicate-high", 0.9, 0.2, 10.2),
                _skeleton_row(2, "t2", 0.5, 0.0, 11.0),
                _skeleton_row(3, "t3", 0.5, 0.0, 9.0),
                _skeleton_row(4, "t4", 0.5, 0.0, 10.0),
                _skeleton_row(5, "outlier", 0.5, 0.0, 100.0),
            ]
        )
        result = run_tracking_postprocess(
            source,
            config=TrackingPostprocessConfig(
                duplicate_distance_threshold=1.0,
                deviation_mode="high_only",
            ),
            bodyparts=["body_c", "nose"],
            width=200,
            height=100,
        )

        self.assertEqual(result.duplicate_removed, 1)
        self.assertEqual(result.length_outliers_invalidated, 1)
        self.assertEqual(len(result.dataframe), 5)
        self.assertTrue(
            result.dataframe.loc[result.dataframe["track"] == "outlier", "nose.x"].isna().all()
        )

    def test_pipeline_can_allow_row_removal_before_column_stage(self) -> None:
        source = pd.DataFrame({"value": [1, 2, 3]}, index=[10, 11, 12])
        result = run_dataframe_pipeline(
            source,
            [
                DataFramePipelineStage(
                    key="repair",
                    label="Repair",
                    output_mode="replace",
                    allow_row_removal=True,
                    transform=lambda df: df.drop(index=[11]),
                ),
                DataFramePipelineStage(
                    key="derived",
                    label="Derived",
                    transform=lambda df: df.assign(doubled=df["value"] * 2),
                ),
            ],
        )

        self.assertEqual(result.dataframe.index.tolist(), [10, 12])
        self.assertEqual(result.dataframe["doubled"].tolist(), [2, 6])
        self.assertEqual(result.stages[0].removed_rows, 1)

    def test_interpolation_fills_original_and_zscore_invalidated_frames(self) -> None:
        rows = [
            _skeleton_row(1, "mouse", 0.5, 0.0, 10.0),
            _skeleton_row(2, "mouse", None, None, None),
            _skeleton_row(3, "mouse", 0.5, 0.0, 11.0),
            _skeleton_row(4, "mouse", 0.5, 0.0, 100.0),
            _skeleton_row(5, "mouse", 0.5, 0.0, 9.0),
            _skeleton_row(6, "mouse", None, None, None),
            _skeleton_row(7, "mouse", 0.5, 0.0, 10.0),
        ]
        for row in (rows[1], rows[5]):
            row["body_c.y"] = None
            row["nose.y"] = None
        source = pd.DataFrame(rows)

        result = run_dataframe_pipeline(
            source,
            [
                DataFramePipelineStage(
                    key="zscore_removal",
                    label="Z-score Invalidation",
                    output_mode="replace",
                    transform=lambda df: invalidate_length_outlier_skeletons(
                        df,
                        bodyparts=["body_c", "nose"],
                        high_z_threshold=3.5,
                        low_z_threshold=3.5,
                        deviation_mode="high_only",
                        width=200,
                        height=100,
                    ).dataframe,
                ),
                DataFramePipelineStage(
                    key="clean_repair",
                    label="Interpolation",
                    output_mode="replace",
                    transform=lambda df: build_interpolation_pipeline_dataframe(
                        df,
                        bodyparts=["body_c", "nose"],
                        region_mask=None,
                        width=200,
                        height=100,
                        removal_mode="none",
                        anchor_bodypart=None,
                        interpolate=True,
                    ),
                ),
            ],
        )

        self.assertEqual(len(result.dataframe), 7)
        self.assertEqual(result.dataframe["frame_idx"].tolist(), list(range(1, 8)))
        self.assertAlmostEqual(float(result.dataframe.loc[1, "nose.x"]), 10.5)
        self.assertAlmostEqual(float(result.dataframe.loc[3, "nose.x"]), 10.0)
        self.assertAlmostEqual(float(result.dataframe.loc[5, "nose.x"]), 9.5)
        self.assertTrue(pd.isna(result.dataframe.loc[3, "instance.score"]))
        self.assertTrue(pd.isna(source.loc[1, "nose.x"]))
        self.assertEqual(float(source.loc[3, "nose.x"]), 100.0)


if __name__ == "__main__":
    unittest.main()
