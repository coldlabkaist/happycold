import unittest
from functools import partial
from pathlib import Path

import pandas as pd

from pipeline import (
    DataFramePipelineStage,
    build_pipeline_output_filename,
    normalize_filename_affix,
    run_dataframe_pipeline,
)
from shared import build_circle_detection_dataframe, build_normalized_dataframe


class PipelineTests(unittest.TestCase):
    def test_replace_stages_update_working_data_and_append_stages_add_columns(self) -> None:
        source = pd.DataFrame(
            {
                "nose.x": [1.0, None, 3.0],
                "nose.y": [2.0, None, 4.0],
                "existing_analysis": ["source", "source", "source"],
            }
        )

        def clean_repair(df: pd.DataFrame) -> pd.DataFrame:
            result = df.copy()
            result["nose.x"] = [1.0, 2.0, 3.0]
            result["nose.y"] = [2.0, 3.0, 4.0]
            return result

        def analyze(df: pd.DataFrame) -> pd.DataFrame:
            result = df.copy()
            result["nose_zone"] = ["left" if value < 2.5 else "right" for value in df["nose.x"]]
            result["existing_analysis"] = ["updated", "updated", "updated"]
            return result

        result = run_dataframe_pipeline(
            source,
            [
                DataFramePipelineStage(
                    key="clean_repair",
                    label="Clean / Repair",
                    output_mode="replace",
                    transform=clean_repair,
                ),
                DataFramePipelineStage(
                    key="analysis",
                    label="Analysis",
                    transform=analyze,
                ),
            ],
        )

        self.assertEqual(result.dataframe["nose.x"].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(result.dataframe["nose.y"].tolist(), [2.0, 3.0, 4.0])
        self.assertEqual(result.dataframe["nose_zone"].tolist(), ["left", "left", "right"])
        pd.testing.assert_series_equal(
            result.dataframe["existing_analysis"],
            source["existing_analysis"],
        )
        self.assertEqual(
            result.dataframe["existing_analysis_analysis"].tolist(),
            ["updated", "updated", "updated"],
        )
        self.assertNotIn("nose.x_processed", result.dataframe.columns)
        self.assertTrue(pd.isna(source.loc[1, "nose.x"]))
        self.assertEqual(set(result.stages[0].replaced_columns), {"nose.x", "nose.y"})

    def test_pipeline_rejects_stages_that_change_rows(self) -> None:
        source = pd.DataFrame({"value": [1, 2, 3]})

        with self.assertRaisesRegex(ValueError, "row count or index"):
            run_dataframe_pipeline(
                source,
                [
                    DataFramePipelineStage(
                        key="bad",
                        label="Bad Stage",
                        transform=lambda df: df.iloc[:2].copy(),
                    )
                ],
            )

    def test_real_analysis_stages_append_one_combined_result(self) -> None:
        source = pd.DataFrame({"nose.x": [25.0], "nose.y": [75.0]})
        result = run_dataframe_pipeline(
            source,
            [
                DataFramePipelineStage(
                    key="circle",
                    label="Circle Detection",
                    transform=partial(
                        build_circle_detection_dataframe,
                        bodyparts=["nose"],
                        center=(50.0, 50.0),
                        radius=40.0,
                        width=100,
                        height=100,
                    ),
                ),
                DataFramePipelineStage(
                    key="square",
                    label="Square Normalization",
                    transform=partial(
                        build_normalized_dataframe,
                        bodyparts=["nose"],
                        quad_points=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
                        width=100,
                        height=100,
                    ),
                ),
            ],
        )

        self.assertEqual(result.dataframe["nose.x"].tolist(), [25.0])
        self.assertEqual(result.dataframe["nose.y"].tolist(), [75.0])
        self.assertEqual(result.dataframe["nose_in_out"].tolist(), ["in"])
        self.assertAlmostEqual(float(result.dataframe["nose.x_normalized"].iloc[0]), 0.25)
        self.assertAlmostEqual(float(result.dataframe["nose.y_normalized"].iloc[0]), 0.75)

    def test_output_filename_uses_optional_prefix_and_processed_default_suffix(self) -> None:
        source = Path("sample.csv")
        self.assertEqual(build_pipeline_output_filename(source), "sample_processed.csv")
        self.assertEqual(
            build_pipeline_output_filename(source, prefix="experiment1"),
            "experiment1_sample_processed.csv",
        )
        self.assertEqual(
            build_pipeline_output_filename(source, prefix="experiment1", suffix="final"),
            "experiment1_sample_final.csv",
        )
        self.assertEqual(build_pipeline_output_filename(source, prefix="", suffix=""), "sample.csv")

    def test_filename_affixes_remove_path_characters(self) -> None:
        self.assertEqual(normalize_filename_affix("  batch/one:*  "), "batch_one")


if __name__ == "__main__":
    unittest.main()
