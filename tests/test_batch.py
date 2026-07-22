from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from batch import BatchItem, run_batch_exports


class BatchRunnerTests(unittest.TestCase):
    def test_shared_pipeline_builds_item_and_collects_skips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "video_a.csv"
            pd.DataFrame(
                {
                    "nose.x": [10.0],
                    "nose.y": [20.0],
                    "tail.x": [30.0],
                    "tail.y": [40.0],
                    "center.x": [20.0],
                    "center.y": [30.0],
                }
            ).to_csv(csv_path, index=False)

            video_a = root / "video_a.mp4"
            video_b = root / "video_b.mp4"
            exported_items: list[BatchItem] = []
            progress_calls: list[tuple[int, int, Path]] = []
            yield_count = 0

            def export_item(item: BatchItem) -> Path:
                exported_items.append(item)
                return root / "video_a_output.csv"

            def yield_events() -> None:
                nonlocal yield_count
                yield_count += 1

            result = run_batch_exports(
                selected_videos=[video_a, video_b],
                source_width=100,
                source_height=50,
                csv_candidates_for=lambda path: [csv_path] if path == video_a else [],
                video_size_for=lambda _path: (200, 100),
                export_item=export_item,
                progress=lambda index, total, path: progress_calls.append((index, total, path)),
                yield_events=yield_events,
            )

            self.assertEqual(result.saved_count, 1)
            self.assertEqual(result.skipped_auto_missing, [video_b])
            self.assertEqual(result.failed, [])
            self.assertEqual(len(exported_items), 1)
            self.assertEqual(exported_items[0].bodyparts, ["nose", "tail", "center"])
            self.assertEqual(exported_items[0].scale_x, 2.0)
            self.assertEqual(exported_items[0].scale_y, 2.0)
            self.assertEqual(len(progress_calls), 2)
            self.assertEqual(yield_count, 2)

    def test_finished_progress_and_safe_cancel_stop_before_next_item(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "video.csv"
            pd.DataFrame({"nose.x": [1.0], "nose.y": [2.0]}).to_csv(
                csv_path,
                index=False,
            )
            videos = [root / "video_a.mp4", root / "video_b.mp4"]
            cancel_requested = False
            completed: list[tuple[int, str, int, int, int]] = []

            def item_finished(index, _total, _path, outcome, saved, skipped, failed):
                nonlocal cancel_requested
                completed.append((index, outcome, saved, skipped, failed))
                cancel_requested = True

            result = run_batch_exports(
                selected_videos=videos,
                source_width=100,
                source_height=100,
                csv_candidates_for=lambda _path: [csv_path],
                video_size_for=lambda _path: (100, 100),
                export_item=lambda item: root / f"{item.video_path.stem}_output.csv",
                item_finished=item_finished,
                should_cancel=lambda: cancel_requested,
            )

            self.assertTrue(result.cancelled)
            self.assertEqual(result.saved_count, 1)
            self.assertEqual(completed, [(1, "saved", 1, 0, 0)])

    def test_shared_pipeline_collects_item_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "video.csv"
            pd.DataFrame({"nose.x": [1.0], "nose.y": [1.0]}).to_csv(csv_path, index=False)
            video_path = root / "video.mp4"

            result = run_batch_exports(
                selected_videos=[video_path],
                source_width=100,
                source_height=100,
                csv_candidates_for=lambda _path: [csv_path],
                video_size_for=lambda _path: None,
                export_item=lambda _item: root / "unused.csv",
            )

            self.assertEqual(result.saved_count, 0)
            self.assertEqual(len(result.failed), 1)
            self.assertIn("Could not read video size", result.failed[0][1])


if __name__ == "__main__":
    unittest.main()
