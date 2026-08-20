from __future__ import annotations

import unittest

import pandas as pd

from trajectory_plot import build_trajectory_figure


class TrajectoryPlotTests(unittest.TestCase):
    def test_build_raw_trajectory_figure_without_qt_dialog(self) -> None:
        df = pd.DataFrame(
            {
                "frame": [1, 2, 1, 2],
                "track": ["a", "a", "b", "b"],
                "nose.x": [10.0, 20.0, 30.0, 40.0],
                "nose.y": [12.0, 22.0, 32.0, 42.0],
            }
        )

        figure = build_trajectory_figure(
            df=df,
            bodyparts=["nose"],
            normalized=False,
            frame_width=100,
            frame_height=80,
        )

        self.assertEqual(len(figure.axes), 1)
        self.assertEqual(len(figure.axes[0].lines), 2)
        figure.clear()

    def test_build_normalized_trajectory_figure_uses_rectified_size(self) -> None:
        df = pd.DataFrame(
            {
                "frame": [1, 2],
                "nose.x_normalized": [0.1, 0.8],
                "nose.y_normalized": [0.2, 0.7],
            }
        )

        figure = build_trajectory_figure(
            df=df,
            bodyparts=["nose"],
            normalized=True,
            frame_width=640,
            frame_height=480,
            normalized_display_size=(200, 100),
        )

        axis = figure.axes[0]
        self.assertEqual(axis.get_xlim(), (0.0, 200.0))
        self.assertEqual(axis.get_ylim(), (100.0, 0.0))
        figure.clear()


if __name__ == "__main__":
    unittest.main()
