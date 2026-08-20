import importlib.util
import sys
import types
import unittest
from pathlib import Path

import pandas as pd

from batch import BatchItem


def _install_pyqt_stubs() -> None:
    if "PyQt6" in sys.modules:
        return

    pyqt6 = types.ModuleType("PyQt6")
    qtcore = types.ModuleType("PyQt6.QtCore")
    qtgui = types.ModuleType("PyQt6.QtGui")
    qtwidgets = types.ModuleType("PyQt6.QtWidgets")

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return _Dummy()

        def __call__(self, *args, **kwargs):
            return _Dummy()

    class _Enum:
        def __getattr__(self, name):
            return 0

    class _Qt:
        AlignmentFlag = _Enum()
        ScrollBarPolicy = _Enum()
        TextElideMode = _Enum()
        CheckState = _Enum()
        KeyboardModifier = _Enum()
        MouseButton = _Enum()
        Key = _Enum()

    qtcore.Qt = _Qt()
    qtcore.QPoint = _Dummy

    qtgui.QColor = _Dummy

    for name in (
        "QCheckBox",
        "QComboBox",
        "QDoubleSpinBox",
        "QFormLayout",
        "QFrame",
        "QGridLayout",
        "QGroupBox",
        "QHBoxLayout",
        "QLabel",
        "QLineEdit",
        "QMessageBox",
        "QPushButton",
        "QScrollArea",
        "QSizePolicy",
        "QSpinBox",
        "QStackedWidget",
        "QVBoxLayout",
        "QWidget",
    ):
        setattr(qtwidgets, name, _Dummy)

    sys.modules.update(
        {
            "PyQt6": pyqt6,
            "PyQt6.QtCore": qtcore,
            "PyQt6.QtGui": qtgui,
            "PyQt6.QtWidgets": qtwidgets,
        }
    )


def _load_pipeline_panel_module():
    _install_pyqt_stubs()
    module_name = "_pipeline_panel_under_test"
    module_path = Path(__file__).resolve().parents[1] / "tab_mixins" / "pipeline_panel.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pipeline_panel = _load_pipeline_panel_module()


class PipelinePanelOrderTests(unittest.TestCase):
    def _item(self) -> BatchItem:
        return BatchItem(
            video_path=Path("sample.mp4"),
            csv_path=Path("sample.csv"),
            source_df=pd.DataFrame({"frame": [0], "nose_x": [1.0], "nose_y": [2.0]}),
            bodyparts=["nose"],
            width=100,
            height=100,
            scale_x=1.0,
            scale_y=1.0,
        )

    def _snapshot(self, enabled_stages, repair_stage_order):
        return pipeline_panel.PipelineSnapshot(
            enabled_stages=frozenset(enabled_stages),
            duplicate_distance_threshold=5.0,
            zscore_high_threshold=3.0,
            zscore_low_threshold=3.0,
            zscore_deviation_mode="axis",
            interpolation_mask=None,
            interpolation_removal_mode="none",
            interpolation_anchor="nose",
            interpolation_enabled=True,
            interpolation_extrapolate=False,
            smoothing_method="anchor_median_smoothing",
            smoothing_anchor="nose",
            smoothing_window_size=3,
            repair_stage_order=tuple(repair_stage_order),
            chamber_mask=None,
            chamber_geometry=None,
            chamber_boundary_mode="manual",
            rooms=(),
            circle_geometry=None,
            masks=(),
            occlusion_quad_points=(),
            square_points=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        )

    def test_ordered_keys_use_custom_repair_order_before_fixed_stages(self):
        snapshot = self._snapshot(
            enabled_stages={
                "smoothing",
                "clean_repair",
                "duplicate_removal",
                "zscore_removal",
                "square",
            },
            repair_stage_order=(
                "smoothing",
                "clean_repair",
                "duplicate_removal",
                "zscore_removal",
            ),
        )

        ordered_keys = pipeline_panel.PipelinePanelMixin._ordered_pipeline_stage_keys(snapshot)

        self.assertEqual(
            ordered_keys,
            (
                "smoothing",
                "clean_repair",
                "duplicate_removal",
                "zscore_removal",
                "square",
            ),
        )

    def test_execute_pipeline_item_applies_repair_stages_in_custom_order(self):
        snapshot = self._snapshot(
            enabled_stages={
                "smoothing",
                "clean_repair",
                "duplicate_removal",
                "zscore_removal",
            },
            repair_stage_order=(
                "smoothing",
                "clean_repair",
                "duplicate_removal",
                "zscore_removal",
            ),
        )
        original_functions = {
            "build_smoothed_dataframe": pipeline_panel.build_smoothed_dataframe,
            "build_interpolation_pipeline_dataframe": pipeline_panel.build_interpolation_pipeline_dataframe,
            "_duplicate_stage_dataframe": pipeline_panel._duplicate_stage_dataframe,
            "_zscore_stage_dataframe": pipeline_panel._zscore_stage_dataframe,
        }

        def append_order(stage_key):
            def transform(dataframe, **kwargs):
                next_df = dataframe.copy()
                current = (
                    next_df["order"].astype(str)
                    if "order" in next_df.columns
                    else pd.Series([""] * len(next_df), index=next_df.index)
                )
                next_df["order"] = current.map(
                    lambda value: f"{value}>{stage_key}" if value else stage_key
                )
                return next_df

            return transform

        try:
            pipeline_panel.build_smoothed_dataframe = append_order("smoothing")
            pipeline_panel.build_interpolation_pipeline_dataframe = append_order("clean_repair")
            pipeline_panel._duplicate_stage_dataframe = append_order("duplicate_removal")
            pipeline_panel._zscore_stage_dataframe = append_order("zscore_removal")

            result = pipeline_panel.PipelinePanelMixin()._execute_pipeline_item(self._item(), snapshot)
        finally:
            for name, original in original_functions.items():
                setattr(pipeline_panel, name, original)

        self.assertEqual(
            [summary.key for summary in result.stages],
            ["smoothing", "clean_repair", "duplicate_removal", "zscore_removal"],
        )
        self.assertEqual(
            result.dataframe["order"].iloc[0],
            "smoothing>clean_repair>duplicate_removal>zscore_removal",
        )


if __name__ == "__main__":
    unittest.main()
