import unittest
from types import SimpleNamespace

import numpy as np
from PyQt6.QtGui import QColor

from shared import RoomRecord
from tab_mixins.chamber_tab import ChamberTabMixin


class _Radio:
    def __init__(self, checked: bool = False) -> None:
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


class _ChamberHarness(ChamberTabMixin):
    def __init__(self, width: int, height: int) -> None:
        self.video_state = SimpleNamespace(width=width, height=height)
        self.chamber_mask = None
        self.chamber_geometry = None
        self.chamber_boundary_mode = "custom"
        self.room_records = {}
        self.selected_room_name = None
        self.chamber_edit_room_radio = _Radio(False)
        self.chamber_edit_chamber_radio = _Radio(True)


class ChamberMaskCompatibilityTests(unittest.TestCase):
    def test_effective_rooms_ignore_masks_from_different_video_size(self) -> None:
        harness = _ChamberHarness(width=1280, height=720)
        harness.chamber_mask = np.ones((540, 720), dtype=np.uint8)
        harness.room_records["old_room"] = RoomRecord(
            name="old_room",
            color=QColor("#ef4444"),
            mask=np.ones((540, 720), dtype=np.uint8),
        )

        rooms = harness._effective_room_records()

        self.assertEqual(len(rooms), 1)
        self.assertEqual(rooms[0].mask.shape, (720, 1280))
        self.assertEqual(int(rooms[0].mask.sum()), 0)
        self.assertEqual(harness._resolved_chamber_boundary_mode(), "unset")

    def test_effective_rooms_ignore_stale_room_but_keep_current_chamber(self) -> None:
        harness = _ChamberHarness(width=1280, height=720)
        harness.chamber_mask = np.ones((720, 1280), dtype=np.uint8)
        harness.room_records["old_room"] = RoomRecord(
            name="old_room",
            color=QColor("#ef4444"),
            mask=np.ones((540, 720), dtype=np.uint8),
        )

        rooms = harness._effective_room_records()

        self.assertEqual(len(rooms), 1)
        self.assertEqual(rooms[0].mask.shape, (720, 1280))
        self.assertEqual(int(rooms[0].mask.sum()), 0)
        self.assertEqual(harness._resolved_chamber_boundary_mode(), "custom")


if __name__ == "__main__":
    unittest.main()
