from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ShortcutSpec:
    sequence: str
    handler_name: str
    args: tuple[Any, ...] = ()


APP_SHORTCUTS: tuple[ShortcutSpec, ...] = (
    ShortcutSpec("Ctrl+B", "_toggle_file_sidebar"),
    ShortcutSpec("Ctrl+Z", "undo_active_mask_edit"),
    ShortcutSpec("Left", "step_frame", (-1,)),
    ShortcutSpec("Right", "step_frame", (1,)),
    ShortcutSpec("D", "_switch_region_edit_mode_shortcut", (False,)),
    ShortcutSpec("T", "_switch_region_edit_mode_shortcut", (True,)),
    ShortcutSpec("E", "_scale_active_region_shortcut", (0.96,)),
    ShortcutSpec("R", "_scale_active_region_shortcut", (1.04,)),
    ShortcutSpec("Ctrl+E", "_rotate_active_region_shortcut", (-4.0,)),
    ShortcutSpec("Ctrl+R", "_rotate_active_region_shortcut", (4.0,)),
    ShortcutSpec("[", "_scale_active_region_shortcut", (0.96,)),
    ShortcutSpec("]", "_scale_active_region_shortcut", (1.04,)),
    ShortcutSpec("Ctrl+[", "_rotate_active_region_shortcut", (-4.0,)),
    ShortcutSpec("Ctrl+]", "_rotate_active_region_shortcut", (4.0,)),
    ShortcutSpec("F1", "_open_context_help"),
    *(
        ShortcutSpec(str((slot_index + 1) % 10), "_select_mask_by_slot", (slot_index,))
        for slot_index in range(10)
    ),
)


ANNOTATE_SHORTCUT_HELP_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Shared",
        (
            ("D", "Draw mode"),
            ("T", "Transform mode"),
            ("E  or  [", "Scale down"),
            ("R  or  ]", "Scale up"),
            ("Ctrl+E  or  Ctrl+[", "Rotate left where supported"),
            ("Ctrl+R  or  Ctrl+]", "Rotate right where supported"),
            ("Ctrl+drag", "Temporarily erase while free drawing where supported"),
            ("Ctrl+Z", "Undo the latest mask content edit in the current tab"),
            ("F1", "Open this shortcut guide"),
        ),
    ),
    (
        "Occlusion only",
        (
            ("1 - 9", "Select mask 1 - 9"),
            ("0", "Select mask 10"),
        ),
    ),
)
