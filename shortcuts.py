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
    ShortcutSpec("F", "_object3d_toggle_face_shortcut"),
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
        "Reference Lines (Direction / 3D Object)",
        (
            ("Click, then move", "Set the start point and preview the line direction"),
            ("Type pixels + Enter", "Set an exact length in original video pixels and confirm"),
            ("Shift", "Constrain the direction to horizontal or vertical while held"),
            ("Fixed", "Keep pixel length while choosing direction or dragging either endpoint"),
            ("Enter / Esc", "Confirm the draft / restore the previous line"),
            ("Ctrl+Z", "Cancel a draft or undo the last committed reference edit"),
        ),
    ),
    (
        "Direction",
        (
            ("Load Masks", "Copy raw shapes from Occlusion or 3D Object; source margins are ignored"),
            ("Import Mask / Folder", "Load a PNG, editable 3D folder, or saved Direction recipe"),
            ("Distance Node", "Choose the point used for Overlap / Near / Far, separately from the heading"),
            ("Object / Near Margin", "Set both offsets from the same base shape; Near must be at least Object"),
            ("Calibration", "Measure physical reference lines or select four perspective points; base masks stay read-only"),
            ("Line / Top / Bottom", "Select Line to draw or adjust a ruler; Top and Bottom assign a ruler to each face"),
            ("Angle = 0 deg", "Test a forward ray from the nose against the Object region"),
            ("Angle > 0 deg", "Test the cone on both sides of the heading when the distance node is Near"),
            ("Left / Right", "Inspect Overlap, Near / Facing, Near / Not Facing, Far, or Unknown"),
        ),
    ),
    (
        "Shared",
        (
            ("D", "Draw mode"),
            ("T", "Transform mode"),
            ("E  or  [", "Scale down"),
            ("R  or  ]", "Scale up"),
            ("Ctrl+E  or  Ctrl+[", "Rotate left where supported"),
            ("Ctrl+R  or  Ctrl+]", "Rotate right where supported"),
            ("Ctrl+drag", "Temporarily erase in Brush or Transform mode where supported"),
            ("Shift+rectangle drag", "Constrain the rectangle to a square"),
            ("Ctrl+Z", "Undo the latest mask content edit in the current tab"),
            ("F1", "Open this shortcut guide"),
        ),
    ),
    (
        "Occlusion / 3D Object",
        (
            ("F (3D Object)", "Switch Top / Bottom while keeping Draw or Transform mode"),
            ("1 - 9", "Select mask or object 1 - 9"),
            ("0", "Select mask or object 10"),
            ("Double-click", "Select the mask, object, or face under the cursor"),
            ("Right-click", "Copy or paste the mask or face under the cursor"),
        ),
    ),
)
