# happycold

`happycold` is a PyQt-based desktop tool for inspecting animal-tracking videos and matching them with pose CSV files.

It supports:

- `Square Normalize`: pick 4 points and export a perspective-normalized CSV
- `Chamber Mark`: define a chamber and named rooms, then export room membership by frame
- `Circle Detection`: draw a circle and classify each keypoint as `in` or `out`
- `Pin Coordinates`: inspect frame coordinates manually
- `Occlusion Detect`: draw named masks and export occlusion flags

## Requirements

- Python 3.11+ recommended
- Packages from [requirements.txt](C:/cold_yj/Programs/happycold/requirements.txt)

Install:

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python happycold.py
```

## Basic Workflow

1. Open a folder containing videos.
2. Select a video from the list.
3. Let the app auto-match a CSV, or load one manually.
4. Choose a tab depending on the task.
5. Save the current result with the button at the bottom right.

## Chamber Mark

`Chamber Mark` is used when each keypoint should be assigned to a named room.

Workflow:

1. Define the chamber area as the union of multiple rectangles or circles.
2. Add a room name.
3. Switch to room editing and draw the room area.
4. Repeat for more rooms.  
   Room pixels cannot overlap other rooms, and only pixels inside the chamber are accepted.
5. Save outputs.

Saved outputs:

- `*_chamber_mark.csv`: per-frame room label columns for each bodypart
- `*_chamber_mask.png`: color mask image of rooms and unassigned chamber area
- `*_chamber_mask_with_frame.png`: current frame with the chamber mask overlaid
- `*_chamber_mask.json`: room names and colors for re-import

## Repo Structure

- [happycold.py](C:/cold_yj/Programs/happycold/happycold.py): main window and viewer
- [happycold_shared.py](C:/cold_yj/Programs/happycold/happycold_shared.py): shared data structures and output builders
- [tab_mixins](C:/cold_yj/Programs/happycold/tab_mixins): per-tab UI and behavior

## Notes

- The app is designed around local video and CSV files.
- Test videos, CSVs, build outputs, and generated masks should not be committed.
