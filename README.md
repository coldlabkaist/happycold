# HappyCoLD

happycold is a desktop application for reviewing and post-processing animal-tracking CSV data alongside video.

Main features:

- Duplicate and Z-score tracking repair, region-based removal, and interpolation
- Perspective normalization with separate normalized coordinate columns
- Chamber, circle, and occlusion annotations
- Editable 3D object occlusion envelopes with named, video-specific length references
- Object-facing direction classification from read-only masks and selectable heading points
- Trajectory, heatmap, and coordinate inspection
- Batch saving and multi-stage pipeline export

## Install

Python 3.11 or newer is recommended.

```bash
git clone https://github.com/coldlabkaist/happycold.git
cd happycold
conda create -n happycold python=3.11
conda activate happycold
pip install -r requirements.txt
```

## Run

```bash
cd (happycold directory)
conda activate happycold
python main.py
```

Open a video folder, select or load a matching CSV, configure the desired tools, and save the result from the output panel.

In Annotate > Occlusion (3D Object), select or add an object first, then choose Line, Top, or Bottom below it. Measure references and edit the two faces with physical margins. Top and bottom references are independently selectable using the same compact face controls. References can be reused across faces and objects. The filled convex envelope is saved using the existing occlusion CSV columns and filename rules. Export Masks uses the same output/masks folder as Occlusion. Object folders preserve both editable faces and references; lengths keep their physical size when changing m/cm/mm units. New videos inherit the object layout but require reference measurements; returning to a video restores its session state. 3D object batch export and Multi Pipeline processing are unavailable.

In 3D Transform mode, the active face remains shaded and outlined while dragging. Use D/T to switch modes, E/R to resize, Ctrl+E/R to rotate, and Ctrl+Z to undo. Right-click a face to copy or paste its mask, and press F1 for the shared annotation shortcuts.

In Annotate > Direction, use Load Masks for read-only base shapes from the Occlusion or 3D Object tab. Import Mask and Import Folder sit above Export Masks and load PNGs, editable 3D folders, or exported Direction recipes. Source margins are ignored. Editable 3D folders preserve the raw top and bottom faces; Direction rebuilds their envelope with its own margins. A standalone PNG without an editable recipe is treated as the supplied base shape. Reload masks after editing a source; existing Direction ranges are retained for matching object names and types.

Select a Distance Node independently of the heading method. For each object, set Object Margin and Near Margin as two offsets from the same base shape; Near Margin must be at least Object Margin. Negative offsets shrink the shape. Pixels uses the existing Occlusion margin, Perspective uses four calibration points, and Physical uses measured references in m/cm/mm. Expand Calibration only when needed. Select Line to use the current reference list, edit its real distance, draw two endpoints, or drag existing endpoints; a reference is created automatically if none exists. Top and Bottom show only the selected face's reference assignment; 2D masks use Object instead. The highlighted target and Measured / Needs line labels show the current task. Reference lines already available in the current video's 3D Object workspace are copied into Direction. Direction edits keep those source references unchanged. 3D physical margins apply to each raw face before connecting them, with separate top and bottom references. Apply to All copies margins and units while preserving each object's calibration. The selected object's Near boundary is dashed cyan; the distance node is purple.

Choose Ear Midpoint to Nose, Neck to Nose, or Body Center to Nose and map the corresponding CSV bodyparts. Angle Margin is the half-angle on either side of the heading, starting at the nose; 0 degrees tests an exact forward ray against occupied pixel cells. Each checked object adds `<object>.direction` and `<object>.direction_reason`, preserving source rows and identities. States are `Overlap` when the Distance Node is inside the Object region; `Far` when outside the Near region; and `NearFacing` / `NearNotFacing` between those regions, according to whether the heading intersects the Object region. The Near region itself is not a facing target. Missing or invalid Distance Node coordinates produce `Unknown`; missing heading points only affect Near rows. A nose inside the Object region with a different Distance Node in Near is also `Unknown`. Reasons retain actual bodypart names, such as `body_c_inside_object_zone` or `body_c_inside_near_zone;missing_nose`. The UI shows readable labels such as `Near / Facing` and `Overlap (body_c)`.

The default filename is `<source>_direction.csv`; current output prefix/suffix controls still apply. Two-dimensional pixel and perspective ranges support batch saving. 3D and Physical ranges are video-specific and disable batch saving. Switching videos keeps the object list, checked/selected objects, raw faces, margins, units, reference assignments and real lengths, bodypart choices, heading method, coordinates, and angle. Masks resize to the new video. Only line coordinates are video-specific: a new video needs fresh measurements, while returning to a video restores its measured lines without reverting the latest settings. Bodyparts missing from a CSV remain unselected until available again, preserving the chosen mapping. Export Masks saves base PNGs, Direction ranges, references, and raw 3D faces for a lossless reload. Known older Direction exports with baked margins require reloading the original Occlusion or editable 3D source. Geometric Occlusion exports include their four reference points for reuse in Direction.

Use Square Drag in Chamber/Room, Occlusion, or 3D face tools, or hold Shift while dragging a rectangle. Room masks still respect chamber boundaries and other rooms.

Interpolation uses frame numbers and processes each animal independently. It fills missing coordinates and restores absent rows between a track's first and last frames. Start/end coordinate gaps are filled only when extrapolation is enabled; synthesized rows have no invented confidence scores.

When track labels repeat within a frame, a stable `instance.id` can separate the animals. If no identity column separates them safely, interpolation stops and displays the conflicting frames and IDs. Batch saving reports the failed file and continues with valid files. The current video's explicitly selected CSV takes precedence over automatic matching.

Run the calculation and Qt export regression tests with `python -m unittest discover -s tests -v`. The tests generate synthetic single- and multi-animal trajectories, missing coordinates, absent frames, ambiguous IDs, and short videos, then check saved CSV coordinates against known trajectories.


## License

See [LICENSE](LICENSE).
