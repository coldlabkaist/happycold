# HappyCoLD

happycold is a desktop application for reviewing and post-processing animal-tracking CSV data alongside video.

Main features:

- Duplicate and Z-score tracking repair, region-based removal, and interpolation
- Perspective normalization with separate normalized coordinate columns
- Chamber, circle, and occlusion annotations
- Editable 3D object occlusion envelopes with named, video-specific length references
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

Use Square Drag in Chamber/Room, Occlusion, or 3D face tools, or hold Shift while dragging a rectangle. Room masks still respect chamber boundaries and other rooms.

Interpolation uses frame numbers and processes each animal independently. It fills missing coordinates and restores absent rows between a track's first and last frames. Start/end coordinate gaps are filled only when extrapolation is enabled; synthesized rows have no invented confidence scores.

When track labels repeat within a frame, a stable `instance.id` can separate the animals. If no identity column separates them safely, interpolation stops and displays the conflicting frames and IDs. Batch saving reports the failed file and continues with valid files. The current video's explicitly selected CSV takes precedence over automatic matching.

Run the calculation and Qt export regression tests with `python -m unittest discover -s tests -v`. The tests generate synthetic single- and multi-animal trajectories, missing coordinates, absent frames, ambiguous IDs, and short videos, then check saved CSV coordinates against known trajectories.


## License

See [LICENSE](LICENSE).
