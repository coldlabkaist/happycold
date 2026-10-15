# HappyCoLD

happycold is a desktop application for reviewing and post-processing animal-tracking CSV data alongside video.

Main features:

- Duplicate and Z-score tracking repair, region-based removal, and interpolation
- Perspective normalization with separate normalized coordinate columns
- Chamber, circle, and occlusion annotations
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

Interpolation uses frame numbers and processes each animal independently. It fills missing coordinates and restores absent rows between a track's first and last frames. Start/end coordinate gaps are filled only when extrapolation is enabled; synthesized rows have no invented confidence scores.

When track labels repeat within a frame, a stable `instance.id` can separate the animals. If no identity column separates them safely, interpolation stops and displays the conflicting frames and IDs. Batch saving reports the failed file and continues with valid files. The current video's explicitly selected CSV takes precedence over automatic matching.

Run the calculation and Qt export regression tests with `python -m unittest discover -s tests -v`. The tests generate synthetic single- and multi-animal trajectories, missing coordinates, absent frames, ambiguous IDs, and short videos, then check saved CSV coordinates against known trajectories.


## License

See [LICENSE](LICENSE).
