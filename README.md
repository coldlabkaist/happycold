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


## License

See [LICENSE](LICENSE).
