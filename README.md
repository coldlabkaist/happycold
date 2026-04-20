# happycold

`happycold` is a set of tools required for a behavioral analysis pipeline, including coordinate normalization, location detection, and occlusion detection. **For happy CoLD!**

It currently supports:

- `Square Normalize`: pick 4 points and export a perspective-normalized CSV
- `Chamber Mark`: define a chamber and named rooms, then export room membership by frame
- `Circle Detection`: draw a circle and classify each keypoint as `in` or `out`
- `Pin Coordinates`: inspect frame coordinates manually
- `Occlusion Detect`: draw named masks and export occlusion flags

## Requirements

- Python 3.11+ recommended
- Packages from [requirements.txt](C:/cold_yj/Programs/happycold/requirements.txt)

Install:

```
git clone https://github.com/coldlabkaist/happycold.git
# git clone -b v1.0.0 https://github.com/coldlabkaist/happycold.git  # for specific version
conda create -n happycold python=3.11 -y
conda activate happycold
cd happycold
pip install -r requirements.txt
```

## Run

```
cd (happycold directory)
conda activate happycold
python happycold.py
```

## Update

```
cd (happycold directory)
git fetch --tags
git checkout (version) # git checkout v1.0.0
```

## Basic Workflow

1. Open a folder containing videos.
2. Select a video from the list.
3. Let the app auto-match a CSV, or load one manually.
4. Choose a tab depending on the task.
5. Save the current result with the button at the bottom right.
