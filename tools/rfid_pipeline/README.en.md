# RFID DLC Tracking Project

This project combines DeepLabCut and RFID technology for animal behavior tracking. It reconstructs animal trajectories and produces visualization videos.

## Project Structure
```
project/
├── utils.py                      # Core utility functions
├── reconstruct_from_pickle.py    # Trajectory reconstruction script
├── make_video.py                 # Video visualization script
├── roi_definitions.json          # ROI region definitions
└── README.md                     # Project documentation
```

## Features

### 1. Trajectory Reconstruction (`reconstruct_from_pickle.py`)
- Reconstruct complete animal trajectories based on RFID tag information
- Use tracklets with RFID tags as anchors
- Connect tracklets before and after anchors via nearest neighbor distance
- Output reconstructed pickle files and chain segment CSVs

**Algorithm Highlights:**
- Anchor selection: choose tracklets with RFID tags and hits ≥ threshold
- Bidirectional expansion: greedily connect nearest tracklets forward/backward
- Temporal constraint: connect only when time gap lies within the specified range

### 2. Video Visualization (`make_video.py`)
- Overlay tracking information on the original video
- Display tracklet trajectories, RFID events, and reconstructed identity chains
- Optionally draw reader locations and ROIs
- Produce a video with legend overlays

**Visualization Elements:**
- Tracklet trajectories: colored lines and ID labels
- RFID events: tag detection prompts
- Identity chains: reconstructed continuous trajectories
- Readers: RFID reader position markers
- ROIs: boundaries of regions of interest

### 3. Utility Library (`utils.py`)
Provides shared core functions:
- **Data I/O:** load and save pickle files
- **DLC processing:** frame index extraction and body center calculation
- **Visualization:** color generation, reader drawing, ROI drawing
- **Geometry:** ROI hit tests and distance calculations

## Usage

### 1. Trajectory Reconstruction
```bash
# Modify path configuration in reconstruct_from_pickle.py
python reconstruct_from_pickle.py
```
Outputs:
- Updated pickle file (with `chain_tag` and `chain_id`)
- `chain_segments.csv`: detailed chain segment information

### 2. Video Generation
```bash
# Modify path configuration in make_video.py
python make_video.py
```
Output:
- `rfid_tracklets_overlay.mp4`: visualization video

## Configuration Parameters

### Reconstruction
- `PCUTOFF = 0.35`: confidence threshold
- `HEAD_TAIL_SAMPLE = 5`: sample frames for head/tail averaging
- `MAX_GAP_FRAMES = 60`: maximum temporal gap
- `ANCHOR_MIN_HITS = 1`: minimum RFID hits for anchors

### Visualization
- `TRAIL_LEN = 15`: tracklet trajectory length
- `CHAIN_TRAIL_LEN = 40`: identity chain trajectory length
- `TAG_HOLD_FRAMES = 3`: duration for displaying RFID tags
- `MAX_FRAMES = None`: maximum output frames (None=all)

## Data Formats

### ROI Definition File (JSON)
```json
{
  "Entrance1": [
    [35, 444],
    [162, 444],
    [162, 634],
    [35, 634]
  ]
}
```

### Reader Center File (TXT/CSV)
```
# Format: row, col, x, y
0, 0, 100.5, 200.3
0, 1, 150.2, 200.1
...
```

## Reconstruction Improvements
Compared to the original version:
1. **Simplified structure:** removed `config.py`; configuration is hard-coded in scripts
2. **Unified utilities:** all shared functions consolidated in `utils.py`
3. **Removed redundancy:** dropped unnecessary complex imports and `__init__.py`
4. **Clear responsibilities:** each script focuses on a single function
5. **Improved documentation:** detailed function and parameter descriptions

## Dependencies
- numpy
- pandas
- opencv-python
- pathlib (standard library)
- json (standard library)

## Notes
- Adjust path configuration in scripts before use
- Ensure input pickle files contain valid DLC tracklet data structures
- Currently only polygon-type JSON is supported for ROI files
