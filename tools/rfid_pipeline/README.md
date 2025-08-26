# RFID Pipeline

Utilities for visualising RFID events alongside DeepLabCut tracklets.

## Video overlay

The script `deeplabcut/newfolder/make_video.py` overlays tracklet and RFID
information on top of a video and writes a new video file.

### Usage

```bash
python deeplabcut/newfolder/make_video.py VIDEO_PATH PICKLE_PATH OUTPUT_VIDEO \
    [--centers-txt CENTERS_TXT] [--roi-file ROI_FILE]
```

### Arguments

- `VIDEO_PATH`: path to the source video.
- `PICKLE_PATH`: tracklet pickle file containing tracking data.
- `OUTPUT_VIDEO`: path for the rendered output video.
- `--centers-txt`: optional text file specifying reader centers; enables reader
  position overlays.
- `--roi-file`: optional JSON file describing ROI polygons; enables ROI
  overlays.
