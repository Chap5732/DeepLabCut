#!/usr/bin/env python3
"""Run the entire RFID tracking pipeline with preset paths."""

from deeplabcut.rfid_tracking import config as cfg
from deeplabcut.rfid_tracking.pipeline import run_pipeline

# Update these paths to match your environment
CONFIG_PATH = "/data/myproject/config.yaml"
VIDEO_PATH = "/data/myproject/video.mp4"
RFID_CSV = "/data/myproject/rfid_events.csv"
CENTERS_TXT = "/data/myproject/readers_centers.txt"
TS_CSV = "/data/myproject/timestamps.csv"
DESTFOLDER = cfg.DESTFOLDER  # Optional output dir; overrides ``config.DESTFOLDER``
MRT_COIL_DIAMETER_PX = cfg.MRT_COIL_DIAMETER_PX  # Override coil diameter if needed


def main() -> None:
    """Execute :func:`run_pipeline` with default arguments."""
    run_pipeline(
        config_path=CONFIG_PATH,
        video_path=VIDEO_PATH,
        rfid_csv=RFID_CSV,
        centers_txt=CENTERS_TXT,
        ts_csv=TS_CSV,
        destfolder=DESTFOLDER,
        mrt_coil_diameter_px=MRT_COIL_DIAMETER_PX,
    )


if __name__ == "__main__":
    main()
