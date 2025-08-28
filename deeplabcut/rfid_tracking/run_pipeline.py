#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line wrapper to run the full RFID tracking pipeline."""

import argparse
import logging

try:  # allow running as script or module
    from . import config as cfg
    from .pipeline import run_pipeline
except ImportError:  # pragma: no cover
    import config as cfg
    from pipeline import run_pipeline


def main() -> None:
    """Parse arguments and execute the RFID tracking pipeline."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run RFID tracking pipeline")
    parser.add_argument("config_path", help="Path to DLC config.yaml")
    parser.add_argument("video_path", help="Video file to analyze")
    parser.add_argument("rfid_csv", help="RFID event CSV")
    parser.add_argument("centers_txt", help="Reader centers text file")
    parser.add_argument("ts_csv", help="Timestamps CSV for alignment")
    parser.add_argument("--shuffle", type=int, default=1, help="DLC shuffle index")
    parser.add_argument(
        "--track_method", default="ellipse", help="Tracklet matching method"
    )
    parser.add_argument(
        "--destfolder",
        default=cfg.DESTFOLDER,
        help="Output directory for intermediates",
    )
    parser.add_argument(
        "--out-subdir",
        default=cfg.OUT_SUBDIR,
        help="Subdirectory within destfolder for intermediate and final outputs",
    )
    parser.add_argument(
        "--trainingsetindex", type=int, default=0, help="DLC training set index"
    )
    parser.add_argument(
        "--output_video", default=None, help="Path for final overlay video"
    )
    parser.add_argument(
        "--config_override",
        default=None,
        help="YAML file to override rfid_tracking.config settings",
    )
    parser.add_argument(
        "--mrt_coil_diameter_px",
        type=float,
        default=None,
        help="RFID coil diameter in pixels; overrides cfg.MRT_COIL_DIAMETER_PX",
    )
    args = parser.parse_args()
    if args.mrt_coil_diameter_px is not None:
        cfg.MRT_COIL_DIAMETER_PX = args.mrt_coil_diameter_px
    params = vars(args)
    params.pop("mrt_coil_diameter_px", None)
    run_pipeline(**params)


if __name__ == "__main__":
    main()
