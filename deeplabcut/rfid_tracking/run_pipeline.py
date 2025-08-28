#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line wrapper to run the full RFID tracking pipeline."""

import argparse
import logging

try:  # allow running as script or module
    from .pipeline import run_pipeline
except ImportError:  # pragma: no cover
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
    parser.add_argument("--track_method", default="ellipse", help="Tracklet matching method")
    parser.add_argument("--destfolder", default=None, help="Output directory for intermediates")
    parser.add_argument("--trainingsetindex", type=int, default=0, help="DLC training set index")
    parser.add_argument("--output_video", default=None, help="Path for final overlay video")
    parser.add_argument(
        "--config_override",
        default=None,
        help="YAML file to override rfid_tracking.config settings",
    )
    args = parser.parse_args()
    run_pipeline(**vars(args))


if __name__ == "__main__":
    main()
