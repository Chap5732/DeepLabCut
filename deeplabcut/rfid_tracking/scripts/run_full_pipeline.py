#!/usr/bin/env python3
"""Example CLI for running the complete RFID tracking pipeline."""

from __future__ import annotations

import argparse

from deeplabcut.rfid_tracking.pipeline import run_pipeline


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full RFID tracking pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config_path", help="Path to DLC project config.yaml")
    parser.add_argument("video_path", help="Video file to analyze")
    parser.add_argument("rfid_csv", help="RFID events CSV file")
    parser.add_argument("centers_txt", help="Reader centers text file")
    parser.add_argument("ts_csv", help="Timestamp CSV used for alignment")
    parser.add_argument("--destfolder", default=None, help="Output directory")
    parser.add_argument(
        "--out-subdir",
        default=None,
        help="Subdirectory inside destfolder for intermediate outputs",
    )
    parser.add_argument("--shuffle", type=int, default=1, help="DLC shuffle index")
    parser.add_argument(
        "--track-method",
        default="ellipse",
        help="Tracklet matching method (ellipse, skeleton, box)",
    )
    parser.add_argument(
        "--trainingsetindex", type=int, default=0, help="DLC training set index"
    )
    parser.add_argument(
        "--output-video",
        default=None,
        help="Path for final overlay video (defaults to auto naming)",
    )
    parser.add_argument(
        "--config-override",
        default=None,
        help="YAML file to override rfid_tracking.config settings",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_pipeline(**vars(args))


if __name__ == "__main__":
    main()

