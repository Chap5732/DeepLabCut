#!/usr/bin/env python3
"""CLI wrapper to generate an RFID overlay video."""

from __future__ import annotations

import argparse

from deeplabcut.rfid_tracking.make_video import main as make_video


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a visualization video with RFID and tracklets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video_path", help="Input video file")
    parser.add_argument(
        "pickle_path",
        help="Tracklet pickle with reconstruction and RFID information",
    )
    parser.add_argument(
        "centers_txt", help="Reader centers text file used for visualization"
    )
    parser.add_argument(
        "--output-video",
        default="./rfid_tracklets_overlay.mp4",
        help="Path for the generated overlay video",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    make_video(
        video_path=args.video_path,
        pickle_path=args.pickle_path,
        centers_txt=args.centers_txt,
        output_video=args.output_video,
    )


if __name__ == "__main__":
    main()

