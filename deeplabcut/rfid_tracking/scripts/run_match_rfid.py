#!/usr/bin/env python3
"""CLI wrapper to match RFID readings to tracklets."""

from __future__ import annotations

import argparse
import logging

from deeplabcut.rfid_tracking import config as cfg
from deeplabcut.rfid_tracking.match_rfid_to_tracklets import main as match_rfid


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match RFID events to DLC tracklets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pickle_path", help="Tracklets pickle file")
    parser.add_argument("rfid_csv", help="RFID events CSV file")
    parser.add_argument("centers_txt", help="Reader centers text file")
    parser.add_argument("ts_csv", help="Timestamps CSV used for alignment")
    parser.add_argument(
        "--out-dir",
        default="./rfid_match_outputs",
        help="Directory to store matching outputs",
    )
    parser.add_argument(
        "--mrt-coil-diameter-px",
        type=float,
        default=None,
        help="RFID coil diameter in pixels (overrides cfg.MRT_COIL_DIAMETER_PX)",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    logger.info("Arguments: %s", args)
    match_rfid(
        pickle_path=args.pickle_path,
        rfid_csv=args.rfid_csv,
        centers_txt=args.centers_txt,
        ts_csv=args.ts_csv,
        out_dir=args.out_dir,
        mrt_coil_diameter_px=args.mrt_coil_diameter_px,
    )


if __name__ == "__main__":
    main()

