#!/usr/bin/env python3
"""CLI wrapper to reconstruct identity chains from a pickle."""

from __future__ import annotations

import argparse

from deeplabcut.rfid_tracking.reconstruct_from_pickle import main as reconstruct


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct trajectories from a tracklet pickle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pickle_in", help="Input tracklets pickle with RFID tags")
    parser.add_argument(
        "--pickle-out",
        default=None,
        help="Path to save reconstructed pickle (defaults to overwrite input)",
    )
    parser.add_argument(
        "--out-subdir",
        default=None,
        help="Optional subdirectory for outputs relative to input pickle",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    reconstruct(
        pickle_in=args.pickle_in,
        pickle_out=args.pickle_out,
        out_subdir=args.out_subdir,
    )


if __name__ == "__main__":
    main()

