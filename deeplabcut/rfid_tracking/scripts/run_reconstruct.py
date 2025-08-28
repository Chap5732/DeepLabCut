#!/usr/bin/env python3
"""Reconstruct trajectories from a pickle using preferred defaults."""

from deeplabcut.rfid_tracking.reconstruct_from_pickle import main as reconstruct

PICKLE_IN = "/data/myproject/tracklets_with_rfid.pickle"
PICKLE_OUT = "/data/myproject/reconstructed_tracklets.pickle"


def main() -> None:
    """Invoke :func:`reconstruct` with predefined paths."""
    reconstruct(pickle_in=PICKLE_IN, pickle_out=PICKLE_OUT)


if __name__ == "__main__":
    main()
