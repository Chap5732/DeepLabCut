#!/usr/bin/env python3
"""Match RFID readings to tracklets using preset file locations."""

from deeplabcut.rfid_tracking.match_rfid_to_tracklets import main as match_rfid

# Default paths used in our local setup
PICKLE_PATH = "/data/myproject/tracklets.pickle"
RFID_CSV = "/data/myproject/rfid_events.csv"
CENTERS_TXT = "/data/myproject/readers_centers.txt"
TS_CSV = "/data/myproject/timestamps.csv"
OUT_DIR = "/data/myproject/rfid_match_outputs"


def main() -> None:
    """Execute :func:`match_rfid` with the predefined parameters."""
    match_rfid(
        pickle_path=PICKLE_PATH,
        rfid_csv=RFID_CSV,
        centers_txt=CENTERS_TXT,
        ts_csv=TS_CSV,
        out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
