#!/usr/bin/env python3
"""Create a visualization video with preset file paths."""

from deeplabcut.rfid_tracking.make_video import main as make_video

VIDEO_PATH = "/data/myproject/video.mp4"
PICKLE_PATH = "/data/myproject/reconstructed_tracklets.pickle"
CENTERS_TXT = "/data/myproject/readers_centers.txt"
OUTPUT_VIDEO = "/data/myproject/output_overlay.mp4"


def main() -> None:
    """Invoke :func:`make_video` using default arguments."""
    make_video(
        video_path=VIDEO_PATH,
        pickle_path=PICKLE_PATH,
        centers_txt=CENTERS_TXT,
        output_video=OUTPUT_VIDEO,
    )


if __name__ == "__main__":
    main()
