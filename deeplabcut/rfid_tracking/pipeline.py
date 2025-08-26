from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import make_video, match_rfid_to_tracklets, reconstruct_from_pickle


def run_pipeline(
    config_path: str,
    video_path: str,
    rfid_csv: str,
    centers_txt: str,
    ts_csv: str,
    shuffle: int = 1,
    track_method: str = "ellipse",
    destfolder: Optional[str] = None,
    trainingsetindex: int = 0,
    output_video: Optional[str] = None,
) -> str:
    """Run the full video + RFID analysis pipeline.

    Parameters
    ----------
    config_path : str
        Path to the DLC project ``config.yaml``.
    video_path : str
        Video file to be analyzed.
    rfid_csv : str
        CSV file containing RFID events.
    centers_txt : str
        Text file with reader (x, y) coordinates.
    ts_csv : str
        CSV with timestamps used to align RFID and video frames.
    shuffle : int, optional
        Training shuffle to use, by default ``1``.
    track_method : str, optional
        Tracklet matching method ("ellipse", "skeleton", or "box").
    destfolder : str, optional
        Directory for intermediate outputs. If ``None``, uses the video folder.
    trainingsetindex : int, optional
        Training set index used for the DLC model, by default ``0``.
    output_video : str, optional
        Path of the final visualization video. If ``None``, a file named
        ``<video>_rfid_tracklets_overlay.mp4`` will be created in ``destfolder``.

    Returns
    -------
    str
        Path to the generated visualization video.
    """
    # Local imports to avoid circular dependency when DLC is imported
    from deeplabcut import analyze_videos, convert_detections2tracklets
    from deeplabcut.utils import auxiliaryfunctions as aux
    from deeplabcut.utils.auxiliaryfunctions import get_scorer_name

    video_path = Path(video_path)
    dest = Path(destfolder) if destfolder else video_path.parent
    videotype = video_path.suffix.lstrip(".")

    # 1) run inference to create assemblies without auto tracking
    analyze_videos(
        config_path,
        [str(video_path)],
        videotype=videotype,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        destfolder=str(dest),
        auto_track=False,
    )

    # 2) convert detections to tracklets
    convert_detections2tracklets(
        config=config_path,
        videos=[str(video_path)],
        videotype=videotype,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        track_method=track_method,
        destfolder=str(dest),
    )

    # Locate the generated tracklet pickle
    cfg = aux.read_config(config_path)
    train_fraction = cfg["TrainingFraction"][trainingsetindex]
    dlc_scorer = get_scorer_name(cfg, shuffle, train_fraction)[0]
    method_suffix = {"ellipse": "el", "box": "bx"}.get(track_method, "sk")
    track_pickle = dest / f"{video_path.stem}{dlc_scorer}_{method_suffix}.pickle"

    # 3) match RFID events to tracklets
    mrf = match_rfid_to_tracklets
    mrf.PICKLE_PATH = str(track_pickle)
    mrf.RFID_CSV = rfid_csv
    mrf.CENTERS_TXT = centers_txt
    mrf.TS_CSV = ts_csv
    mrf.OUT_DIR = None
    mrf.main()

    # 4) reconstruct identity chains
    rec = reconstruct_from_pickle
    rec.PICKLE_IN = str(track_pickle)
    rec.PICKLE_OUT = None
    rec.OUT_SUBDIR = None
    rec.main()

    # 5) generate visualization video
    mkv = make_video
    mkv.VIDEO_PATH = str(video_path)
    mkv.PICKLE_PATH = str(track_pickle)
    mkv.CENTERS_TXT = centers_txt
    mkv.OUTPUT_VIDEO = (
        str(Path(output_video))
        if output_video
        else str(dest / f"{video_path.stem}_rfid_tracklets_overlay.mp4")
    )
    mkv.main()

    return mkv.OUTPUT_VIDEO
