from __future__ import annotations

from pathlib import Path
from typing import Optional

from .make_video import main as make_video
from .match_rfid_to_tracklets import main as match_rfid_to_tracklets
from .reconstruct_from_pickle import main as reconstruct_from_pickle


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

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    rfid_csv = Path(rfid_csv)
    if not rfid_csv.exists():
        raise FileNotFoundError(f"RFID CSV file not found: {rfid_csv}")

    centers_txt = Path(centers_txt)
    if not centers_txt.exists():
        raise FileNotFoundError(f"Reader centers file not found: {centers_txt}")

    ts_csv = Path(ts_csv)
    if not ts_csv.exists():
        raise FileNotFoundError(f"Timestamps CSV file not found: {ts_csv}")

    dest = Path(destfolder) if destfolder else video_path.parent
    videotype = video_path.suffix.lstrip(".")

    # 1) run inference to create assemblies without auto tracking
    analyze_videos(
        str(config_path),
        [str(video_path)],
        videotype=videotype,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        destfolder=str(dest),
        auto_track=False,
    )

    # 2) convert detections to tracklets
    valid_methods = {"ellipse", "skeleton", "box"}
    if track_method not in valid_methods:
        raise ValueError(
            f"Unsupported track_method '{track_method}'. Supported methods are: "
            f"{', '.join(sorted(valid_methods))}."
        )

    convert_detections2tracklets(
        config=str(config_path),
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
    match_rfid_to_tracklets(
        pickle_path=str(track_pickle),
        rfid_csv=str(rfid_csv),
        centers_txt=str(centers_txt),
        ts_csv=str(ts_csv),
        out_dir=None,
    )

    # 4) reconstruct identity chains
    reconstruct_from_pickle(
        pickle_in=str(track_pickle),
        pickle_out=None,
        out_subdir=None,
    )

    # 5) generate visualization video
    out_vid = (
        Path(output_video)
        if output_video
        else dest / f"{video_path.stem}_rfid_tracklets_overlay.mp4"
    )
    make_video(
        video_path=str(video_path),
        pickle_path=str(track_pickle),
        centers_txt=str(centers_txt),
        output_video=str(out_vid),
    )

    return str(out_vid)
