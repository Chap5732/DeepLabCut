from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from . import config as cfg
from .make_video import main as make_video
from .match_rfid_to_tracklets import main as match_rfid_to_tracklets
from .reconstruct_from_pickle import main as reconstruct_from_pickle

logger = logging.getLogger(__name__)


def run_pipeline(
    config_path: str,
    video_path: str,
    rfid_csv: Optional[str] = None,
    centers_txt: Optional[str] = None,
    ts_csv: Optional[str] = None,
    shuffle: int = 1,
    track_method: str = "ellipse",
    destfolder: Optional[str] = cfg.DESTFOLDER,
    out_subdir: str | None = cfg.OUT_SUBDIR,
    trainingsetindex: int = 0,
    output_video: Optional[str] = None,
    config_override: str | Path | None = None,
) -> str:
    """Run the full video + RFID analysis pipeline.

    Parameters
    ----------
    config_path : str
        Path to the DLC project ``config.yaml``.
    video_path : str
        Video file to be analyzed.
    rfid_csv : str, optional
        CSV file containing RFID events. If ``None``,
        ``cfg.MRT_RFID_CSV`` from :mod:`rfid_tracking.config` is used.
    centers_txt : str, optional
        Text file with reader (x, y) coordinates. If ``None``,
        ``cfg.MRT_CENTERS_TXT`` is used.
    ts_csv : str, optional
        CSV with timestamps used to align RFID and video frames. If ``None``,
        ``cfg.MRT_TS_CSV`` is used.
    shuffle : int, optional
        Training shuffle to use, by default ``1``.
    track_method : str, optional
        Tracklet matching method ("ellipse", "skeleton", or "box").
    destfolder : str, optional
        Directory for intermediate outputs. Defaults to ``cfg.DESTFOLDER``;
        if ``None``, uses the video folder.
    out_subdir : str, optional
        Subdirectory inside ``destfolder`` for intermediate and reconstruction
        outputs. If ``None``, files are written directly to ``destfolder``.
    trainingsetindex : int, optional
        Training set index used for the DLC model, by default ``0``.
    output_video : str, optional
        Path of the final visualization video. If ``None``, a file named
        ``<video>_rfid_tracklets_overlay.mp4`` will be created in
        ``destfolder`` or ``destfolder/out_subdir``.
    config_override : str | Path, optional
        YAML file to override values in :mod:`rfid_tracking.config` before
        running the pipeline.

    Notes
    -----
    Default file paths for ``rfid_csv``, ``centers_txt``, and ``ts_csv`` are
    defined in :mod:`rfid_tracking.config` (``config.py``).

    Returns
    -------
    str
        Path to the generated visualization video.
    """
    # Local imports to avoid circular dependency when DLC is imported
    from deeplabcut import analyze_videos, convert_detections2tracklets
    from deeplabcut.utils import auxiliaryfunctions as aux
    from deeplabcut.utils.auxiliaryfunctions import get_scorer_name

    cfg.load_config(config_override)

    if config_path is None:
        raise ValueError(
            "config_path is required; provide via command line or configuration file"
        )
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if video_path is None:
        raise ValueError(
            "video_path is required; provide via command line or configuration file"
        )
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    rfid_csv = (
        Path(rfid_csv)
        if rfid_csv is not None
        else (Path(cfg.MRT_RFID_CSV) if cfg.MRT_RFID_CSV is not None else None)
    )
    if rfid_csv is None:
        raise ValueError(
            "rfid_csv is None; specify path via command line or set MRT_RFID_CSV in config"
        )
    if not rfid_csv.exists():
        raise FileNotFoundError(f"RFID CSV file not found: {rfid_csv}")

    centers_txt = (
        Path(centers_txt)
        if centers_txt is not None
        else (Path(cfg.MRT_CENTERS_TXT) if cfg.MRT_CENTERS_TXT is not None else None)
    )
    if centers_txt is None:
        raise ValueError(
            "centers_txt is None; specify path via command line or set MRT_CENTERS_TXT in config"
        )
    if not centers_txt.exists():
        raise FileNotFoundError(f"Reader centers file not found: {centers_txt}")

    ts_csv = (
        Path(ts_csv)
        if ts_csv is not None
        else (Path(cfg.MRT_TS_CSV) if cfg.MRT_TS_CSV is not None else None)
    )
    if ts_csv is None:
        raise ValueError(
            "ts_csv is None; specify path via command line or set MRT_TS_CSV in config"
        )
    if not ts_csv.exists():
        raise FileNotFoundError(f"Timestamps CSV file not found: {ts_csv}")

    dest = Path(destfolder) if destfolder else video_path.parent
    videotype = video_path.suffix.lstrip(".")

    # Load project config and prepare inference settings
    dlc_cfg = aux.read_config(config_path)
    inference_cfg = (
        dlc_cfg.get("inference_cfg") or dlc_cfg.get("inferencecfg") or {}
    ).copy()
    gate_params = {
        "velocity_gate_cms": getattr(cfg, "V_GATE_CMS", None),
        "px_per_cm": getattr(cfg, "PX_PER_CM", None),
        "fps": getattr(cfg, "FPS", None),
        "max_px_gate": getattr(cfg, "MAX_PX_GATE", None),
    }
    for key, val in gate_params.items():
        if val is not None:
            inference_cfg.setdefault(key, val)
    logger.info("inference_cfg: %s", inference_cfg)
    logger.info(
        "Gating parameters: %s",
        {
            k: inference_cfg.get(k)
            for k in (
                "max_px_gate",
                "velocity_gate_cms",
                "px_per_cm",
                "fps",
                "gate_last_position",
                "pcutoff",
                "topktoretain",
            )
        },
    )

    # 1) run inference to create assemblies without auto tracking
    logger.info(
        "Starting video analysis: %s (shuffle=%s, trainingsetindex=%s)",
        video_path,
        shuffle,
        trainingsetindex,
    )
    analyze_videos(
        str(config_path),
        [str(video_path)],
        videotype=videotype,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        destfolder=str(dest),
        auto_track=False,
    )
    logger.info("Finished video analysis: %s", video_path)

    # 2) convert detections to tracklets
    valid_methods = {"ellipse", "skeleton", "box"}
    if track_method not in valid_methods:
        raise ValueError(
            f"Unsupported track_method '{track_method}'. Supported methods are: "
            f"{', '.join(sorted(valid_methods))}."
        )
    logger.info(
        "Converting detections to tracklets for %s using %s method (shuffle=%s, trainingsetindex=%s)",
        video_path,
        track_method,
        shuffle,
        trainingsetindex,
    )
    convert_detections2tracklets(
        config=str(config_path),
        videos=[str(video_path)],
        videotype=videotype,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        track_method=track_method,
        destfolder=str(dest),
        inferencecfg=inference_cfg,
    )
    logger.info("Finished converting detections to tracklets for %s", video_path)

    # Locate the generated tracklet pickle
    train_fraction = dlc_cfg["TrainingFraction"][trainingsetindex]
    dlc_scorer = get_scorer_name(dlc_cfg, shuffle, train_fraction)[0]
    method_suffix = {"ellipse": "el", "box": "bx"}.get(track_method, "sk")
    track_pickle = dest / f"{video_path.stem}{dlc_scorer}_{method_suffix}.pickle"

    # 3) match RFID events to tracklets
    logger.info("Matching RFID events from %s to tracklets: %s", rfid_csv, track_pickle)
    mrt_out_dir = (
        dest / out_subdir / "rfid_match_outputs"
        if out_subdir
        else dest / "rfid_match_outputs"
    )
    match_rfid_to_tracklets(
        pickle_path=str(track_pickle),
        rfid_csv=str(rfid_csv),
        centers_txt=str(centers_txt),
        ts_csv=str(ts_csv),
        out_dir=str(mrt_out_dir),
    )
    logger.info("Finished matching RFID events for %s", track_pickle)

    # 4) reconstruct identity chains
    logger.info("Reconstructing identity chains from %s", track_pickle)
    reconstruct_from_pickle(
        pickle_in=str(track_pickle),
        pickle_out=None,
        out_subdir=out_subdir,
    )
    if out_subdir:
        track_pickle = track_pickle.parent / out_subdir / track_pickle.name
    logger.info("Finished reconstructing identity chains for %s", track_pickle)

    # 5) generate visualization video
    out_vid = (
        Path(output_video)
        if output_video
        else (
            dest / out_subdir / f"{video_path.stem}_rfid_tracklets_overlay.mp4"
            if out_subdir
            else dest / f"{video_path.stem}_rfid_tracklets_overlay.mp4"
        )
    )
    logger.info(
        "Generating visualization video: %s using tracklets %s",
        out_vid,
        track_pickle,
    )
    make_video(
        video_path=str(video_path),
        pickle_path=str(track_pickle),
        centers_txt=str(centers_txt),
        output_video=str(out_vid),
    )
    logger.info("Finished generating visualization video: %s", out_vid)

    return str(out_vid)
