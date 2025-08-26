from pathlib import Path

import numpy as np
import pandas as pd

from deeplabcut.newfolder.match_rfid_to_tracklets import (
    assign_tag_for_one_tracklet,
    parse_rfid_csv,
    parse_timestamps_csv,
    time_to_nearest_frame,
)
from deeplabcut.pose_estimation_pytorch.apis.tracklets import build_tracklets

RFID_DEMO = Path("deeplabcut/rfid_visual_demo")
RFID_CSV = RFID_DEMO / "rfid_data_20250813_055827.csv"
TS_CSV = RFID_DEMO / "record_20250813_053913_timestamps.csv"


def test_detection_to_tracklet_conversion_runs_without_errors():
    assemblies_data = {
        0: [
            np.array([[10, 20, 0.9, -1], [30, 40, 0.8, -1]]),
            np.array([[13, 23, 0.9, -1], [33, 43, 0.8, -1]]),
        ],
        1: [
            np.array([[9, 19, 0.9, -1], [29, 41, 0.8, -1]]),
            np.array([[15, 21, 0.9, -1], [35, 45, 0.8, -1]]),
        ],
    }
    tracklets = build_tracklets(
        assemblies_data=assemblies_data,
        track_method="box",
        inference_cfg={"max_age": 3, "min_hits": 1, "topktoretain": 1, "pcutoff": 0.5},
        joints=["nose", "ear"],
        scorer="DLC",
        num_frames=3,
        unique_bodyparts=None,
        identity_only=False,
    )
    assert isinstance(tracklets, dict)
    assert "header" in tracklets


def _top_tag_from_demo():
    df = parse_rfid_csv(str(RFID_CSV))
    counts = df["tag"].value_counts()
    return counts.idxmax(), counts.to_dict()


def test_rfid_matching_assigns_expected_tag():
    tag, _ = _top_tag_from_demo()
    frames_arr, times_arr = parse_timestamps_csv(str(TS_CSV))
    df = parse_rfid_csv(str(RFID_CSV))
    subset = df[df["tag"] == tag].head(30)
    counts = subset["tag"].value_counts().to_dict()
    frames_dict = {
        tag: [
            (time_to_nearest_frame(t, frames_arr, times_arr), 1.0)
            for t in subset["time"].head(10)
        ]
    }
    assigned, _ = assign_tag_for_one_tracklet(
        counts_dict=counts,
        frames_dict=frames_dict,
        confidence_threshold=0.5,
        min_reads_threshold=5,
        dominant_ratio_threshold=2.0,
    )
    assert assigned == tag


def test_reconstruction_preserves_continuity_across_gaps():
    import sys

    sys.path.append(str(Path("deeplabcut/newfolder")))
    from reconstruct_from_pickle import reconstruct_by_timespeed_gate

    tag, _ = _top_tag_from_demo()
    arr = lambda x, y: np.array([[x, y, 0.9]])
    dd = {
        "header": None,
        "tk1": {0: arr(0, 0), 1: arr(1, 1), "tag": tag, "rfid_counts": {tag: 5}},
        "tk2": {5: arr(2, 2), 6: arr(3, 3)},
        "tk3": {10: arr(4, 4), 11: arr(5, 5)},
    }
    dd_out, _ = reconstruct_by_timespeed_gate(dd)
    chain_id = dd_out["tk1"].get("chain_id")
    for tk in ("tk1", "tk2", "tk3"):
        assert dd_out[tk].get("chain_id") == chain_id
        assert dd_out[tk].get("chain_tag") == tag
