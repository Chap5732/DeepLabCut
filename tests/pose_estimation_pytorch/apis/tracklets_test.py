import numpy as np
import pandas as pd
import pytest

from deeplabcut.core import trackingutils
from deeplabcut.pose_estimation_pytorch.apis.tracklets import (
    build_tracklets,
    rle_break_log,
)


@pytest.mark.parametrize(
    "assemblies_data, inference_cfg, joints, scorer, num_frames, unique_bodyparts",
    [
        (
            # assemblies_data
            {
                "single": {
                    0: np.array([[1, 2, 0.9]]),
                    1: np.array([[1, 3, 0.7]]),
                    2: np.array([[0, 1, 0.9]]),
                },
                0: [
                    np.array([[10, 20, 0.9, -1], [30, 40, 0.8, -1]]),
                    np.array([[13, 23, 0.9, -1], [33, 43, 0.8, -1]]),
                ],
                1: [
                    np.array([[9, 19, 0.9, -1], [29, 41, 0.8, -1]]),
                    np.array([[15, 21, 0.9, -1], [35, 45, 0.8, -1]]),
                ],
                2: [
                    np.array([[13, 23, 0.9, -1], [33, 43, 0.8, -1]]),
                    np.array([[10, 20, 0.9, -1], [30, 40, 0.8, -1]]),
                ],
            },
            # inference_cfg
            {"max_age": 3, "min_hits": 1, "topktoretain": 1, "pcutoff": 0.5},
            # joints
            ["nose", "ear"],
            # scorer
            "DLC",
            # num_frames
            3,
            # unique_bodyparts
            ["led"],
        ),
        (
            # assemblies_data
            {
                0: [
                    np.array([[10, 20, 0.9, -1], [30, 40, 0.8, -1]]),
                    np.array([[13, 23, 0.9, -1], [33, 43, 0.8, -1]]),
                ],
                1: [
                    np.array([[9, 19, 0.9, -1], [29, 41, 0.8, -1]]),
                    np.array([[15, 21, 0.9, -1], [35, 45, 0.8, -1]]),
                ],
                2: [
                    np.array([[13, 23, 0.9, -1], [33, 43, 0.8, -1]]),
                    np.array([[10, 20, 0.9, -1], [30, 40, 0.8, -1]]),
                ],
            },
            # inference_cfg
            {"max_age": 3, "min_hits": 1, "topktoretain": 1, "pcutoff": 0.5},
            # joints
            ["nose", "ear"],
            # scorer
            "DLC",
            # num_frames
            3,
            # unique_bodyparts
            None,
        ),
    ],
)
def test_build_tracklets(
    assemblies_data: dict,
    inference_cfg: dict,
    joints: list,
    scorer: str,
    num_frames: int,
    unique_bodyparts: list,
):
    # Run the function
    tracklets = build_tracklets(
        assemblies_data=assemblies_data,
        track_method="box",
        inference_cfg=inference_cfg,
        joints=joints,
        scorer=scorer,
        num_frames=num_frames,
        unique_bodyparts=unique_bodyparts,
        identity_only=False,
    )

    # # Assertions
    assert "header" in tracklets
    assert isinstance(tracklets["header"], pd.MultiIndex)
    if unique_bodyparts:
        assert "single" in tracklets
    else:
        assert not "single" in tracklets

    assert isinstance(tracklets, dict)


def test_build_tracklets_report_csv(tmp_path, monkeypatch):
    class DummySORTBox(trackingutils.SORTBox):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.break_log = {
                0: [
                    {"frame": 0, "reason": "dummy", "assembly": -1},
                    {"frame": 1, "reason": "dummy", "assembly": -1},
                    {"frame": 2, "reason": "other", "assembly": 0},
                ]
            }

    monkeypatch.setattr(trackingutils, "SORTBox", DummySORTBox)

    report_path = tmp_path / "report.csv"
    assemblies_data = {0: [np.array([[10, 20, 0.9, -1], [30, 40, 0.8, -1]])]}
    build_tracklets(
        assemblies_data=assemblies_data,
        track_method="box",
        inference_cfg={"max_age": 3, "min_hits": 1, "topktoretain": 1, "pcutoff": 0.5},
        joints=["nose", "ear"],
        scorer="DLC",
        num_frames=3,
        report_path=report_path,
    )

    df = pd.read_csv(report_path)
    assert list(df.columns) == [
        "tracklet_id",
        "start_frame",
        "end_frame",
        "break_reason",
        "assembly",
    ]
    assert len(df) == 2
    assert df.loc[0].to_dict() == {
        "tracklet_id": 0,
        "start_frame": 0,
        "end_frame": 1,
        "break_reason": "dummy",
        "assembly": -1,
    }
    assert df.loc[1].to_dict() == {
        "tracklet_id": 0,
        "start_frame": 2,
        "end_frame": 2,
        "break_reason": "other",
        "assembly": 0,
    }


def test_rle_break_log():
    events = [
        {"frame": 0, "reason": "a", "assembly": -1},
        {"frame": 1, "reason": "a", "assembly": -1},
        {"frame": 3, "reason": "b", "assembly": 2},
        {"frame": 4, "reason": "b", "assembly": 2},
        {"frame": 5, "reason": "a", "assembly": 3},
    ]

    encoded = rle_break_log(events)
    assert encoded == [
        {"start_frame": 0, "end_frame": 1, "break_reason": "a", "assembly": -1},
        {"start_frame": 3, "end_frame": 4, "break_reason": "b", "assembly": 2},
        {"start_frame": 5, "end_frame": 5, "break_reason": "a", "assembly": 3},
    ]
