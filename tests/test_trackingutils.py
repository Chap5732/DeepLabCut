#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import numpy as np
import pytest

from deeplabcut.core import trackingutils


def test_compute_v_gate_pxpf():
    assert trackingutils.compute_v_gate_pxpf(1, 2, 4) == 0.5
    assert trackingutils.compute_v_gate_pxpf(None, 2, 4) is None
    assert trackingutils.compute_v_gate_pxpf(1, -2, 4) is None
    assert trackingutils.compute_v_gate_pxpf(1, 2, 0) is None


@pytest.fixture()
def ellipse():
    params = {"x": 0, "y": 0, "width": 2, "height": 4, "theta": np.pi / 2}
    return trackingutils.Ellipse(**params)


def test_ellipse(ellipse):
    assert ellipse.aspect_ratio == 2
    np.testing.assert_equal(
        ellipse.contains_points(np.asarray([[0, 0], [10, 10]])), [True, False]
    )


def test_ellipse_similarity(ellipse):
    assert ellipse.calc_similarity_with(ellipse) == 1


def test_ellipse_fitter():
    fitter = trackingutils.EllipseFitter()
    assert fitter.fit(np.random.rand(2, 2)) is None
    xy = np.asarray([[-2, 0], [2, 0], [0, 1], [0, -1]], dtype=float)
    assert fitter.fit(xy) is not None
    fitter.sd = 0
    el = fitter.fit(xy)
    assert np.isclose(el.parameters, [0, 0, 4, 2, 0]).all()
    fitter = trackingutils.EllipseFitter(min_n_valid=5)
    assert fitter.fit(xy) is None
    xy5 = np.vstack([xy, [0, 0]])
    assert fitter.fit(xy5) is not None


def test_ellipse_tracker(ellipse):
    centroid = (ellipse.x, ellipse.y)
    tracker1 = trackingutils.EllipseTracker(ellipse.parameters, centroid)
    tracker2 = trackingutils.EllipseTracker(ellipse.parameters, centroid)
    assert tracker1.id != tracker2.id
    tracker1.update(ellipse.parameters, centroid)
    assert tracker1.hit_streak == 1
    state = tracker1.predict()
    np.testing.assert_equal(ellipse.parameters, state)
    _ = tracker1.predict()
    assert tracker1.hit_streak == 0


def test_sort_ellipse():
    tracklets = dict()
    mot_tracker = trackingutils.SORTEllipse(1, 1, 0.6)
    poses = np.random.rand(2, 10, 3)
    pre_tsu = {trk.id: trk.time_since_update for trk in mot_tracker.trackers}
    track_out = mot_tracker.track(poses[..., :2])
    trackers = track_out[0] if isinstance(track_out, tuple) else track_out
    assert trackers.shape == (2, 7)
    time_since_updates = {
        trk.id: pre_tsu.get(trk.id, 0) for trk in mot_tracker.trackers
    }
    trackingutils.fill_tracklets(
        tracklets, trackers, poses, imname=0, time_since_updates=time_since_updates
    )
    assert all(id_ in tracklets for id_ in trackers[:, -2])
    assert all(np.array_equal(tracklets[n][0], pose) for n, pose in enumerate(poses))
    assert "time_since_update" in tracklets
    assert all(
        tracklets["time_since_update"][n][0] == 0 for n in range(trackers.shape[0])
    )


def test_sort_ellipse_min_n_valid():
    mot_tracker = trackingutils.SORTEllipse(1, 1, 0.6, min_n_valid=5)
    pose = np.array([[-2, 0], [2, 0], [0, 1], [0, -1]], dtype=float)[None, ...]
    ret = mot_tracker.track(pose)[0]
    assert ret.size == 0
    assert len(mot_tracker.trackers) == 0


def _ellipse_pose(offset):
    base = np.array([[-2, 0], [2, 0], [0, 1], [0, -1]], dtype=float)
    return base + np.asarray(offset)


@pytest.mark.parametrize("gate_kwargs", [{"max_px_gate": 5}, {"v_gate_pxpf": 5}])
def test_sort_ellipse_gates_zero_iou(gate_kwargs):
    mot_tracker = trackingutils.SORTEllipse(5, 1, 0, **gate_kwargs)
    pose = _ellipse_pose((0, 0))[None, ...]
    mot_tracker.track(pose)
    far_pose = _ellipse_pose((10, 10))[None, ...]
    ret = mot_tracker.track(far_pose)[0]
    assert ret.size == 0
    assert len(mot_tracker.trackers) == 2
    assert mot_tracker.trackers[0].time_since_update == 1
    ret = mot_tracker.track(pose)[0]
    assert ret.shape[0] == 1
    assert ret[0, -2] == 0


def test_sort_ellipse_max_px_gate():
    mot_tracker = trackingutils.SORTEllipse(5, 1, 0.1, max_px_gate=5)
    pose = _ellipse_pose((0, 0))[None, ...]
    mot_tracker.track(pose)
    far_pose = _ellipse_pose((10, 10))[None, ...]
    ret = mot_tracker.track(far_pose)[0]
    assert ret.size == 0
    assert len(mot_tracker.trackers) == 2
    assert mot_tracker.trackers[0].time_since_update == 1
    ret = mot_tracker.track(pose)[0]
    assert ret.shape[0] == 1
    assert ret[0, -2] == 0


def test_sort_ellipse_max_px_gate_scaled_by_dt():
    mot_tracker = trackingutils.SORTEllipse(5, 1, 0.0, max_px_gate=5)
    pose = _ellipse_pose((0, 0))[None, ...]
    mot_tracker.track(pose)
    # Skip a frame to increase the time since update to 2 frames
    mot_tracker.track(np.empty((0, 4, 2)))
    near_pose = _ellipse_pose((8, 0))[None, ...]
    ret = mot_tracker.track(near_pose)[0]
    assert ret.shape[0] <= 1
    if ret.shape[0]:
        assert ret[0, -2] == 0


def test_sort_ellipse_max_dt_for_gating():
    mot_tracker = trackingutils.SORTEllipse(
        20, 1, 0.0, max_px_gate=5, max_dt_for_gating=3
    )
    pose = _ellipse_pose((0, 0))[None, ...]
    mot_tracker.track(pose)
    for _ in range(10):
        mot_tracker.track(np.empty((0, 4, 2)))
    near_pose = _ellipse_pose((20, 0))[None, ...]
    ret = mot_tracker.track(near_pose)[0]
    assert ret.size == 0
    assert len(mot_tracker.trackers) == 2


def test_sort_ellipse_v_gate_pxpf():
    mot_tracker = trackingutils.SORTEllipse(5, 1, 0.1, v_gate_pxpf=5)
    pose = _ellipse_pose((0, 0))[None, ...]
    mot_tracker.track(pose)
    far_pose = _ellipse_pose((10, 10))[None, ...]
    ret = mot_tracker.track(far_pose)[0]
    assert ret.size == 0
    assert len(mot_tracker.trackers) == 2
    assert mot_tracker.trackers[0].time_since_update == 1
    ret = mot_tracker.track(pose)[0]
    assert ret.shape[0] == 1
    assert ret[0, -2] == 0


@pytest.mark.parametrize("gate_key", ["max_px_gate", "v_gate_pxpf"])
@pytest.mark.parametrize("gate_last_position", [None, False, True])
def test_sort_ellipse_rejects_large_displacement(gate_key, gate_last_position):
    kwargs = {}
    if gate_last_position is not None:
        kwargs["gate_last_position"] = gate_last_position
    mot_tracker = trackingutils.SORTEllipse(5, 1, 0.0, **kwargs)
    pose = _ellipse_pose((0, 0))[None, ...]
    mot_tracker.track(pose)
    far_pose = _ellipse_pose((100, 0))[None, ...]
    mot_tracker.track(far_pose)
    setattr(mot_tracker, gate_key, 5)
    mismatch = _ellipse_pose((200, 0))[None, ...]
    ret = mot_tracker.track(mismatch)[0]
    enabled = True if gate_last_position is None else gate_last_position
    if enabled:
        assert ret.size == 0
        assert len(mot_tracker.trackers) == 2
        assert mot_tracker.trackers[0].time_since_update == 1
    else:
        assert ret.shape[0] == 1
        assert ret[0, -2] == 0


def test_tracking_ellipse(real_assemblies, real_tracklets):
    tracklets_ref = real_tracklets.copy()
    _ = tracklets_ref.pop("header", None)
    tracklets = dict()
    mot_tracker = trackingutils.SORTEllipse(1, 1, 0.6)
    for ind, assemblies in real_assemblies.items():
        animals = np.stack([ass.data for ass in assemblies])
        track_out = mot_tracker.track(animals[..., :2])
        trackers = track_out[0] if isinstance(track_out, tuple) else track_out
        trackingutils.fill_tracklets(tracklets, trackers, animals, ind)
    assert len(tracklets) == len(tracklets_ref)
    assert [len(tracklet) for tracklet in tracklets.values()] == [
        len(tracklet) for tracklet in tracklets_ref.values()
    ]
    assert all(
        t.shape[1] == 4 for tracklet in tracklets.values() for t in tracklet.values()
    )


def test_box_tracker():
    bbox = 0, 0, 100, 100
    tracker1 = trackingutils.BoxTracker(bbox)
    tracker2 = trackingutils.BoxTracker(bbox)
    assert tracker1.id != tracker2.id
    tracker1.update(bbox)
    assert tracker1.hit_streak == 1
    state = tracker1.predict()
    np.testing.assert_equal(bbox, state)
    _ = tracker1.predict()
    assert tracker1.hit_streak == 0


def test_tracking_box(real_assemblies, real_tracklets):
    tracklets_ref = real_tracklets.copy()
    _ = tracklets_ref.pop("header", None)
    tracklets = dict()
    mot_tracker = trackingutils.SORTBox(1, 1, 0.1)
    for ind, assemblies in real_assemblies.items():
        animals = np.stack([ass.data for ass in assemblies])
        bboxes = trackingutils.calc_bboxes_from_keypoints(animals)
        trackers = mot_tracker.track(bboxes)
        trackingutils.fill_tracklets(tracklets, trackers, animals, ind)
    assert len(tracklets) == len(tracklets_ref)
    assert [len(tracklet) for tracklet in tracklets.values()] == [
        len(tracklet) for tracklet in tracklets_ref.values()
    ]
    assert all(
        t.shape[1] == 4 for tracklet in tracklets.values() for t in tracklet.values()
    )


def test_tracking_montblanc(
    real_assemblies_montblanc,
    real_tracklets_montblanc,
):
    tracklets_ref = real_tracklets_montblanc.copy()
    _ = tracklets_ref.pop("header", None)
    tracklets = dict()
    tracklets["single"] = real_assemblies_montblanc[1]
    mot_tracker = trackingutils.SORTEllipse(1, 1, 0.6)
    for ind, assemblies in real_assemblies_montblanc[0].items():
        animals = np.stack([ass.data for ass in assemblies])
        track_out = mot_tracker.track(animals[..., :2])
        trackers = track_out[0] if isinstance(track_out, tuple) else track_out
        trackingutils.fill_tracklets(tracklets, trackers, animals, ind)
    assert len(tracklets) == len(tracklets_ref)
    assert [len(tracklet) for tracklet in tracklets.values()] == [
        len(tracklet) for tracklet in tracklets_ref.values()
    ]
    for k, assemblies in tracklets.items():
        ref = tracklets_ref[k]
        for ind, data in assemblies.items():
            frame = f"frame{str(ind).zfill(3)}" if k != "single" else ind
            np.testing.assert_equal(data, ref[frame])


def test_calc_bboxes_from_keypoints():
    # Test bounding box from a single keypoint
    xy = np.asarray([[[0, 0, 1]]])
    np.testing.assert_equal(
        trackingutils.calc_bboxes_from_keypoints(xy, 10), [[-10, -10, 10, 10, 1]]
    )
    np.testing.assert_equal(
        trackingutils.calc_bboxes_from_keypoints(xy, 20, 10), [[-10, -20, 30, 20, 1]]
    )

    width = 200
    height = width * 2
    xyp = np.zeros((1, 2, 3))
    xyp[:, 1, :2] = width, height
    xyp[:, 1, 2] = 1
    with pytest.raises(ValueError):
        _ = trackingutils.calc_bboxes_from_keypoints(xyp[..., :2])

    bboxes = trackingutils.calc_bboxes_from_keypoints(xyp)
    np.testing.assert_equal(bboxes, [[0, 0, width, height, 0.5]])

    slack = 20
    bboxes = trackingutils.calc_bboxes_from_keypoints(xyp, slack=slack)
    np.testing.assert_equal(
        bboxes, [[-slack, -slack, width + slack, height + slack, 0.5]]
    )

    offset = 50
    bboxes = trackingutils.calc_bboxes_from_keypoints(xyp, offset=offset)
    np.testing.assert_equal(bboxes, [[offset, 0, width + offset, height, 0.5]])
