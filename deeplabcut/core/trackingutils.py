# -*- coding: utf-8 -*-
#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Licensed under GNU Lesser General Public License v3.0
#

import abc
import logging
import math
import warnings
from collections import defaultdict

import numpy as np
from filterpy.common import kinematic_kf
from filterpy.kalman import KalmanFilter
from matplotlib import patches
from numba import jit
from numba.core.errors import NumbaPerformanceWarning
from scipy.optimize import linear_sum_assignment
from scipy.stats import mode
from tqdm import tqdm

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

logger = logging.getLogger(__name__)

TRACK_METHODS = {
    "box": "_bx",
    "ctd": "_ctd",
    "skeleton": "_sk",
    "ellipse": "_el",
    "transformer": "_tr",
}


def calc_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    wh = w * h
    return wh / (
        (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        - wh
    )


class BaseTracker:
    """Base class for a constant-velocity Kalman filter-based tracker."""

    n_trackers = 0

    def __init__(self, dim, dim_z):
        self.kf = kinematic_kf(dim, 1, dim_z=dim_z, order_by_dim=False)
        self.id = self.__class__.n_trackers
        self.__class__.n_trackers += 1
        self.time_since_update = 0
        self.age = 0
        self.hits = 0
        self.hit_streak = 0

    def update(self, z):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(z)

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.state

    @property
    def state(self):
        return self.kf.x.squeeze()[: self.kf.dim_z]

    @state.setter
    def state(self, state):
        self.kf.x[: self.kf.dim_z] = state


class Ellipse:
    def __init__(self, x, y, width, height, theta):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.theta = theta  # in radians
        self._geometry = None

    @property
    def parameters(self):
        return self.x, self.y, self.width, self.height, self.theta

    @property
    def aspect_ratio(self):
        return max(self.width, self.height) / min(self.width, self.height)

    def calc_similarity_with(self, other_ellipse):
        max_dist = max(
            self.height, self.width, other_ellipse.height, other_ellipse.width
        )
        dist = math.hypot(self.x - other_ellipse.x, self.y - other_ellipse.y)
        if max_dist == 0:
            max_dist = 1.0
        cost1 = 1 - min(dist / max_dist, 1.0)
        cost2 = abs(math.cos(self.theta - other_ellipse.theta))
        return 0.8 * cost1 + 0.2 * cost2 * cost1

    def contains_points(self, xy, tol=0.1):
        ca = math.cos(self.theta)
        sa = math.sin(self.theta)
        x_demean = xy[:, 0] - self.x
        y_demean = xy[:, 1] - self.y
        return (
            ((ca * x_demean + sa * y_demean) ** 2 / (0.5 * self.width) ** 2)
            + ((sa * x_demean - ca * y_demean) ** 2 / (0.5 * self.height) ** 2)
        ) <= 1 + tol

    def draw(self, show_axes=True, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.transforms import Affine2D

        if ax is None:
            ax = plt.subplot(111, aspect="equal")
        el = patches.Ellipse(
            xy=(self.x, self.y),
            width=self.width,
            height=self.height,
            angle=np.rad2deg(self.theta),
            **kwargs,
        )
        ax.add_patch(el)
        if show_axes:
            major = Line2D([-self.width / 2, self.width / 2], [0, 0], lw=3, zorder=3)
            minor = Line2D([0, 0], [-self.height / 2, self.height / 2], lw=3, zorder=3)
            trans = (
                Affine2D().rotate(self.theta).translate(self.x, self.y) + ax.transData
            )
            major.set_transform(trans)
            minor.set_transform(trans)
            ax.add_artist(major)
            ax.add_artist(minor)


class EllipseFitter:
    def __init__(self, sd=2, min_n_valid=None):
        self.sd = sd
        self.min_n_valid = min_n_valid
        self.x = None
        self.y = None
        self.params = None
        self._coeffs = None

    def fit(self, xy):
        mask = np.isfinite(xy).all(axis=1)
        if self.min_n_valid is not None and mask.sum() < self.min_n_valid:
            return None
        self.x, self.y = xy[mask].T
        if len(self.x) < 3:
            return None
        if self.sd:
            self.params = self._fit_error(self.x, self.y, self.sd)
        else:
            self._coeffs = self._fit(self.x, self.y)
            self.params = self.calc_parameters(self._coeffs)
        if not np.isnan(self.params).any():
            return Ellipse(*self.params)
        return None

    @staticmethod
    @jit(nopython=True)
    def _fit(x, y):
        D1 = np.vstack((x * x, x * y, y * y))
        D2 = np.vstack((x, y, np.ones_like(x)))
        S1 = D1 @ D1.T
        S2 = D1 @ D2.T
        S3 = D2 @ D2.T
        T = -np.linalg.inv(S3) @ S2.T
        temp = S1 + S2 @ T
        M = np.zeros_like(temp)
        M[0] = temp[2] * 0.5
        M[1] = -temp[1]
        M[2] = temp[0] * 0.5
        E, V = np.linalg.eig(M)
        cond = 4 * V[0] * V[2] - V[1] ** 2
        a1 = V[:, cond > 0][:, 0]
        a2 = T @ a1
        return np.hstack((a1, a2))

    @staticmethod
    @jit(nopython=True)
    def _fit_error(x, y, sd):
        cov = np.cov(x, y)
        E, V = np.linalg.eigh(cov)  # ascending
        height, width = 2 * sd * np.sqrt(E)
        a, b = V[:, 1]
        rotation = math.atan2(b, a) % np.pi
        return [np.mean(x), np.mean(y), width, height, rotation]

    @staticmethod
    @jit(nopython=True)
    def calc_parameters(coeffs):
        a, b, c, d, f, g = coeffs
        b *= 0.5
        d *= 0.5
        f *= 0.5

        x0 = (c * d - b * f) / (b * b - a * c)
        y0 = (a * f - b * d) / (b * b - a * c)

        num = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        den1 = (b * b - a * c) * (np.sqrt((a - c) ** 2 + 4 * b * b) - (a + c))
        den2 = (b * b - a * c) * (-np.sqrt((a - c) ** 2 + 4 * b * b) - (a + c))
        major = np.sqrt(num / den1)
        minor = np.sqrt(num / den2)

        if b == 0:
            phi = 0 if a < c else np.pi / 2
        else:
            if a < c:
                phi = np.arctan(2 * b / (a - c)) / 2
            else:
                phi = np.pi / 2 + np.arctan(2 * b / (a - c)) / 2

        return [x0, y0, 2 * major, 2 * minor, phi]


class EllipseTracker(BaseTracker):
    def __init__(self, params, centroid):
        super().__init__(dim=5, dim_z=5)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[5:, 5:] *= 1000.0  # velocity unobservable initially
        self.kf.P *= 10.0
        self.kf.Q[5:, 5:] *= 0.01
        self.state = params
        # gating based on last confirmed (not prediction)
        self.last_confirmed_state = np.asarray(params).copy()
        self.last_confirmed_centroid = np.asarray(centroid).copy()

    @BaseTracker.state.setter
    def state(self, params):
        state = np.asarray(params).reshape((-1, 1))
        super(EllipseTracker, type(self)).state.fset(self, state)

    def update(self, params, centroid):
        super().update(np.asarray(params))
        self.last_confirmed_state = self.state.copy()
        self.last_confirmed_centroid = np.asarray(centroid).copy()


class SkeletonTracker(BaseTracker):
    def __init__(self, n_bodyparts):
        super().__init__(dim=n_bodyparts * 2, dim_z=n_bodyparts)
        self.kf.Q[self.kf.dim_z :, self.kf.dim_z :] *= 10
        self.kf.R[self.kf.dim_z :, self.kf.dim_z :] *= 0.01
        self.kf.P[self.kf.dim_z :, self.kf.dim_z :] *= 1000

    def update(self, pose):
        flat = pose.reshape((-1, 1))
        empty = np.isnan(flat).squeeze()
        if empty.any():
            H = self.kf.H.copy()
            H[empty] = 0
            flat[empty] = 0
            self.kf.update(flat, H=H)
        else:
            super().update(flat)

    @BaseTracker.state.setter
    def state(self, pose):
        curr_pose = pose.copy()
        empty = np.isnan(curr_pose).all(axis=1)
        if empty.any():
            fill = np.nanmean(pose, axis=0)
            curr_pose[empty] = fill
        super(SkeletonTracker, type(self)).state.fset(self, curr_pose.reshape((-1, 1)))


class BoxTracker(BaseTracker):
    def __init__(self, bbox):
        super().__init__(dim=4, dim_z=4)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.state = bbox

    def update(self, bbox):
        super().update(self.convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        return super().predict()

    @property
    def state(self):
        return self.convert_x_to_bbox(self.kf.x)[0]

    @state.setter
    def state(self, bbox):
        state = self.convert_bbox_to_z(bbox)
        super(BoxTracker, type(self)).state.fset(self, state)

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array(
                [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
            ).reshape((1, 4))
        else:
            return np.array(
                [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
            ).reshape((1, 5))

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))


class SORTBase(metaclass=abc.ABCMeta):
    def __init__(self):
        self.n_frames = 0
        self.trackers = []

    @abc.abstractmethod
    def track(self, *args, **kwargs):
        pass


def _allowed_disp(max_px_gate, v_gate_pxpf, dt):
    """Return allowed displacement based on gates.
    - Absolute gate: fixed radius (NOT scaled by dt).
    - Velocity gate: scaled by dt.
    - If both provided: take the stricter one (min).
    """
    limits = []
    if max_px_gate is not None and max_px_gate > 0:
        limits.append(float(max_px_gate))
    if v_gate_pxpf is not None and v_gate_pxpf > 0:
        limits.append(float(v_gate_pxpf) * float(dt))
    if not limits:
        return None
    return min(limits)


class SORTBox(SORTBase):
    def __init__(self, max_age, min_hits, iou_threshold):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        BoxTracker.n_trackers = 0
        super().__init__()

    def track(self, bboxes):
        self.n_frames += 1
        trks = np.zeros((len(self.trackers), 5))
        for i, trk in enumerate(self.trackers):
            trks[i, :4] = trk.predict()
        if len(bboxes) == 0:
            matched = np.empty((0, 2), dtype=int)
            unmatched_dets = []
            unmatched_trks = list(range(len(trks)))
        else:
            iou_matrix = np.zeros((len(bboxes), len(trks)))
            for d, det in enumerate(bboxes):
                for t, trk in enumerate(trks):
                    iou_matrix[d, t] = calc_iou(det, trk[:4])
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            unmatched_dets = [d for d in range(len(bboxes)) if d not in row_ind]
            unmatched_trks = [t for t in range(len(trks)) if t not in col_ind]
            matches = []
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] < self.iou_threshold:
                    unmatched_dets.append(r)
                    unmatched_trks.append(c)
                else:
                    matches.append([r, c])
            matched = (
                np.asarray(matches, dtype=int)
                if matches
                else np.empty((0, 2), dtype=int)
            )

        for m in matched:
            self.trackers[m[1]].update(bboxes[m[0]])
        for i in unmatched_dets:
            self.trackers.append(BoxTracker(bboxes[i]))

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.state
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.n_frames <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


class SORTEllipse(SORTBase):
    def __init__(
        self,
        max_age,
        min_hits,
        iou_threshold,
        sd=2,
        max_px_gate=None,
        v_gate_pxpf=None,
        verbose=False,
        gate_last_position=True,
        max_dt_for_gating=5,
        min_n_valid=None,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold  # actually a similarity threshold
        self.fitter = EllipseFitter(sd, min_n_valid)
        self.max_px_gate = max_px_gate
        self.v_gate_pxpf = v_gate_pxpf  # px per frame
        self.verbose = verbose
        self.gate_last_position = gate_last_position
        self.max_dt_for_gating = max(1, int(max_dt_for_gating))
        self.min_n_valid = min_n_valid
        EllipseTracker.n_trackers = 0
        super().__init__()
        self.break_log = defaultdict(list)
        logger.info(
            "SORTEllipse init: max_px_gate=%s v_gate_pxpf=%s gate_last=%s max_dt=%s max_age=%s min_hits=%s sim_thr=%s min_n_valid=%s",
            self.max_px_gate,
            self.v_gate_pxpf,
            self.gate_last_position,
            self.max_dt_for_gating,
            self.max_age,
            self.min_hits,
            self.iou_threshold,
            self.min_n_valid,
        )

    def track(self, poses, identities=None):
        self.n_frames += 1
        frame_breaks = []

        # 1) 预测所有 tracker
        trackers = np.zeros((len(self.trackers), 6))
        for i in range(len(trackers)):
            trackers[i, :5] = self.trackers[i].predict()
        empty = np.isnan(trackers).any(axis=1)
        valid_idx = np.flatnonzero(~empty)
        trackers = trackers[valid_idx]
        unmatched_trackers = np.flatnonzero(empty).tolist()
        ellipses, centroids, pred_ids, det_indices = [], [], [], []
        unmatched_detections = []
        gated_trackers = set()

        # 2) 拟合检测（椭圆 + 质心）
        for i, pose in enumerate(poses):
            el = self.fitter.fit(pose)
            if el is None:
                unmatched_detections.append(i)
                continue
            centroid = np.nanmean(pose[:, :2], axis=0)
            if not np.isfinite(centroid).all():
                unmatched_detections.append(i)
                continue
            ellipses.append(el)
            centroids.append(centroid.astype(float))
            det_indices.append(i)
            if identities is not None:
                try:
                    pred_ids.append(mode(identities[i])[0][0])
                except Exception:
                    pred_ids.append(None)

        if not len(trackers):
            # 没有活动tracker，全部当作未匹配检测，后面逐个spawn
            matches = np.empty((0, 2), dtype=int)
            unmatched_detection_idx = list(range(len(ellipses)))
            unmatched_detections.extend(det_indices[j] for j in unmatched_detection_idx)
            unmatched_detections = np.array(unmatched_detections, dtype=int)
            unmatched_detection_idx = np.array(unmatched_detection_idx, dtype=int)
            unmatched_trackers = np.array(unmatched_trackers, dtype=int)
        else:
            # 3) 构造代价矩阵（加入硬门控）
            ellipses_trackers = [Ellipse(*t[:5]) for t in trackers]
            last_confirmed = [
                Ellipse(*self.trackers[idx].last_confirmed_state) for idx in valid_idx
            ]
            cost_matrix = np.zeros((len(ellipses), len(ellipses_trackers)))

            for i, (el, centroid) in enumerate(zip(ellipses, centroids)):
                for j, (el_track, el_last) in enumerate(
                    zip(ellipses_trackers, last_confirmed)
                ):
                    tracker = self.trackers[valid_idx[j]]

                    if self.gate_last_position:
                        dist = float(
                            np.linalg.norm(centroid - tracker.last_confirmed_centroid)
                        )
                    else:
                        dist = math.hypot(
                            centroid[0] - el_track.x, centroid[1] - el_track.y
                        )

                    dt = min(max(tracker.time_since_update, 1), self.max_dt_for_gating)
                    allowed = _allowed_disp(self.max_px_gate, self.v_gate_pxpf, dt)

                    if allowed is not None and dist > allowed:
                        cost_matrix[i, j] = -1e6  # 硬屏蔽
                        gated_trackers.add(tracker.id)
                        logger.debug(
                            "[GATE BLOCK][matrix] trk=%s dt=%d dist=%.1f allowed=%.1f frame=%s",
                            tracker.id,
                            dt,
                            dist,
                            allowed,
                            self.n_frames,
                        )
                        continue

                    cost = el.calc_similarity_with(el_track)
                    if identities is not None and j < len(self.trackers):
                        id_match = (
                            2.0
                            if (
                                hasattr(tracker, "id_")
                                and pred_ids
                                and pred_ids[i] == tracker.id_
                            )
                            else 1.0
                        )
                        cost *= id_match
                    cost_matrix[i, j] = cost

            # 可行性检查（便于调试）
            feasible_rows = (cost_matrix > -1e5).any(axis=1) if cost_matrix.size else []
            feasible_cols = (cost_matrix > -1e5).any(axis=0) if cost_matrix.size else []
            if self.verbose:
                bad_rows = (
                    np.where(~feasible_rows)[0].tolist() if len(feasible_rows) else []
                )
                bad_cols = (
                    np.where(~feasible_cols)[0].tolist() if len(feasible_cols) else []
                )
                if bad_rows or bad_cols:
                    logger.debug(
                        "No feasible rows (det idx): %s ; no feasible cols (trk local idx): %s",
                        bad_rows,
                        bad_cols,
                    )

            # 4) 匈牙利匹配
            if len(ellipses) and len(ellipses_trackers):
                row_indices, col_indices = linear_sum_assignment(
                    cost_matrix, maximize=True
                )
            else:
                row_indices, col_indices = np.array([], dtype=int), np.array(
                    [], dtype=int
                )

            unmatched_detection_idx = [
                i for i in range(len(ellipses)) if i not in row_indices
            ]
            unmatched_trackers.extend(
                valid_idx[j] for j in range(len(trackers)) if j not in col_indices
            )

            matches_list = []
            for row, col in zip(row_indices, col_indices):
                val = cost_matrix[row, col]
                if val < self.iou_threshold:
                    unmatched_detection_idx.append(row)
                    unmatched_trackers.append(valid_idx[col])
                else:
                    matches_list.append([row, valid_idx[col]])

            matches = (
                np.asarray(matches_list, dtype=int)
                if matches_list
                else np.empty((0, 2), dtype=int)
            )

            # 5) 匹配后复检（再次硬门控，保持同一规则）
            if self.gate_last_position and len(matches):
                keep = []
                for det_ind, trk_ind in matches:
                    tracker = self.trackers[trk_ind]
                    prev_centroid = tracker.last_confirmed_centroid
                    det_centroid = centroids[det_ind]
                    disp = float(np.linalg.norm(det_centroid - prev_centroid))
                    dt = min(max(tracker.time_since_update, 1), self.max_dt_for_gating)
                    allowed = _allowed_disp(self.max_px_gate, self.v_gate_pxpf, dt)
                    if allowed is not None and disp > allowed:
                        unmatched_detection_idx.append(det_ind)
                        unmatched_trackers.append(trk_ind)
                        gated_trackers.add(tracker.id)
                        logger.debug(
                            "[GATE BLOCK][post] trk=%s dt=%d dist=%.1f allowed=%.1f frame=%s",
                            tracker.id,
                            dt,
                            disp,
                            allowed,
                            self.n_frames,
                        )
                    else:
                        keep.append([det_ind, trk_ind])
                        logger.debug(
                            "[GATE PASS ][post] trk=%s dt=%d dist=%.1f allowed=%.1f frame=%s",
                            tracker.id,
                            dt,
                            disp,
                            allowed if allowed is not None else float("inf"),
                            self.n_frames,
                        )
                matches = (
                    np.asarray(keep, dtype=int)
                    if len(keep)
                    else np.empty((0, 2), dtype=int)
                )

            unmatched_trackers = np.unique(unmatched_trackers)
            unmatched_detection_idx = np.unique(unmatched_detection_idx)
            unmatched_detections.extend(det_indices[j] for j in unmatched_detection_idx)
            unmatched_detections = np.unique(unmatched_detections)

        if not len(ellipses):
            for trk in self.trackers:
                event = {"frame": self.n_frames, "reason": "all_nan", "assembly": -1}
                self.break_log[trk.id].append(event)
                frame_breaks.append((trk.id, event))
        else:
            for idx in unmatched_trackers:
                trk = self.trackers[idx]
                if trk.id in gated_trackers:
                    event = {"frame": self.n_frames, "reason": "gated", "assembly": -1}
                    self.break_log[trk.id].append(event)
                    frame_breaks.append((trk.id, event))
                else:
                    event = {
                        "frame": self.n_frames,
                        "reason": "iou_fail",
                        "assembly": -1,
                    }
                    self.break_log[trk.id].append(event)
                    frame_breaks.append((trk.id, event))

        if self.verbose:
            logger.info(
                "[frame %s] um_det=%s  um_trk=%s",
                self.n_frames,
                unmatched_detections.tolist()
                if len(np.atleast_1d(unmatched_detections))
                else [],
                unmatched_trackers.tolist()
                if len(np.atleast_1d(unmatched_trackers))
                else [],
            )

        # 6) 更新已匹配 tracker
        animalindex = []
        for t, tracker in enumerate(self.trackers):
            if t not in unmatched_trackers:
                # 找到匹配到该全局 tracker 的检测下标
                if len(matches) == 0:
                    animalindex.append(-1)
                    continue
                idx = np.where(matches[:, 1] == t)[0]
                if idx.size == 0:
                    animalindex.append(-1)
                    continue
                det_local = matches[idx[0], 0]
                animalindex.append(det_indices[det_local])
                tracker.update(ellipses[det_local].parameters, centroids[det_local])
            else:
                animalindex.append(-1)

        # 7) 为未匹配检测创建新 tracker
        for i in (
            unmatched_detection_idx if "unmatched_detection_idx" in locals() else []
        ):
            trk = EllipseTracker(ellipses[i].parameters, centroids[i])
            if identities is not None and det_indices[i] < len(identities):
                try:
                    trk.id_ = mode(identities[det_indices[i]])[0][0]
                except Exception:
                    pass
            self.trackers.append(trk)
            animalindex.append(det_indices[i])

        # 8) 输出
        i = len(self.trackers)
        ret = []
        for trk in reversed(self.trackers):
            d = trk.state
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.n_frames <= self.min_hits
            ):
                ret.append(
                    np.concatenate((d, [trk.id, int(animalindex[i - 1])])).reshape(
                        1, -1
                    )
                )
            i -= 1
            if trk.time_since_update > self.max_age:
                det_idx = int(animalindex[i]) if i < len(animalindex) else -1
                event = {
                    "frame": self.n_frames,
                    "reason": "max_age",
                    "assembly": det_idx,
                }
                self.break_log[trk.id].append(event)
                frame_breaks.append((trk.id, event))
                self.trackers.pop(i)
        if len(ret) > 0:
            out = np.concatenate(ret)
        else:
            out = np.empty((0, 7))
        return out, frame_breaks


class SORTSkeleton(SORTBase):
    def __init__(self, n_bodyparts, max_age=20, min_hits=3, oks_threshold=0.5):
        self.n_bodyparts = n_bodyparts
        self.max_age = max_age
        self.min_hits = min_hits
        self.oks_threshold = oks_threshold
        SkeletonTracker.n_trackers = 0
        super().__init__()

    @staticmethod
    def weighted_hausdorff(x, y):
        cmax = 0
        for i in range(x.shape[0]):
            no_break_occurred = True
            cmin = np.inf
            for j in range(y.shape[0]):
                d = (x[i, 0] - y[j, 0]) ** 2 + (x[i, 1] - y[j, 1]) ** 2
                if d < cmax:
                    no_break_occurred = False
                    break
                if d < cmin:
                    cmin = d
            if cmin != np.inf and cmin > cmax and no_break_occurred:
                cmax = cmin
        return np.sqrt(cmax)

    @staticmethod
    def object_keypoint_similarity(x, y):
        mask = ~np.isnan(x * y).all(axis=1)
        xx = x[mask]
        yy = y[mask]
        dist = np.linalg.norm(xx - yy, axis=1)
        scale = np.sqrt(np.product(np.ptp(yy, axis=0)))
        oks = np.exp(-0.5 * (dist / (0.05 * scale)) ** 2)
        return np.mean(oks)

    def calc_pairwise_hausdorff_dist(self, poses, poses_ref):
        mat = np.zeros((len(poses), len(poses_ref)))
        for i, pose in enumerate(poses):
            for j, pose_ref in enumerate(poses_ref):
                mat[i, j] = self.weighted_hausdorff(pose, pose_ref)
        return mat

    def calc_pairwise_oks(self, poses, poses_ref):
        mat = np.zeros((len(poses), len(poses_ref)))
        for i, pose in enumerate(poses):
            for j, pose_ref in enumerate(poses_ref):
                mat[i, j] = self.object_keypoint_similarity(pose, pose_ref)
        return mat

    def track(self, poses):
        self.n_frames += 1

        if not len(self.trackers):
            for pose in poses:
                tracker = SkeletonTracker(self.n_bodyparts)
                tracker.state = pose
                self.trackers.append(tracker)

        poses_ref = []
        for i, tracker in enumerate(self.trackers):
            pose_ref = tracker.predict()
            poses_ref.append(pose_ref.reshape((-1, 2)))

        mat = self.calc_pairwise_hausdorff_dist(poses, poses_ref)
        row_indices, col_indices = linear_sum_assignment(mat, maximize=False)

        unmatched_poses = [p for p, _ in enumerate(poses) if p not in row_indices]
        unmatched_trackers = [
            t for t, _ in enumerate(poses_ref) if t not in col_indices
        ]
        matches = np.c_[row_indices, col_indices]

        animalindex = []
        for t, tracker in enumerate(self.trackers):
            if t not in unmatched_trackers:
                ind = matches[matches[:, 1] == t, 0][0]
                animalindex.append(ind)
                tracker.update(poses[ind])
            else:
                animalindex.append(-1)

        for i in unmatched_poses:
            tracker = SkeletonTracker(self.n_bodyparts)
            tracker.state = poses[i]
            self.trackers.append(tracker)
            animalindex.append(i)

        states = []
        i = len(self.trackers)
        for tracker in reversed(self.trackers):
            i -= 1
            if tracker.time_since_update > self.max_age:
                self.trackers.pop()
                continue
            state = tracker.predict()
            states.append(np.r_[state, [tracker.id, int(animalindex[i])]])
        if len(states) > 0:
            return np.stack(states)
        return np.empty((0, self.n_bodyparts * 2 + 2))


def fill_tracklets(tracklets, trackers, animals, imname, time_since_updates=None):
    """Populate the ``tracklets`` structure with tracker outputs."""
    for content in trackers:
        tracklet_id, pred_id = content[-2:].astype(int)
        if tracklet_id not in tracklets:
            tracklets[tracklet_id] = {}
        if time_since_updates is not None:
            tsu = tracklets.setdefault("time_since_update", {})
            tsu.setdefault(tracklet_id, {})[imname] = time_since_updates.get(
                tracklet_id, np.nan
            )
        if pred_id != -1:
            tracklets[tracklet_id][imname] = np.asarray(animals[pred_id])
        else:  # use tracker prediction
            xy = np.asarray(content[:-2])
            pred = np.insert(xy, range(2, len(xy) + 1, 2), 1)
            tracklets[tracklet_id][imname] = np.asarray(pred)


def calc_bboxes_from_keypoints(data, slack=0, offset=0):
    data = np.asarray(data)
    if data.shape[-1] < 3:
        raise ValueError("Data should be of shape (n_animals, n_bodyparts, 3)")

    if data.ndim != 3:
        data = np.expand_dims(data, axis=0)
    bboxes = np.full((data.shape[0], 5), np.nan)
    bboxes[:, :2] = np.nanmin(data[..., :2], axis=1) - slack  # X1, Y1
    bboxes[:, 2:4] = np.nanmax(data[..., :2], axis=1) + slack  # X2, Y2
    bboxes[:, -1] = np.nanmean(data[..., 2], axis=1)  # mean confidence
    bboxes[:, [0, 2]] += offset
    return bboxes


def reconstruct_all_ellipses(data, sd):
    xy = data.droplevel("scorer", axis=1).drop("likelihood", axis=1, level=-1)
    if "single" in xy:
        xy.drop("single", axis=1, level="individuals", inplace=True)
    animals = xy.columns.get_level_values("individuals").unique()
    nrows = xy.shape[0]
    ellipses = np.full((len(animals), nrows, 5), np.nan)
    fitter = EllipseFitter(sd)
    for n, animal in enumerate(animals):
        _data = xy.xs(animal, axis=1, level="individuals").values.reshape(
            (nrows, -1, 2)
        )
        for i, coords in enumerate(tqdm(_data)):
            el = fitter.fit(coords.astype(np.float64))
            if el is not None:
                ellipses[n, i] = el.parameters
    return ellipses


def compute_v_gate_pxpf(v_gate_cms=None, px_per_cm=None, fps=None):
    """Return velocity gate in pixels per frame (px/frame)."""
    try:
        if v_gate_cms is not None and px_per_cm is not None and fps is not None:
            if v_gate_cms > 0 and px_per_cm > 0 and fps > 0:
                return float(v_gate_cms * (px_per_cm / fps))
    except Exception:
        pass
    return None
