#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import Optional, Tuple

import numpy as np


def frame_idx_from_key(k) -> int:
    """从帧键提取帧索引"""
    if isinstance(k, (int, np.integer)):
        return int(k)
    s = str(k)
    m = re.findall(r"\d+", s)
    if not m:
        raise ValueError(f"无法从帧键 '{k}' 提取数字")
    return int(m[0])


def find_mouse_center_index(header) -> Optional[int]:
    """查找mouse_center在bodyparts中的索引"""
    try:
        bps = header.get_level_values("bodyparts").unique().to_list()
    except Exception:
        return None

    norm = lambda s: re.sub(r"[\s_]+", "", str(s).lower())
    targets = {"mouse_center", "mousecenter", "center", "bodycenter", "mousecentre"}

    for i, bp in enumerate(bps):
        if norm(str(bp)) in targets:
            return i
    return None


def body_center_from_arr(arr: np.ndarray, mc_idx: Optional[int], pcutoff: float) -> Optional[Tuple[float, float]]:
    """从DLC数组提取身体中心点"""
    if arr is None or not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[1] < 3:
        return None

    # 优先使用mouse_center
    if mc_idx is not None and mc_idx < arr.shape[0]:
        x, y, lk = arr[mc_idx, 0], arr[mc_idx, 1], arr[mc_idx, 2]
        if np.isfinite(x) and np.isfinite(y) and float(lk) >= pcutoff:
            return float(x), float(y)

    # 回退到加权平均
    xy = arr[:, :2]
    lk = arr[:, 2]
    mask = np.isfinite(xy).all(axis=1) & np.isfinite(lk) & (lk >= pcutoff)
    if not np.any(mask):
        return None

    w = np.clip(lk[mask], 0.0, None) + 1e-12
    cx = float(np.nansum(xy[mask, 0] * w) / np.nansum(w))
    cy = float(np.nansum(xy[mask, 1] * w) / np.nansum(w))

    if np.isfinite(cx) and np.isfinite(cy):
        return cx, cy
    return None


__all__ = [
    "frame_idx_from_key",
    "find_mouse_center_index",
    "body_center_from_arr",
]
