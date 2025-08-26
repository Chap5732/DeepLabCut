#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


# ================== Basic I/O ==================
# ================== 基础 I/O ==================
def load_tracklets_pickle(pkl_path: str) -> dict:
    """Load tracklets pickle file
    加载tracklets pickle文件
    """
    with open(pkl_path, "rb") as f:
        dd = pickle.load(f)
    if not isinstance(dd, dict) or "header" not in dd:
        raise ValueError("pickle结构异常：未找到 'header'")
    return dd


def save_pickle_safely(obj: Any, target: Path):
    """Safely save pickle file (temporary file + backup)
    安全保存pickle文件（使用临时文件+备份）
    """
    target = Path(target)
    tmp = target.with_suffix(target.suffix + ".tmp")
    bak = target.with_suffix(target.suffix + ".bak")

    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        if target.exists():
            if bak.exists():
                bak.unlink()
            target.replace(bak)
    except Exception:
        pass
    tmp.replace(target)


# ================== DLC utilities ==================
# ================== DLC工具 ==================
def frame_idx_from_key(k) -> int:
    """Extract frame index from key
    从帧键提取帧索引
    """
    if isinstance(k, (int, np.integer)):
        return int(k)
    s = str(k)
    m = re.findall(r"\d+", s)
    if not m:
        raise ValueError(f"无法从帧键 '{k}' 提取数字")
    return int(m[0])


def find_mouse_center_index(header) -> Optional[int]:
    """Find index of ``mouse_center`` among bodyparts
    查找mouse_center在bodyparts中的索引
    """
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


def body_center_from_arr(
    arr: np.ndarray, mc_idx: Optional[int], pcutoff: float
) -> Optional[Tuple[float, float]]:
    """Extract body center from a DLC array
    从DLC数组提取身体中心点
    """
    if (
        arr is None
        or not isinstance(arr, np.ndarray)
        or arr.ndim != 2
        or arr.shape[1] < 3
    ):
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


def color_for_id(any_id) -> Tuple[int, int, int]:
    """Generate a BGR color for an ID
    为ID生成BGR颜色
    """
    h = abs(hash(str(any_id))) % 360
    s, v = 0.8, 1.0
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    R, G, B = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
    return (B, G, R)  # BGR for OpenCV


# ================== Reader visualization ==================
# ================== 读卡器可视化 ==================
def parse_centers(txt_path: str):
    """Parse reader center position file
    解析读卡器中心位置文件
    """
    df = pd.read_csv(txt_path, header=None, comment="#", sep=None, engine="python")
    vals = df.values
    centers = {}

    if vals.shape[1] >= 4:
        # Format: row, col, x, y
        # 格式: row, col, x, y
        rows, cols = [], []
        for r, c, x, y, *_ in vals:
            r, c = int(r), int(c)
            centers[(r, c)] = (float(x), float(y))
            rows.append(r)
            cols.append(c)

        row_min, row_max = min(rows), max(rows)
        col_min, col_max = min(cols), max(cols)
        nrows = row_max - row_min + 1
        ncols = col_max - col_min + 1

        meta = dict(
            row_min=row_min,
            row_max=row_max,
            col_min=col_min,
            col_max=col_max,
            nrows=nrows,
            ncols=ncols,
            base_row=row_min,
            base_col=col_min,
        )
        return centers, meta

    # Simple format: infer grid automatically
    # 简单格式自动推导网格
    n = vals.shape[0]
    side = int(round(math.sqrt(n))) or 1
    if side * side != n:
        side = int(math.ceil(math.sqrt(n)))

    for idx in range(n):
        r = idx // side
        c = idx % side
        if vals.shape[1] == 3:
            x, y = vals[idx, 1], vals[idx, 2]
        else:
            x, y = vals[idx, 0], vals[idx, 1]
        centers[(r, c)] = (float(x), float(y))

    meta = dict(
        row_min=0,
        row_max=side - 1,
        col_min=0,
        col_max=side - 1,
        nrows=side,
        ncols=side,
        base_row=0,
        base_col=0,
    )
    return centers, meta


def centers_to_reader_positions_column_major(
    centers: Dict[tuple, int], meta: Dict[str, int]
):
    """Convert center positions to column-major reader IDs
    将中心位置转换为列优先编号的读卡器位置
    """
    reader_positions = {}
    nrows = meta["nrows"]
    br, bc = meta["base_row"], meta["base_col"]

    for (row, col), (x, y) in centers.items():
        r0 = row - br
        c0 = col - bc
        reader_id = c0 * nrows + r0 + 1
        reader_positions[reader_id] = (x, y)

    return reader_positions


def draw_readers_on_frame(
    frame,
    reader_positions,
    circle_radius=50,
    circle_color=(0, 255, 0),
    circle_thickness=2,
    center_color=(0, 0, 255),
    center_radius=3,
    text_color=(255, 255, 255),
    text_scale=0.6,
    text_thickness=1,
):
    """Draw reader positions on a frame
    在帧上绘制读卡器位置
    """
    canvas = frame.copy()

    for rid, (x, y) in reader_positions.items():
        x, y = int(round(x)), int(round(y))

        # Draw reader circle
        # 绘制读卡器圆圈
        cv2.circle(canvas, (x, y), circle_radius, circle_color, circle_thickness)
        cv2.circle(canvas, (x, y), center_radius, center_color, -1)

        # Add ID label
        # 添加ID标签
        text = str(rid)
        (tw, th), base = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
        )
        tx = x - tw // 2
        ty = y + th // 2

        # Semi-transparent background
        # 半透明背景
        ov = canvas.copy()
        cv2.rectangle(
            ov, (tx - 2, ty - th - 2), (tx + tw + 2, ty + base + 2), (0, 0, 0), -1
        )
        canvas = cv2.addWeighted(canvas, 0.7, ov, 0.3, 0)

        cv2.putText(
            canvas,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )

    return canvas


# ================== ROI handling ==================
# ================== ROI处理 ==================
def load_rois(roi_path: str):
    """Load ROI definition file (supports JSON)
    加载ROI定义文件（支持JSON格式）
    """
    if not roi_path or not Path(roi_path).exists():
        return []

    p = Path(roi_path)
    rois = []
    ROI_COLOR_DEFAULT = (80, 160, 240)  # BGR

    def _parse_color(colval):
        if colval is None:
            return ROI_COLOR_DEFAULT
        if isinstance(colval, (list, tuple)) and len(colval) == 3:
            return tuple(int(v) for v in colval)
        return ROI_COLOR_DEFAULT

    if p.suffix.lower() == ".json":
        text = p.read_text(encoding="utf-8").strip()
        try:
            parsed = json.loads(text)

            # Check if format is label->polygon
            # 检查是否为 label->polygon 格式
            if isinstance(parsed, dict) and all(
                isinstance(v, list) for v in parsed.values()
            ):
                for label, pts_list in parsed.items():
                    if all(
                        isinstance(pt, (list, tuple)) and len(pt) >= 2
                        for pt in pts_list
                    ):
                        pts = [(int(pt[0]), int(pt[1])) for pt in pts_list]
                        if len(pts) >= 3:
                            rois.append(
                                {
                                    "type": "poly",
                                    "label": str(label),
                                    "pts": pts,
                                    "color": ROI_COLOR_DEFAULT,
                                }
                            )
        except Exception:
            pass

    return rois


def draw_rois(frame, rois, line_thick: int = 2, alpha_fill: float = 0.12):
    """Draw ROI areas on a frame
    在帧上绘制ROI区域
    """
    if not rois:
        return frame

    canvas = frame.copy()
    overlay = canvas.copy()

    for roi in rois:
        color = tuple(int(c) for c in roi.get("color", (80, 160, 240)))

        if roi["type"] == "rect":
            x1, y1, x2, y2 = (
                int(roi["x1"]),
                int(roi["y1"]),
                int(roi["x2"]),
                int(roi["y2"]),
            )
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_thick, cv2.LINE_AA)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cx = (x1 + x2) // 2
            cy = min(y1, y2) - 6
        elif roi["type"] == "circle":
            x, y, r = int(roi["x"]), int(roi["y"]), int(roi["r"])
            cv2.circle(canvas, (x, y), r, color, line_thick, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), r, color, -1)
            cx, cy = x, y - r - 8
        else:  # poly
            pts = np.array(roi["pts"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                canvas,
                [pts],
                isClosed=True,
                color=color,
                thickness=line_thick,
                lineType=cv2.LINE_AA,
            )
            cv2.fillPoly(overlay, [pts], color)
            arr = np.array(roi["pts"], dtype=np.float32)
            cx = int(arr[:, 0].mean())
            cy = int(arr[:, 1].mean()) - 10

        # Draw label
        # 绘制标签
        label = roi.get("label", "")
        if label:
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            bx1 = max(cx - tw // 2 - 4, 0)
            by1 = max(cy - th - 8, 0)
            bx2 = min(bx1 + tw + 8, canvas.shape[1] - 1)
            by2 = min(by1 + th + base + 6, canvas.shape[0] - 1)

            lab_overlay = canvas.copy()
            cv2.rectangle(lab_overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
            canvas = cv2.addWeighted(canvas, 0.75, lab_overlay, 0.25, 0)
            cv2.putText(
                canvas,
                label,
                (bx1 + 4, by2 - base - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # Apply semi-transparent fill
    canvas = cv2.addWeighted(canvas, 1.0, overlay, alpha_fill, 0)
    return canvas


# ================== ROI hit testing ==================
# ================== ROI命中测试 ==================
def point_in_rect(x: int, y: int, x1: int, y1: int, x2: int, y2: int) -> bool:
    """Test whether point lies inside rectangle
    测试点是否在矩形内
    """
    xa, xb = sorted([x1, x2])
    ya, yb = sorted([y1, y2])
    return (xa <= x <= xb) and (ya <= y <= yb)


def point_in_poly(x: int, y: int, pts: List[Tuple[int, int]]) -> bool:
    """Test whether point lies inside polygon using ray casting
    使用ray casting算法测试点是否在多边形内
    """
    inside = False
    n = len(pts)
    if n < 3:
        return False

    xj, yj = pts[-1]
    for i in range(n):
        xi, yi = pts[i]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        ):
            inside = not inside
        xj, yj = xi, yi

    return inside


def point_in_any_roi(
    pt: Optional[Tuple[float, float]], rois: List[Dict[str, Any]]
) -> bool:
    """Test whether a point lies inside any ROI
    测试点是否在任何ROI内
    """
    if not rois or pt is None:
        return False

    x = int(round(pt[0]))
    y = int(round(pt[1]))

    for r in rois:
        t = r["type"]
        if t == "rect":
            if point_in_rect(
                x, y, int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
            ):
                return True
        elif t == "circle":
            dx = x - int(r["x"])
            dy = y - int(r["y"])
            rr = int(r["r"])
            if (dx * dx + dy * dy) <= rr * rr:
                return True
        else:  # poly
            if point_in_poly(x, y, [(int(px), int(py)) for px, py in r["pts"]]):
                return True

    return False
