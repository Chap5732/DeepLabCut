#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RFID（按坐标重排→行优先 id）→ tracklet 累计读数 + 保守/智能 tag 分配
- 读取 readers_centers.txt（一列 144 个坐标），按空间重排为 12×12 网格（上→下，左→右）
- 用行优先 id（第 0 行 0..11，第 1 行 12..23，…）映射到圈心坐标
- RFID 事件：时间→最近帧；逐帧在半径内竞争，支持“唯一最近邻”过滤
- 写出：counts_by_track_tag.csv（累计命中）、tag_assignments.csv（分配结果）、meta.json（参数与诊断）
- 写回 pickle：每个 tracklet 节点加入 rfid_counts / rfid_frames / （可选）tag / rfid_hint
"""

from __future__ import annotations
import os
import re
import json
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ==========================
# ====== 用户配置区 =========
# ==========================

# ---- 路径 ----
PICKLE_PATH = "/ssd01/user_acc_data/oppa/deeplabcut/projects/MiceTrackerFor20-Oppa-2024-12-08/analyze_videos/shuffle3/demo1/velocity_gating/demoDLC_HrnetW32_MiceTrackerFor20Dec8shuffle3_detector_best-250_snapshot_best-190_el.pickle"
RFID_CSV    = "/ssd01/user_acc_data/oppa/analysis/data/jc0813/rfid_data_20250813_055827.csv"
CENTERS_TXT = "/ssd01/user_acc_data/oppa/analysis/data/jc0813/readers_centers.txt"
TS_CSV      = "/ssd01/user_acc_data/oppa/analysis/data/jc0813/record_20250813_053913_timestamps.csv"
OUT_DIR     = None  # None -> 与 pickle 同目录创建 rfid_match_outputs/

# ---- 阵列/网格配置（很关键）----
N_ROWS  = 12           # 行数
N_COLS  = 12           # 列数
ID_BASE = 0            # RFID CSV 中的 id 是否从 0 开始（若从 1 开始则设为 1）
Y_TOP_TO_BOTTOM = True # True: 画面上方 y 小、下方 y 大；若相反改 False

# ---- 几何与匹配门槛 ----
PCUTOFF = 0.35            # mouse center / 回退质心 的置信度阈值
RFID_FRAME_RANGE = 10     # 每帧纳入竞争的时间窗口（±range//2）
COIL_DIAMETER_PX = 170.0
HIT_MARGIN       = 1.00
HIT_RADIUS_PX    = (COIL_DIAMETER_PX / 2.0) * HIT_MARGIN  # 命中半径（像素）

UNIQUE_NEIGHBOR_ONLY = True   # 是否要求最近邻“足够唯一”
AMBIG_MARGIN_PX      = 75.0   # 最近与次近差距 < 此值 → 视为不唯一（丢弃事件）

LOW_FREQ_TAG_MIN_COUNT = 2    # 过滤：全局出现次数 ≤ 此阈值 的 tag 忽略
MIN_VALID_FRAMES_PER_TK = 1   # tracklet 至少有这么多有效帧才参与匹配

# ---- Tag 分配阈值（保守）----
TAG_CONFIDENCE_THRESHOLD = 0.70   # 主导占比 ≥ 70%
TAG_MIN_READS            = 20     # 总读数 ≥ 20
TAG_DOMINANT_RATIO       = 3.0    # 第一名/第二名 ≥ 3
# 当总读数 < TAG_MIN_READS 且top占比≥该值时，仍然指派
LOW_READS_HIGH_PURITY_ASSIGN = True
LOW_READS_PURITY_THRESHOLD   = 0.90

# ---- 逐帧稳健性（可关）----
USE_FRAME_STABILITY_CHECK = False   # 打开逐帧稳健性判据
BURST_GAP_FRAMES          = 150    # ~5秒@30fps，命中间隔≥此值计作新波段
MIN_BURSTS_IF_LOWHITS     = 2      # 当总读数 < 200 时，要求 ≥2 个波段
LOWHITS_THRESHOLD         = 200

# ==========================
# ====== 工具函数 ===========
# ==========================

def ensure_out_dir(pickle_path: str) -> Path:
    out = Path(OUT_DIR) if OUT_DIR else (Path(pickle_path).parent / "rfid_match_outputs")
    out.mkdir(parents=True, exist_ok=True)
    return out

def load_centers_txt(txt_path: str) -> np.ndarray:
    arr = np.loadtxt(txt_path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("readers_centers.txt 必须是 N×2 的 x,y 列")
    return arr  # 原始顺序后续会重排

def build_row_major_index(centers: np.ndarray, n_rows: int, n_cols: int, y_top_to_bottom: bool=True) -> tuple[np.ndarray, np.ndarray]:
    """
    将一列 N=n_rows*n_cols 个 (x,y) 位置按空间重排为 n_rows×n_cols 网格（上→下、左→右）。
    返回：
      - idx_map: 长度 N 的一维数组，idx_map[row*n_cols + col] = 原 txt 中的行号（0 基）
      - centers_grid: 形状 (n_rows, n_cols, 2) 的 (x,y) 网格（行优先）
    """
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers 必须是 N×2 的数组")
    N = centers.shape[0]
    if N != n_rows * n_cols:
        raise ValueError(f"点的数量 {N} 与 n_rows*n_cols={n_rows*n_cols} 不一致")

    xs = centers[:, 0].copy()
    ys = centers[:, 1].copy()
    y_for_sort = ys if y_top_to_bottom else -ys

    order = np.lexsort((xs, y_for_sort))  # 先 y 升序，再 x 升序
    idx_map = np.empty(N, dtype=int)
    for r in range(n_rows):
        block = order[r*n_cols:(r+1)*n_cols]
        block_sorted = block[np.argsort(xs[block])]
        idx_map[r*n_cols:(r+1)*n_cols] = block_sorted

    centers_grid = centers[idx_map].reshape(n_rows, n_cols, 2)

    # 轻量自检
    row_means_y = centers_grid[:, :, 1].mean(axis=1)
    if not np.all(np.diff(row_means_y) > 0):
        print("[WARN] 行均值 y 未严格递增，可能存在轻微噪声/方向差异，仍按行优先继续。")

    return idx_map, centers_grid

def parse_rfid_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    # 时间列（自动识别）
    time_keys = ["time","timestamp","datetime","ts","client_receive_time","client_recieve_time"]
    time_k = next((cols.get(k) for k in time_keys if k in cols), None)
    if time_k is None:
        raise ValueError(f"RFID CSV 未找到时间列；候选：{time_keys}")
    t_series = pd.to_datetime(df[time_k], errors="coerce", utc=False)
    if t_series.isna().all():
        t_num = pd.to_numeric(df[time_k], errors="coerce")
        if t_num.isna().all():
            raise ValueError("RFID 时间列既不是可解析时间，也不是数值秒。")
        time_sec = t_num.astype(float)
    else:
        time_sec = (t_series.astype("int64") / 1e9).astype(float)

    # 标签列
    tag_k = next((cols.get(k) for k in ["tag","epc","tid","card_id","data"] if k in cols), None)
    if tag_k is None:
        raise ValueError("RFID CSV 未找到标签列（tag/epc/tid/card_id/data）")
    tag_series = df[tag_k].astype(str).str.strip()

    # 读取器 id（保留原始基准：0基或1基）
    id_list = []
    for _, row in df.iterrows():
        id_val = None
        for k in ("id","reader_id"):
            if k in row and pd.notna(row[k]):
                try:
                    id_val = int(row[k]); break
                except Exception:
                    pass
        if id_val is None:
            for k in df.columns:
                v = row[k]
                if isinstance(v, str):
                    m = re.search(r"/id/(\d+)", v) or re.search(r"id[=:](\d+)", v)
                    if m:
                        try:
                            id_val = int(m.group(1)); break
                        except Exception:
                            pass
        id_list.append(id_val)

    out = pd.DataFrame({
        "time": time_sec,
        "tag": tag_series,
        "id_raw": id_list,
    })
    out = out.dropna(subset=["time","tag"])
    return out

def parse_timestamps_csv(ts_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(ts_path)
    cols = {c.lower(): c for c in df.columns}

    frame_k = next((cols.get(k) for k in ["frame","frame_id","fid","index"] if k in cols), None)
    if frame_k is None:
        raise ValueError("timestamps.csv 未找到帧列（frame/frame_id/fid/index）")

    time_keys = ["time","timestamp","datetime","ts","client_receive_time","client_recieve_time"]
    time_k = next((cols.get(k) for k in time_keys if k in cols), None)
    if time_k is None:
        raise ValueError(f"timestamps.csv 未找到时间列；候选：{time_keys}")

    t_series = pd.to_datetime(df[time_k], errors="coerce", utc=False)
    if t_series.isna().all():
        t_num = pd.to_numeric(df[time_k], errors="coerce")
        if t_num.isna().all():
            raise ValueError("timestamps.csv 时间列解析失败。")
        times = t_num.to_numpy(dtype="float64")
    else:
        times = (t_series.astype("int64") / 1e9).to_numpy(dtype="float64")

    frames = pd.to_numeric(df[frame_k], errors="coerce").astype("int64").to_numpy()
    ok = np.isfinite(times) & np.isfinite(frames)
    frames = frames[ok]; times = times[ok]
    order = np.argsort(times, kind="mergesort")
    frames = frames[order]; times = times[order]
    uniq_t, idx = np.unique(times, return_index=True)
    frames = frames[idx]; times = uniq_t
    return frames, times

def time_to_nearest_frame(t: float, frames: np.ndarray, times: np.ndarray) -> int:
    if t <= times[0]:  return int(frames[0])
    if t >= times[-1]: return int(frames[-1])
    k = int(np.searchsorted(times, t, side="left"))
    return int(frames[k-1] if abs(t - times[k-1]) <= abs(t - times[k]) else frames[k])

def load_tracklets_pickle(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        dd = pickle.load(f)
    if not isinstance(dd, dict) or "header" not in dd:
        raise ValueError("pickle 结构异常：未找到 'header'")
    return dd

def frame_idx_from_key(k) -> int:
    if isinstance(k, (int, np.integer)):
        return int(k)
    s = str(k)
    m = re.findall(r"\d+", s)
    if not m:
        raise ValueError(f"无法从帧键 '{k}' 提取数字")
    return int(m[0])

def find_mouse_center_index(header) -> int | None:
    try:
        bps = header.get_level_values("bodyparts").unique().to_list()
    except Exception:
        return None
    norm = lambda s: re.sub(r"[\s_]+", "", str(s).lower())
    targets = {"mouse_center", "center", "bodycenter", "mousecentre"}
    for i, bp in enumerate(bps):
        if norm(bp) in targets:
            return i
    return None

def mean_confidence(arr: np.ndarray) -> float:
    try:
        p = arr[:, 2]
        m = float(np.nanmean(p))
        return m if np.isfinite(m) else 0.0
    except Exception:
        return 0.0

def body_center_from_arr(arr: np.ndarray, mc_idx: int | None, pcutoff: float) -> tuple[float,float] | None:
    if arr is None or not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[1] < 3:
        return None
    if mc_idx is not None and mc_idx < arr.shape[0]:
        x, y, lk = arr[mc_idx, 0], arr[mc_idx, 1], arr[mc_idx, 2]
        if np.isfinite(x) and np.isfinite(y) and float(lk) >= pcutoff:
            return float(x), float(y)
    xy = arr[:, :2]
    lk = arr[:, 2]
    mask = np.isfinite(xy).all(axis=1) & np.isfinite(lk) & (lk >= pcutoff)
    if not np.any(mask):
        return None
    w = np.clip(lk[mask], 0.0, None) + 1e-12
    cx = float(np.nansum(xy[mask,0] * w) / np.nansum(w))
    cy = float(np.nansum(xy[mask,1] * w) / np.nansum(w))
    if np.isfinite(cx) and np.isfinite(cy):
        return cx, cy
    return None

# ---- 逐帧稳健性 & 分配 ----

def _count_bursts_from_frames(frames_list, gap_frames=150) -> int:
    """[(frame, conf), ...] -> 估算独立命中波段数"""
    if not frames_list:
        return 0
    fs = sorted(int(f) for f, _ in frames_list)
    bursts = 1
    for i in range(1, len(fs)):
        if fs[i] - fs[i-1] >= gap_frames:
            bursts += 1
    return bursts

def assign_tag_for_one_tracklet(
    counts_dict: dict[str, int],
    frames_dict: dict[str, list[tuple[int, float]]] | None,
    confidence_threshold: float = TAG_CONFIDENCE_THRESHOLD,
    min_reads_threshold: int = TAG_MIN_READS,
    dominant_ratio_threshold: float = TAG_DOMINANT_RATIO,
    use_stability: bool = USE_FRAME_STABILITY_CHECK,
    lowhits_threshold: int = LOWHITS_THRESHOLD,
    burst_gap_frames: int = BURST_GAP_FRAMES,
    min_bursts_if_lowhits: int = MIN_BURSTS_IF_LOWHITS,
) -> tuple[str | None, dict]:
    """
    保守分配 + 可选逐帧稳健性（波段）判据。
    新增：当 total_reads < TAG_MIN_READS 且 top_ratio ≥ LOW_READS_PURITY_THRESHOLD 时，直接指派。
    """
    diag = {}
    if not counts_dict:
        diag["reason"] = "no_counts"
        return None, diag

    total_reads = int(sum(counts_dict.values()))

    # 排序：按计数降序；计数相同用平均置信度打破
    def avg_conf(tg: str) -> float:
        lst = frames_dict.get(tg, []) if frames_dict else []
        return float(np.mean([c for _, c in lst])) if lst else 0.0

    items = sorted(counts_dict.items(),
                   key=lambda kv: (kv[1], avg_conf(kv[0])),
                   reverse=True)
    top_tag, top_count = items[0][0], int(items[0][1])
    top_ratio = top_count / max(total_reads, 1)

    # === 新增“低读数但高纯度直通” ===
    if total_reads < min_reads_threshold:
        if LOW_READS_HIGH_PURITY_ASSIGN and top_ratio >= LOW_READS_PURITY_THRESHOLD:
            diag.update({
                "reason": "assigned_low_reads_high_purity",
                "total_reads": total_reads,
                "top": [top_tag, top_count],
                "top_ratio": round(top_ratio, 3)
            })
            return str(top_tag), diag
        else:
            diag["reason"] = "low_reads"
            diag["total_reads"] = total_reads
            return None, diag
    # === 直通逻辑结束 ===

    # 规则 1：绝对主导（占比）
    if top_ratio >= confidence_threshold:
        pass
    else:
        # 规则 2：相对主导（倍差）
        second = int(items[1][1]) if len(items) > 1 else 0
        if second == 0 or (top_count / max(second, 1)) >= dominant_ratio_threshold:
            pass
        else:
            diag.update({
                "reason": "ambiguous",
                "total_reads": total_reads,
                "top": [top_tag, top_count],
                "second": [items[1][0], second],
                "top_ratio": round(top_ratio, 3)
            })
            return None, diag

    # 规则 3（可选）：逐帧稳健性（仅对低命中段 < lowhits_threshold 更严格）
    if use_stability and total_reads < lowhits_threshold:
        bursts = _count_bursts_from_frames(frames_dict.get(top_tag, []), gap_frames=burst_gap_frames) \
                 if frames_dict else 0
        if bursts < min_bursts_if_lowhits:
            diag.update({
                "reason": "unstable_single_burst",
                "bursts": bursts,
                "total_reads": total_reads
            })
            return None, diag
        diag["bursts"] = bursts

    diag.update({
        "reason": "assigned",
        "total_reads": total_reads,
        "top": [top_tag, top_count],
        "top_ratio": round(top_ratio, 3)
    })
    return str(top_tag), diag

# ==========================
# ====== 主流程 =============
# ==========================

def main():
    out_dir = ensure_out_dir(PICKLE_PATH)

    # 1) 加载并按坐标重排为行优先网格
    centers_raw = load_centers_txt(CENTERS_TXT)
    idx_map, centers_grid = build_row_major_index(centers_raw, N_ROWS, N_COLS, Y_TOP_TO_BOTTOM)
    centers = centers_raw  # 通过 idx_map 访问

    # 自检：打印四角
    print("[grid] top-left     id 0           ->", centers_grid[0, 0])
    print("[grid] top-right    id 11          ->", centers_grid[0, N_COLS-1])
    print("[grid] bottom-left  id", (N_ROWS-1)*N_COLS, "->", centers_grid[N_ROWS-1, 0])
    print("[grid] bottom-right id", N_ROWS*N_COLS-1,  "->", centers_grid[N_ROWS-1, N_COLS-1])

    # 2) 读 timestamps / RFID / tracklets
    frames_arr, times_arr = parse_timestamps_csv(TS_CSV)
    df_rfid = parse_rfid_csv(RFID_CSV)
    dd = load_tracklets_pickle(PICKLE_PATH)

    # 低频 tag 过滤（≤ 阈值）
    tag_counts_all = df_rfid["tag"].value_counts()
    low_tags = set(tag_counts_all[tag_counts_all <= LOW_FREQ_TAG_MIN_COUNT].index.tolist())
    if low_tags:
        df_rfid = df_rfid[~df_rfid["tag"].isin(low_tags)].copy()

    # 解析 header 找 mouse center 索引
    header = dd.get("header", None)
    mc_idx = find_mouse_center_index(header)

    # 3) 构建: frame -> [(tk, (cx,cy), mean_conf)]
    frame2points: dict[int, list[tuple[object, tuple[float,float], float]]] = defaultdict(list)
    valid_frame_count: dict[object, int] = defaultdict(int)
    for tk, node in dd.items():
        if tk in ("header", "single"):
            continue
        for fkey, arr in node.items():
            try:
                fi = frame_idx_from_key(fkey)
            except Exception:
                continue
            if not isinstance(arr, np.ndarray):
                continue
            c = body_center_from_arr(arr, mc_idx, PCUTOFF)
            if c is None:
                continue
            conf = mean_confidence(arr)
            frame2points[fi].append((tk, c, conf))
            valid_frame_count[tk] += 1

    # 剔除有效帧过少的 tracklet
    bad_tks = {tk for tk, cnt in valid_frame_count.items() if cnt < MIN_VALID_FRAMES_PER_TK}
    if bad_tks:
        for f in list(frame2points.keys()):
            frame2points[f] = [t for t in frame2points[f] if t[0] not in bad_tks]

    # 4) RFID 事件预处理：映射到最近帧，绑定圈心
    N_centers = centers.shape[0]
    if N_centers != N_ROWS * N_COLS:
        raise ValueError(f"CENTERS 总点数 {N_centers} != N_ROWS*N_COLS={N_ROWS*N_COLS}")

    events = []  # list of (f_event, tag, (rx,ry))
    events_stats = {"events_out_of_ts_range": 0, "events_no_reader_xy": 0}

    for _, r in df_rfid.iterrows():
        t = float(r["time"])
        tag = str(r["tag"])
        id_raw = r.get("id_raw", None)
        if id_raw is None:
            events_stats["events_no_reader_xy"] += 1
            continue

        try:
            std_id = int(id_raw) - int(ID_BASE)  # 行优先 id（0 基）
        except Exception:
            events_stats["events_no_reader_xy"] += 1
            continue

        if not (0 <= std_id < N_ROWS * N_COLS):
            events_stats["events_no_reader_xy"] += 1
            continue

        file_idx = int(idx_map[std_id])
        if not (0 <= file_idx < N_centers):
            events_stats["events_no_reader_xy"] += 1
            continue

        if t < times_arr[0] or t > times_arr[-1]:
            events_stats["events_out_of_ts_range"] += 1
            continue

        f_event = time_to_nearest_frame(t, frames_arr, times_arr)
        rx, ry = float(centers[file_idx, 0]), float(centers[file_idx, 1])
        events.append((int(f_event), tag, (rx, ry)))

    # 5) 建索引: frame -> [event_idx,...]
    frame_to_events: dict[int, list[int]] = defaultdict(list)
    for ei, (fe, _, _) in enumerate(events):
        frame_to_events[fe].append(ei)

    # 6) 逐帧扫描并竞争分配（含诊断）
    half_win = int(RFID_FRAME_RANGE // 2)
    tk_tag_counts: dict[object, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tk_tag_frames: dict[object, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    stats = defaultdict(int)

    all_frames_sorted = sorted(frame2points.keys())
    for f in all_frames_sorted:
        cand_event_indices = []
        for ff in range(f - half_win, f + half_win + 1):
            if ff in frame_to_events:
                cand_event_indices.extend(frame_to_events[ff])
        if not cand_event_indices:
            stats["no_events_in_window"] += 1
            continue
        cand_event_indices.sort(key=lambda ei: abs(events[ei][0] - f))

        pts = frame2points.get(f, [])
        if not pts:
            stats["no_track_points_this_frame"] += 1
            continue

        assigned_this_frame: set[object] = set()

        for ei in cand_event_indices:
            _, tag, (rx, ry) = events[ei]
            dlist: list[tuple[float, object, float]] = []  # (dist, tk, conf)
            for tk, (cx, cy), conf in pts:
                if tk in assigned_this_frame:
                    continue
                d = float(((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5)
                if d <= HIT_RADIUS_PX:
                    dlist.append((d, tk, conf))

            if not dlist:
                stats["no_candidate_in_radius"] += 1
                continue

            dlist.sort(key=lambda x: x[0])
            if UNIQUE_NEIGHBOR_ONLY and len(dlist) >= 2:
                if (dlist[1][0] - dlist[0][0]) < AMBIG_MARGIN_PX:
                    stats["ambiguous_drop"] += 1
                    continue

            best_d, best_tk, best_conf = dlist[0]
            assigned_this_frame.add(best_tk)
            tk_tag_counts[best_tk][tag] += 1
            tk_tag_frames[best_tk][tag].append((f, float(best_conf)))
            stats["assigned"] += 1

    # 7) 输出累计读数 CSV
    rows_counts = []
    tag_names_all = set()
    for tk, per_tag in tk_tag_counts.items():
        for tg, cnt in per_tag.items():
            rows_counts.append({"tracklet": tk, "tag": tg, "total_hits": int(cnt)})
            tag_names_all.add(tg)
    df_counts_long = pd.DataFrame(rows_counts)
    if df_counts_long.empty:
        df_counts_long = pd.DataFrame(columns=["tracklet","tag","total_hits"])

    if not df_counts_long.empty:
        df_counts_wide = df_counts_long.pivot_table(
            index="tracklet", columns="tag", values="total_hits",
            fill_value=0, aggfunc="sum"
        ).sort_index(axis=0)
    else:
        df_counts_wide = pd.DataFrame()

    df_counts_wide.to_csv(out_dir / "counts_by_track_tag.csv", encoding="utf-8")

    # 8) 组装 meta.json（先写基础参数与诊断）
    meta = {
        "params": {
            "pcutoff": PCUTOFF,
            "rfid_frame_range": RFID_FRAME_RANGE,
            "hit_radius_px": HIT_RADIUS_PX,
            "unique_neighbor_only": UNIQUE_NEIGHBOR_ONLY,
            "ambig_margin_px": AMBIG_MARGIN_PX,
            "low_freq_tag_min_count": LOW_FREQ_TAG_MIN_COUNT,
            "min_valid_frames_per_tk": MIN_VALID_FRAMES_PER_TK,
            "grid": {
                "n_rows": N_ROWS,
                "n_cols": N_COLS,
                "id_base": ID_BASE,
                "y_top_to_bottom": bool(Y_TOP_TO_BOTTOM),
            },
        },
        "events_total": int(len(events)),
        **{"events_out_of_ts_range": int(events_stats.get("events_out_of_ts_range", 0)),
           "events_no_reader_xy": int(events_stats.get("events_no_reader_xy", 0))},
        "assign_stats_stage_match": {k:int(v) for k,v in stats.items()},
    }

    # 9) 写回 pickle + Tag 分配（一次完成）
    orig_path = Path(PICKLE_PATH)
    backup_path = orig_path.with_name(orig_path.name + ".bak")
    tmp_path = orig_path.with_name(orig_path.name + ".tmp")

    dd_out = dict(dd)
    assign_stats = {"high_confidence": 0, "medium_confidence": 0, "low_reads": 0, "ambiguous": 0, "unstable": 0}
    assignments = {}

    for tk, node in list(dd_out.items()):
        if tk in ("header", "single"):
            continue

        # 写回计数与逐帧
        counts_dict = {tg: int(cnt) for tg, cnt in tk_tag_counts.get(tk, {}).items()}
        frames_dict = {tg: [(int(f), float(c)) for (f, c) in lst] for tg, lst in tk_tag_frames.get(tk, {}).items()}
        node["rfid_counts"] = counts_dict if counts_dict else {}
        node["rfid_frames"] = frames_dict if frames_dict else {}

        # 分配 tag（保守 + 可选逐帧稳健性）
        assigned_tag, hint = assign_tag_for_one_tracklet(counts_dict, frames_dict)

        if assigned_tag is not None:
            node["tag"] = str(assigned_tag)
            top_ratio = hint.get("top_ratio", 0.0)
            if top_ratio >= TAG_CONFIDENCE_THRESHOLD:
                assign_stats["high_confidence"] += 1
            else:
                assign_stats["medium_confidence"] += 1
            assignments[tk] = assigned_tag
        else:
            reason = hint.get("reason", "")
            if reason == "low_reads":
                assign_stats["low_reads"] += 1
            elif reason == "unstable_single_burst":
                assign_stats["unstable"] += 1
            else:
                assign_stats["ambiguous"] += 1
            node.pop("tag", None)

        # 写 hint（便于溯源）
        node["rfid_hint"] = {
            "assign_mode": "conservative_counts_with_optional_bursts",
            "params": {
                "confidence_threshold": TAG_CONFIDENCE_THRESHOLD,
                "min_reads_threshold": TAG_MIN_READS,
                "dominant_ratio_threshold": TAG_DOMINANT_RATIO,
                "use_frame_stability": USE_FRAME_STABILITY_CHECK,
                "burst_gap_frames": BURST_GAP_FRAMES,
                "min_bursts_if_lowhits": MIN_BURSTS_IF_LOWHITS,
                "lowhits_threshold": LOWHITS_THRESHOLD,
            },
            **hint
        }

        dd_out[tk] = node

    with open(tmp_path, "wb") as f:
        pickle.dump(dd_out, f, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        if orig_path.exists():
            if backup_path.exists():
                backup_path.unlink()
            os.replace(orig_path, backup_path)
    except Exception as e:
        print(f"[WARN] 备份原始 pickle 失败：{e}（继续覆盖）")

    os.replace(tmp_path, orig_path)

    # 10) 输出分配结果 CSV
    assign_rows = []
    all_tks = sorted(tk_tag_counts.keys(), key=lambda x: (isinstance(x, str), x))
    for tk in all_tks:
        counts = tk_tag_counts.get(tk, {})
        total = int(sum(counts.values()))
        assigned = assignments.get(tk)
        if assigned and total > 0:
            conf = counts.get(assigned, 0) / total
            vals_sorted = sorted(counts.values(), reverse=True)
            dom_ratio = (vals_sorted[0] / max(vals_sorted[1], 1)) if len(vals_sorted) >= 2 else float("inf")
        else:
            conf, dom_ratio = 0.0, 0.0
        assign_rows.append({
            "tracklet": tk,
            "assigned_tag": assigned if assigned else "UNASSIGNED",
            "total_reads": total,
            "confidence": round(conf, 3),
            "dominance_ratio": ("INF" if dom_ratio == float("inf") else round(dom_ratio, 2)),
            "tag_distribution": "|".join([f"{tag}:{cnt}" for tag, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)])
        })
    pd.DataFrame(assign_rows).to_csv(out_dir / "tag_assignments.csv", index=False, encoding="utf-8")

    # 11) 更新 meta.json（加入分配统计）
    meta["tag_assignment"] = {
        "strategy": "conservative_counts_with_optional_bursts",
        "params": {
            "confidence_threshold": TAG_CONFIDENCE_THRESHOLD,
            "min_reads_threshold": TAG_MIN_READS,
            "dominant_ratio_threshold": TAG_DOMINANT_RATIO,
            "use_frame_stability": USE_FRAME_STABILITY_CHECK,
            "burst_gap_frames": BURST_GAP_FRAMES,
            "min_bursts_if_lowhits": MIN_BURSTS_IF_LOWHITS,
            "lowhits_threshold": LOWHITS_THRESHOLD,
        },
        "stats": {k: int(v) for k, v in assign_stats.items()},
        "assignments_count": int(len(assignments)),
        "total_tracklets": int(len(tk_tag_counts)),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 12) 打印总结
    print("\n[TAG ASSIGNMENT] 分配结果：")
    print(f"  高置信度分配: {assign_stats['high_confidence']} tracklets")
    print(f"  中等置信度分配: {assign_stats['medium_confidence']} tracklets")
    print(f"  读数不足: {assign_stats['low_reads']} tracklets")
    print(f"  单段不稳: {assign_stats['unstable']} tracklets")
    print(f"  竞争激烈: {assign_stats['ambiguous']} tracklets")
    tot = len(tk_tag_counts)
    print(f"  总分配率: {len(assignments)}/{tot} = {len(assignments)/max(tot,1)*100:.1f}%")

    print("\n[FILES] 输出文件：")
    print(f"  counts_by_track_tag: {out_dir/'counts_by_track_tag.csv'}")
    print(f"  tag_assignments:     {out_dir/'tag_assignments.csv'}")
    print(f"  meta.json:           {out_dir/'meta.json'}")
    print(f"  pickle 已更新（backup: {backup_path}）")

if __name__ == "__main__":
    main()
