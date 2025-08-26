#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
轨迹重建（时间–速度门控 80 cm/s + 同步推进 + 贪心让步 + 近锚点堤坝止水）
- 以 RFID 锚点为“水源”，仅向【非锚点】tracklet 扩散
- 单步“时间–速度门控”：d <= v_gate(px/frame) * gap 且 gap <= MAX_GAP_FRAMES
- 同步推进（wavefront）：一回合内所有水源各走一步
- 冲突裁决：最小代价优先 -> 歧义冻结(δ) -> 反“抢食”让步
- 同标签近邻锚点（MAX_GAP_FRAMES 内）视为“堤坝”，对应方向不扩散
- 写回 chain_tag / chain_id / chain_origin；支持重复运行并覆盖输出
"""

from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, List, Any
import time
import numpy as np
import pandas as pd

from utils import (
    load_tracklets_pickle, save_pickle_safely, frame_idx_from_key,
    find_mouse_center_index, body_center_from_arr
)

# ================== 配置参数 ==================
# 输入输出路径
PICKLE_IN  = "/ssd01/user_acc_data/oppa/deeplabcut/projects/MiceTrackerFor20-Oppa-2024-12-08/analyze_videos/shuffle3/demo1/velocity_gating/demoDLC_HrnetW32_MiceTrackerFor20Dec8shuffle3_detector_best-250_snapshot_best-190_el.pickle"
PICKLE_OUT = None                # None=覆盖输入；或给出新路径
OUT_SUBDIR = "CAP15"             # 输出到输入pickle同目录下的子目录；设 None 则不用子目录

# 相机与门控
FPS        = 30.0                # 帧/秒
PX_PER_CM  = 14.0                # 像素/厘米（相机标定）
V_GATE_CMS = 80.0                # cm/s

# 重建参数
PCUTOFF            = 0.35
HEAD_TAIL_SAMPLE   = 5
MAX_GAP_FRAMES     = 60
ANCHOR_MIN_HITS    = 1

# 同步推进裁决参数（只作用于冲突裁决，不改变门控本质）
EPS_GAP       = 0.5              # cost = d + EPS_GAP * gap（设 0 可关闭）
DELTA_PX_CAP  = 15.0             # 歧义冻结像素上限（代价差可接受阈值）
DELTA_PROP    = 0.10             # 歧义冻结相对门槛（门控上限的 10%）

# 邻近锚点“堤坝止水”
STOP_NEAR_ANCHOR = True

# 可重复运行支持
RESET_PREVIOUS   = True          # 运行前清理旧的 chain_* 字段
LOG_RUN_METADATA = True          # 记录本次运行参数（安全写法）


# ================== 工具函数 ==================
def _euclid(a, b) -> float:
    if a is None or b is None:
        return float("inf")
    dx, dy = a[0] - b[0], a[1] - b[1]
    return float((dx * dx + dy * dy) ** 0.5)

def _head_center(tk_centers: Dict[int, Tuple[float,float]], start_f: int, n: int = HEAD_TAIL_SAMPLE):
    fs = sorted([f for f in tk_centers.keys() if f >= start_f])[:n]
    if not fs:
        return None
    xs = [tk_centers[f][0] for f in fs]
    ys = [tk_centers[f][1] for f in fs]
    return float(np.mean(xs)), float(np.mean(ys))

def _tail_center(tk_centers: Dict[int, Tuple[float,float]], end_f: int, n: int = HEAD_TAIL_SAMPLE):
    fs = sorted([f for f in tk_centers.keys() if f <= end_f])[-n:]
    if not fs:
        return None
    xs = [tk_centers[f][0] for f in fs]
    ys = [tk_centers[f][1] for f in fs]
    return float(np.mean(xs)), float(np.mean(ys))

def _safe_center_from_arr(arr: np.ndarray, mc_idx, pcutoff: float):
    """优先用 utils.body_center_from_arr；失败则回退到 p>=pcutoff 的关键点均值"""
    try:
        c = body_center_from_arr(arr, mc_idx, pcutoff)
        if c is not None:
            return (float(c[0]), float(c[1]))
    except Exception:
        pass
    try:
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[1] >= 3:
            mask = a[:, 2] >= pcutoff
            if np.any(mask):
                return (float(a[mask, 0].mean()), float(a[mask, 1].mean()))
    except Exception:
        return None
    return None

def summarize_tracklets(dd: Dict, pcutoff: float):
    """
    提取每个 tracklet 的：
      start, end, centers, head, tail, tag, anchor_strength, is_anchor
    并为每个 tag 建立锚点时间索引 tag_to_anchors[tag] = [(start, end, tk), ...]
    —— 健壮性：
       - header 不是 dict 时按 None 处理
       - 仅处理 node 为 dict 的条目；其它类型一律跳过
       - 帧键解析：先用 frame_idx_from_key；失败则若键是 int 直接使用
       - 中心点计算：utils 失败则回退到“p>=pcutoff 的均值”
    """
    header = dd.get("header", None)
    header_for_idx = header if isinstance(header, dict) else None

    # 防御式获取 mc_idx
    try:
        mc_idx = find_mouse_center_index(header_for_idx)
    except Exception:
        try:
            mc_idx = find_mouse_center_index(None)  # 如果 utils 支持
        except Exception:
            mc_idx = None  # 兜底

    tk_info: Dict[Any, Dict[str, Any]] = {}
    tag_to_anchors: Dict[str, List[Tuple[int,int,Any]]] = defaultdict(list)

    for tk, node in dd.items():
        if tk in ("header", "single"):
            continue
        if not isinstance(node, dict):
            continue  # 仅处理 dict 节点

        centers = {}
        fmin, fmax = None, None

        for fkey, arr in node.items():
            if fkey in ("rfid_frames", "rfid_counts", "rfid_hint", "tag", "chain_tag", "chain_id"):
                continue

            # 解析帧号
            fi = None
            try:
                fi = frame_idx_from_key(fkey)
            except Exception:
                if isinstance(fkey, (int, np.integer)):
                    fi = int(fkey)
                else:
                    continue

            if not isinstance(arr, np.ndarray):
                continue

            c = _safe_center_from_arr(arr, mc_idx, pcutoff)
            if c is None:
                continue

            centers[int(fi)] = c
            fmin = int(fi) if fmin is None else min(fmin, int(fi))
            fmax = int(fi) if fmax is None else max(fmax, int(fi))

        if fmin is None:
            continue  # 该节点没有有效中心点

        head = _head_center(centers, fmin)
        tail = _tail_center(centers, fmax)

        tg = node.get("tag")
        tg = str(tg) if tg else None

        counts = node.get("rfid_counts", {}) or {}
        strength = int(counts.get(tg, 0)) if tg and isinstance(counts, dict) else 0
        is_anchor = bool(tg) and (strength >= ANCHOR_MIN_HITS)

        tk_info[tk] = dict(
            start=int(fmin), end=int(fmax), centers=centers,
            head=head, tail=tail, tag=tg, anchor_strength=strength,
            is_anchor=is_anchor
        )

        if is_anchor:
            tag_to_anchors[tg].append((int(fmin), int(fmax), tk))

    for tg in tag_to_anchors:
        tag_to_anchors[tg].sort(key=lambda x: x[0])

    ordered = sorted(tk_info.keys(), key=lambda t: tk_info[t]["start"])
    return tk_info, ordered, tag_to_anchors

def _has_next_anchor_within(tag: str, seed_end: int, tag_to_anchors, max_gap: int) -> bool:
    for s, e, tk in tag_to_anchors.get(tag, []):
        if s > seed_end and (s - seed_end) <= max_gap:
            return True
    return False

def _has_prev_anchor_within(tag: str, seed_start: int, tag_to_anchors, max_gap: int) -> bool:
    for s, e, tk in reversed(tag_to_anchors.get(tag, [])):
        if e < seed_start and (seed_start - e) <= max_gap:
            return True
    return False


# ================== 时间–速度门控与同步推进 ==================
def _v_gate_px_per_frame() -> float:
    if FPS <= 0 or PX_PER_CM <= 0 or V_GATE_CMS <= 0:
        raise ValueError("请正确设置 FPS、PX_PER_CM、V_GATE_CMS（均需>0）")
    return float(V_GATE_CMS * (PX_PER_CM / FPS))

def _delta_threshold_px(gap: int, v_gate_pxpf: float) -> float:
    # 歧义冻结阈值 δ：min(DELTA_PX_CAP, DELTA_PROP * (门控上限))
    return float(min(DELTA_PX_CAP, DELTA_PROP * (v_gate_pxpf * max(1, gap))))

def _collect_candidates(curr_tk, tk_info, assigned, direction: str, v_gate_pxpf: float):
    """
    收集 curr_tk 在指定方向上的、通过时间–速度门控的候选：
      - 仅非锚点、未被占用
      - gap <= MAX_GAP_FRAMES, 且 d <= v_gate_pxpf * gap
    返回 [(cand_tk, d, gap, cost), ...] 按 cost 升序
    """
    info_c = tk_info[curr_tk]

    if direction == "forward":
        t_edge = info_c["end"]
        p_from = info_c["tail"]
        def ok(ti):
            g = ti["start"] - t_edge
            return (g > 0) and (g <= MAX_GAP_FRAMES)
        def dist_gap(ti):
            g = ti["start"] - t_edge
            d = _euclid(p_from, ti["head"])
            return d, g
    else:  # backward
        t_edge = info_c["start"]
        p_from = info_c["head"]
        def ok(ti):
            g = t_edge - ti["end"]
            return (g > 0) and (g <= MAX_GAP_FRAMES)
        def dist_gap(ti):
            g = t_edge - ti["end"]
            d = _euclid(ti["tail"], p_from)
            return d, g

    pool = []
    for tk, ti in tk_info.items():
        if tk in assigned or ti["is_anchor"]:
            continue
        if not ok(ti):
            continue
        d, g = dist_gap(ti)
        if d <= v_gate_pxpf * g:
            cost = d + EPS_GAP * g
            pool.append((tk, d, g, cost))

    pool.sort(key=lambda x: x[3])  # cost 升序
    return pool

def reconstruct_by_timespeed_gate(dd: Dict):
    """
    主流程：按“时间–速度门控 + 同步推进 + 贪心让步”进行链重建
    """
    v_gate_pxpf = _v_gate_px_per_frame()
    tk_info, ordered, tag_to_anchors = summarize_tracklets(dd, PCUTOFF)

    # 1) seeds -> chains & sources
    class Source:
        __slots__ = ("chain_id","tag","direction","cur_tk","active","seed_tk")
        def __init__(self, chain_id, tag, direction, cur_tk, seed_tk):
            self.chain_id  = chain_id
            self.tag       = tag
            self.direction = direction  # "forward" or "backward"
            self.cur_tk    = cur_tk
            self.active    = True
            self.seed_tk   = seed_tk

    seeds_all: List[Tuple[str,int,Any]] = []
    for tk in ordered:
        info = tk_info[tk]
        if info["is_anchor"]:
            seeds_all.append((info["tag"], info["start"], tk))
    seeds_all.sort(key=lambda x: (x[0], x[1]))

    assigned: set = set()
    chain_tracks: Dict[str, List[Any]] = {}     # chain_id -> ordered tk list
    chain_meta: Dict[str, Dict[str, Any]] = {}  # {chain_id: {"tag":tag,"seeds":set()}}
    sources: List[Source] = []
    tag_chain_count = defaultdict(int)

    def allow_dirs_for_seed(tag: str, seed_tk) -> Tuple[bool,bool]:
        sstart, send = tk_info[seed_tk]["start"], tk_info[seed_tk]["end"]
        allow_f = allow_b = True
        if STOP_NEAR_ANCHOR:
            if _has_next_anchor_within(tag, send, tag_to_anchors, MAX_GAP_FRAMES):
                allow_f = False
            if _has_prev_anchor_within(tag, sstart, tag_to_anchors, MAX_GAP_FRAMES):
                allow_b = False
        return allow_f, allow_b

    for tag, _s, seed in seeds_all:
        if seed in assigned:
            continue
        tag_chain_count[tag] += 1
        chain_id = f"{tag}_chain{tag_chain_count[tag]:02d}"
        chain_tracks[chain_id] = [seed]
        chain_meta[chain_id] = {"tag": tag, "seeds": {seed}}
        assigned.add(seed)
        allow_f, allow_b = allow_dirs_for_seed(tag, seed)
        if allow_f:
            sources.append(Source(chain_id, tag, "forward", seed, seed))
        if allow_b:
            sources.append(Source(chain_id, tag, "backward", seed, seed))

    # 2) wavefront 同步推进
    round_idx = 0
    while True:
        round_idx += 1
        active_sources = [i for i,s in enumerate(sources) if s.active]
        if not active_sources:
            break

        proposals_by_cand = defaultdict(list)  # cand_tk -> list of {source_idx,cost,gap,has_alt}
        source_candidates_cache = {}

        # 每个 source 提名其 top1 候选（保留是否有替代）
        for si in active_sources:
            s = sources[si]
            pool = _collect_candidates(s.cur_tk, tk_info, assigned, s.direction, v_gate_pxpf)
            source_candidates_cache[si] = pool
            if not pool:
                s.active = False
                continue
            cand_tk, d, g, cost = pool[0]
            has_alt = (len(pool) > 1)
            proposals_by_cand[cand_tk].append({
                "source_idx": si, "cost": cost, "gap": g, "has_alt": has_alt
            })

        if not proposals_by_cand:
            break

        # 3) 冲突裁决（A 方案增强版）
        assignments = {}        # source_idx -> cand_tk
        used_sources = set()
        used_candidates = set()

        # 候选按最佳代价升序处理
        cand_order = []
        for cand_tk, lst in proposals_by_cand.items():
            best_cost = min(x["cost"] for x in lst)
            cand_order.append((best_cost, cand_tk))
        cand_order.sort(key=lambda x: x[0])

        for _, cand_tk in cand_order:
            if cand_tk in used_candidates:
                continue
            lst = [x for x in proposals_by_cand[cand_tk] if x["source_idx"] not in used_sources]
            if not lst:
                continue
            lst.sort(key=lambda x: x["cost"])
            winner = lst[0]

            # 歧义冻结（前两名差 < δ）
            if len(lst) >= 2:
                second = lst[1]
                delta = _delta_threshold_px(winner["gap"], v_gate_pxpf)
                if abs(second["cost"] - winner["cost"]) < delta:
                    # 冻结该候选
                    continue

            # 反“抢食”让步：winner 有替代、存在无替代对手 -> 让给无替代且 cost 最小者
            no_alt_competitors = [x for x in lst[1:] if not x["has_alt"]]
            if winner["has_alt"] and no_alt_competitors:
                chosen = min(no_alt_competitors, key=lambda x: x["cost"])
            else:
                chosen = winner

            si = chosen["source_idx"]
            assignments[si] = cand_tk
            used_sources.add(si)
            used_candidates.add(cand_tk)

        if not assignments:
            break  # 本回合没有任何分配，终止

        # 4) 执行分配：推进前沿、占用候选、写入链
        for si, cand_tk in assignments.items():
            s = sources[si]
            assigned.add(cand_tk)
            if s.direction == "forward":
                chain_tracks[s.chain_id].append(cand_tk)
            else:
                chain_tracks[s.chain_id].insert(0, cand_tk)
            s.cur_tk = cand_tk

    # 5) 写回属性并汇总 CSV
    chain_rows = []
    for chain_id, tks in chain_tracks.items():
        tag = chain_meta[chain_id]["tag"]
        seeds_set = chain_meta[chain_id]["seeds"]
        for tk in tks:
            node = dd[tk]
            if isinstance(node, dict):  # 仅对 dict 节点写回链字段
                node["chain_tag"] = tag
                node["chain_id"]  = chain_id
                node["chain_origin"] = ("anchor" if tk in seeds_set else "propagated")
                dd[tk] = node

            info = tk_info[tk]
            chain_rows.append([chain_id, tag, tk, info["start"], info["end"]])

    df = pd.DataFrame(chain_rows, columns=["chain_id","chain_tag","tracklet","start","end"])
    return dd, df


# ================== 运行入口 ==================
def main():
    p_in = Path(PICKLE_IN)
    if OUT_SUBDIR:
        out_dir = p_in.parent / OUT_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        p_out = out_dir / p_in.name  # 保留原文件名
    else:
        p_out = p_in if PICKLE_OUT is None else Path(PICKLE_OUT)

    if not p_in.exists():
        print(f"错误：输入文件不存在: {p_in}")
        return
    if str(p_in).endswith(".json"):
        print(f"错误：PICKLE_IN 指向 JSON 而非 .pickle：{p_in}")
        return
    if FPS <= 0 or PX_PER_CM <= 0 or V_GATE_CMS <= 0:
        print("错误：请正确设置 FPS、PX_PER_CM、V_GATE_CMS（均需>0）")
        return

    print(f"正在加载 pickle: {p_in}")
    dd = load_tracklets_pickle(str(p_in))

    # —— 可重复运行：清理旧链字段 ——
    if RESET_PREVIOUS:
        cleared = 0
        for tk, node in dd.items():
            if tk in ("header", "single"):
                continue
            if isinstance(node, dict):
                for k in ("chain_tag", "chain_id", "chain_origin", "chain_tag_conflict"):
                    if k in node:
                        node.pop(k, None)
                        cleared += 1
        print(f"[reset] 清理旧链字段完成（条目计数≈{cleared}）")

    # —— 记录本次运行参数（安全写法：不修改非 dict 的 header） ——
    if LOG_RUN_METADATA:
        meta = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": "time-speed gate=80cm/s + wavefront + greedy",
            "fps": FPS, "px_per_cm": PX_PER_CM, "v_gate_cmps": V_GATE_CMS,
            "max_gap": MAX_GAP_FRAMES, "eps_gap": EPS_GAP,
            "stop_near_anchor": STOP_NEAR_ANCHOR,
        }
        hdr = dd.get("header", None)
        if isinstance(hdr, dict):
            hist = hdr.setdefault("reconstruct_history", [])
            if isinstance(hist, list):
                hist.append(meta)
            else:
                dd.setdefault("_reconstruct_history", [])
                dd["_reconstruct_history"].append(meta)
        else:
            dd.setdefault("_reconstruct_history", [])
            dd["_reconstruct_history"].append(meta)

    print("开始基于【时间–速度门控(80 cm/s) + 同步推进 + 贪心让步】的轨迹重建…")
    dd2, df_chains = reconstruct_by_timespeed_gate(dd)

    # 保存链段 CSV
    out_csv = p_out.parent / "chain_segments.csv"
    df_chains.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] 链段信息已保存: {out_csv}")

    # 保存更新后的 pickle（覆盖或另存）
    save_pickle_safely(dd2, p_out)
    print(f"[OK] pickle 文件已更新: {p_out}")
    print(f"重建完成：共生成 {len(df_chains)} 条链段记录")

if __name__ == "__main__":
    main()
