#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视频可视化脚本：在原视频上叠加tracklet轨迹、RFID事件和重建链信息
- 支持绘制读卡器位置、ROI区域
- 显示tracklet轨迹和RFID标签检测事件
- 显示重建后的身份链和图例（与tracklet标签不重叠）
"""

from __future__ import annotations
from pathlib import Path
from collections import defaultdict, deque
import argparse
import cv2
import numpy as np
from utils import (
    load_tracklets_pickle, frame_idx_from_key, find_mouse_center_index,
    body_center_from_arr, color_for_id, parse_centers,
    centers_to_reader_positions_column_major, draw_readers_on_frame,
    load_rois, draw_rois
)

# ================== 配置参数 ==================
# 读卡器可视化
DRAW_READERS = True
CENTERS_TXT_DEFAULT = Path.cwd() / "readers_centers.txt"

# ROI可视化
DRAW_ROIS = True
ROI_FILE_DEFAULT = Path.cwd() / "roi_definitions.json"

# 轨迹参数
PCUTOFF = 0.35                # 置信度阈值
TRAIL_LEN = 15                # tracklet轨迹长度
TAG_HOLD_FRAMES = 3           # RFID标签显示持续帧数

# 身份链可视化
SHOW_CHAIN = True             # 是否显示重建的身份链
CHAIN_FALLBACK_ID = True      # 是否使用chain_id作为回退
CHAIN_TRAIL_LEN = 40          # 身份链轨迹长度
CHAIN_LINE_THICK = 3          # 身份链线条粗细
CHAIN_POINT_R = 5             # 身份链端点半径

# 图例设置
DRAW_LEGEND = True            # 是否绘制图例
LEGEND_COLS = 2               # 图例列数
LEGEND_POS = (20, 40)         # 图例位置

# 输出限制
MAX_FRAMES = None             # 最大输出帧数（None=全部）

def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay tracklet and RFID info on video"
    )
    parser.add_argument(
        "--video", default=Path.cwd() / "video.mp4", help="Input video file"
    )
    parser.add_argument(
        "--pickle",
        default=Path.cwd() / "tracklets.pickle",
        help="Tracklets pickle file",
    )
    parser.add_argument(
        "--output-video",
        default=Path.cwd() / "rfid_tracklets_overlay.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--centers-txt",
        default=CENTERS_TXT_DEFAULT,
        help="RFID reader centers file",
    )
    parser.add_argument(
        "--roi-file",
        default=ROI_FILE_DEFAULT,
        help="ROI definition file",
    )
    args = parser.parse_args()
    video = Path(args.video)
    pickle_path = Path(args.pickle)
    if not video.is_file():
        parser.error(f"Video file not found: {video}")
    if not pickle_path.is_file():
        parser.error(f"Pickle file not found: {pickle_path}")
    return args, video, pickle_path

# ================== 小工具 ==================
def draw_label_with_bg(img, text, org, font_scale=0.6, thickness=2, fg=(255,255,255), bg=(0,0,0), alpha=0.4, padding=4):
    """在文字下绘制半透明背景框，提升可读性"""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = org
    x0, y0 = x - padding, y - th - padding
    x1, y1 = x + tw + padding, y + base + padding
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg, thickness, cv2.LINE_AA)

def safe_find_mouse_center_index(header):
    """header 非 dict 时按 None 处理，避免 utils 内部假设"""
    if not isinstance(header, dict):
        header = None
    return find_mouse_center_index(header)

# ================== 数据处理 ==================
def build_per_frame_struct(dd: dict, pcutoff: float):
    """构建按帧组织的数据结构"""
    header = dd.get("header", None)
    mc_idx = safe_find_mouse_center_index(header)

    # tracklet相关数据
    frame2tk_center = defaultdict(list)
    tk_trail = {}
    frames_set = set()
    chain_key_of_tk = {}

    # 处理每个tracklet
    for tk, node in dd.items():
        if tk in ("header", "single"):
            continue
        if not isinstance(node, dict):
            continue

        tk_trail[tk] = deque(maxlen=TRAIL_LEN)

        # 提取每帧的中心点
        for fkey, arr in node.items():
            if fkey in ("rfid_frames", "rfid_counts", "tag", "rfid_hint", "chain_tag", "chain_id"):
                continue
            try:
                fi = frame_idx_from_key(fkey)
            except Exception:
                continue
            if not isinstance(arr, np.ndarray):
                continue

            c = body_center_from_arr(arr, mc_idx, pcutoff)
            if c is None:
                continue

            frame2tk_center[int(fi)].append((tk, (float(c[0]), float(c[1]))))
            frames_set.add(int(fi))

        # 获取身份链键（chain_tag 优先，其次原始 tag，最后 fallback 到 chain_id）
        chain_key = node.get("chain_tag") or node.get("tag")
        if chain_key is None and CHAIN_FALLBACK_ID:
            chain_key = node.get("chain_id")
        chain_key_of_tk[tk] = str(chain_key) if chain_key is not None else None

    # 构建RFID事件映射
    tk_rfid_events = defaultdict(lambda: defaultdict(list))
    for tk, node in dd.items():
        if tk in ("header", "single"):
            continue
        if not isinstance(node, dict):
            continue
        rf = node.get("rfid_frames", None)
        if not rf:
            continue
        for tag, lst in rf.items():
            for fr_conf in lst:
                try:
                    fr = int(fr_conf[0])
                except Exception:
                    continue
                tk_rfid_events[tk][fr].append(str(tag))

    # 构建身份链的每帧数据（一个 chain 在同一帧只显示一次）
    frame2chain_center = defaultdict(list)
    chain_seen_in_frame = defaultdict(set)

    for f in sorted(frames_set):
        for tk, c in frame2tk_center[f]:
            ck = chain_key_of_tk.get(tk)
            if ck is None:
                continue
            if ck in chain_seen_in_frame[f]:
                continue
            frame2chain_center[f].append((ck, c))
            chain_seen_in_frame[f].add(ck)

    # 初始化身份链轨迹
    chain_trail = {}
    for f, items in frame2chain_center.items():
        for ck, _ in items:
            if ck not in chain_trail:
                chain_trail[ck] = deque(maxlen=CHAIN_TRAIL_LEN)

    frames_sorted = sorted(frames_set)
    return (frames_sorted, frame2tk_center, tk_rfid_events, tk_trail,
            chain_key_of_tk, frame2chain_center, chain_trail)

# ================== 绘制函数 ==================
def draw_tracklets_layer(frame, frame_idx, pts_in_frame, tk_rfid_events, tk_trail):
    """绘制tracklet轨迹层（ID 放在点下方；RFID 提示再更下方）"""
    canvas = frame.copy()

    # 更新轨迹
    for tk, (x, y) in pts_in_frame:
        tk_trail.setdefault(tk, deque(maxlen=TRAIL_LEN))
        tk_trail[tk].append((int(round(x)), int(round(y))))

    # 绘制每个tracklet
    for tk, (x, y) in pts_in_frame:
        xi, yi = int(round(x)), int(round(y))
        color = color_for_id(tk)

        # 轨迹线
        trail = list(tk_trail[tk])
        for i in range(1, len(trail)):
            cv2.line(canvas, trail[i-1], trail[i], color, 2, cv2.LINE_AA)

        # 当前位置点
        cv2.circle(canvas, (xi, yi), 5, color, -1, cv2.LINE_AA)

        # tracklet ID（放下方）
        id_text = f"{tk}"
        (tw, th), base = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_org = (xi - tw // 2, yi + th + 8)  # 下方
        draw_label_with_bg(canvas, id_text, text_org, font_scale=0.6, thickness=2,
                           fg=(255,255,255), bg=(0,0,0), alpha=0.35, padding=4)

        # RFID 标签（再更下方）
        tags_to_show = []
        for df in range(TAG_HOLD_FRAMES):
            fr = frame_idx - df
            if fr in tk_rfid_events.get(tk, {}):
                tags_to_show.extend(tk_rfid_events[tk][fr])
                break

        if tags_to_show:
            tag_text = f"{', '.join(sorted(set(tags_to_show)))} tag received"
            (tw2, th2), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_org2 = (xi - tw2 // 2, yi + th + 8 + th2 + 8)  # 再往下
            draw_label_with_bg(canvas, tag_text, text_org2, font_scale=0.6, thickness=2,
                               fg=(255,255,255), bg=(0,0,0), alpha=0.35, padding=4)

    return canvas

def draw_chain_layer(frame, frame_idx, frame2chain_center, chain_trail):
    """绘制身份链轨迹层（chain ID 放上方；颜色固定且与 tracklet 解耦）"""
    canvas = frame.copy()
    items = frame2chain_center.get(frame_idx, [])

    # 更新身份链轨迹
    for ck, (x, y) in items:
        chain_trail.setdefault(ck, deque(maxlen=CHAIN_TRAIL_LEN))
        chain_trail[ck].append((int(round(x)), int(round(y))))

    # 绘制每个身份链
    for ck, _ in items:
        color = color_for_id(f"chain:{ck}")  # 与 tracklet 颜色解耦，保证每个 chain 固定颜色
        trail = list(chain_trail[ck])

        # 轨迹线
        for i in range(1, len(trail)):
            cv2.line(canvas, trail[i-1], trail[i], color, CHAIN_LINE_THICK, cv2.LINE_AA)

        # 当前位置点和标签（标签放上方）
        if trail:
            xi, yi = trail[-1]
            cv2.circle(canvas, (xi, yi), CHAIN_POINT_R, color, -1, cv2.LINE_AA)
            label = str(ck)
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_org = (xi - tw // 2, yi - CHAIN_POINT_R - 10)  # 上方
            draw_label_with_bg(canvas, label, text_org, font_scale=0.6, thickness=2,
                               fg=(255,255,255), bg=color, alpha=0.35, padding=4)

    return canvas

def draw_chain_legend(frame, chain_trail, pos=(20, 40), cols=2):
    """绘制身份链图例（颜色与链颜色一致）"""
    items = sorted(chain_trail.keys(), key=lambda x: str(x))
    if not items:
        return frame

    canvas = frame.copy()
    x0, y0 = pos
    cell_h = 26
    cell_w = 220

    for i, ck in enumerate(items):
        r = i // cols
        c = i % cols
        x = x0 + c * cell_w
        y = y0 + r * cell_h

        color = color_for_id(f"chain:{ck}")
        cv2.circle(canvas, (x, y), 8, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(ck), (x + 14, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas

# ================== 主函数 ==================
def main():
    """主函数"""
    args, video_path, pickle_path = parse_args()
    centers_txt = Path(args.centers_txt)
    roi_file = Path(args.roi_file)

    # 加载数据
    print("正在加载tracklet数据...")
    dd = load_tracklets_pickle(str(pickle_path))

    print("构建每帧数据结构...")
    (frames_sorted, frame2tk_center, tk_rfid_events, tk_trail,
     chain_key_of_tk, frame2chain_center, chain_trail) = build_per_frame_struct(dd, PCUTOFF)

    print(f"发现 {len(tk_trail)} 个tracklet，包含 {len(frames_sorted)} 个有效帧")
    print(f"检测到 {len([k for k in set(chain_key_of_tk.values()) if k])} 个身份键")

    # 加载读卡器位置（可选）
    reader_positions = None
    if DRAW_READERS and centers_txt.exists():
        centers, meta = parse_centers(str(centers_txt))
        reader_positions = centers_to_reader_positions_column_major(centers, meta)
        print(f"读取到 {len(reader_positions)} 个读卡器位置")

    # 加载ROI（可选）
    rois = (
        load_rois(str(roi_file))
        if (DRAW_ROIS and roi_file and roi_file.exists())
        else []
    )
    if rois:
        print(f"加载了 {len(rois)} 个ROI区域")

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("错误：无法打开视频文件")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 设置输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(args.output_video), fourcc, fps if fps > 0 else 25.0, (W, H)
    )

    max_frames = T if MAX_FRAMES is None else min(T, MAX_FRAMES)
    print(f"视频信息: {W}x{H}, {fps:.2f}fps, 共 {T} 帧；将输出 {max_frames} 帧")

    # 处理视频帧
    valid_frames = set(frame2tk_center.keys())
    fidx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fidx += 1
        if fidx > max_frames:
            break

        canvas = frame

        # 绘制ROI区域
        if rois:
            canvas = draw_rois(canvas, rois)

        # 绘制读卡器位置
        if reader_positions is not None:
            canvas = draw_readers_on_frame(canvas, reader_positions)

        # 绘制tracklet轨迹
        if fidx in valid_frames:
            pts = frame2tk_center[fidx]
            canvas = draw_tracklets_layer(canvas, fidx, pts, tk_rfid_events, tk_trail)

        # 绘制身份链
        if SHOW_CHAIN:
            canvas = draw_chain_layer(canvas, fidx, frame2chain_center, chain_trail)
            if DRAW_LEGEND:
                canvas = draw_chain_legend(canvas, chain_trail, pos=LEGEND_POS, cols=LEGEND_COLS)

        # 写入输出视频
        out.write(canvas)

        # 进度显示
        if fidx % 50 == 0:
            print(f"已处理 {fidx}/{max_frames} 帧 ({fidx/max_frames*100:.1f}%)")

    # 清理资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[OK] 完成！输出视频: {args.output_video}")

if __name__ == "__main__":
    main()
