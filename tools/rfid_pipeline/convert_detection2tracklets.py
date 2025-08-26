#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最简脚本：把 DLC 的检测结果转成 tracklets（*.pickle）
前提：你已经跑过 deeplabcut.analyze_videos（已有 <video>*h5）
"""

from pathlib import Path
import argparse
import deeplabcut as dlc
from deeplabcut.utils import auxiliaryfunctions as aux

# ===========================
# 默认参数（可通过命令行覆盖）
# ===========================
CONFIG_PATH  = "/ssd01/user_acc_data/oppa/deeplabcut/projects/MiceTrackerFor20-Oppa-2024-12-08/config.yaml"
TRACK_METHOD = "ellipse"   # "ellipse" / "skeleton" / "box"
SHUFFLE      = 3
DESTFOLDER   = "/ssd01/user_acc_data/oppa/deeplabcut/projects/MiceTrackerFor20-Oppa-2024-12-08/analyze_videos/shuffle3/demo1/"
VIDEO_INPUT  = "/ssd01/user_acc_data/oppa/deeplabcut/videos/test/demo.mp4"
VIDEOTYPE    = "mp4"       # 目录模式生效；文件模式无所谓
# ===========================


def collect_videos(input_path: str):
    p = Path(input_path)
    if p.is_dir():
        if VIDEOTYPE:
            vids = sorted(str(x) for x in p.glob(f"*.{VIDEOTYPE.lstrip('.')}"))
        else:
            exts = (".mp4", ".avi", ".mov", ".mpeg", ".mkv")
            vids = sorted(str(x) for x in p.iterdir() if x.suffix.lower() in exts)
    else:
        vids = [str(p)]
    if not vids:
        raise FileNotFoundError(f"在 {input_path} 下未找到视频（后缀：{VIDEOTYPE or '常见扩展'}）")
    return vids


def main():
    videos = collect_videos(VIDEO_INPUT)
    print(f"[INFO] config: {CONFIG_PATH}")
    print(f"[INFO] videos: {len(videos)} 个（track_method={TRACK_METHOD}）")
    print(f"[INFO] destfolder: {DESTFOLDER or '(视频目录)'}")

    # 1) 优先使用项目 config.yaml 中的 inference_cfg / inferencecfg
    try:
        cfg = aux.read_config(CONFIG_PATH)
        base_inferencecfg = (cfg.get("inference_cfg") or cfg.get("inferencecfg") or {}).copy()
    except Exception:
        base_inferencecfg = {}

    # 2) 如果项目中没有，就提供 DLC 需要的“最小键”。不额外覆盖其它阈值，让 DLC 用默认。
    if not base_inferencecfg:
        base_inferencecfg = {
            "variant": "paf",                # 组装方式（常见）
            "method": "munkres",             # 匹配方法（匈牙利）
            "pafthreshold": 0.05,            # PAF 连接阈值（保守默认）
            "minimalnumberofconnections": 1, # 骨架最小连接数
            "withid": False,                 # 若关键点含 identity 通道才需 True；一般 False
        }

    # 3) 调用 DLC：不再强行覆盖任何 tracking 阈值，保持默认
    dlc.convert_detections2tracklets(
        config=CONFIG_PATH,
        videos=videos,
        videotype=VIDEOTYPE,
        shuffle=SHUFFLE,
        track_method=TRACK_METHOD,
        destfolder=DESTFOLDER,
        inferencecfg=base_inferencecfg
    )

    print("\n[OK] 完成。请在对应输出目录查看 *.pickle（tracklets）文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DLC detections to tracklets")
    parser.add_argument("--config", default=CONFIG_PATH, help="路径到 config.yaml")
    parser.add_argument("--track-method", default=TRACK_METHOD, help="tracking 方法")
    parser.add_argument("--shuffle", type=int, default=SHUFFLE, help="shuffle 编号")
    parser.add_argument("--destfolder", default=DESTFOLDER, help="输出目录")
    parser.add_argument("--video-input", default=VIDEO_INPUT, help="视频文件或目录")
    parser.add_argument("--videotype", default=VIDEOTYPE, help="目录模式下的视频后缀")
    args = parser.parse_args()

    CONFIG_PATH = args.config
    TRACK_METHOD = args.track_method
    SHUFFLE = args.shuffle
    DESTFOLDER = args.destfolder
    VIDEO_INPUT = args.video_input
    VIDEOTYPE = args.videotype

    main()
