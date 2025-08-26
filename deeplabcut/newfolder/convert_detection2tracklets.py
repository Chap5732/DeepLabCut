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
# 默认参数
# ===========================
DEFAULT_CONFIG_PATH = "/ssd01/user_acc_data/oppa/deeplabcut/projects/MiceTrackerFor20-Oppa-2024-12-08/config.yaml"
DEFAULT_TRACK_METHOD = "ellipse"   # "ellipse" / "skeleton" / "box"
DEFAULT_SHUFFLE = 3
DEFAULT_DESTFOLDER = "/ssd01/user_acc_data/oppa/deeplabcut/projects/MiceTrackerFor20-Oppa-2024-12-08/analyze_videos/shuffle3/demo1/"
DEFAULT_VIDEO_INPUT = "/ssd01/user_acc_data/oppa/deeplabcut/videos/test/demo.mp4"
DEFAULT_VIDEOTYPE = "mp4"       # 目录模式生效；文件模式无所谓
# ===========================


def collect_videos(input_path: str, videotype: str):
    p = Path(input_path)
    if p.is_dir():
        if videotype:
            vids = sorted(str(x) for x in p.glob(f"*.{videotype.lstrip('.')}"))
        else:
            exts = (".mp4", ".avi", ".mov", ".mpeg", ".mkv")
            vids = sorted(str(x) for x in p.iterdir() if x.suffix.lower() in exts)
    else:
        vids = [str(p)]
    if not vids:
        raise FileNotFoundError(f"在 {input_path} 下未找到视频（后缀：{videotype or '常见扩展'}）")
    return vids


def parse_args():
    parser = argparse.ArgumentParser(description="Convert DLC detections to tracklets")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="path to config.yaml")
    parser.add_argument("--track-method", default=DEFAULT_TRACK_METHOD)
    parser.add_argument("--shuffle", type=int, default=DEFAULT_SHUFFLE)
    parser.add_argument("--destfolder", default=DEFAULT_DESTFOLDER)
    parser.add_argument("--video-input", default=DEFAULT_VIDEO_INPUT)
    parser.add_argument("--videotype", default=DEFAULT_VIDEOTYPE)
    return parser.parse_args()


def main():
    args = parse_args()
    videos = collect_videos(args.video_input, args.videotype)
    print(f"[INFO] config: {args.config}")
    print(f"[INFO] videos: {len(videos)} 个（track_method={args.track_method}）")
    print(f"[INFO] destfolder: {args.destfolder or '(视频目录)'}")

    # 1) 优先使用项目 config.yaml 中的 inference_cfg / inferencecfg
    try:
        cfg = aux.read_config(args.config)
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
        config=args.config,
        videos=videos,
        videotype=args.videotype,
        shuffle=args.shuffle,
        track_method=args.track_method,
        destfolder=args.destfolder,
        inferencecfg=base_inferencecfg,
    )

    print("\n[OK] 完成。请在对应输出目录查看 *.pickle（tracklets）文件。")


if __name__ == "__main__":
    main()
