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


def collect_videos(input_path: str, videotype: str | None):
    p = Path(input_path)
    if p.is_dir():
        if videotype:
            vids = sorted(str(x) for x in p.glob(f"*.{videotype.lstrip('.')}") )
        else:
            exts = (".mp4", ".avi", ".mov", ".mpeg", ".mkv")
            vids = sorted(str(x) for x in p.iterdir() if x.suffix.lower() in exts)
    else:
        vids = [str(p)]
    if not vids:
        raise FileNotFoundError(
            f"在 {input_path} 下未找到视频（后缀：{videotype or '常见扩展'}）"
        )
    return vids


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DLC detections to tracklets"
    )
    parser.add_argument(
        "--config",
        default=Path.cwd() / "config.yaml",
        help="Path to DLC project config.yaml",
    )
    parser.add_argument(
        "--track-method",
        default="ellipse",
        choices=["ellipse", "skeleton", "box"],
        help="Tracking method",
    )
    parser.add_argument(
        "--shuffle", type=int, default=3, help="Training shuffle index"
    )
    parser.add_argument(
        "--destfolder",
        default=None,
        help="Destination folder for outputs",
    )
    parser.add_argument(
        "--video-input",
        default=str(Path.cwd()),
        help="Video file or directory",
    )
    parser.add_argument(
        "--videotype",
        default=None,
        help="Restrict directory search to this extension",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        parser.error(f"Config file not found: {config_path}")
    return args, config_path


def main():
    args, config_path = parse_args()
    videos = collect_videos(args.video_input, args.videotype)
    destfolder = Path(args.destfolder) if args.destfolder else None
    if destfolder:
        destfolder.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] config: {config_path}")
    print(f"[INFO] videos: {len(videos)} 个（track_method={args.track_method}）")
    print(f"[INFO] destfolder: {destfolder or '(视频目录)'}")

    # 1) 优先使用项目 config.yaml 中的 inference_cfg / inferencecfg
    try:
        cfg = aux.read_config(str(config_path))
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
        config=str(config_path),
        videos=videos,
        videotype=args.videotype,
        shuffle=args.shuffle,
        track_method=args.track_method,
        destfolder=str(destfolder) if destfolder else None,
        inferencecfg=base_inferencecfg,
    )

    print("\n[OK] 完成。请在对应输出目录查看 *.pickle（tracklets）文件。")


if __name__ == "__main__":
    main()
