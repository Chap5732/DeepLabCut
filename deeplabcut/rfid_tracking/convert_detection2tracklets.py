#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最简脚本：把 DLC 的检测结果转成 tracklets（*.pickle）
前提：你已经跑过 deeplabcut.analyze_videos（已有 <video>*h5）
"""

from pathlib import Path
from typing import Optional

import click
import deeplabcut as dlc
from deeplabcut.utils import auxiliaryfunctions as aux

try:  # 允许作为脚本或模块运行
    from . import config
except ImportError:  # pragma: no cover
    import config

DEFAULTS_PATH = config.CONVERT_DEFAULTS


def collect_videos(input_path: str, videotype: Optional[str]):
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
        raise FileNotFoundError(
            f"在 {input_path} 下未找到视频（后缀：{videotype or '常见扩展'}）"
        )
    return vids


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config-path", type=click.Path(exists=True, dir_okay=False), help="项目 config.yaml 路径")
@click.option("--track-method", type=click.Choice(["ellipse", "skeleton", "box"]), help="tracklet 匹配方法")
@click.option("--shuffle", type=int, help="训练 shuffle 编号")
@click.option("--destfolder", type=click.Path(), help="输出目录")
@click.option("--video-input", type=click.Path(), help="视频文件或目录")
@click.option("--videotype", help="目录模式下的视频后缀")
@click.option(
    "--defaults",
    type=click.Path(exists=True, dir_okay=False),
    default=DEFAULTS_PATH,
    show_default=True,
    help="默认参数 YAML 文件",
)
def main(config_path, track_method, shuffle, destfolder, video_input, videotype, defaults):
    defaults_dict = {}
    if defaults and Path(defaults).is_file():
        defaults_dict = aux.read_config(defaults)

    config_path = config_path or defaults_dict.get("config_path")
    track_method = track_method or defaults_dict.get("track_method", "ellipse")
    shuffle = shuffle or defaults_dict.get("shuffle", 1)
    destfolder = destfolder or defaults_dict.get("destfolder")
    video_input = video_input or defaults_dict.get("video_input")
    videotype = videotype or defaults_dict.get("videotype")

    if not config_path or not video_input:
        raise click.UsageError("必须指定 CONFIG_PATH 和 VIDEO_INPUT（可在默认配置文件中设置）。")

    videos = collect_videos(video_input, videotype)
    print(f"[INFO] config: {config_path}")
    print(f"[INFO] videos: {len(videos)} 个（track_method={track_method}）")
    print(f"[INFO] destfolder: {destfolder or '(视频目录)'}")

    try:
        cfg = aux.read_config(config_path)
        base_inferencecfg = (cfg.get("inference_cfg") or cfg.get("inferencecfg") or {}).copy()
    except Exception:
        base_inferencecfg = {}

    if not base_inferencecfg:
        base_inferencecfg = {
            "variant": "paf",                # 组装方式（常见）
            "method": "munkres",             # 匹配方法（匈牙利）
            "pafthreshold": 0.05,            # PAF 连接阈值（保守默认）
            "minimalnumberofconnections": 1, # 骨架最小连接数
            "withid": False,                 # 若关键点含 identity 通道才需 True；一般 False
        }

    dlc.convert_detections2tracklets(
        config=config_path,
        videos=videos,
        videotype=videotype,
        shuffle=shuffle,
        track_method=track_method,
        destfolder=destfolder,
        inferencecfg=base_inferencecfg,
    )

    print("\n[OK] 完成。请在对应输出目录查看 *.pickle（tracklets）文件。")


if __name__ == "__main__":
    main()
