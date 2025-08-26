#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Minimal script: convert DLC detections into tracklets (``*.pickle``)
最简脚本：把 DLC 的检测结果转成 tracklets（*.pickle）

Prerequisite: run ``deeplabcut.analyze_videos`` beforehand to obtain
``<video>*h5`` files
前提：你已经跑过 ``deeplabcut.analyze_videos``（已有 ``<video>*h5``）
"""

from pathlib import Path

import deeplabcut as dlc
from deeplabcut.utils import auxiliaryfunctions as aux

# ===========================
# Hard-coded parameters (edit as needed)
# 硬编码参数（按需修改）
# ===========================
CONFIG_PATH = "/ssd01/user_acc_data/oppa/deeplabcut/projects/MiceTrackerFor20-Oppa-2024-12-08/config.yaml"
TRACK_METHOD = "ellipse"  # "ellipse" / "skeleton" / "box"
SHUFFLE = 3
DESTFOLDER = "/ssd01/user_acc_data/oppa/deeplabcut/projects/MiceTrackerFor20-Oppa-2024-12-08/analyze_videos/shuffle3/demo1/"
VIDEO_INPUT = "/ssd01/user_acc_data/oppa/deeplabcut/videos/test/demo.mp4"
VIDEOTYPE = (
    "mp4"  # Effective only when VIDEO_INPUT is a directory; ignored for single files
)
# 目录模式生效；文件模式无所谓
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
        raise FileNotFoundError(
            f"No videos found in {input_path} (extension: {VIDEOTYPE or 'common'})"
            f" 在 {input_path} 下未找到视频（后缀：{VIDEOTYPE or '常见扩展'}）"
        )
    return vids


def main():
    videos = collect_videos(VIDEO_INPUT)
    print(f"[INFO] config: {CONFIG_PATH}")
    print(f"[INFO] videos: {len(videos)} (track_method={TRACK_METHOD})")
    print(f"[INFO] destfolder: {DESTFOLDER or '(video directory)'}")

    # 1) Prefer ``inference_cfg``/``inferencecfg`` from the project config.yaml
    #    优先使用项目 config.yaml 中的 ``inference_cfg`` / ``inferencecfg``
    try:
        cfg = aux.read_config(CONFIG_PATH)
        base_inferencecfg = (
            cfg.get("inference_cfg") or cfg.get("inferencecfg") or {}
        ).copy()
    except Exception:
        base_inferencecfg = {}

    # 2) If the project lacks these settings, supply the minimal keys DLC needs.
    #    Do not override additional thresholds so DLC uses its defaults.
    #    如果项目中没有，就提供 DLC 需要的“最小键”，不额外覆盖其它阈值，让 DLC 用默认。
    if not base_inferencecfg:
        base_inferencecfg = {
            "variant": "paf",  # Assembly method (common)
            "method": "munkres",  # Matching algorithm (Hungarian)
            "pafthreshold": 0.05,  # PAF connection threshold (conservative default)
            "minimalnumberofconnections": 1,  # Minimum skeleton connections
            "withid": False,  # Set True only if keypoints include an identity channel
            # 若关键点含 identity 通道才需 True；一般 False
        }

    # 3) Call DLC without forcing tracking thresholds; keep defaults
    #    调用 DLC：不再强行覆盖任何 tracking 阈值，保持默认
    dlc.convert_detections2tracklets(
        config=CONFIG_PATH,
        videos=videos,
        videotype=VIDEOTYPE,
        shuffle=SHUFFLE,
        track_method=TRACK_METHOD,
        destfolder=DESTFOLDER,
        inferencecfg=base_inferencecfg,
    )

    print(
        "\n[OK] Done. Check the output directory for *.pickle tracklet files.\n"
        "完成。请在对应输出目录查看 *.pickle（tracklets）文件。"
    )


if __name__ == "__main__":
    main()
