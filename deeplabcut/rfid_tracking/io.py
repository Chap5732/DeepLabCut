#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from pathlib import Path
from typing import Any


def load_tracklets_pickle(pkl_path: str) -> dict:
    """加载tracklets pickle文件"""
    with open(pkl_path, "rb") as f:
        dd = pickle.load(f)
    if not isinstance(dd, dict) or "header" not in dd:
        raise ValueError("pickle结构异常：未找到 'header'")
    return dd


def save_pickle_safely(obj: Any, target: Path):
    """安全保存pickle文件（使用临时文件+备份）"""
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


__all__ = ["load_tracklets_pickle", "save_pickle_safely"]
