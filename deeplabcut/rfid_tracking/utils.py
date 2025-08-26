#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility utilities for RFID tracking.

This module now re-exports functions from the new submodules:
`io`, `dlc_tools`, and `visualization`. Existing imports from
`utils` continue to work.
"""

from .io import load_tracklets_pickle, save_pickle_safely
from .dlc_tools import frame_idx_from_key, find_mouse_center_index, body_center_from_arr
from .visualization import (
    color_for_id,
    parse_centers,
    centers_to_reader_positions_column_major,
    draw_readers_on_frame,
    load_rois,
    draw_rois,
    point_in_rect,
    point_in_poly,
    point_in_any_roi,
)

__all__ = [
    "load_tracklets_pickle",
    "save_pickle_safely",
    "frame_idx_from_key",
    "find_mouse_center_index",
    "body_center_from_arr",
    "color_for_id",
    "parse_centers",
    "centers_to_reader_positions_column_major",
    "draw_readers_on_frame",
    "load_rois",
    "draw_rois",
    "point_in_rect",
    "point_in_poly",
    "point_in_any_roi",
]
