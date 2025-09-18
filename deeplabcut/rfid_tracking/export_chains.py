"""Utilities for exporting reconstructed RFID chains as DLC-style tables."""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, MutableMapping, Optional

import numpy as np
import pandas as pd

from .io import load_tracklets_pickle

logger = logging.getLogger(__name__)

_FRAME_RE = re.compile(r"frame(\d+)$")


def _as_multiindex(header: object) -> pd.MultiIndex:
    """Return the tracklet header as a :class:`~pandas.MultiIndex`.

    The tracklet pickle stores the DLC column metadata under the ``"header"``
    key.  Recent pipelines already persist it as a ``MultiIndex`` with levels
    named ``("scorer", "bodyparts", "coords")``.  For compatibility with older
    pickles we also accept sequences of tuples.
    """

    if isinstance(header, pd.MultiIndex):
        return header

    try:
        tuples = list(header)  # type: ignore[arg-type]
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError("Unsupported header format in tracklet pickle") from exc

    if not tuples:
        raise ValueError("Tracklet header is empty; cannot export chains")

    return pd.MultiIndex.from_tuples(  # type: ignore[arg-type]
        tuples, names=["scorer", "bodyparts", "coords"]
    )


def _get_level(mi: pd.MultiIndex, name: str, pos: int) -> pd.Index:
    """Robustly fetch a level from ``mi`` by name or positional index."""

    if name in mi.names:
        return mi.get_level_values(name)
    return mi.get_level_values(pos)


def _flatten_frame(values: np.ndarray, expected: int, chain_tag: str, frame: int) -> np.ndarray:
    """Convert a frame matrix into a flat vector matching the DLC header."""

    if values.ndim != 2:
        raise ValueError(
            f"Frame data for chain '{chain_tag}' frame {frame} must be 2D; got shape {values.shape}"
        )

    if values.shape[1] < 3:
        raise ValueError(
            "Frame data must contain at least three columns (x, y, likelihood)"
        )

    trimmed = values[:, :3]
    flat = trimmed.reshape(-1)
    if flat.size != expected:
        raise ValueError(
            "Frame data length does not match header columns for chain "
            f"'{chain_tag}' frame {frame}: expected {expected}, got {flat.size}"
        )

    return flat.astype(float, copy=False)


def export_chain_tracks(
    pickle_path: str | Path,
    *,
    output_path: str | Path | None = None,
    export_csv: bool = False,
) -> Optional[Path]:
    """Export reconstructed RFID identity chains into a DLC-style table.

    Parameters
    ----------
    pickle_path : str or :class:`~pathlib.Path`
        Path to the reconstructed tracklet pickle containing ``chain_tag``
        assignments.
    output_path : str or :class:`~pathlib.Path`, optional
        Destination for the generated ``.h5`` file.  When ``None`` (the
        default) the file is written alongside ``pickle_path`` using the naming
        pattern ``<stem>_rfid_tracks.h5``.
    export_csv : bool, optional
        When ``True`` a CSV mirror of the table is produced next to the ``.h5``
        file.  The default is ``False``.

    Returns
    -------
    :class:`~pathlib.Path` or ``None``
        Path to the created ``.h5`` file.  ``None`` is returned when the input
        pickle does not contain any tracklets with a ``chain_tag`` assignment.
    """

    pkl_path = Path(pickle_path)
    dd = load_tracklets_pickle(str(pkl_path))

    header = _as_multiindex(dd["header"])
    expected_len = len(header)

    frame_store: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)
    frames: set[int] = set()

    for tk, node in dd.items():
        if tk == "header" or not isinstance(node, MutableMapping):
            continue

        chain_tag = node.get("chain_tag")
        if not chain_tag:
            continue

        for key, value in node.items():
            match = _FRAME_RE.match(str(key))
            if not match:
                continue

            frame = int(match.group(1))
            frames.add(frame)

            arr = np.asarray(value)
            flat = _flatten_frame(arr, expected_len, chain_tag, frame)

            existing = frame_store[chain_tag].get(frame)
            if existing is not None:
                prev_llh = np.nanmean(existing[2::3])
                new_llh = np.nanmean(flat[2::3])
                if math.isnan(prev_llh) or new_llh > prev_llh:
                    logger.debug(
                        "Replacing frame %s for chain %s with higher likelihood sample", frame, chain_tag
                    )
                    frame_store[chain_tag][frame] = flat
                else:
                    logger.debug(
                        "Retaining existing frame %s for chain %s with higher likelihood sample", frame, chain_tag
                    )
                continue

            frame_store[chain_tag][frame] = flat

    if not frame_store:
        logger.warning("No chain_tag assignments found in %s; skipping export", pkl_path)
        return None

    frame_index = pd.Index(sorted(frames), name="frame")

    scorer_level = _get_level(header, "scorer", 0)
    bodypart_level = _get_level(header, "bodyparts", 1)
    coord_level = _get_level(header, "coords", header.nlevels - 1)

    parts: List[pd.DataFrame] = []
    for chain_tag, frame_map in sorted(frame_store.items()):
        values = np.full((len(frame_index), expected_len), np.nan, dtype=float)
        frame_to_pos = {frame: idx for idx, frame in enumerate(frame_index)}

        for frame, flat in frame_map.items():
            pos = frame_to_pos.get(frame)
            if pos is None:
                continue
            values[pos, :] = flat

        df_chain = pd.DataFrame(values, index=frame_index, columns=header)
        df_chain.columns = pd.MultiIndex.from_arrays(
            [
                scorer_level,
                pd.Index([chain_tag] * expected_len, name="individuals"),
                bodypart_level,
                coord_level,
            ],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
        parts.append(df_chain)

    export_df = pd.concat(parts, axis=1)

    if output_path is None:
        output_path = pkl_path.with_name(f"{pkl_path.stem}_rfid_tracks.h5")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_hdf(out_path, key="df_with_missing")

    if export_csv:
        csv_path = out_path.with_suffix(".csv")
        export_df.to_csv(csv_path)
        logger.info("Exported RFID tracks CSV: %s", csv_path)

    logger.info("Exported RFID tracks HDF5: %s", out_path)
    return out_path


__all__ = ["export_chain_tracks"]

