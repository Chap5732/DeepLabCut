import pickle
from pathlib import Path

import numpy as np

import importlib.util
import sys
import types


def _import_mrt() -> types.ModuleType:
    root = Path(__file__).resolve().parents[1]
    dlc_pkg = types.ModuleType("deeplabcut")
    dlc_pkg.__path__ = [str(root / "deeplabcut")]
    sys.modules.setdefault("deeplabcut", dlc_pkg)

    rfid_pkg = types.ModuleType("deeplabcut.rfid_tracking")
    rfid_pkg.__path__ = [str(root / "deeplabcut/rfid_tracking")]
    sys.modules.setdefault("deeplabcut.rfid_tracking", rfid_pkg)

    spec = importlib.util.spec_from_file_location(
        "deeplabcut.rfid_tracking.match_rfid_to_tracklets",
        root / "deeplabcut/rfid_tracking/match_rfid_to_tracklets.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


mrt = _import_mrt()


def _write_simple_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text("\n".join([header] + rows))


def test_resume_after_partial_pickle(tmp_path):
    centers_txt = tmp_path / "centers.txt"
    centers_txt.write_text("0 0\n1 0\n")

    rfid_csv = tmp_path / "rfid.csv"
    _write_simple_csv(rfid_csv, "time,tag,id", ["0,A,0", "1,A,0"])

    ts_csv = tmp_path / "timestamps.csv"
    _write_simple_csv(ts_csv, "frame,time", ["0,0", "1,1"])

    pickle_path = tmp_path / "tracklets.pickle"
    dd = {
        "header": {},
        0: {0: np.array([[0.0, 0.0, 1.0]])},
        1: "corrupted",
    }
    with open(pickle_path, "wb") as f:
        pickle.dump(dd, f)

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text("\n".join([
        "MRT_N_ROWS: 1",
        "MRT_N_COLS: 2",
        "MRT_LOW_FREQ_TAG_MIN_COUNT: 0",
    ]))

    out_dir = tmp_path / "out"
    mrt.main(
        pickle_path=str(pickle_path),
        rfid_csv=str(rfid_csv),
        centers_txt=str(centers_txt),
        ts_csv=str(ts_csv),
        out_dir=str(out_dir),
        config_path=str(config_yaml),
    )

    with open(pickle_path, "rb") as f:
        dd2 = pickle.load(f)
    assert dd2[1] == "corrupted"
