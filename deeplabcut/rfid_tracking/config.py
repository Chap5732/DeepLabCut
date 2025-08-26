from pathlib import Path

# Base directory for RFID tracking scripts
BASE_DIR = Path(__file__).resolve().parent

# ================== 默认路径 ==================
# 路径可根据实际项目调整
PICKLE_IN = "/ssd01/user_acc_data/oppa/deeplabcut/projects/MiceTrackerFor20-Oppa-2024-12-08/analyze_videos/shuffle3/demo1/velocity_gating/demoDLC_HrnetW32_MiceTrackerFor20Dec8shuffle3_detector_best-250_snapshot_best-190_el.pickle"
PICKLE_OUT = None            # None 表示覆盖输入
OUT_SUBDIR = "CAP15"          # 输出子目录; 设为 None 则直接写入同级目录

VIDEO_PATH = "/ssd01/user_acc_data/oppa/deeplabcut/videos/test/demo.mp4"
PICKLE_PATH = PICKLE_IN      # make_video 默认使用同一 pickle
OUTPUT_VIDEO = "/ssd01/user_acc_data/oppa/deeplabcut/projects/MiceTrackerFor20-Oppa-2024-12-08/analyze_videos/shuffle3/demo1/velocity_gating/CAP15/demo_tracked.mp4"

CENTERS_TXT = "/ssd01/user_acc_data/oppa/analysis/data/jc0813/readers_centers.txt"
ROI_FILE = "/ssd01/user_acc_data/oppa/analysis/rfid_dlc_tracking/version2_tracking/roi_definitions.json"

# convert_detection2tracklets 默认参数文件
CONVERT_DEFAULTS = BASE_DIR / "convert_detection2tracklets_config.yaml"

# ================== 门控与重建参数 ==================
FPS = 30.0                # 帧率 (frames/second)
PX_PER_CM = 14.0          # 像素与厘米换算
V_GATE_CMS = 80.0         # 最大速度门限 (cm/s)

PCUTOFF = 0.35
HEAD_TAIL_SAMPLE = 5
MAX_GAP_FRAMES = 60
ANCHOR_MIN_HITS = 1

EPS_GAP = 0.5
DELTA_PX_CAP = 15.0
DELTA_PROP = 0.10

STOP_NEAR_ANCHOR = True
RESET_PREVIOUS = True
LOG_RUN_METADATA = True

# ================== 可视化参数 ==================
TRAIL_LEN = 15
TAG_HOLD_FRAMES = 3

SHOW_CHAIN = True
CHAIN_FALLBACK_ID = True
CHAIN_TRAIL_LEN = 40
CHAIN_LINE_THICK = 3
CHAIN_POINT_R = 5

DRAW_LEGEND = True
LEGEND_COLS = 2
LEGEND_POS = (20, 40)

DRAW_READERS = True
DRAW_ROIS = True

MAX_FRAMES = None

# ================== match_rfid_to_tracklets 默认参数 ==================
# 路径可在此修改，或通过 load_mrt_config 从 YAML 覆盖
MRT_PICKLE_PATH = PICKLE_IN
MRT_RFID_CSV = "/ssd01/user_acc_data/oppa/analysis/data/jc0813/rfid_data_20250813_055827.csv"
MRT_CENTERS_TXT = CENTERS_TXT
MRT_TS_CSV = "/ssd01/user_acc_data/oppa/analysis/data/jc0813/record_20250813_053913_timestamps.csv"
MRT_OUT_DIR = None  # None -> 与 pickle 同目录创建 rfid_match_outputs/

MRT_N_ROWS = 12
MRT_N_COLS = 12
MRT_ID_BASE = 0
MRT_Y_TOP_TO_BOTTOM = True

MRT_PCUTOFF = 0.35
MRT_RFID_FRAME_RANGE = 10
MRT_COIL_DIAMETER_PX = 170.0
MRT_HIT_MARGIN = 1.00
MRT_HIT_RADIUS_PX = (MRT_COIL_DIAMETER_PX / 2.0) * MRT_HIT_MARGIN

MRT_UNIQUE_NEIGHBOR_ONLY = True
MRT_AMBIG_MARGIN_PX = 75.0

MRT_LOW_FREQ_TAG_MIN_COUNT = 2
MRT_MIN_VALID_FRAMES_PER_TK = 1

MRT_TAG_CONFIDENCE_THRESHOLD = 0.70
MRT_TAG_MIN_READS = 20
MRT_TAG_DOMINANT_RATIO = 3.0
MRT_LOW_READS_HIGH_PURITY_ASSIGN = True
MRT_LOW_READS_PURITY_THRESHOLD = 0.90

MRT_USE_FRAME_STABILITY_CHECK = False
MRT_BURST_GAP_FRAMES = 150
MRT_MIN_BURSTS_IF_LOWHITS = 2
MRT_LOWHITS_THRESHOLD = 200


def load_mrt_config(yaml_path: str) -> None:
    """Override MRT_* defaults from a YAML file."""
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    for key, value in data.items():
        key = key if key.startswith("MRT_") else f"MRT_{key}".upper()
        if key in globals():
            globals()[key] = value

