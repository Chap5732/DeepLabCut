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
