# RFID DLC 追踪项目

这是一个基于DeepLabCut和RFID技术的动物行为追踪分析项目，用于重建动物移动轨迹并生成可视化视频。

## 项目结构

```
project/
├── utils.py                              # 核心工具函数库
├── convert_detection2tracklets.py        # 检测结果转 tracklets
├── convert_detection2tracklets_config.yaml  # 默认参数
├── config.py                             # 默认路径与门控参数
├── reconstruct_from_pickle.py            # 轨迹重建脚本
├── make_video.py                         # 视频可视化脚本
├── roi_definitions.json                  # ROI区域定义文件
└── README.md                             # 项目说明
```

## 功能概述

### 1. 轨迹重建 (`reconstruct_from_pickle.py`)
- 基于RFID标签信息重建完整的动物移动轨迹
- 以已有RFID标签的tracklet为锚点
- 通过最近邻距离算法连接前后时间段的tracklet
- 输出重建后的pickle文件和链段信息CSV

**核心算法：**
- 锚点选择：选择有RFID标签且命中次数≥阈值的tracklet
- 双向扩展：从锚点向前/向后贪心连接距离最近的tracklet
- 时间约束：只连接时间间隔在指定范围内的tracklet

### 2. 视频可视化 (`make_video.py`)
- 在原视频上叠加各种追踪信息
- 支持显示tracklet轨迹、RFID事件、重建后的身份链
- 可选绘制读卡器位置和ROI区域
- 生成带有图例的可视化视频

**可视化元素：**
- Tracklet轨迹：彩色轨迹线和ID标签
- RFID事件：标签检测事件提示
- 身份链：重建后的完整轨迹
- 读卡器：RFID读卡器位置标记
- ROI区域：感兴趣区域边界

### 3. 工具函数库 (`utils.py`)
包含所有共用的核心函数：
- **数据I/O**：pickle文件加载/保存
- **DLC处理**：帧索引提取、身体中心计算
- **可视化**：颜色生成、读卡器绘制、ROI绘制
- **几何计算**：ROI命中测试、距离计算

## 使用方法

### 一键全流程分析
```python
from deeplabcut import run_rfid_pipeline

run_rfid_pipeline(
    config_path="path/to/config.yaml",
    video_path="path/to/video.mp4",
    rfid_csv="path/to/rfid.csv",
    centers_txt="path/to/readers_centers.txt",
    ts_csv="path/to/timestamps.csv",
    destfolder="path/to/output"  # 可选
)
```

该函数依次调用：

1. `deeplabcut.analyze_videos(..., auto_track=False)`
2. `deeplabcut.convert_detections2tracklets`
3. `deeplabcut.match_rfid_to_tracklets`
4. `deeplabcut.reconstruct_from_pickle`
5. `deeplabcut.make_video`

### 0. 检测结果转 tracklets
```bash
# 如需修改默认参数，先编辑 convert_detection2tracklets_config.yaml
python convert_detection2tracklets.py --config-path <项目config.yaml> --video-input <视频或目录>
```

常用选项：
- `--track-method`：`ellipse` / `skeleton` / `box`
- `--shuffle`：训练 shuffle 编号
- `--destfolder`：输出目录（默认跟随视频路径）
- `--videotype`：目录模式下的视频后缀

### 1. 轨迹重建
```bash
# 根据实际情况修改 config.py 中的路径与门控参数
python reconstruct_from_pickle.py
```

输出文件：
- 更新后的pickle文件（包含chain_tag和chain_id）
- `chain_segments.csv`：链段详细信息

### 2. 视频生成
```bash
# 根据实际情况修改 config.py 中的路径与可视化参数
python make_video.py
```

输出文件：
- `rfid_tracklets_overlay.mp4`：可视化视频

## 配置参数

`config.py` 集中定义了默认路径与门控参数，包含以下常见设置：

### 路径相关
- `PICKLE_IN` / `PICKLE_OUT`：输入和输出的 tracklet pickle 文件
- `VIDEO_PATH` / `OUTPUT_VIDEO`：原始视频与生成视频路径
- `CENTERS_TXT`：读卡器中心文件
- `ROI_FILE`：ROI 区域定义文件

### 门控与重建参数
- `FPS`：相机帧率
- `PX_PER_CM`：像素与厘米换算
- `V_GATE_CMS`：速度门限
- `PCUTOFF`：置信度阈值
- `HEAD_TAIL_SAMPLE`：头尾平均中心采样帧数
- `MAX_GAP_FRAMES`：最大时间间隔
- `ANCHOR_MIN_HITS`：锚点最少 RFID 命中数

### 可视化参数
- `TRAIL_LEN`：tracklet 轨迹长度
- `CHAIN_TRAIL_LEN`：身份链轨迹长度
- `TAG_HOLD_FRAMES`：RFID 标签显示持续帧数
- `MAX_FRAMES`：最大输出帧数（`None` 表示全部）

## 数据格式

### ROI定义文件 (JSON)
```json
{
  "Entrance1": [
    [35, 444],
    [162, 444],
    [162, 634],
    [35, 634]
  ]
}
```

### 读卡器中心文件 (TXT/CSV)
```
# 格式：row, col, x, y
0, 0, 100.5, 200.3
0, 1, 150.2, 200.1
...
```

## 重构改进

相比原版本的主要改进：
1. **集中配置**：新增 `config.py`，所有路径与门控参数统一管理
2. **统一工具函数**：所有共用函数集中在 `utils.py` 中
3. **删除冗余**：移除了不必要的复杂导入逻辑和 `__init__.py`
4. **清晰职责**：每个脚本专注单一功能
5. **改进文档**：添加详细的函数和参数说明

## 依赖库

- numpy
- pandas
- opencv-python
- pathlib (标准库)
- json (标准库)

## 注意事项

- 使用前需要根据实际数据路径修改 `config.py`
- 确保输入的 pickle 文件包含正确的 DLC tracklet 数据结构
- ROI 文件格式目前只支持 polygon 类型的 JSON 格式
