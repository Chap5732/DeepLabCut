# RFID DLC Tracking Project
# RFID DLC 追踪项目

This project combines DeepLabCut and RFID technology to reconstruct animal
trajectories and generate visualized videos.
本项目基于 DeepLabCut 和 RFID 技术，用于重建动物轨迹并生成可视化视频。

## Project Structure
## 项目结构

```
project/
├── utils.py                      # Core utility functions 核心工具函数库
├── reconstruct_from_pickle.py    # Trajectory reconstruction script 轨迹重建脚本
├── make_video.py                 # Video visualization script 视频可视化脚本
├── roi_definitions.json          # ROI region definitions ROI 区域定义
└── README.md                     # Project documentation 项目说明
```

## Features
## 功能概述

### 1. Trajectory Reconstruction (`reconstruct_from_pickle.py`)
### 1. 轨迹重建 (`reconstruct_from_pickle.py`)

- Reconstruct full animal trajectories using RFID tag information
- 基于 RFID 标签信息重建完整的动物轨迹
- Anchor tracklets with RFID tags and connect neighbors via nearest distance
- 以已有 RFID 标签的 tracklet 为锚点，通过最近邻距离连接前后 tracklet
- Output updated pickle and chain segment CSV
- 输出更新的 pickle 文件及链段信息 CSV

### 2. Video Visualization (`make_video.py`)
### 2. 视频可视化 (`make_video.py`)

- Overlay tracklet trajectories, RFID events, and reconstructed chains on the
  original video
- 在原视频上叠加 tracklet 轨迹、RFID 事件和重建链信息
- Optionally draw reader locations and ROI regions
- 可选绘制读卡器位置和 ROI 区域
- Produce videos with legend explanations
- 生成带图例的可视化视频

### 3. Utility Functions (`utils.py`)
### 3. 工具函数库 (`utils.py`)

Shared helper functions including:
包含的通用函数：
- **Data I/O** – load/save pickle files 数据 I/O：pickle 文件加载和保存
- **DLC handling** – frame index extraction, body center computation DLC 处理：帧索引提取、身体中心计算
- **Visualization** – color generation, reader and ROI drawing 可视化：颜色生成、读卡器和 ROI 绘制
- **Geometry** – ROI hit tests and distance utilities 几何计算：ROI 命中测试、距离计算

## Usage
## 使用方法

### 1. Trajectory Reconstruction
### 1. 轨迹重建

```bash
# Edit paths inside reconstruct_from_pickle.py
python reconstruct_from_pickle.py
```

Outputs:
输出文件：
- Updated pickle with `chain_tag` and `chain_id`
- 更新后的 pickle（包含 `chain_tag` 和 `chain_id`）
- `chain_segments.csv` with detailed chain information
- `chain_segments.csv`：链段详细信息

### 2. Video Generation
### 2. 视频生成

```bash
# Edit paths inside make_video.py
python make_video.py
```

Outputs:
输出文件：
- `rfid_tracklets_overlay.mp4` visualized video
- `rfid_tracklets_overlay.mp4` 可视化视频

## Configuration Parameters
## 配置参数

### Reconstruction
### 重建参数

- `PCUTOFF = 0.35` – confidence threshold 置信度阈值
- `HEAD_TAIL_SAMPLE = 5` – head/tail averaging frames 头尾平均帧数
- `MAX_GAP_FRAMES = 60` – maximum gap between tracklets 最大时间间隔
- `ANCHOR_MIN_HITS = 1` – minimum RFID hits for anchor 锚点最少 RFID 命中数

### Visualization
### 可视化参数

- `TRAIL_LEN = 15` – tracklet trail length tracklet 轨迹长度
- `CHAIN_TRAIL_LEN = 40` – identity chain trail length 身份链轨迹长度
- `TAG_HOLD_FRAMES = 3` – frames to display RFID tag RFID 标签显示帧数
- `MAX_FRAMES = None` – process all frames if `None` 最大输出帧数（`None` 表示全部）

## Data Formats
## 数据格式

### ROI Definition (JSON)
### ROI 定义文件 (JSON)

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

### Reader Centers File (TXT/CSV)
### 读卡器中心文件 (TXT/CSV)

```
# format: row, col, x, y
0, 0, 100.5, 200.3
0, 1, 150.2, 200.1
...
```

## Improvements Over Original
## 重构改进

1. **Simpler structure** – configuration hard-coded in scripts 简化结构：配置直接硬编码
2. **Unified utilities** – common functions consolidated in `utils.py` 统一工具函数：集中在 `utils.py`
3. **Removed redundancy** – unnecessary imports and `__init__.py` removed 删除冗余：移除多余导入和 `__init__.py`
4. **Clear responsibilities** – each script focuses on a single task 清晰职责：每个脚本专注单一功能
5. **Improved docs** – detailed function and parameter descriptions 改进文档：添加详细说明

## Dependencies
## 依赖库

- numpy
- pandas
- opencv-python
- pathlib (standard library)
