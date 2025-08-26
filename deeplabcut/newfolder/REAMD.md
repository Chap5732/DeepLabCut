# RFID DLC 追踪项目

这是一个基于DeepLabCut和RFID技术的动物行为追踪分析项目，用于重建动物移动轨迹并生成可视化视频。

## 项目结构

```
project/
├── utils.py                      # 核心工具函数库
├── convert_detection2tracklets.py # 检测结果转 tracklet 脚本
├── match_rfid_to_tracklets.py    # RFID 匹配脚本
├── reconstruct_from_pickle.py    # 轨迹重建脚本
├── make_video.py                 # 视频可视化脚本
├── roi_definitions.json          # ROI区域定义文件
└── README.md                     # 项目说明
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

### 1. 检测结果转 tracklet
```bash
python convert_detection2tracklets.py --config config.yaml --video-input demo.mp4 --destfolder ./tracks/
```

### 2. RFID 匹配
```bash
python match_rfid_to_tracklets.py --pickle tracklets.pickle --rfid-csv rfid.csv --centers-txt readers_centers.txt --timestamps-csv timestamps.csv
```

### 3. 轨迹重建
```bash
python reconstruct_from_pickle.py --pickle-in tracklets.pickle --out-subdir CAP15
```

输出文件：
- 更新后的pickle文件（包含chain_tag和chain_id）
- `chain_segments.csv`：链段详细信息

### 4. 视频生成
```bash
python make_video.py --video demo.mp4 --pickle tracklets.pickle --output-video rfid_tracklets_overlay.mp4
```

输出文件：
- `rfid_tracklets_overlay.mp4`：可视化视频

## 配置参数

### 重建参数
- `PCUTOFF = 0.35`：置信度阈值
- `HEAD_TAIL_SAMPLE = 5`：头尾平均中心采样帧数
- `MAX_GAP_FRAMES = 60`：最大时间间隔
- `ANCHOR_MIN_HITS = 1`：锚点最少RFID命中数

### 可视化参数
- `TRAIL_LEN = 15`：tracklet轨迹长度
- `CHAIN_TRAIL_LEN = 40`：身份链轨迹长度
- `TAG_HOLD_FRAMES = 3`：RFID标签显示持续帧数
- `MAX_FRAMES = None`：最大输出帧数（None=全部）

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
1. **简化结构**：删除了config.py，配置直接硬编码在各脚本中
2. **统一工具函数**：所有共用函数集中在utils.py中
3. **删除冗余**：移除了不必要的复杂导入逻辑和`__init__.py`
4. **清晰职责**：每个脚本专注单一功能
5. **改进文档**：添加详细的函数和参数说明

## 依赖库

- numpy
- pandas  
- opencv-python
- pathlib (标准库)
- json (标准库)

## 注意事项

- 使用前需要根据实际数据路径修改各脚本中的路径配置
- 确保输入的pickle文件包含正确的DLC tracklet数据结构
- ROI文件格式目前只支持polygon类型的JSON格式