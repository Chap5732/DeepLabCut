# RFID 管线

该目录提供一套基于 **DeepLabCut** 与 **RFID** 的处理脚本，用于将检测结果转为带身份的轨迹并生成可视化视频。

## 文件结构

```
rfid_pipeline/
├── convert_detection2tracklets.py  # DLC 检测转为 tracklets
├── match_rfid_to_tracklets.py      # RFID 读数匹配 tracklets
├── reconstruct_from_pickle.py      # 依据 RFID 重建完整轨迹
├── make_video.py                   # 生成叠加信息的视频
├── utils.py                        # 共用工具函数
├── roi_definitions.json            # 示例 ROI 定义
└── README.md                       # 本说明文件
```

## 处理流程

1. **convert_detection2tracklets.py**：将 `analyze_videos` 的检测结果转换为 tracklets pickle。
2. **match_rfid_to_tracklets.py**：读取 RFID 数据并为每个 tracklet 统计命中次数、可选地分配标签。
3. **reconstruct_from_pickle.py**：以 RFID 标签为锚点重建轨迹链，写回 `chain_tag`/`chain_id` 等信息。
4. **make_video.py**：在原始视频上叠加轨迹、RFID 事件和重建结果，输出可视化视频。

各脚本均包含示例路径，使用前请根据实际数据位置修改。

## 工具函数

`utils.py` 集中实现了 pickle I/O、几何计算、颜色与可视化辅助函数等，可被上述脚本复用。

