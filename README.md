
# 华南雷达回波短临预报数据集说明文档
**South China Radar Echo Dataset for Precipitation Nowcasting**

**创建日期**: 2025-11-19
**数据来源**: 华南地区雷达组合反射率拼图（涵盖多次台风过程）
**适用模型**: ConvLSTM, TrajGRU, PredRNN, SimVP 等时空序列预测模型
## 数据概述：
本数据集收录了华南地区整合雷达回波反射图2025.6.15日到2025.11.17 的32428张雷达回波图

关于雷达回波的科普：
### 雷达反射率回波图的核心作用

雷达反射率回波图（以dBZ为单位）是天气雷达最主要的产品，直接测量大气中降水粒子（雨滴、冰晶、冰雹等）对电磁波的反射强度，是气象学中监测和短时预报（0-2小时）降水与强对流天气的最关键工具。它能实时显示降水分布、强度和移动趋势，比卫星云图更精准地捕捉水汽凝结物，尤其适合灾害性天气预警。

**主要价值**包括：  
- 快速识别暴雨、冰雹、雷暴大风等强对流；  
- 支持短临降水估测（QPE）和外推预报（nowcasting）；  
- 揭示台风、飑线、雷暴等中尺度系统的内部结构，帮助预报员判断发展阶段和潜在威胁。

---

## 1. 数据集概览 (Statistics)

本数据集经过严格的空间清洗与时间序列化处理，旨在用于 **1小时输入 -> 1小时预测** 的任务。

*   **总序列样本数 (Samples)**: **20,140** 组
*   **序列长度 (Sequence Length)**: **20 帧** (前10帧输入，后10帧预测)
*   **时间分辨率**: 6 分钟/帧
*   **图像分辨率**: $512 \times 512$ (单通道灰度)
*   **存储格式**: PNG 图片 + TXT 索引 (Lazy Loading 模式)

**处理统计**:
*   **总中断切分段数**: 45 段 (遇到 >30分钟中断强制切分)
*   **光流插值补全帧数**: 331 帧 (文件名带有 `_interp` 后缀)

---

## 2. 目录结构 (File Structure)

请确保训练代码能够访问以下目录结构：

```text
processed_data/
├── images/                  # [核心数据] 存放约 32,000+ 张处理后的 PNG 图片
│   ├── 20250615_134800.png
│   ├── 20250615_135400.png
│   ├── ...
│   └── 20250615_141200_interp.png  # (光流法生成的插值帧)
│
└── train_list.txt           # [核心索引] 每一行代表一个样本，包含20个文件名
```

**索引文件格式 (`train_list.txt`)**:
每一行由逗号分隔的 20 个文件名组成：
```text
img1.png, img2.png, ..., img10.png, ..., img20.png
```

---

## 3. 预处理技术细节 (Preprocessing Methodology)

为了保证物理一致性并消除地图噪声，我们执行了以下 **V3版** 处理流水线：

### A. 空间域处理 (Spatial)
1.  **KDTree 颜色映射**: 精确将 RGB 映射为 dBZ 物理数值，剔除了背景杂色。
2.  **非对称形态学修复**: 针对大湾区密集的扁长形地名文字，采用了 **$4 \times 8$ 矩形卷积核** 进行闭运算，消除了“对流空洞”且未破坏云层结构。
3.  **中值滤波**: 去除了细碎的椒盐噪声。
4.  **等比例缩放**: 图像被缩放至 $512 \times 512$，边缘填充背景色，保证了台风螺旋结构的几何真实性。

### B. 时间域处理 (Temporal)
1.  **光流插值 (Optical Flow)**: 针对 $<30$ 分钟的微小缺失，使用 Farneback 算法生成插值帧，保证运动连贯性。
2.  **数据不平衡处理**: 自动丢弃了回波像素占比 $<5\%$ 的“晴空”序列，聚焦于降水事件。

---

## 4. 如何加载数据 (For Developers)

**重要提示**：
1.  图片已保存为 `uint8` (0-255) 的 PNG 格式以节省空间。**读取时必须除以 255.0 进行归一化。**
2.  读取时需要增加 Channel 维度，使其变为 `(Batch, 20, 1, 512, 512)`。

请直接在训练脚本中使用以下 `Dataset` 类：

```python
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np

class RadarLazyDataset(Dataset):
    def __init__(self, root_dir, list_file):
        """
        Args:
            root_dir (str): './processed_data/images' 的路径
            list_file (str): './processed_data/train_list.txt' 的路径
        """
        self.root_dir = root_dir
        self.sequences = []
        
        # 加载索引到内存
        with open(list_file, 'r') as f:
            for line in f:
                self.sequences.append(line.strip().split(','))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        file_names = self.sequences[idx]
        frames = []
        
        for fname in file_names:
            path = os.path.join(self.root_dir, fname)
            # 以灰度模式读取 (512, 512)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            frames.append(img)
            
        # 堆叠 -> float32
        seq_data = np.array(frames, dtype=np.float32) # Shape: (20, 512, 512)
        
        # 1. 归一化 (0-255 -> 0.0-1.0)
        seq_data = seq_data / 255.0
        
        # 2. 增加 Channel 维度 -> (20, 1, 512, 512)
        seq_data = np.expand_dims(seq_data, axis=1)
        
        # 3. 切分为 Input (前10) 和 Target (后10)
        input_seq = torch.from_numpy(seq_data[:10])
        target_seq = torch.from_numpy(seq_data[10:])
        
        return input_seq, target_seq

# --- 使用示例 ---
# dataset = RadarLazyDataset('./processed_data/images', './processed_data/train_list.txt')
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
```

---

## 5. 常见问题 (FAQ)

*   **Q: 为什么有些文件名带有 `_interp`？**
    *   A: 这些是光流法生成的插值帧。它们也是有效的训练数据，无需特殊处理，正常读取即可。
*   **Q: 显存不够怎么办？**
    *   A: 目前分辨率为 $512 \times 512$。如果 Batch Size=4 导致 OOM (Out of Memory)，请尝试将 Batch Size 降为 2 或 1，或者开启混合精度训练 (AMP)，或者用腾讯云平台上的跑。
*   **Q: 数据集加载速度慢？**
    *   A: 请确保 `processed_data` 文件夹位于 SSD (固态硬盘) 上，并增加 DataLoader 的 `num_workers` （多线程加载，注意使用方式）


## 6. 📂 代码文件结构说明 (Code Structure)

本项目包含雷达数据预处理、分析及评估的全套代码。由于原始雷达拼图数据（Raw Data）体积巨大（>25GB），**未包含在代码库中**。以下脚本仅供展示处理逻辑和算法实现细节。

### 核心代码功能
*   **`picture_analysis.py`**:
    *   **用途**: **[分析专用]** 用于对原始数据集进行时间轴扫描。
    *   **功能**: 检测时间序列的中断情况（缺失帧），统计缺失时长分布，并生成中断分析报告。这是制定插值和切分策略的依据。
    
*   **`data_process.py`**:
    *   **用途**: **[核心流水线]** 完整的数据预处理 ETL 脚本。
    *   **功能**: 包含 KDTree 颜色映射、非对称形态学修复（去除文字干扰）、光流法插值（Optical Flow）、序列切分及 Lazy Loading 索引生成。
    *   *注：需配合原始数据运行，此处仅作逻辑展示。*

*   **`pre_process_eval.py`**:
    *   **用途**: **[质量评估]** 用于对处理后的数据进行多维度“体检”。
    *   **产出**: 生成直方图 (`eval_histogram.png`)、时序平滑度曲线 (`eval_smoothness.png`) 及单样本演示 (`eval_sample.gif`)，验证数据清洗的物理一致性。

*   **`show_gif.py`** :
    *   **用途**: **[可视化展示]** 批量随机抽取处理好的序列，生成带有时间标注的高清 GIF 动图，用于直观感受数据质量。

### 📊 评估结果示例 (Artifacts)
*   `eval_histogram.png`: 像素分布直方图，展示数据的物理反射率分布。
*   `eval_smoothness.png`: 帧间差分图，验证时间序列的连贯性。
*   `eval_sample.gif`: 单个样本的动态预览。

---