import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import Dataset, DataLoader

# ================= 配置区域 =================
CONFIG = {
    'images_dir': './processed_data/images',       # PNG 图片目录
    'list_file': './processed_data/train_list.txt', # 序列索引文件
    'sample_count_for_stats': 100,                 # 用于统计分布的样本数
    'batch_size_speed_test': 16,                   # 测速用的 Batch Size
    'img_size': (512, 512)
}

# ================= 0. 复用 Lazy Dataset 类 =================
class RadarEvalDataset(Dataset):
    def __init__(self, root_dir, list_file):
        self.root_dir = root_dir
        self.sequences = []
        
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"索引文件不存在: {list_file}")
            
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
            # 读取为灰度图 (0-255)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # 容错处理：如果图片损坏，生成全黑图避免报错，但打印警告
                print(f"Warning: Corrupted image {path}")
                img = np.zeros(CONFIG['img_size'], dtype=np.uint8)
            frames.append(img)
        
        # 转为 Float32 (0.0 - 1.0)
        seq_data = np.array(frames, dtype=np.float32) / 255.0
        return seq_data  # Shape: (20, 512, 512)

# ================= 1. 视觉图灵测试 (生成 GIF) =================
def generate_gif(dataset, sample_idx, output_name="eval_sample.gif"):
    print(f"\n[1/4] 生成动态 GIF 样本 (Index: {sample_idx})...")
    try:
        seq = dataset[sample_idx] # (20, 512, 512)
    except IndexError:
        print("样本索引越界，跳过。")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    
    # 使用 jet 颜色表模拟雷达图，vmin=0(背景), vmax=1(强回波)
    im = ax.imshow(seq[0], cmap='jet', vmin=0, vmax=1, animated=True)
    
    title = ax.text(0.5, 1.01, "", transform=ax.transAxes, ha="center")

    def update(frame_idx):
        im.set_array(seq[frame_idx])
        # 前10帧是Input，后10帧是Target
        phase = "INPUT (Past)" if frame_idx < 10 else "TARGET (Future)"
        title.set_text(f"Sample {sample_idx} | Frame {frame_idx+1}/20 | {phase}")
        return [im, title]
    
    ani = animation.FuncAnimation(fig, update, frames=20, interval=200, blit=True) # 5fps
    ani.save(output_name, writer='pillow', fps=5)
    print(f" -> GIF 已保存至: {output_name}")
    print("    请检查：1.文字是否去除干净？ 2.云团移动是否平滑？ 3.是否有闪烁的黑块？")

# ================= 2. 物理分布检查 (直方图) =================
def check_statistics(dataset):
    print(f"\n[2/4] 统计像素分布 (抽样 {CONFIG['sample_count_for_stats']} 个序列)...")
    
    pixel_values = []
    count = min(len(dataset), CONFIG['sample_count_for_stats'])
    
    for i in range(count):
        seq = dataset[i]
        # 只统计非零像素（排除背景），否则直方图会被0淹没
        valid = seq[seq > 0.01].flatten()
        if len(valid) > 0:
            # 为了节省内存，每个样本只随机抽 1000 个点
            sample = np.random.choice(valid, min(len(valid), 1000))
            pixel_values.extend(sample)
            
    if not pixel_values:
        print(" -> 警告：采样数据全为 0，数据集可能是空的或全黑！")
        return

    plt.figure(figsize=(10, 5))
    plt.hist(pixel_values, bins=50, color='blue', alpha=0.7, log=True)
    plt.title("Radar Reflectivity Distribution (Log Scale)")
    plt.xlabel("Normalized Value (0.0 - 1.0)")
    plt.ylabel("Frequency (Log)")
    plt.grid(True, alpha=0.3)
    plt.savefig("eval_histogram.png")
    print(" -> 分布图已保存至: eval_histogram.png")
    print("    标准：应该呈长尾分布。如果有特定数值的异常尖峰，可能是某个地图颜色没洗干净。")

# ================= 3. 时序平滑度 (检测闪烁/断裂) =================
def check_temporal_smoothness(dataset, sample_idx):
    print(f"\n[3/4] 检查时序平滑度 (Index: {sample_idx})...")
    seq = dataset[sample_idx] # (20, 512, 512)
    
    diffs = []
    # 计算相邻帧的平均绝对误差 (MAE)
    for i in range(19):
        diff = np.mean(np.abs(seq[i+1] - seq[i]))
        diffs.append(diff)
        
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, 20), diffs, marker='o', linestyle='-', color='red')
    plt.axvline(x=10, color='green', linestyle='--', label='Input/Target Split')
    plt.title(f"Frame-to-Frame Difference (Sample {sample_idx})")
    plt.xlabel("Frame Transition (t -> t+1)")
    plt.ylabel("Mean Absolute Difference")
    plt.legend()
    plt.savefig("eval_smoothness.png")
    print(" -> 曲线图已保存至: eval_smoothness.png")
    print("    标准：曲线应平缓波动。如果第10帧附近有巨型尖峰，说明序列切分不连续。")

# ================= 4. I/O 性能压力测试 =================
def benchmark_io(dataset):
    print(f"\n[4/4] I/O 性能压力测试 (Lazy Loading)...")
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size_speed_test'], shuffle=False, num_workers=4)
    
    start_time = time.time()
    total_batches = 5
    
    print(f" -> 尝试读取 {total_batches} 个 Batch (Batch Size={CONFIG['batch_size_speed_test']})...")
    for i, batch in enumerate(loader):
        # 这里 batch 是 (B, 20, 512, 512)
        if i >= total_batches:
            break
        print(f"    Batch {i+1} Loaded. Shape: {batch.shape}")
        
    end_time = time.time()
    total_time = end_time - start_time
    img_per_sec = (total_batches * CONFIG['batch_size_speed_test'] * 20) / total_time
    
    print(f" -> 耗时: {total_time:.2f} 秒")
    print(f" -> 吞吐量: {img_per_sec:.1f} Frames/sec")
    
    if img_per_sec > 100:
        print(" -> 评价：I/O 速度优秀，足以支撑 GPU 训练。")
    else:
        print(" -> 评价：I/O 速度较慢，建议把数据放在固态硬盘(SSD)上。")

# ================= 主程序 =================
if __name__ == "__main__":
    if not os.path.exists(CONFIG['list_file']):
        print("错误：找不到 train_list.txt，请先运行预处理脚本！")
    else:
        # 初始化数据集
        ds = RadarEvalDataset(CONFIG['images_dir'], CONFIG['list_file'])
        print(f"数据集加载成功，共有 {len(ds)} 个可用序列。")
        
        if len(ds) > 0:
            # 随机选一个样本进行可视化
            rand_idx = np.random.randint(0, len(ds))
            
            generate_gif(ds, rand_idx)
            check_statistics(ds)
            check_temporal_smoothness(ds, rand_idx)
            benchmark_io(ds)
            
            print("\n=== 评估结束 ===")
        else:
            print("数据集为空！请检查预处理逻辑。")