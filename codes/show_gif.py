import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# ================= 配置区域 =================
CONFIG = {
    'images_dir': './processed_data/images',       # 处理后的PNG文件夹
    'list_file': './processed_data/train_list.txt', # 序列索引文件
    'save_dir': './vis_results',                   # GIF保存路径
    'num_save': 5,                                 # 要随机生成多少个GIF
    'fps': 5,                                      # 播放速度 (帧/秒)
    'img_size': (512, 512)
}

def load_sequence(filenames, root_dir):
    """读取一个序列的所有图片"""
    frames = []
    for fname in filenames:
        path = os.path.join(root_dir, fname)
        # 读取灰度图 (0-255)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # 容错：生成全黑帧
            img = np.zeros(CONFIG['img_size'], dtype=np.uint8)
        frames.append(img)
    # 转为 0.0-1.0 用于显示
    return np.array(frames, dtype=np.float32) / 255.0

def create_gif(seq_data, seq_idx, save_path):
    """生成美观的 GIF 动图"""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=0.9, wspace=None, hspace=None)
    ax.axis('off')
    
    # 使用 jet 颜色映射，模拟雷达回波风格
    im = ax.imshow(seq_data[0], cmap='jet', vmin=0, vmax=1, animated=True)
    
    # 添加标题文本对象
    title_text = ax.text(0.5, 1.02, "", transform=ax.transAxes, ha="center", 
                         fontsize=12, fontweight='bold', color='black')

    def update(frame_idx):
        im.set_array(seq_data[frame_idx])
        
        # 区分输入阶段(0-9)和预测阶段(10-19)
        if frame_idx < 10:
            phase = "INPUT (Past 1H)"
            color = "blue"
        else:
            phase = "PREDICTION (Future 1H)"
            color = "red"
            
        title_text.set_text(f"Sample {seq_idx} | T={frame_idx+1} | {phase}")
        title_text.set_color(color)
        
        return [im, title_text]

    # 生成动画
    ani = animation.FuncAnimation(fig, update, frames=len(seq_data), interval=200, blit=True)
    
    # 保存
    ani.save(save_path, writer='pillow', fps=CONFIG['fps'])
    plt.close(fig) # 关闭画布释放内存
    print(f" -> Saved: {save_path}")

def main():
    # 1. 检查路径
    if not os.path.exists(CONFIG['list_file']):
        print(f"错误：找不到索引文件 {CONFIG['list_file']}")
        return
    
    if not os.path.exists(CONFIG['save_dir']):
        os.makedirs(CONFIG['save_dir'])
        print(f"创建输出目录: {CONFIG['save_dir']}")

    # 2. 加载所有序列索引
    print("正在读取序列索引...")
    with open(CONFIG['list_file'], 'r') as f:
        all_sequences = [line.strip().split(',') for line in f]
    
    total_seqs = len(all_sequences)
    print(f"共有 {total_seqs} 个可用序列。")
    
    if total_seqs < CONFIG['num_save']:
        print("警告：可用序列少于请求生成的数量。")
        indices = range(total_seqs)
    else:
        # 随机抽取
        indices = random.sample(range(total_seqs), CONFIG['num_save'])

    # 3. 循环生成
    print(f"开始生成 {len(indices)} 个随机演示 GIF ...")
    
    for i, idx in enumerate(indices):
        filenames = all_sequences[idx]
        
        print(f"[{i+1}/{len(indices)}] 处理序列 ID: {idx} ...")
        seq_data = load_sequence(filenames, CONFIG['images_dir'])
        
        save_name = os.path.join(CONFIG['save_dir'], f"demo_sample_{idx}.gif")
        create_gif(seq_data, idx, save_name)

    print("\n=== 全部完成！请在 vis_results 文件夹中查看 ===")

if __name__ == "__main__":
    main()