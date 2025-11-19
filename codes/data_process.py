import os
import glob
import numpy as np
import cv2
import pandas as pd
from datetime import datetime, timedelta
from scipy.spatial import KDTree

# =================配置区域=================
CONFIG = {
    'raw_dir': './weather_radar_huanan_final',   # 原始图片文件夹
    'output_dir': './processed_data',   # 根目录
    'images_dir': 'images',             # 存放处理后PNG的子目录
    'list_file': 'train_list.txt',      # 存放序列索引的文件名
    
    'img_size': (512, 512),             # 目标尺寸
    'crop_roi': (0, 742, 231, 878),     # (y1, y2, x1, x2)
    'dbz_threshold': 15,                # 过滤阈值
    'max_gap_fill': 30,                 # 最大插值间隔
    'std_interval': 6,                  # 标准时间间隔
    'seq_len': 20,                      # 序列长度
    'min_echo_ratio': 0.05              # 最小回波占比
}

# =================颜色定义 (保持不变)=================
COLOR_MAP_RAW = {
    (255, 255, 255): 0, (0, 0, 0): 0, (200, 200, 200): 0, (171, 175, 171): 0,
    (1, 1, 1): 0, (203, 116, 191): 0, (141, 167, 178): 0, (52, 52, 52): 0,
    (105, 103, 103): 0, (109, 128, 138): 0, (179, 230, 255): 0, (76, 200, 255): 0,
    (255, 240, 225): 0, (204, 204, 204): 0, (229, 229, 229): 0, (57, 53, 51): 0,
    (177,187,177): 0, (116,212,253): 0, (68,189,255): 0, (120, 102, 56): 0,
    (109, 165, 145): 0, (72,72,63): 0, (143, 182, 105): 0, (88, 114, 158): 0,
    (95, 86, 96): 0, (112, 90, 54): 0, # 以上都是用于过滤背景内容
    # dBZ
    (65, 157, 241): 10, (100, 231, 235): 15, (109, 250, 61): 20, (0, 216, 0): 25,
    (1, 144, 0): 30, (255, 255, 0): 35, (231, 192, 0): 40, (255, 144, 0): 45,
    (255, 0, 0): 50, (214, 0, 0): 55, (192, 0, 0): 60, (255, 0, 240): 65,
    (150, 0, 180): 70
}

palette_colors = np.array(list(COLOR_MAP_RAW.keys()))
palette_values = np.array(list(COLOR_MAP_RAW.values()))
color_tree = KDTree(palette_colors)

# ================= 工具函数 (保持不变) =================
def rgb_to_dbz_kdtree(img_rgb):
    h, w, c = img_rgb.shape
    pixels = img_rgb.reshape(-1, 3)
    _, idx = color_tree.query(pixels, k=1)
    dbz_flat = palette_values[idx]
    return dbz_flat.reshape(h, w).astype(np.float32)

def resize_pad(img, target_size, pad_color=(255, 255, 255)):
    h, w = img.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    x_center = (target_w - new_w) // 2
    y_center = (target_h - new_h) // 2
    canvas[y_center:y_center+new_h, x_center:x_center+new_w] = resized
    return canvas

def optical_flow_interpolation(frame_prev, frame_next, num_frames_to_insert):
    prev_gray = frame_prev.astype(np.uint8)
    next_gray = frame_next.astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None, 
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    generated_frames = []
    h, w = frame_prev.shape
    grid_y, grid_x = np.mgrid[:h, :w].astype(np.float32)
    for i in range(1, num_frames_to_insert + 1):
        alpha = i / (num_frames_to_insert + 1)
        map_x = grid_x - flow[..., 0] * alpha
        map_y = grid_y - flow[..., 1] * alpha
        interp_frame = cv2.remap(frame_prev, map_x, map_y, cv2.INTER_LINEAR)
        generated_frames.append(interp_frame)
    return generated_frames

# ================= Dataset Builder (核心修改) =================

class RadarDatasetBuilder:
    def __init__(self):
        # 初始化目录
        self.save_dir = os.path.join(CONFIG['output_dir'], CONFIG['images_dir'])
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        self.list_path = os.path.join(CONFIG['output_dir'], CONFIG['list_file'])
        # 清空旧的列表文件
        open(self.list_path, 'w').close()
        
        self.stats = {"processed": 0, "interpolated": 0, "segments": 0, "sequences": 0}

    def save_frame(self, dbz_data, filename):
        """
        保存单帧图像为PNG。
        dbz_data: 0.0 - 1.0 的 float32
        保存为: 0 - 255 的 uint8 灰度图
        """
        # 转换为 0-255
        img_uint8 = (dbz_data * 255).astype(np.uint8)
        save_path = os.path.join(self.save_dir, filename)
        cv2.imwrite(save_path, img_uint8)
        
        # 返回非零像素数量，用于后续快速过滤，不用重新读图
        non_zero_count = np.count_nonzero(img_uint8)
        return filename, non_zero_count

    def load_and_clean(self, path):
        """读取并清洗，不保存，返回数据"""
        img = cv2.imread(path)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Crop
        y1, y2, x1, x2 = CONFIG['crop_roi']
        img = img[y1:y2, x1:x2]
        if img.size == 0: return None
        
        # 2. Resize & Pad
        img = resize_pad(img, CONFIG['img_size'])
        
        # 3. Mapping
        dbz = rgb_to_dbz_kdtree(img)
        
        # 4. Cleaning (Morphology 4x8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,8)) # 保持你的设置
        dbz = cv2.morphologyEx(dbz, cv2.MORPH_CLOSE, kernel)
        
        # 5. Denoise
        dbz = cv2.medianBlur(dbz, 3)

        # 6. Normalize
        dbz[dbz < CONFIG['dbz_threshold']] = 0
        dbz = dbz / 70.0
        
        return dbz

    def process(self):
        file_list = sorted(glob.glob(os.path.join(CONFIG['raw_dir'], "*.png")))
        if not file_list:
            print("No images found!")
            return

        print(f"Found {len(file_list)} images. Processing and saving to disk...")
        
        # current_segment 存储元组: (filename, data_float, non_zero_count)
        # data_float只保留最近的一帧用于插值，之前的可以扔掉以省内存
        current_segment_meta = [] # 只存 (filename, non_zero_count)
        
        last_time = None
        last_frame_data = None # 缓存上一帧数据用于光流
        
        for idx, fpath in enumerate(file_list):
            basename = os.path.basename(fpath)
            try:
                curr_time = datetime.strptime(basename.split('.')[0], "%Y%m%d_%H%M%S")
            except ValueError: continue

            curr_frame_data = self.load_and_clean(fpath)
            if curr_frame_data is None: continue
            
            # 生成当前帧的文件名 (直接用原名)
            curr_filename = basename
            
            if last_time is None:
                # 第一帧，保存并初始化
                fname, nz = self.save_frame(curr_frame_data, curr_filename)
                current_segment_meta.append((fname, nz))
            else:
                diff_min = (curr_time - last_time).total_seconds() / 60.0
                
                # A. 正常连续
                if abs(diff_min - CONFIG['std_interval']) < 1.0:
                    fname, nz = self.save_frame(curr_frame_data, curr_filename)
                    current_segment_meta.append((fname, nz))
                
                # B. 微小中断 -> 插值
                elif CONFIG['std_interval'] < diff_min <= CONFIG['max_gap_fill']:
                    missing_count = int(round(diff_min / CONFIG['std_interval'])) - 1
                    print(f"[Interp] {diff_min:.1f} min gap. +{missing_count} frames.")
                    
                    # 光流插值
                    prev_val = last_frame_data * 70.0
                    curr_val = curr_frame_data * 70.0
                    interp_frames = optical_flow_interpolation(prev_val, curr_val, missing_count)
                    
                    # 保存插值帧
                    for i, f in enumerate(interp_frames):
                        # 构造插值文件名: 原时间 + 间隔
                        interp_time = last_time + timedelta(minutes=CONFIG['std_interval'] * (i+1))
                        interp_name = interp_time.strftime("%Y%m%d_%H%M%S") + "_interp.png"
                        
                        f_norm = f / 70.0
                        fname, nz = self.save_frame(f_norm, interp_name)
                        current_segment_meta.append((fname, nz))
                    
                    # 保存当前帧
                    fname, nz = self.save_frame(curr_frame_data, curr_filename)
                    current_segment_meta.append((fname, nz))
                    self.stats["interpolated"] += missing_count
                
                # C. 严重中断 -> 结算旧段，开启新段
                else:
                    self.flush_segment(current_segment_meta)
                    current_segment_meta = [] # 清空列表
                    # 保存新段的第一帧
                    fname, nz = self.save_frame(curr_frame_data, curr_filename)
                    current_segment_meta.append((fname, nz))
                    
            last_time = curr_time
            last_frame_data = curr_frame_data # 更新缓存
            
            if idx % 50 == 0: print(f"Processed {idx}/{len(file_list)}...")

        # 循环结束，结算最后一段
        self.flush_segment(current_segment_meta)
        
        print("\n=== Processing Report ===")
        print(f"Saved Images Directory: {self.save_dir}")
        print(f"Sequence List File: {self.list_path}")
        print(f"Total Segments: {self.stats['segments']}")
        print(f"Total Interpolated Frames: {self.stats['interpolated']}")
        print(f"Total Valid Sequences Written: {self.stats['sequences']}")

    def flush_segment(self, segment_meta):
        """
        segment_meta: list of (filename, non_zero_count)
        不再处理图像数据，只处理文件名列表
        """
        seg_len = len(segment_meta)
        window = CONFIG['seq_len']
        if seg_len < window: return
        
        self.stats['segments'] += 1
        
        valid_sequences = []
        min_pixels = CONFIG['img_size'][0] * CONFIG['img_size'][1] * window * CONFIG['min_echo_ratio']
        
        with open(self.list_path, 'a') as f:
            for i in range(seg_len - window + 1):
                # 取出这20帧的元数据
                seq_meta = segment_meta[i : i + window]
                
                # 检查回波占比 (利用之前算好的 non_zero_count)
                total_nz = sum([item[1] for item in seq_meta])
                
                if total_nz < min_pixels:
                    continue # 丢弃空序列
                
                # 提取文件名并写入文件
                filenames = [item[0] for item in seq_meta]
                line = ",".join(filenames) + "\n"
                f.write(line)
                self.stats['sequences'] += 1

if __name__ == "__main__":
    builder = RadarDatasetBuilder()
    builder.process()