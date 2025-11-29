import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
from lstm_cnn_model import LSTMCNN, SimpleViT

# ================= 配置参数 =================
CONFIG = {
    'images_dir': './processed_data/images',   # 图片目录
    'list_file': './processed_data/train_list.txt',  # 序列索引文件
    'img_size': (256, 256),                  # 图片尺寸
    'seq_len': 20,                          # 总序列长度
    'input_len': 10,                        # 输入序列长度
    'target_len': 10,                       # 目标序列长度
    'batch_size': 8,                        # 增加批次大小以提高GPU利用率
    'num_epochs': 10,                       # 训练轮数
    'learning_rate': 0.001,                 # 学习率
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 设备
    'checkpoint_dir': './checkpoints',       # 模型保存目录
    'use_amp': True,                        # 使用混合精度训练
    'model_type': 'vit',               # 模型类型: 'lstm_cnn' 或 'vit'
    'val_split': 0.1,                       # 验证集比例
    # 性能优化参数
    'num_workers': 6,                       # 数据加载工作线程数
    'pin_memory': True,                     # 固定内存以加速数据传输
    'persistent_workers': True,             # 保持工作线程存活
    'prefetch_factor': 2,                   # 预取因子
    'gradient_accumulation_steps': 1,       # 梯度累积步数
    # ViT模型参数
    'vit_embed_dim': 256,                   # ViT嵌入维度
    'vit_num_heads': 4,                     # ViT注意力头数
    'vit_depth': 3,                         # ViT层数
    'vit_dropout': 0.1                      # ViT dropout比例
}

# ================= RadarDataset 类 =================
class RadarLazyDataset(Dataset):
    def __init__(self, root_dir, list_file):
        """
        优化的雷达回波序列数据集懒加载实现
        
        Args:
            root_dir (str): 图片根目录路径
            list_file (str): 序列索引文件路径
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
        
        # 预先分配内存以提高效率
        frames = np.zeros((len(file_names), CONFIG['img_size'][0], CONFIG['img_size'][1]), dtype=np.float32)
        
        for i, fname in enumerate(file_names):
            path = os.path.join(self.root_dir, fname)
            # 以灰度模式读取
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # 容错处理 - 直接使用已分配的零值
                continue
            
            # 如果配置了不同尺寸，进行调整
            if img.shape[:2] != CONFIG['img_size']:
                img = cv2.resize(img, CONFIG['img_size'])
                
            # 直接归一化并存储，避免额外的数组创建
            frames[i] = img / 255.0
        
        # 增加 Channel 维度 -> (20, 1, 256, 256)
        seq_data = np.expand_dims(frames, axis=1)
        
        # 切分为 Input (前10) 和 Target (后10)
        input_seq = torch.from_numpy(seq_data[:CONFIG['input_len']])
        target_seq = torch.from_numpy(seq_data[CONFIG['input_len']:])
        
        return input_seq, target_seq

# ================= 模型初始化函数 =================
def get_model():
    """
    根据配置初始化模型
    """
    if CONFIG['model_type'] == 'lstm_cnn':
        # 初始化LSTM-CNN模型
        model = LSTMCNN(
            input_channels=1,
            encoder_dims=[8, 16, 32],  # 编码器通道配置
            convlstm_dims=[32],        # ConvLSTM隐藏层维度
            decoder_dims=[32, 16, 8],  # 解码器通道配置
            kernel_size=(3, 3)
        )
    elif CONFIG['model_type'] == 'vit':
        # 初始化优化版ViT模型
        model = SimpleViT(
            img_size=CONFIG['img_size'][0],
            patch_size=16,
            in_channels=1,
            embed_dim=CONFIG['vit_embed_dim'],  # 使用配置的嵌入维度
            num_heads=CONFIG['vit_num_heads'],  # 使用配置的头数
            depth=CONFIG['vit_depth'],          # 使用配置的层数
            out_channels=1,
            future_frames=CONFIG['target_len'],
            dropout=CONFIG['vit_dropout']        # 使用配置的dropout比例
        )
    else:
        raise ValueError(f"未知的模型类型: {CONFIG['model_type']}")
    
    # 尝试使用更快的层标准化实现（如果可用）
    try:
        from apex.normalization import FusedLayerNorm
        # 替换模型中的LayerNorm为FusedLayerNorm（如果使用apex）
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.LayerNorm):
                norm_layer = FusedLayerNorm(module.normalized_shape, module.eps)
                parent = model
                for part in name.split('.')[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, name.split('.')[-1], norm_layer)
        print("已使用FusedLayerNorm加速")
    except ImportError:
        print("未找到apex，继续使用标准LayerNorm")
    
    return model

# ================= 训练函数 =================
def train_model():
    # 创建保存目录
    if not os.path.exists(CONFIG['checkpoint_dir']):
        os.makedirs(CONFIG['checkpoint_dir'])
    
    # 设置设备
    device = torch.device(CONFIG['device'])
    print(f"使用设备: {device}")
    print(f"模型类型: {CONFIG['model_type']}")
    
    # 加载数据集
    print("加载数据集...")
    dataset = RadarLazyDataset(CONFIG['images_dir'], CONFIG['list_file'])
    
    # 分割训练集、验证集和测试集（8:1:1）
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # 使用random_split进行三部分分割
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器 - 使用优化的参数
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=CONFIG['persistent_workers'],
        prefetch_factor=CONFIG['prefetch_factor']
    )
    
    # 验证和测试使用较少的worker以节省资源
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=min(2, CONFIG['num_workers']),
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=CONFIG['persistent_workers']
    )
    
    # 创建测试集数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=min(2, CONFIG['num_workers']),
        pin_memory=CONFIG['pin_memory'],
        persistent_workers=CONFIG['persistent_workers']
    )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    print(f"Batch Size: {CONFIG['batch_size']}")
    
    # 初始化模型
    model = get_model()
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # 设置混合精度训练
    scaler = torch.cuda.amp.GradScaler() if CONFIG['use_amp'] and device.type == 'cuda' else None
    
    # 开始训练
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for i, (inputs, targets) in enumerate(tqdm(train_loader, desc="训练")):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # 梯度累积
            with torch.set_grad_enabled(True):
                # 使用混合精度训练
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets) / CONFIG['gradient_accumulation_steps']
                    
                    # 缩放损失并反向传播
                    scaler.scale(loss).backward()
                else:
                    # 普通训练
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / CONFIG['gradient_accumulation_steps']
                    loss.backward()
                
                # 只有在累积了足够的梯度后才更新参数
                if (i + 1) % CONFIG['gradient_accumulation_steps'] == 0 or (i + 1) == len(train_loader):
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * CONFIG['gradient_accumulation_steps'] * inputs.size(0)
            
            # 清理不需要的中间变量
            del inputs, targets, outputs, loss
            
            # 只在需要时清理缓存，避免频繁调用
            if i % 100 == 0:
                torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        train_loss /= len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        # 测试集评估
        test_loss = 0.0
        
        # 先验证
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="验证"):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                # 使用混合精度验证
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                
                # 清理不需要的中间变量
                del inputs, targets, outputs, loss
            
            # 再测试
            for inputs, targets in tqdm(test_loader, desc="测试"):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                # 使用混合精度测试
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
                
                # 清理不需要的中间变量
                del inputs, targets, outputs, loss
            
            # 验证和测试结束后清理缓存
            torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        val_loss /= len(val_loader.dataset)
        
        print(f"训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}, 测试损失: {test_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(CONFIG['checkpoint_dir'], f"best_{CONFIG['model_type']}_model.pth")
            torch.save(model.cpu().state_dict(), model_path)
            model.to(device)  # 保存后转回GPU
            print(f"保存最佳模型到 {model_path}，验证损失: {best_val_loss:.6f}")
        
        # 每10个epoch保存一次
        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(CONFIG['checkpoint_dir'], f"{CONFIG['model_type']}_model_epoch_{epoch+1}.pth")
            torch.save(model.cpu().state_dict(), model_path)
            model.to(device)  # 保存后转回GPU
            
        # 每个epoch结束后清理缓存
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.6f}")

if __name__ == "__main__":
    train_model()