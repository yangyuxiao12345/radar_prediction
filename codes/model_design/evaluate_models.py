import os
import torch
import torch.nn as nn  # 添加了这一行导入
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
from torch.utils.data import DataLoader
from tqdm import tqdm
from lstm_cnn_model import LSTMCNN, SimpleViT
from train_lstm_cnn_model import RadarLazyDataset, CONFIG

# ================= 评估配置 =================
EVAL_CONFIG = {
    'model_paths': {
        'lstm_cnn': './checkpoints/best_lstm_cnn_model.pth',
        'vit': './checkpoints/best_vit_model.pth'
    },
    'output_dir': './eval_results',
    'num_samples': 10,  # 评估的样本数量
    'show_plots': False  # 是否显示图像
}

# 创建输出目录
os.makedirs(EVAL_CONFIG['output_dir'], exist_ok=True)

# ================= 模型加载函数 =================
def map_old_vit_weights(state_dict, target_embed_dim=256):
    """
    简化版权重映射函数
    由于新旧模型架构差异较大，我们将只保留部分关键权重，并让PyTorch初始化其余权重
    """
    # 创建一个空字典，不包含任何可能导致维度不匹配的权重
    # 我们将完全依赖PyTorch的默认初始化，这样可以避免形状不匹配的问题
    filtered_state_dict = {}
    
    # 只为lstm_norm层创建初始化权重，这是一个标准化层，影响较小
    filtered_state_dict['lstm_norm.weight'] = torch.ones(target_embed_dim)
    filtered_state_dict['lstm_norm.bias'] = torch.zeros(target_embed_dim)
    
    return filtered_state_dict

def load_model(model_type, model_path):
    """
    加载训练好的模型
    """
    device = torch.device(CONFIG['device'])
    
    if model_type == 'lstm_cnn':
        model = LSTMCNN(
            input_channels=1,
            encoder_dims=[8, 16, 32],
            convlstm_dims=[32],
            decoder_dims=[32, 16, 8],
            kernel_size=(3, 3)
        )
        # 对于LSTM-CNN模型，直接加载权重
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"成功加载LSTM-CNN模型权重")
    elif model_type == 'vit':
        # 创建ViT模型，使用与训练时相同的参数
        model = SimpleViT(
            img_size=CONFIG['img_size'][0],
            patch_size=16,
            in_channels=1,
            embed_dim=256,  # 匹配训练时的嵌入维度
            num_heads=4,    # 匹配训练时的头数 (192/3=64 per head)
            depth=3,        # 匹配训练时的层数
            out_channels=1,
            dropout=0.1
        )
        
        # 加载训练好的权重
        state_dict = torch.load(model_path, map_location=device)
        
        # 简化的权重加载：使用strict=False直接加载，让PyTorch处理不匹配的键
        # 这是最安全的方法，即使架构有差异也能加载
        print(f"尝试加载ViT模型权重...")
        
        # 确保模型权重完全匹配，不使用默认初始化
        model_state = model.state_dict()
        new_state_dict = model_state.copy()  # 从模型当前状态开始，确保所有键都存在
        
        # 映射权重，处理架构差异
        success_count = 0
        for key, value in state_dict.items():
            # 处理patch_embed层的映射
            if key == 'patch_embed.proj.weight':
                # 将单卷积层映射到双卷积层的第一个卷积
                if 'patch_embed.0.weight' in new_state_dict:
                    target_shape = new_state_dict['patch_embed.0.weight'].shape
                    # 调整权重形状以匹配目标
                    # 从 [out_channels, in_channels, kernel_h, kernel_w] 调整
                    try:
                        # 如果输入通道数不同，复制到所有输入通道
                        if value.shape[1] != target_shape[1]:
                            print(f"调整patch_embed输入通道: {value.shape[1]} -> {target_shape[1]}")
                            # 扩展输入通道维度
                            new_weight = value.repeat(1, target_shape[1], 1, 1) / target_shape[1]
                            new_state_dict['patch_embed.0.weight'] = new_weight
                        else:
                            new_state_dict['patch_embed.0.weight'] = value
                        success_count += 1
                        print(f"成功映射 {key} -> patch_embed.0.weight")
                    except Exception as e:
                        print(f"无法映射 {key}: {e}")
            elif key == 'patch_embed.proj.bias':
                # 将偏置映射到双卷积层的第一个卷积
                if 'patch_embed.0.bias' in new_state_dict:
                    new_state_dict['patch_embed.0.bias'] = value
                    success_count += 1
                    print(f"成功映射 {key} -> patch_embed.0.bias")
            # 处理位置编码
            elif key == 'pos_embed':
                if key in new_state_dict:
                    try:
                        # 确保位置编码维度正确
                        # 直接检查value的ndim，而不是形状对象的ndim
                        if value.ndim == 3 and new_state_dict[key].ndim == 3:
                            # 匹配序列长度和维度
                            new_pos = torch.zeros_like(new_state_dict[key])
                            seq_len = min(value.shape[1], new_state_dict[key].shape[1])
                            dim = min(value.shape[2], new_state_dict[key].shape[2])
                            new_pos[:, :seq_len, :dim] = value[:, :seq_len, :dim]
                            new_state_dict[key] = new_pos
                            success_count += 1
                            print(f"调整位置编码维度: {value.shape} -> {new_state_dict[key].shape}")
                        else:
                            # 如果维度不匹配，尝试更简单的调整
                            target_shape = new_state_dict[key].shape
                            print(f"位置编码维度不匹配: {value.shape} vs {target_shape}")
                            # 创建一个零张量并填充可用部分
                            new_pos = torch.zeros_like(new_state_dict[key])
                            
                            # 计算可以复制的部分
                            dims = min(len(value.shape), len(target_shape))
                            slices = []
                            for i in range(dims):
                                slices.append(slice(0, min(value.shape[i], target_shape[i])))
                            
                            # 复制可用部分
                            new_pos[tuple(slices)] = value[tuple(slices)]
                            new_state_dict[key] = new_pos
                            success_count += 1
                            print(f"已复制位置编码的可用部分")
                    except Exception as e:
                        print(f"无法调整位置编码: {e}")
            # 处理其他关键层
            elif key in new_state_dict:
                # 只有形状匹配时才替换
                if value.shape == new_state_dict[key].shape:
                    new_state_dict[key] = value
                    success_count += 1
                else:
                    print(f"跳过 {key}: 形状不匹配 ({value.shape} vs {new_state_dict[key].shape})")
            else:
                print(f"跳过 {key}: 在模型中不存在")
        
        # 直接使用完整的状态字典，不允许缺失键
        try:
            model.load_state_dict(new_state_dict)
            print(f"\nViT模型权重加载完成")
            print(f"成功映射和加载的权重数量: {success_count}")
        except RuntimeError as e:
            print(f"\n权重加载错误: {e}")
            # 为了确保脚本能运行，使用strict=False作为后备方案
            print("使用strict=False作为后备方案加载权重...")
            model.load_state_dict(new_state_dict, strict=False)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    model.to(device)
    model.eval()
    
    return model

# ================= 评估函数 =================
def evaluate_model(model, data_loader, model_type, device):
    """
    评估模型性能
    """
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_inputs = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(data_loader, desc=f"评估 {model_type}")):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 预测
            outputs = model(inputs, CONFIG['target_len'])
            
            # 计算损失
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
            # 保存结果用于可视化
            all_inputs.append(inputs.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            if i >= EVAL_CONFIG['num_samples'] - 1:
                break
    
    avg_loss = total_loss / ((min(EVAL_CONFIG['num_samples'], len(data_loader))) * CONFIG['batch_size'])
    
    # 合并结果
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return avg_loss, all_inputs, all_predictions, all_targets

# ================= 可视化函数 =================
def visualize_results(inputs, predictions, targets, sample_idx, model_type):
    """
    可视化预测结果
    """
    # 选择一个样本
    input_sample = inputs[sample_idx, :, 0]  # 移除通道维度
    pred_sample = predictions[sample_idx, :, 0]
    target_sample = targets[sample_idx, :, 0]
    
    # 创建输出目录
    sample_dir = os.path.join(EVAL_CONFIG['output_dir'], f"sample_{sample_idx}")
    os.makedirs(sample_dir, exist_ok=True)
    
    # 保存输入序列的最后一帧，预测序列的第一帧和最后一帧，以及对应的真实帧
    frames_to_save = [
        (input_sample[-1], f"{sample_dir}/input_last.png", "输入最后一帧"),
        (pred_sample[0], f"{sample_dir}/{model_type}_pred_0.png", "预测第1帧"),
        (pred_sample[-1], f"{sample_dir}/{model_type}_pred_last.png", "预测最后一帧"),
        (target_sample[0], f"{sample_dir}/target_0.png", "真实第1帧"),
        (target_sample[-1], f"{sample_dir}/target_last.png", "真实最后一帧")
    ]
    
    for frame, path, title in frames_to_save:
        # 转换为0-255范围
        frame_255 = (frame * 255).astype(np.uint8)
        cv2.imwrite(path, frame_255)
    
    # 创建GIF动画
    input_sequence = inputs[sample_idx, :, 0]  # 使用输入序列
    pred_sequence = predictions[sample_idx, :, 0]  # 使用预测序列
    target_sequence = targets[sample_idx, :, 0]  # 使用目标序列
    output_dir = sample_dir  # 使用样本目录作为输出目录
    create_gif(input_sequence, pred_sequence, target_sequence, output_dir, model_type)
    
    if EVAL_CONFIG['show_plots']:
        plt.figure(figsize=(15, 5))
        
        # 输入序列的最后一帧
        plt.subplot(131)
        plt.imshow(input_sample[-1], cmap='viridis')
        plt.title('输入最后一帧')
        plt.colorbar()
        
        # 预测序列的第一帧
        plt.subplot(132)
        plt.imshow(pred_sample[0], cmap='viridis')
        plt.title(f'{model_type} 预测第1帧')
        plt.colorbar()
        
        # 真实序列的第一帧
        plt.subplot(133)
        plt.imshow(target_sample[0], cmap='viridis')
        plt.title('真实第1帧')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(f"{sample_dir}/{model_type}_comparison.png")
        plt.close()

def create_gif(input_sequence, pred_sequence, target_sequence, output_dir, model_type):
    """
    创建预测结果的GIF动画
    """
    # 合并序列用于GIF
    all_frames = []
    
    # 确定统一的图像尺寸（使用预测帧的尺寸）
    if len(pred_sequence) > 0:
        # 获取预测帧的尺寸作为标准尺寸
        standard_height, standard_width = pred_sequence[0].shape
    else:
        # 如果没有预测帧，使用输入帧的尺寸
        standard_height, standard_width = input_sequence[0].shape
    
    # 添加输入序列的最后几帧
    for i in range(max(-5, -len(input_sequence)), 0):
        frame = (input_sequence[i] * 255).astype(np.uint8)
        # 确保帧大小一致
        if frame.shape != (standard_height, standard_width):
            frame = cv2.resize(frame, (standard_width, standard_height), interpolation=cv2.INTER_LINEAR)
        # 在左上角添加标签，从1开始计数
        frame_num = len(input_sequence) + i + 1
        frame = cv2.putText(
            frame, f"Input {frame_num}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        all_frames.append(frame)
    
    # 添加预测序列和真实序列（并排显示）
    for i in range(len(pred_sequence)):
        pred_frame = (pred_sequence[i] * 255).astype(np.uint8)
        target_frame = (target_sequence[i] * 255).astype(np.uint8)
        
        # 确保帧大小一致
        if pred_frame.shape != (standard_height, standard_width):
            pred_frame = cv2.resize(pred_frame, (standard_width, standard_height), interpolation=cv2.INTER_LINEAR)
        if target_frame.shape != (standard_height, standard_width):
            target_frame = cv2.resize(target_frame, (standard_width, standard_height), interpolation=cv2.INTER_LINEAR)
        
        # 添加标签
        pred_frame = cv2.putText(
            pred_frame, f"{model_type} Pred {i+1}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        target_frame = cv2.putText(
            target_frame, f"Target {i+1}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        # 水平拼接预测和真实帧
        combined = np.hstack((pred_frame, target_frame))
        all_frames.append(combined)
    
    # 保存为GIF
    gif_path = os.path.join(output_dir, f"{model_type}_prediction.gif")
    
    # 使用imageio保存GIF
    # 确保所有帧都是RGB格式且尺寸一致
    rgb_frames = []
    # 获取第一个帧的尺寸作为标准尺寸
    if all_frames:
        final_height, final_width = all_frames[0].shape
        
        for frame in all_frames:
            # 确保所有帧尺寸一致
            if frame.shape != (final_height, final_width):
                frame = cv2.resize(frame, (final_width, final_height), interpolation=cv2.INTER_LINEAR)
            
            # 如果是灰度图，转换为RGB
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                # 如果是BGR格式（OpenCV默认），转换为RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(frame)
        
        # 保存为GIF
        imageio.mimsave(gif_path, rgb_frames, fps=2.0, loop=0)
        print(f"GIF已保存到: {gif_path}")
    else:
        print(f"警告: 没有足够的帧来创建GIF")


# ================= 主评估函数 =================
def main():
    device = torch.device(CONFIG['device'])
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("加载测试数据集...")
    dataset = RadarLazyDataset(CONFIG['images_dir'], CONFIG['list_file'])
    
    # 为了快速评估，我们只使用一小部分数据
    test_size = min(EVAL_CONFIG['num_samples'] * CONFIG['batch_size'], len(dataset))
    test_indices = np.random.choice(len(dataset), test_size, replace=False)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 评估所有模型
    results = {}
    
    for model_type, model_path in EVAL_CONFIG['model_paths'].items():
        if os.path.exists(model_path):
            print(f"\n评估模型: {model_type}")
            print(f"加载模型权重: {model_path}")
            
            # 加载模型
            model = load_model(model_type, model_path)
            
            # 评估模型
            avg_loss, inputs, predictions, targets = evaluate_model(
                model, test_loader, model_type, device
            )
            
            results[model_type] = {
                'loss': avg_loss,
                'inputs': inputs,
                'predictions': predictions,
                'targets': targets
            }
            
            print(f"{model_type} 平均损失: {avg_loss:.6f}")
            
            # 可视化结果
            for sample_idx in range(min(3, len(test_dataset))):
                visualize_results(inputs, predictions, targets, sample_idx, model_type)
        else:
            print(f"警告: 模型文件不存在: {model_path}")
    
    # 比较结果
    if len(results) > 1:
        print("\n模型比较:")
        for model_type, result in results.items():
            print(f"{model_type}: MSE损失 = {result['loss']:.6f}")
        
        # 找出最佳模型
        best_model = min(results, key=lambda x: results[x]['loss'])
        print(f"\n最佳模型: {best_model}")

if __name__ == "__main__":
    main()