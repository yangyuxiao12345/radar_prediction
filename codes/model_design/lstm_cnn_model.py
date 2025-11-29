import torch
import torch.nn as nn

# ================= 编码器-解码器结构的 LSTM-CNN 模型 =================
class Encoder(nn.Module):
    """
    CNN编码器，用于提取空间特征
    由多个卷积层和池化层构成，逐步压缩特征图尺寸，增加通道数
    """
    def __init__(self, input_channels=1, hidden_dims=[8, 16, 32]):
        super(Encoder, self).__init__()
        
        layers = []
        in_channels = input_channels
        
        # 构建编码器层
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*layers)
        self.out_channels = hidden_dims[-1]
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, channels, height, width] -> 输入序列
        返回编码后的特征序列
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # 对每个时间步的帧单独进行编码
        encoded_frames = []
        for t in range(seq_len):
            frame = x[:, t]
            encoded = self.encoder(frame)
            encoded_frames.append(encoded)
        
        # 堆叠编码后的帧 [batch_size, seq_len, hidden_dims[-1], h, w]
        return torch.stack(encoded_frames, dim=1)

class ConvLSTMCell(nn.Module):
    """
    ConvLSTM单元，用于处理空间-时间特征
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        # 组合输入和隐藏状态进行卷积
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # 门控机制
        i = torch.sigmoid(cc_i)  # 输入门
        f = torch.sigmoid(cc_f)  # 遗忘门
        o = torch.sigmoid(cc_o)  # 输出门
        g = torch.tanh(cc_g)     # 候选状态
        
        # 更新细胞状态和隐藏状态
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        )

class Decoder(nn.Module):
    """
    CNN解码器，用于恢复特征图到原始尺寸
    由上采样层和卷积层构成
    """
    def __init__(self, input_channels, hidden_dims=[32, 16, 8], output_channels=1):
        super(Decoder, self).__init__()
        
        layers = []
        in_channels = input_channels
        
        # 构建解码器层
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, h_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = h_dim
        
        # 输出层，恢复到原始通道数
        layers.append(nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())  # 归一化到[0,1]范围
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        x: [batch_size, seq_len, channels, height, width] -> 编码后的特征序列
        返回解码后的图像序列
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # 对每个时间步的特征单独进行解码
        decoded_frames = []
        for t in range(seq_len):
            feature = x[:, t]
            decoded = self.decoder(feature)
            decoded_frames.append(decoded)
        
        # 堆叠解码后的帧 [batch_size, seq_len, 1, h, w]
        return torch.stack(decoded_frames, dim=1)

class LSTMCNN(nn.Module):
    """
    完整的LSTM-CNN模型，包含编码器、ConvLSTM和解码器
    """
    def __init__(self, input_channels=1, encoder_dims=[8, 16, 32], 
                 convlstm_dims=[32], decoder_dims=[32, 16, 8], 
                 kernel_size=(3, 3)):
        super(LSTMCNN, self).__init__()
        
        # 编码器
        self.encoder = Encoder(input_channels, encoder_dims)
        
        # ConvLSTM层
        self.convlstm_layers = nn.ModuleList()
        input_dim = encoder_dims[-1]
        
        for hidden_dim in convlstm_dims:
            self.convlstm_layers.append(
                ConvLSTMCell(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size
                )
            )
            input_dim = hidden_dim
        
        # 解码器
        self.decoder = Decoder(convlstm_dims[-1], decoder_dims, output_channels=1)
        
    def forward(self, x, future_frames=10):
        """
        x: [batch_size, seq_len, channels, height, width] -> 输入序列
        future_frames: 需要预测的未来帧数
        返回预测的未来帧序列
        """
        batch_size, seq_len, _, height, width = x.shape
        device = x.device
        
        # 1. 编码输入序列
        encoded = self.encoder(x)
        _, _, channels, enc_height, enc_width = encoded.shape
        
        # 2. 初始化ConvLSTM隐藏状态
        hidden_states = []
        for i in range(len(self.convlstm_layers)):
            hidden_states.append(
                self.convlstm_layers[i].init_hidden(batch_size, (enc_height, enc_width))
            )
        
        # 3. 处理编码后的序列，更新隐藏状态
        for t in range(seq_len):
            current_input = encoded[:, t]
            for i in range(len(self.convlstm_layers)):
                hidden_states[i] = self.convlstm_layers[i](current_input, hidden_states[i])
                current_input = hidden_states[i][0]  # 使用隐藏状态作为下一层的输入
        
        # 4. 预测未来帧
        predictions = []
        current_frame = encoded[:, -1]
        
        for _ in range(future_frames):
            # 使用当前帧和隐藏状态预测下一帧
            for i in range(len(self.convlstm_layers)):
                hidden_states[i] = self.convlstm_layers[i](current_frame, hidden_states[i])
                current_frame = hidden_states[i][0]
            
            # 保存当前预测的特征
            predictions.append(current_frame)
        
        # 5. 解码预测的特征得到图像
        # 堆叠预测的特征 [batch_size, future_frames, channels, h, w]
        pred_features = torch.stack(predictions, dim=1)
        pred_images = self.decoder(pred_features)
        
        return pred_images

# ================= 简化版ViT模型 =================
class PatchEmbedding(nn.Module):
    """
    将图像分割为patch并嵌入
    """
    def __init__(self, img_size=256, patch_size=16, in_channels=1, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 使用卷积层作为patch embedding
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        """
        x: [batch_size, channels, height, width]
        返回: [batch_size, num_patches, embed_dim]
        """
        batch_size = x.shape[0]
        x = self.proj(x)  # [batch_size, embed_dim, num_patches^(1/2), num_patches^(1/2)]
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        return x

class ViTLayer(nn.Module):
    """
    ViT的基本层，包含自注意力和前馈网络
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(ViTLayer, self).__init__()
        
        # 自注意力层 - 增加dropout提高稳定性
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=dropout  # 添加注意力dropout
        )
        self.dropout1 = nn.Dropout(dropout)  # 残差连接后dropout
        
        # 前馈网络 - 增加dropout和更强的MLP结构
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # 自注意力路径
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        attn_output = self.dropout1(attn_output)
        x = x + attn_output  # 残差连接
        
        # 前馈网络路径
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output  # 残差连接
        
        return x

class SimpleViT(nn.Module):
    """
    增强版ViT模型，用于雷达图像预测
    """
    def __init__(self, img_size=256, patch_size=16, in_channels=1, 
                 embed_dim=256, num_heads=4, depth=3,  # 增加模型复杂度
                 out_channels=1, future_frames=10, dropout=0.1):
        super(SimpleViT, self).__init__()
        
        # 增强的Patch嵌入 - 增加一个卷积层
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=patch_size, stride=patch_size)
        )
        
        num_patches = (img_size // patch_size) ** 2
        
        # 位置编码 - 使用更好的初始化
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        
        # ViT层 - 增加层数并添加dropout
        self.blocks = nn.ModuleList([
            ViTLayer(dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        # 增强的时间序列处理
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True
        )
        # LSTM输出归一化
        self.lstm_norm = nn.LayerNorm(embed_dim)
        
        # 增强的输出层 - 使用多层投影
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, patch_size * patch_size * out_channels)
        )
        
        # 配置参数
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim  # 添加embed_dim作为属性
        self.future_frames = future_frames
        self.out_channels = out_channels
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        # 初始化卷积层权重
        for m in self.patch_embed.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 初始化LSTM权重
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
        
        # 初始化输出层权重
        for m in self.output_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, future_frames=None):
        """
        优化的前向传播，移除嵌套循环，批量处理所有帧和patch
        
        x: [batch_size, seq_len, channels, height, width] -> 输入序列
        future_frames: 需要预测的未来帧数
        返回预测的未来帧序列
        """
        if future_frames is None:
            future_frames = self.future_frames
            
        batch_size, seq_len, channels, height, width = x.shape
        
        # ========== 优化1: 批量处理所有帧的patch嵌入 ==========
        # 重塑为 [batch_size*seq_len, channels, height, width]
        x_reshaped = x.view(-1, channels, height, width)
        # 批量嵌入所有帧 - 当前形状: [batch_size*seq_len, embed_dim, grid_size, grid_size]
        embedded_all = self.patch_embed(x_reshaped)
        
        # 计算网格大小
        grid_size = self.img_size // self.patch_size
        
        # 重塑为 [batch_size*seq_len, num_patches, embed_dim] 以匹配位置编码
        # num_patches = grid_size * grid_size
        embedded_all = embedded_all.flatten(2)  # [batch_size*seq_len, embed_dim, num_patches]
        embedded_all = embedded_all.transpose(1, 2)  # [batch_size*seq_len, num_patches, embed_dim]
        
        # 确保位置编码形状匹配
        # self.pos_embed 应该是 [1, num_patches, embed_dim]
        # 添加位置编码
        embedded_all = embedded_all + self.pos_embed
        
        # 重塑回 [batch_size, seq_len, num_patches, embed_dim]
        num_patches = grid_size * grid_size
        embedded_sequences = embedded_all.view(batch_size, seq_len, num_patches, self.embed_dim)
        
        # ========== 优化2: 移除patch循环，使用reshape批量处理 ==========
        # 重塑为 [batch_size*num_patches, seq_len, embed_dim]
        batch_size_patches = batch_size * embedded_sequences.shape[2]
        vit_input = embedded_sequences.permute(0, 2, 1, 3).reshape(batch_size_patches, seq_len, -1)
        
        # 批量应用ViT层
        x_vit = vit_input
        for block in self.blocks:
            x_vit = block(x_vit)
        x_vit = self.norm(x_vit)
        
        # ========== 优化3: 高效的LSTM处理 ==========
        # 使用LSTM处理时间序列（已经是正确形状）
        lstm_output, (h_n, c_n) = self.lstm(x_vit)
        
        # ========== 优化4: 高效的未来帧预测 ==========
        # 使用LSTM的最后状态进行预测，避免循环
        future_outputs = []
        last_hidden = lstm_output[:, -1:]
        h_state, c_state = h_n, c_n
        
        # 只进行一次前向传播预测所有未来帧
        for _ in range(future_frames):
            lstm_pred, (h_state, c_state) = self.lstm(last_hidden, (h_state, c_state))
            future_outputs.append(lstm_pred)
            last_hidden = lstm_pred
        
        # 堆叠未来预测 [batch_size*num_patches, future_frames, embed_dim]
        future_outputs = torch.cat(future_outputs, dim=1)
        
        # ========== 优化5: 批量投影和图像重构 ==========
        # 投影回patch空间
        patch_size_sq = self.patch_size * self.patch_size
        future_patches = self.output_proj(future_outputs)
        future_patches = future_patches.view(-1, future_frames, patch_size_sq * self.out_channels)
        
        # 重构图像
        grid_size = self.img_size // self.patch_size
        
        # 批量处理所有未来帧
        # 重塑为 [batch_size, future_frames, grid_size, grid_size, patch_size, patch_size, channels]
        future_patches_reshaped = future_patches.view(
            batch_size, grid_size*grid_size, future_frames, self.patch_size, self.patch_size, self.out_channels
        ).permute(0, 2, 1, 3, 4, 5).contiguous()
        
        # 重塑为网格形式 [batch_size, future_frames, grid_size, grid_size, patch_size, patch_size, channels]
        future_patches_reshaped = future_patches_reshaped.view(
            batch_size, future_frames, grid_size, grid_size, self.patch_size, self.patch_size, self.out_channels
        )
        
        # 转置并重塑为图像 [batch_size, future_frames, channels, height, width]
        future_images = future_patches_reshaped.permute(
            0, 1, 6, 2, 4, 3, 5
        ).contiguous().view(
            batch_size, future_frames, self.out_channels, self.img_size, self.img_size
        )
        
        # 使用sigmoid激活归一化到[0,1]
        future_images = torch.sigmoid(future_images)
        
        return future_images