import copy
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Models.Attention import *


def Encoder_factory(config):
    model = EEG2Rep(config, num_classes=config['num_labels'])
    return model


class EEG2Rep(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        """
         channel_size: number of EEG channels
         seq_len: number of timepoints in a window
        """
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']  # d_x default=16
        # Shapelet Embedding Layer -----------------------------------------------------------
        config['pooling_size'] = 10  # Shapelet 降采样步长 (原版是2，这里稍微调大配合滑动窗口)
        self.seq_len = int(seq_len / config['pooling_size'])  # Number of patches (l')
        # 定义 Shapelet 字典，确保总数相加等于 emb_size default=16
        L_raw = config['Data_shape'][2]
        shapelet_fractions = {
            0.1: 5,
            0.4: 5,
            0.7: 6
        }
        # 动态计算 Shapelet 长度并构建字典
        shapelet_dict = {}
        for frac, num in shapelet_fractions.items():
            # 乘以比例并取整。使用 max(1, ...) 防止在极短序列中出现长度为 0 的致命报错
            actual_len = max(1, int(L_raw * frac))
            shapelet_dict[actual_len] = num

        self.share_frontend = config.get('share_frontend', True)  # 是否共享shapelet参数
        # 学生网络前端 (Context Frontend)
        self.student_frontend = ShapeletEmbedding(
            in_channels=channel_size, 
            shapelets_size_and_len=shapelet_dict, 
            pool_size=config['pooling_size'], 
            to_cuda=True
        )
        
        # 老师网络前端 (Target Frontend)，使用深度拷贝并阻断梯度
        if not self.share_frontend:
            # 策略 B: EMA 更新
            self.teacher_frontend = copy.deepcopy(self.student_frontend)
            for param in self.teacher_frontend.parameters():
                param.requires_grad = False
        else:
            # 策略 A: 共享前端
            self.teacher_frontend = self.student_frontend
            

        self.PositionalEncoding = PositionalEmbedding(self.seq_len, emb_size)
        # -------------------------------------------------------------------------
        self.momentum = config['momentum']
        self.device = config['device']
        self.mask_ratio = config['mask_ratio']
        self.mask_len = int(config['mask_ratio'] * self.seq_len)
        self.mask_token = nn.Parameter(torch.randn(emb_size, ))

        # Transformer Encoder
        self.contex_encoder = Encoder(config)
        self.target_encoder = copy.deepcopy(self.contex_encoder)
        self.Predictor = Predictor(emb_size, config['num_heads'], config['dim_ff'], 1, config['pre_layers'])
        self.predict_head = nn.Linear(emb_size, config['num_labels'])
        self.Norm = nn.LayerNorm(emb_size)
        self.Norm2 = nn.LayerNorm(emb_size)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def copy_weight(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.contex_encoder.parameters(), self.target_encoder.parameters()):
                param_b.data = param_a.data
            if not self.share_frontend:
                for (param_a, param_b) in zip(self.student_frontend.parameters(), self.teacher_frontend.parameters()):
                    param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.contex_encoder.parameters(), self.target_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data
            # Shapelet  embedding shared/EMA updated
            if not self.share_frontend:
                for (param_q, param_k) in zip(self.student_frontend.parameters(), self.teacher_frontend.parameters()):
                    param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data
    
    def linear_prob(self, x):
        with (torch.no_grad()):
            patches = self.student_frontend(x)
            patches = self.Norm(patches)
            patches = patches + self.PositionalEncoding(patches)
            patches = self.Norm2(patches)
            out = self.contex_encoder(patches)
            out = out.transpose(2, 1)
            out = self.gap(out)
            return out.squeeze()

    def pretrain_forward(self, x):
        B, C, L_raw = x.shape
        
        # 1. shapelet embedding
        patches_raw = self.student_frontend(x)  # (Batch, l, d_x)
        patches = self.Norm(patches_raw)
        patches = patches + self.PositionalEncoding(patches)
        patches = self.Norm2(patches)

        # 2. 生成全长度的 Mask Token 占位符，并赋予位置编码
        rep_mask_token_full = self.mask_token.repeat(patches.shape[0], patches.shape[1], 1)
        rep_mask_token_full = rep_mask_token_full + self.PositionalEncoding(rep_mask_token_full)

        # 3. Shapelet Guided Masking：计算可见索引和遮挡索引
        # 注意：使用 patches_raw 来计算得分，保留最纯粹的物理动作强度
        v_index, m_index, ids_restore = Shapelet_Guided_Masking(patches_raw, mask_ratio=self.mask_ratio, noise_scale=0.2)
        
        # 将 2D 索引扩展为 3D，以便在特征维度进行 Gather 抽取
        v_index_expanded = v_index.unsqueeze(-1).expand(-1, -1, patches.size(2))
        m_index_expanded = m_index.unsqueeze(-1).expand(-1, -1, patches.size(2))

        # 4. 抽取相应的特征流
        visible = torch.gather(patches, dim=1, index=v_index_expanded)
        rep_mask_token = torch.gather(rep_mask_token_full, dim=1, index=m_index_expanded)
        
        # 学生网络仅对可见上下文进行编码
        rep_contex = self.contex_encoder(visible)
        
        # 5. (Teacher) 网络提供标准答案
        with torch.no_grad():
            if self.share_frontend:
                y_patches_raw = patches_raw.detach()
            else:
                y_patches_raw = self.teacher_frontend(x)
            y_patches = self.Norm(y_patches_raw)
            y_patches = y_patches + self.PositionalEncoding(y_patches)
            y_patches = self.Norm2(y_patches)
            
            # 老师网络对完整序列进行编码
            rep_target = self.target_encoder(y_patches)
            
            # 从teacher完整特征图中，找出mask的那部分作为 Target
            rep_mask = torch.gather(rep_target, dim=1, index=m_index_expanded)
            
        # 学生尝试根据上下文预测被遮挡的特征
        rep_mask_prediction = self.Predictor(rep_contex, rep_mask_token)
        
        return [rep_mask, rep_mask_prediction, rep_contex, rep_target]

    def forward(self, x):
        patches = self.student_frontend(x)
        patches = self.Norm(patches)
        patches = patches + self.PositionalEncoding(patches)
        patches = self.Norm2(patches)
        out = self.contex_encoder(patches)
        return self.predict_head(torch.mean(out, dim=1))

# Shapelet Embedding 取代 CNN InputEmbedding
class ShapeletEmbedding(nn.Module):
    """
    用于替代原版 EEG2Rep 的 CNN InputEmbedding。
    输入: (Batch, Channels, L_raw)
    输出: (Batch, L_patch, emb_size)
    """
    def __init__(self, in_channels, shapelets_size_and_len, pool_size=10, to_cuda=True):
        super(ShapeletEmbedding, self).__init__()
        self.to_cuda = to_cuda
        self.in_channels = in_channels
        self.pool_size = pool_size
        
        # 确保总 Shapelet 数量等于 Transformer 的 emb_size (默认是 16)
        self.emb_size = sum(shapelets_size_and_len.values())
        
        # 将各尺度的 Shapelet 均分给三种物理度量 (Euclidean, Cosine, Cross-Corr)
        self.blocks = nn.ModuleList()
        
        for size, num in shapelets_size_and_len.items():
            num_euclid = num // 3
            num_cosine = num // 3
            num_cc = num - num_euclid - num_cosine
            
            if num_euclid > 0:
                self.blocks.append(LocalEuclideanBlock(size, num_euclid, in_channels, pool_size, to_cuda))
            if num_cosine > 0:
                self.blocks.append(LocalCosineBlock(size, num_cosine, in_channels, pool_size, to_cuda))
            if num_cc > 0:
                self.blocks.append(LocalCrossCorrBlock(size, num_cc, in_channels, pool_size, to_cuda))
                
        # 独立的 BatchNorm1d，防止不同度量方式的数值量级互相“霸凌”
        self.bn = nn.BatchNorm1d(self.emb_size)
        
        if self.to_cuda:
            self.cuda()

    def forward(self, x, masking=False):
        # x shape: (B, C, L_raw)
        out_features = []
        
        # 1. 遍历所有不同尺度和度量的 Shapelet 探测器
        for block in self.blocks:
            # 每个 block 输出形状: (B, num_shapelets_in_block, L_patch)
            out_features.append(block(x))
            
        # 2. 在特征通道维度拼接所有 Shapelet 的激活得分
        # 拼接后形状: (B, emb_size, L_patch)
        x_out = torch.cat(out_features, dim=1)
        
        # 3. 归一化 (让欧氏、余弦、互相关的数值分布对齐，极大地加速收敛)
        x_out = self.bn(x_out)
        
        # 4. 终极接口适配：转置对齐 Transformer
        # 最终形状: (B, L_patch, emb_size)
        x_out = x_out.transpose(1, 2).contiguous()
        
        return x_out


# ==========================================
# 底层三大物理度量模块 (带 Padding 和 局部池化)
# ==========================================

class LocalEuclideanBlock(nn.Module):
    def __init__(self, shapelet_size, num_shapelets, in_channels, pool_size, to_cuda):
        super().__init__()
        self.size = shapelet_size
        self.pool_size = pool_size
        
        shapelets = torch.randn(in_channels, num_shapelets, shapelet_size, requires_grad=True)
        self.shapelets = nn.Parameter(shapelets.cuda() if to_cuda else shapelets)

    def forward(self, x):
        # 1. 右侧补零，确保 unfold 后序列长度与原序列 L_raw 一致
        x_pad = F.pad(x, (0, self.size - 1))
        # 2. 滑动窗口扫描
        x_unfold = x_pad.unfold(2, self.size, 1).contiguous()
        
        # 3. 计算欧氏距离
        dist = torch.cdist(x_unfold, self.shapelets, p=2)
        dist = torch.sum(dist, dim=1, keepdim=True).transpose(2, 3).squeeze(1) # shape: (B, K, L_raw)
        
        # 4. 核心魔改：高斯反转！把“距离最小”变成“得分最高”
        # 这样就能和另外两个度量标准统一使用 MaxPool
        activation = torch.exp(-dist) 
        
        # 5. 局部池化 (降采样生成 Patch)
        out = F.max_pool1d(activation, kernel_size=self.pool_size, stride=self.pool_size)
        return out


class LocalCosineBlock(nn.Module):
    def __init__(self, shapelet_size, num_shapelets, in_channels, pool_size, to_cuda):
        super().__init__()
        self.size = shapelet_size
        self.pool_size = pool_size
        self.relu = nn.ReLU()
        
        shapelets = torch.randn(in_channels, num_shapelets, shapelet_size, requires_grad=True)
        self.shapelets = nn.Parameter(shapelets.cuda() if to_cuda else shapelets)

    def forward(self, x):
        x_pad = F.pad(x, (0, self.size - 1))
        x_unfold = x_pad.unfold(2, self.size, 1).contiguous()
        
        # L2 归一化
        x_norm = x_unfold / x_unfold.norm(p=2, dim=3, keepdim=True).clamp(min=1e-8)
        s_norm = self.shapelets / self.shapelets.norm(p=2, dim=2, keepdim=True).clamp(min=1e-8)
        
        # 计算余弦相似度
        sim = torch.matmul(x_norm, s_norm.transpose(1, 2))
        sim = torch.sum(sim, dim=1, keepdim=True).transpose(2, 3).squeeze(1) / x.shape[1]
        
        # 忽略负相关，只看正向激活
        sim = self.relu(sim)
        
        # 局部池化
        out = F.max_pool1d(sim, kernel_size=self.pool_size, stride=self.pool_size)
        return out


class LocalCrossCorrBlock(nn.Module):
    def __init__(self, shapelet_size, num_shapelets, in_channels, pool_size, to_cuda):
        super().__init__()
        self.size = shapelet_size
        self.pool_size = pool_size
        
        # 互相关本质上就是卷积
        self.conv = nn.Conv1d(in_channels, num_shapelets, kernel_size=shapelet_size)
        if to_cuda:
            self.conv.cuda()

    def forward(self, x):
        # 卷积的 padding 需要放在左右两边或单边
        x_pad = F.pad(x, (0, self.size - 1))
        corr = self.conv(x_pad) # shape: (B, K, L_raw)
        
        # 局部池化
        out = F.max_pool1d(corr, kernel_size=self.pool_size, stride=self.pool_size)
        return out

class InputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']  # d_x (input embedding dimension)
        k = 7
        # Embedding Layer -----------------------------------------------------------
        self.depthwise_conv = nn.Conv2d(in_channels=1, out_channels=emb_size, kernel_size=(channel_size, 1))
        self.spatial_padding = nn.ReflectionPad2d((int(np.floor((k - 1) / 2)), int(np.ceil((k - 1) / 2)), 0, 0))
        self.spatialwise_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, k))
        self.spatialwise_conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, k))
        self.SiLU = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, config['pooling_size']), stride=(1, config['pooling_size']))

    def forward(self, x):
        out = x.unsqueeze(1)
        out = self.depthwise_conv(out)  # (bs, embedding, 1 , T)
        out = out.transpose(1, 2)  # (bs, 1, embedding, T)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv1(out)  # (bs, 1, embedding, T)
        out = self.SiLU(out)
        out = self.maxpool(out)  # (bs, 1, embedding, T // m)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv2(out)
        out = out.squeeze(1)  # (bs, embedding, T // m)
        out = out.transpose(1, 2)  # (bs, T // m, embedding)
        patches = self.SiLU(out)
        return patches


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        d_model = config['emb_size']
        attn_heads = config['num_heads']
        # d_ffn = 4 * d_model
        d_ffn = config['dim_ff']
        layers = config['layers']
        dropout = config['dropout']
        enable_res_parameter = True
        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x


def Semantic_Subsequence_Preserving(time_step_indices, chunk_count, target_percentage):
    # Get the total number of time steps
    total_time_steps = len(time_step_indices)
    # Calculate the desired total time steps for the selected chunks
    target_total_time_steps = int(total_time_steps * target_percentage)

    # Calculate the size of each chunk
    chunk_size = target_total_time_steps // chunk_count

    # Randomly select starting points for each chunk with minimum distance
    start_points = [random.randint(0, total_time_steps - chunk_size)]
    # Randomly select starting points for each subsequent chunk with minimum distance
    for _ in range(chunk_count - 1):
        next_start_point = random.randint(0, total_time_steps - chunk_size)
        start_points.append(next_start_point)

    # Select non-overlapping chunks using indices
    selected_chunks_indices = [time_step_indices[start:start + chunk_size] for start in start_points]

    return selected_chunks_indices

def Shapelet_Guided_Masking(x, mask_ratio=0.4, noise_scale=0.2):
    """
    显著性引导的靶向掩码策略 (Shapelet-Guided Masking)
    替代原版 EEG2Rep 中的 Semantic_Subsequence_Preserving。
    
    参数:
        x: [B, L_patch, emb_size] Shapelet前端吐出的特征图
        mask_ratio: 要掩盖的时间块比例 (默认0.4代表掩盖40%)
        noise_scale: 随机扰动系数。越大越接近随机掩盖，越小越死板地只遮最高分。
                     设为0.2可以保证大概率遮住高潮，但偶尔留出破绽，防止模型信息饥饿。
    
    返回:
        v_index: [B, len_keep] 保留（可见）的时间块索引
        m_index: [B, len_mask] 被掩盖的时间块索引
        ids_restore: [B, L_patch] 用于最后将预测结果和可见结果拼回原顺序的恢复索引
    """
    B, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    
    # 1. 计算每个时间块 (Patch) 的“显著性总得分”
    # 我们沿着 Shapelet 维度(dim=-1)求和，得到当前时刻的整体动作剧烈程度
    # scores shape: [B, L]
    scores = torch.sum(x, dim=-1)
    
    # 2. 引入概率扰动 (极其核心的 Trick！)
    # 生成与 scores 同分布的随机噪声。
    # 这样可以打破绝对的确定性，让每次 Epoch 遮住的位置有微小变化，极大地增强泛化能力。
    noise = torch.rand(B, L, device=x.device) * scores.max(dim=-1, keepdim=True)[0] * noise_scale
    saliency = scores + noise
    
    # 3. 排序决胜负
    # torch.argsort 默认是从小到大排序。
    # 因为我们要把“得分最高”的遮住，所以排序后，排在前面的（分低）保留，排在后面的（分高）掩盖。
    ids_shuffle = torch.argsort(saliency, dim=1)  # [B, L]
    ids_restore = torch.argsort(ids_shuffle, dim=1) # [B, L] 用于打乱后的还原
    
    # 4. 划分可见域与掩盖域
    v_index = ids_shuffle[:, :len_keep]  # 得分较低的 (1-mask_ratio) 部分，保留
    m_index = ids_shuffle[:, len_keep:]  # 得分最高的 mask_ratio 部分，残忍掩盖！
    
    return v_index, m_index, ids_restore


class Predictor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Predictor, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter) for i in range(layers)])

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)
        return rep_mask_token


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


