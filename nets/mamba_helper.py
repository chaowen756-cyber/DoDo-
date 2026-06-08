import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba
except ImportError:
    print("请先安装 mamba-ssm: pip install mamba-ssm")
    Mamba = None # 占位符

class VSSBlock(nn.Module):
    """
    适配视觉任务的 Mamba 模块 (Visual State Space Block)
    将 2D 图像展平为序列输入 Mamba，处理后再恢复。
    """
    def __init__(self, in_channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError("Mamba module not found.")
            
        self.norm = nn.LayerNorm(in_channels)
        
        # Mamba 核心层
        # d_model: 输入维度
        # d_state: SSM 状态维度 (类似隐藏层大小)
        # d_conv: 局部 1D 卷积宽度
        # expand: 内部维度扩展系数
        self.mamba = Mamba(
            d_model=in_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x):
        # x input: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 1. 维度变换: [B, C, H, W] -> [B, H*W, C] (为了 LayerNorm 和 Mamba)
        x_in = x.flatten(2).transpose(1, 2) 
        
        # 2. 残差连接 + Norm + Mamba
        # Mamba 要求输入是 [B, L, C]
        x_out = self.mamba(self.norm(x_in))
        
        # 3. 恢复维度: [B, H*W, C] -> [B, C, H, W]
        x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        
        return x + x_out # Residual connection