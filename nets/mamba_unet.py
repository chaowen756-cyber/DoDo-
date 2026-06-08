import torch
import torch.nn as nn
from .mamba_helper import VSSBlock

def _make_norm(channels, norm_type='batch'):
    if norm_type == 'group':
        num_groups = min(8, channels)
        if channels % num_groups != 0:
            num_groups = 4
            while num_groups > 1 and channels % num_groups != 0:
                num_groups -= 1
        return nn.GroupNorm(num_groups, channels)
    return nn.BatchNorm2d(channels)


class DoubleConv(nn.Module):
    """传统的 CNN 卷积块"""
    def __init__(self, in_channels, out_channels, norm_type='batch'):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            _make_norm(out_channels, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            _make_norm(out_channels, norm_type),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class MambaEncoderBlock(nn.Module):
    """封装 VSSBlock 和下采样"""
    def __init__(self, in_channels, out_channels, use_mamba=True):
        super().__init__()
        self.use_mamba = use_mamba
        if use_mamba:
            self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.feature_extract = VSSBlock(out_channels)
        else:
            self.feature_extract = DoubleConv(in_channels, out_channels)
            
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):
        if self.use_mamba:
            x = self.channel_proj(x)
        feat = self.feature_extract(x)
        pool = self.downsample(feat)
        return feat, pool

class MambaDualHeadUNet(nn.Module):
    def __init__(self, in_channels=32, out_hs_channels=25, scheme='hybrid',
                 depth_shallow_skip_mode='lowpass', norm_type='batch'):
        super().__init__()

        # [32, 64, 128, 256, 512, 1024]
        dims = [32, 64, 128, 256, 512, 1024]
        self.enc_channels = dims

        # ================= Encoder =================
        self.encoders = nn.ModuleList()

        # L1: 32 -> 64
        use_mamba_l1 = False if scheme == 'hybrid' else True
        self.encoders.append(MambaEncoderBlock(dims[0], dims[1], use_mamba=use_mamba_l1))

        # L2-L4: Mamba
        self.encoders.append(MambaEncoderBlock(dims[1], dims[2], use_mamba=True)) # 64->128
        self.encoders.append(MambaEncoderBlock(dims[2], dims[3], use_mamba=True)) # 128->256
        self.encoders.append(MambaEncoderBlock(dims[3], dims[4], use_mamba=True)) # 256->512

        # Bottleneck: 512 -> 1024
        self.bottleneck = nn.Sequential(
            nn.Conv2d(dims[4], dims[5], kernel_size=1),
            VSSBlock(dims[5])
        )

        # ================= Decoder (Depth Branch) =================
        # Up 4: 1024 -> 512
        self.up_depth_4 = nn.ConvTranspose2d(dims[5], dims[4], 2, stride=2)
        self.conv_depth_4 = DoubleConv(dims[5], dims[4], norm_type=norm_type) # 512+512 -> 512

        # Up 3: 512 -> 256
        self.up_depth_3 = nn.ConvTranspose2d(dims[4], dims[3], 2, stride=2)
        self.conv_depth_3 = DoubleConv(dims[4], dims[3], norm_type=norm_type) # 256+256 -> 256

        # Up 2: 256 -> 128
        self.up_depth_2 = nn.ConvTranspose2d(dims[3], dims[2], 2, stride=2)
        self.conv_depth_2 = DoubleConv(dims[3], dims[2], norm_type=norm_type) # 128+128 -> 128

        # Up 1: 128 -> 64 (恢复到了原始分辨率)
        self.up_depth_1 = nn.ConvTranspose2d(dims[2], dims[1], 2, stride=2)
        self.conv_depth_1 = DoubleConv(dims[2], dims[1], norm_type=norm_type) # 64+64 -> 64

        # [ARCH-MOD-20260403] 深度分支浅层 skip 解耦配置：
        # - full: 保持原始直连
        # - lowpass: 对浅层 skip 低通+投影后再注入（默认，抑制颜色纹理泄漏）
        # - drop: 完全关闭浅层 skip
        self.depth_shallow_skip_mode = depth_shallow_skip_mode
        self.depth_skip_lowpass = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.depth_skip_proj = nn.Conv2d(dims[1], dims[1], kernel_size=1, bias=False)
        self.depth_skip_gate = nn.Parameter(torch.tensor(0.0))

        # Output Head: 直接接收 64 通道
        self.depth_out = nn.Conv2d(dims[1], 1, kernel_size=1)

        # ================= Decoder (HS Branch) =================
        self.up_hs_4 = nn.ConvTranspose2d(dims[5], dims[4], 2, stride=2)
        self.conv_hs_4 = DoubleConv(dims[5], dims[4], norm_type=norm_type)

        self.up_hs_3 = nn.ConvTranspose2d(dims[4], dims[3], 2, stride=2)
        self.conv_hs_3 = DoubleConv(dims[4], dims[3], norm_type=norm_type)

        self.up_hs_2 = nn.ConvTranspose2d(dims[3], dims[2], 2, stride=2)
        self.conv_hs_2 = DoubleConv(dims[3], dims[2], norm_type=norm_type)

        self.up_hs_1 = nn.ConvTranspose2d(dims[2], dims[1], 2, stride=2)
        self.conv_hs_1 = DoubleConv(dims[2], dims[1], norm_type=norm_type)

        # Output Head: 直接接收 64 通道
        self.hs_out = nn.Conv2d(dims[1], out_hs_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, 32, H, W]
        
        # Encoder
        skips = []
        curr = x
        for enc in self.encoders:
            skip, curr = enc(curr)
            skips.append(skip) 
            # skips[0]: [64, H, W]
            # skips[1]: [128, H/2, W/2]
            # skips[2]: [256, H/4, W/4]
            # skips[3]: [512, H/8, W/8]
        
        bot = self.bottleneck(curr) # [1024, H/16, W/16]
        
        # --- Decoder (Depth) ---
        d4 = self.up_depth_4(bot)
        d4 = torch.cat([d4, skips[3]], dim=1) # H/8
        d4 = self.conv_depth_4(d4)
        
        d3 = self.up_depth_3(d4)
        d3 = torch.cat([d3, skips[2]], dim=1) # H/4
        d3 = self.conv_depth_3(d3)
        
        d2 = self.up_depth_2(d3)
        d2 = torch.cat([d2, skips[1]], dim=1) # H/2
        d2 = self.conv_depth_2(d2)
        
        d1 = self.up_depth_1(d2)

        # [ARCH-MOD-20260403] 深度头浅层 skip 解耦：避免深度预测过度跟随高光谱纹理。
        # [ARCH-OLD-20260403] d1 = torch.cat([d1, skips[0]], dim=1) # H (原始分辨率)
        if self.depth_shallow_skip_mode == 'full':
            depth_skip_l1 = skips[0]
        elif self.depth_shallow_skip_mode == 'drop':
            depth_skip_l1 = torch.zeros_like(skips[0])
        elif self.depth_shallow_skip_mode == 'lowpass':
            depth_skip_l1 = self.depth_skip_lowpass(skips[0])
            depth_skip_l1 = self.depth_skip_proj(depth_skip_l1)
            depth_skip_l1 = depth_skip_l1 * torch.sigmoid(self.depth_skip_gate)
        else:
            raise ValueError(f"Unknown depth_shallow_skip_mode: {self.depth_shallow_skip_mode}")

        d1 = torch.cat([d1, depth_skip_l1], dim=1) # H (原始分辨率)
        d1 = self.conv_depth_1(d1)
        
        # 此时 d1 已经是 [B, 64, H, W]，直接输出
        depth_logits = self.depth_out(d1)
        
        # --- Decoder (HS) ---
        h4 = self.up_hs_4(bot)
        h4 = torch.cat([h4, skips[3]], dim=1)
        h4 = self.conv_hs_4(h4)
        
        h3 = self.up_hs_3(h4)
        h3 = torch.cat([h3, skips[2]], dim=1)
        h3 = self.conv_hs_3(h3)
        
        h2 = self.up_hs_2(h3)
        h2 = torch.cat([h2, skips[1]], dim=1)
        h2 = self.conv_hs_2(h2)
        
        h1 = self.up_hs_1(h2)
        h1 = torch.cat([h1, skips[0]], dim=1)
        h1 = self.conv_hs_1(h1)
        
        hs_logits = self.hs_out(h1)
        
        return depth_logits, hs_logits