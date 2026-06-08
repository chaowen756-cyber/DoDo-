import torch
import torch.nn as nn

from models.outputs_container import OutputsContainer
# 引入新定义的 Mamba U-Net
from nets.mamba_unet import MambaDualHeadUNet, _make_norm

class SimpleModelHS(nn.Module):
    """
    修改后的SimpleModelHS，集成了 Mamba 结构的 DualHeadUNet。
    """

    def __init__(self, hparams, *args, **kargs):
        super().__init__()
        self.preinverse = hparams.preinverse

        # 获取方案配置，默认为 'hybrid' (方案 B), 可选 'pure' (方案 A)
        mamba_scheme = getattr(hparams, 'mamba_scheme', 'hybrid')
        # [ARCH-MOD-20260403] 深度浅层 skip 解耦模式。
        depth_shallow_skip_mode = getattr(hparams, 'depth_shallow_skip_mode', 'lowpass')
        decoder_norm = getattr(hparams, 'decoder_norm', 'batch')

        # ================= 参数配置 =================
        hs_channels = hparams.hs_channels  # 25
        measurement_channels = getattr(hparams, 'measurement_channels', hs_channels) or hs_channels
        decoder_in_channels = getattr(hparams, 'decoder_in_channels', measurement_channels)
        depth_ch = 1
        base_ch = hparams.model_base_ch # 32
        n_depths = hparams.n_depths
        # For preinverse=False: captimgs may carry extra depth channel
        # For preinverse=True: decoder depth input is not supported, captimgs unchanged
        self._expected_measurement_channels = (
            measurement_channels if hparams.preinverse else decoder_in_channels
        )

        # ================= 1. 适配层 (Stem) =================
        preinv_input_ch = hs_channels * n_depths + hs_channels

        # 负责将 DOE 压缩数据映射到 Mamba 的输入维度 (base_ch)
        self.base_input_layers = nn.Sequential(
            nn.Conv2d(preinv_input_ch, base_ch, kernel_size=3, padding=1, bias=False),
            _make_norm(base_ch, decoder_norm),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
            _make_norm(base_ch, decoder_norm),
            nn.ReLU(),
        )

        if self.preinverse:
            self.input_adapter = self.base_input_layers
        else:
            self.input_adapter = nn.Sequential(
                nn.Conv2d(decoder_in_channels, base_ch, kernel_size=3, padding=1, bias=False),
                _make_norm(base_ch, decoder_norm),
                nn.ReLU(),
                nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
                _make_norm(base_ch, decoder_norm),
                nn.ReLU(),
            )

        # ================= 2. 核心骨干 (MambaDualHeadUNet) =================
        print(f"Building Backbone with Scheme: {mamba_scheme} (Mamba), decoder_norm={decoder_norm}")

        self.backbone = MambaDualHeadUNet(
            in_channels=base_ch, # 32
            out_hs_channels=hs_channels,
            scheme=mamba_scheme,  # 'hybrid' or 'pure'
            depth_shallow_skip_mode=depth_shallow_skip_mode,
            norm_type=decoder_norm,
        )

        # ================= 3. 激活函数 =================
        self.sigmoid = nn.Sigmoid()

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): # Mamba 内部可能有 Linear
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, captimgs, pinv_volumes, *args, **kwargs):
        if captimgs.shape[1] != self._expected_measurement_channels:
            raise ValueError(
                f"captimgs.shape[1]={captimgs.shape[1]} does not match "
                f"expected measurement_channels={self._expected_measurement_channels}"
            )
        b_sz = captimgs.shape[0]
        h_sz, w_sz = captimgs.shape[2], captimgs.shape[3]

        # --- 1. 数据预处理与拼接 ---
        if self.preinverse:
            if pinv_volumes.ndim == 4:
                pinv_reshaped = pinv_volumes
            elif pinv_volumes.ndim == 5:
                _, c, d, _, _ = pinv_volumes.shape
                pinv_reshaped = pinv_volumes.reshape(b_sz, c*d, h_sz, w_sz)
            else:
                raise ValueError(f"pinv_volumes 维度错误: {pinv_volumes.shape}")
            
            inputs = torch.cat([captimgs, pinv_reshaped], dim=1)
        else:
            inputs = captimgs

        # --- 2. 适配层 ---
        # [Batch, 32, H, W]
        x = self.input_adapter(inputs)

        # --- 3. Mamba 双头处理 ---
        depth_logits, hs_logits = self.backbone(x)

        # --- 4. 激活 ---
        est_depthmaps = self.sigmoid(depth_logits)
        est_images = self.sigmoid(hs_logits)

        # 调整深度图维度
        if est_depthmaps.ndim == 4 and est_depthmaps.shape[1] == 1:
            est_depthmaps = est_depthmaps.squeeze(1)

        outputs = OutputsContainer(
            est_images=est_images,
            est_depthmaps=est_depthmaps,
        )
        return outputs