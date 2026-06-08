软 Diopter 分层的精确实现方案
本文档旨在指导 Coding Agent 按照“不改主光路顺序，只替换 depth partitioner”的原则，实现基于屈光度域的软分层逻辑。
1. 核心原则保持总体框架：
每个深度层独立传播 -> 传感器强度响应 -> 各层强度累加。软赋权替代硬切分：将原先的 mask_k = (z in bin_k) 这种硬切分，改为按 Inverse Depth (Diopter) 的三角基函数做软赋权。物理语义一致性：EDoF-HS 等深景深 DOE 系统使用 diopter 定义训练范围和随机采样（正文里写了训练对象从 $1/z \in [1/z_{\min}, 1/z_{\max}]$ 的均匀 dioptric distribution 中采样）。此修改确保了仿真逻辑与系统物理特性的一致。

2. 接口约定建议新增独立模块 depth_partition.py，暴露如下明确接口。forward_dodo.py 不需要知道内部细节，仅需获取 weights 和 centers_z。Pythonclass SoftDiopterPartitioner(nn.Module):
    def __init__(
        self,
        depth_min_m: float,
        depth_max_m: float,
        num_layers: int,
        overlap: float = 1.0,
        eps: float = 1e-8,
        clamp_depth: bool = True,
    ):
        ...

    def forward(
        self,
        depth_map: torch.Tensor,   # [B, 1, H, W], meters
        valid_mask: torch.Tensor | None = None,  # [B, 1, H, W], {0,1}
    ) -> dict:
        """
        return {
            "weights": Tensor[B, K, 1, H, W],
            "centers_u": Tensor[K],   # diopters
            "centers_z": Tensor[K],   # meters
        }
        """
3. 数学与张量形状假设输入光谱张量为
 $x: [B, L, H, W]$，深度图为 $z: [B, 1, H, W]$。定义屈光度：$$u = \frac{1}{z}, \qquad u_{\min}=\frac{1}{z_{\max}}, \qquad u_{\max}=\frac{1}{z_{\min}}$$均匀采样：对 $K = \text{num\_depth\_layers}$ 个中心做均匀采样：$$u_k = \operatorname{linspace}(u_{\min}, u_{\max}, K)$$计算带宽：若 $K > 1$，中心间距 $\Delta u_{\text{center}} = u_{k+1}-u_k$。三角权重半宽 $\Delta u = \text{overlap} \cdot \Delta u_{\text{center}}$。注：默认 overlap = 1.0，若需过渡更柔和可设为 1.25 或 1.5。4. 推荐实现代码此版本在米制深度图上清洗后转 diopter，并确保有效像素跨层权重和为 1。Pythonimport torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiopterPartitioner(nn.Module):
    def __init__(
        self,
        depth_min_m: float,
        depth_max_m: float,
        num_layers: int,
        overlap: float = 1.0,
        eps: float = 1e-8,
        clamp_depth: bool = True,
    ):
        super().__init__()
        assert depth_min_m > 0
        assert depth_max_m > depth_min_m
        assert num_layers >= 1
        assert overlap > 0

        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.num_layers = int(num_layers)
        self.overlap = float(overlap)
        self.eps = float(eps)
        self.clamp_depth = bool(clamp_depth)

        u_min = 1.0 / self.depth_max_m
        u_max = 1.0 / self.depth_min_m
        centers_u = torch.linspace(u_min, u_max, self.num_layers)
        centers_z = 1.0 / centers_u

        if self.num_layers > 1:
            du_center = centers_u[1] - centers_u[0]
        else:
            du_center = torch.tensor(u_max - u_min if u_max > u_min else 1.0)

        self.register_buffer("centers_u", centers_u)
        self.register_buffer("centers_z", centers_z)
        self.register_buffer("du_center", du_center)

    def forward(self, depth_map, valid_mask=None):
        # depth_map: [B,1,H,W]
        assert depth_map.ndim == 4 and depth_map.shape[1] == 1

        z = depth_map.float()
        finite = torch.isfinite(z)

        if valid_mask is None:
            valid = finite.float()
        else:
            valid = (finite & (valid_mask > 0)).float()

        if self.clamp_depth:
            z = z.clamp(min=self.depth_min_m, max=self.depth_max_m)
        else:
            z = z.clamp(min=self.depth_min_m, max=self.depth_max_m)

        u = 1.0 / z  # [B,1,H,W]

        centers = self.centers_u.view(1, self.num_layers, 1, 1)
        du = (self.du_center * self.overlap).view(1, 1, 1, 1)

        # broadcast: [B,K,H,W]
        tri = F.relu(1.0 - torch.abs(u - centers) / (du + self.eps))
        tri = tri * valid  # invalid pixels stay zero

        denom = tri.sum(dim=1, keepdim=True) + self.eps
        weights = tri / denom  # [B,K,H,W]
        weights = weights.unsqueeze(2)  # [B,K,1,H,W]

        return {
            "weights": weights,
            "centers_u": self.centers_u,
            "centers_z": self.centers_z,
        }
5. 集成说明（forward_dodo 接线）替换“按 hard bin 建 mask”的代码段，保留 for 循环主逻辑：Pythonpart = self.depth_partitioner(depth_map, valid_mask=final_mask)
weights = part["weights"]        # [B,K,1,H,W]
centers_z = part["centers_z"]    # [K]

y_sum = None
for k in range(self.num_depth_layers):
    wk = weights[:, k]           # [B,1,H,W]
    xk = spectral * wk           # [B,L,H,W]

    # 下面仍然沿用每层独立传播的逻辑
    # 注意：prop1_layers[k] 必须绑定到 centers_z[k]
    yk = self.forward_single_depth_layer(xk, depth_layer_idx=k)

    y_sum = yk if y_sum is None else (y_sum + yk)
6. 三个易错点范围方向：记住 $u = 1/z$，所以远处 $z$ 大，对应的 $u$ 小；近处 $z$ 小，对应的 $u$ 大。因此中心必须从 $1/z_{\max}$ 排列到 $1/z_{\min}$，而不是反过来。否则 $centers\_z = 1/centers\_u$ 会变成逆序层，导致传播参数错位。无效深度与越界深度：建议把 0、NaN、Inf、负值都视为 invalid，并在 valid_mask 中归零；越界但有限的深度，可以先 clamp 到合法区间，再做权重归一化。这样不会把一个异常深度点错误地分配给全部层，也不会让归一化分母变成 0。不要为了省显存而牺牲数值稳定性：权重计算建议固定在 float32，即使主训练是 AMP/mixed precision。因为 inverse depth 和 small $du$ 的组合，在半精度下非常容易在深度边界处抖动，进而让 DOE 梯度更噪。7. 必做单元测试权重归一化：在随机合法深度图上检查 weights.sum(dim=1) 在有效区内是否接近 1。层中心响应：把深度图设成某个层中心 $z_k$，检查该层权重是否最大且相邻层非零。平滑性测试：把一个平面点目标放在层边界附近，确认随着深度连续移动，输出测量是平滑变化而不是 bin 跳变。回归测试：保留旧的 hard_meter 模式做 ablation，保证两个模式接口完全一致，便于实验对照。