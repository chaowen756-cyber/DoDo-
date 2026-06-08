目标

当前 dodo_depth 前向模型使用 hard depth bin：每个像素根据 meter-space depth 被分配到唯一深度层，然后每层独立传播，最后 sensor-plane intensity 相加。

本次改动只做一件事：

将 hard depth bin 替换为 soft diopter bin。

也就是把原来的二值 mask mask_k ∈ {0,1} 替换成连续权重 w_k ∈ [0,1]。

不要修改 DOE、decoder、SensingLayer、loss、训练流程和 optical regularization。
不要引入 DOE2、CFA、CCA、PSF regularization、occlusion-aware propagation。
这些是后续任务。

数学定义

输入 depth map 为：

depth: [B, 1, H, W]

先转成 diopter / inverse depth：

u = 1.0 / depth

给定有效深度范围：

z_min, z_max

定义 diopter 范围：

u_min = 1.0 / z_max
u_max = 1.0 / z_min

定义 K = num_depth_layers 个均匀 diopter center：

centers_u = torch.linspace(u_min, u_max, K)
du = (u_max - u_min) / (K - 1)
z_centers = 1.0 / centers_u

每个像素到每个 diopter center 的三角权重为：

raw_w_k = relu(1 - abs(u - centers_u[k]) / bandwidth)
w_k = raw_w_k / (sum_j raw_w_j + eps)

默认：

bandwidth = du
eps = 1e-8

如果配置里支持 warmup，可以允许：

bandwidth_scale = 1.0  # 可选，初期可设 1.2~1.5
bandwidth = du * bandwidth_scale
目标张量 shape

假设输入 hyperspectral image 是：

x: [B, C, H, W]
depth: [B, 1, H, W]
valid_mask: [B, 1, H, W] or None

soft diopter binner 输出：

weights: [B, K, H, W]
z_centers: [K]
centers_u: [K]

进入每个传播层时：

x_k = x * weights[:, k:k+1, :, :]   # [B, C, H, W]

后续 prop1 -> doe1 -> prop2 -> doe2(optional) -> prop3 -> sensing 逻辑保持不变。

推荐新增模块

在 forward_dodo.py 中新增一个小模块，名字可以是：

class SoftDiopterBinner(nn.Module):

或者如果项目风格不希望新增 class，也可以新增函数：

def compute_soft_diopter_weights(depth, z_min, z_max, num_layers, valid_mask=None, eps=1e-8, bandwidth_scale=1.0):
    ...

推荐 class 版本。

接口建议：

class SoftDiopterBinner(nn.Module):
    def __init__(
        self,
        z_min: float,
        z_max: float,
        num_layers: int,
        eps: float = 1e-8,
        bandwidth_scale: float = 1.0,
    ):
        ...

    def forward(
        self,
        depth: torch.Tensor,          # [B,1,H,W]
        valid_mask: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ):
        """
        Returns:
            weights: [B,K,H,W]
            z_centers: [K]
            debug: optional dict
        """

注意：

centers_u
z_centers
du

应该注册为 buffer，而不是普通 Python float/list，保证 .to(device) 和 checkpoint 正常工作：

self.register_buffer("centers_u", centers_u)
self.register_buffer("z_centers", z_centers)
self.register_buffer("du", torch.tensor(du))
数值稳定要求

实现时必须处理这些情况：

1. depth clamp

不要直接做 1.0 / depth，必须先 clamp：

depth_safe = depth.clamp(min=z_min, max=z_max)
u = 1.0 / depth_safe
u = u.clamp(min=u_min, max=u_max)
2. valid mask

如果有 valid_mask，先乘 mask，再归一化：

raw_w = raw_w * valid_mask
weights = raw_w / (raw_w.sum(dim=1, keepdim=True) + eps)

如果某个像素 invalid，所有层权重应为 0。

可以这样做：

weights = weights * valid_mask

并在 debug 中检查：

weight_sum = weights.sum(dim=1, keepdim=True)

valid 区域内 weight_sum ≈ 1，invalid 区域内 weight_sum = 0。

3. dtype / device

所有新 tensor 必须与 depth.device 和 depth.dtype 对齐。
如果 buffer 已注册，一般会自动随 module 迁移，但计算时仍要注意：

centers_u = self.centers_u.to(device=depth.device, dtype=depth.dtype)
4. K = 1

如果 num_depth_layers == 1，直接返回全 1 权重：

weights = torch.ones(B, 1, H, W, device=depth.device, dtype=depth.dtype)

如果有 valid mask，则乘 valid mask。

替换位置

在 DepthAwareDoDoForwardModel 中找到原来生成 hard depth mask 的代码，大概形式可能类似：

for k in range(num_depth_layers):
    mask_k = ...
    x_k = spectral * mask_k
    y_k = propagate_layer(x_k, depth_center[k])
    y_sum += y_k

替换为：

weights, z_centers, debug = self.diopter_binner(
    depth_map,
    valid_mask=valid_mask,
    return_debug=return_aux,
)

y_sum = 0
for k in range(self.num_depth_layers):
    x_k = spectral * weights[:, k:k+1, :, :]
    y_k = self.propagate_one_depth_layer(
        x_k,
        z_center=z_centers[k],
        ...
    )
    y_sum = y_sum + y_k

传播层使用的 depth center 必须改成：

z_centers[k] = 1.0 / centers_u[k]

不要继续使用 meter-space 均匀 depth center。

配置项

新增配置项，保留向后兼容：

depth_layering_mode: "soft_diopter"  # choices: hard_depth, soft_diopter
soft_diopter_eps: 1.0e-8
soft_diopter_bandwidth_scale: 1.0

默认建议：

depth_layering_mode: "soft_diopter"

但为了安全迁移，第一次 patch 可以默认仍为：

depth_layering_mode: "hard_depth"

然后在实验配置里显式打开：

depth_layering_mode: "soft_diopter"
forward aux debug

如果原 forward 支持 return_aux=True，请在 aux 里加入：

aux["depth_weights"] = weights.detach()
aux["depth_weight_sum"] = weights.sum(dim=1, keepdim=True).detach()
aux["diopter_centers"] = centers_u.detach()
aux["depth_centers"] = z_centers.detach()
aux["depth_layering_mode"] = "soft_diopter"

不要默认保存未 detach 的大张量，避免显存泄漏。
只有明确需要梯度诊断时才保留 graph。

单元测试 / smoke test

请新增或修改测试，至少覆盖：

test 1: shape

输入：

x: [2, 31, 64, 64]
depth: [2, 1, 64, 64]
K = 8

检查：

weights.shape == [2, 8, 64, 64]
z_centers.shape == [8]
test 2: energy conservation

valid pixels 内：

weights.sum(dim=1) ≈ 1

invalid pixels 内：

weights.sum(dim=1) == 0

容差：

atol = 1e-5
test 3: boundary behavior

depth = z_min 时，应该主要落在 u_max 对应的近距离层。
depth = z_max 时，应该主要落在 u_min 对应的远距离层。

test 4: differentiability

构造：

depth.requires_grad_(True)
weights = binner(depth)
loss = weights.mean()
loss.backward()

检查：

depth.grad is not None
torch.isfinite(depth.grad).all()

注意：因为 clamp 和 relu 在边界处可能局部梯度为 0，这是可以接受的；测试重点是不能断图、不能 NaN。

test 5: forward compatibility

打开 depth_layering_mode="soft_diopter" 后，完整 model forward 能跑通，输出 shape 与 hard depth 版本一致。

不要做的事

本次 patch 不要做：

- 不要改 reconstruction loss
- 不要加 optical regularizer
- 不要加 r90 / PSF loss
- 不要改 SensingLayer
- 不要改 decoder input channel
- 不要加 DOE2
- 不要加 CFA / CCA
- 不要重构成 occlusion-aware propagation
- 不要把传播逻辑从 wave propagation 改成 PSF convolution

这次只做 soft diopter layering。

验收标准

本次修改完成后，应满足：

hard_depth 和 soft_diopter 两种模式都能运行。
soft_diopter 输出的 measurement shape 与原来完全一致。
valid 区域内所有 depth layer 权重和为 1。
无效区域权重和为 0。
训练时 DOE 参数仍然能收到梯度。
不引入 NaN / Inf。
原有 hard depth baseline 不被破坏。
至少跑一个小 batch smoke training step，确认 loss.backward() 成功。
实现优先级

第一版优先正确性，不追求稀疏双层优化。

也就是说，第一版直接显式构造：

weights: [B, K, H, W]

等确认训练稳定后，再考虑只激活相邻两层的 sparse interpolation 优化。