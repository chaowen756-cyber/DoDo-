# PSF 卷积 DOE 可编码性预优化

## 目的

在不加载数据集、不构造重建网络的情况下，直接回答一个问题：当前
`consistent_grid_v1` 光路能否在给定物理高度预算内生成同时具备空间带宽、
波长区分和深度区分能力的 PSF bank。

该实验保持 Prop1 点源传播、10 mm 一致网格、Prop3、pupil、完整场归一化和
129×129 PSF 裁剪不变。它不是联合训练，也不引入 50 mm/6.45 μm 相机参数。

## 公平对照

- `rank9`：当前 12 项来源、正交后 rank=9、每模态 3 μm pupil RMS。
- `free150`：150 项 Zernike，每模态约 1 μm pupil RMS。
- 两者均以 0.6 μm pupil RMS 随机初始化，并约束在同一个3 μm pupil-RMS
  球内。首次越界执行安全回缩；到达边界后先移除 Adam 候选更新的向外法向
  分量，再做数值回缩，从而保留沿 RMS 边界的有效优化方向。
- rank-9 与 free-150 的差异因此主要是表达容量，而不是允许的总高度预算。

前两轮正式实验已经确认 free-150 在两个seed、3000步下收敛到相同指标上限：
相邻波长 optical cosine 仍约0.9898，MTF@0.05 p10仍约0.011。为检验瓶颈是否
来自平滑Zernike表达空间，新增：

- `pixelphase`：128×128逐像素未包裹参考相位，共16384个自由度；
- 默认参考波长550 nm，光学前向把相位包裹到一个物理2π高度周期，再按现有
  DOE材料色散计算所有25个波长；
- 初始化不套用外部焦距/像元尺寸，而是用当前Prop1参考输入场和Prop3严格离散
  伴随，在约1 m、550 nm处构造相位共轭聚焦载波；
- 加入0.05 rad随机扰动，使不同seed仍能检验非凸稳定性。

这一实现与Baek等补充材料中“优化未包裹相位、以Fresnel DOE初始化、最终相位
包裹成物理高度”的处理一致，并进一步让包裹物理高度直接参与宽带训练，避免
优化结束后再包裹造成非参考波长PSF退化。

优化目标包括：

1. 单色点源 RGB PSF 的任务加权 Fisher A-optimality；
2. PSF MTF floor，保护 0.02–0.15 cycle/pixel 空间带宽；
3. 不含 RGB 响应、逐核尺度不敏感的单色 PSF 光学形状分离：波长使用
   1/2/4 band offset，深度使用1层 offset；
4. RGB 传感器加权的波长/深度分离作为真实观测辅助指标；
5. 宽松的 r16/r24 能量护栏，防止通过无限扩散伪造区分度。

光学形状分离对每个单色 PSF 做 L2 归一化，只比较空间形状。仅改变某波长
总强度或利用 RGB 光谱响应本身不再能降低该项；色散引起的质心移动仍保留，
因为它是 PSF 卷积测量中的有效编码线索。正式目标默认 optical spectral/depth
权重为 `5/2`，sensor spectral/depth 为 `0/0.5`；sensor-weighted Fisher 始终
保留，防止只改善光学图样却脱离真实 RGB 观测。

训练历史同时记录：

- `loss/train_total`：含 separation warm-up 的实际反传目标；
- `loss/full_total`：始终按 separation scale=1 计算的固定完整目标。

最佳 checkpoint 始终按 `loss/full_total` 选择。因此 warm-up 前后可以直接
比较，不会再把逐渐加入的损失项误判为训练发散。

Fisher 项以 Baek 等人在同一 HS-D 任务中使用的 DOE 初始化方法为基础。
实现始终对完整 `(x,y,depth,wavelength)` Fisher 矩阵求逆，因此 x/y 仍作为
nuisance 参数参与深度/波长 CRLB；损失权重设为 `(0.1,0.1,1,1)`，避免上一轮
把大部分自由度用于 x/y 定位。四个导数均按一个离散采样 bin 定标，避免米、
纳米等单位选择任意改变 A-optimality；PSF 不做逐核归一化，因此低 capture
不能伪装成高信息量。

## 优化器依据

- Baek 等人的 Fisher DOE 初始化使用 Adam，并从 Fresnel DOE 开始；其论文
  没有给出 Fisher 阶段的独立学习率。论文给出的 `1e-4` 是约3948万参数网络
  与 DOE 相位联合训练的学习率，不能直接套到这里的9/150维系数空间。
- D-Flat 的不同相位/元表面示例使用过 `1e-3` 到 `1e-1`，进一步说明学习率
  取决于参数化尺度，而不是固定的光学常数。
- 当前实现使用 Adam、系数学习率 `1e-2`、梯度裁剪、cosine 衰减和 RMS
  边界切向更新。完整16深度的20步预检中两种 DOE 均稳定下降，
  因而不再无依据地改成联合训练的 `1e-4`。
- 没有把 Fresnel 初始化设为默认：当前 rank-9 基底对离散二次高度的相对拟合
  误差约为0.866；free-150 中同一0.6 μm预算下，随机初始化 A-optimality
  `2.020e6` 优于较好符号的 Fresnel `2.218e6`。若按容量采用不同初始化还会
  混入额外变量，因此继续使用固定 seed 的随机初始化进行公平对照。

## 当前实验命令

正式实验改用已在200步预检中明显超过free-150 3000步结果的pixel-phase，
并用seed 123/456在两张卡上复核。完整命令中的关键参数如下；聊天交付给出
两张卡的可直接复制版本：

```bash
/home/wenchao/conda_envs/ld_clean/bin/python scripts/preoptimize_psf_doe.py \
  --modes pixelphase --steps 1000 --lr 1e-1 \
  --final_lr_ratio 0.05 --gradient_clip_norm 1.0 \
  --separation_warmup_steps 100 \
  --phase_reference_depth_m 1.0 \
  --phase_reference_wavelength_nm 550 \
  --phase_initial_noise_std_rad 0.05 \
  --fisher_weight 1.0 --mtf_weight 20.0 \
  --optical_spectral_weight 5.0 --optical_depth_weight 2.0 \
  --sensor_spectral_weight 0.0 --sensor_depth_weight 0.5 \
  --optical_spectral_offsets 1 2 4 --optical_depth_offsets 1
```

## 输出

每个 `mode/seed_*` 目录包含：

- `best_doe.pt`：可复用的 DOE checkpoint、构造配置和最佳指标；
- `best_coefficients.npy`、`best_heightmap_m.npy`；
- pixel-phase额外保存 `best_unwrapped_phase_rad.npy` 与
  `best_unwrapped_heightmap_m.npy`；其中 `best_heightmap_m.npy` 始终是实际参与
  宽带前向的单周期包裹物理高度；
- `best_heightmap.png`、`best_psf_montage.png`；
- `history.jsonl`、`summary.json`；
- 可选的 `best_psf_bank.pt`。

根目录包含 `command.txt` 和跨模式的 `comparison.json`。

加载预优化系数时使用：

```python
from util.doe_preoptimization import load_preoptimized_doe_

load_preoptimized_doe_(camera.doe1, "/path/to/best_doe.pt")
```

## 判读

`summary.json` 会分别报告 MTF、波长 margin 和深度 margin 是否满足。重点观察：

- `mtf/005_p10`、`mtf/005_mean` 是否显著高于当前约 0.011/0.031；
- `fisher/task_a_optimality_mean` 和 depth/wavelength CRLB 是否显著下降；
- `fisher/a_optimality_mean` 与最小特征值用于和上一轮完整 Fisher 横向比较；
- `spectral/adjacent_cosine_mean` 是否显著低于当前约 0.974；
- `depth/adjacent_cosine_mean` 是否显著低于当前约 0.965；
- `optical_spectral/adjacent_cosine_mean` 和
  `optical_spectral/offset_2/4_cosine_mean` 是否下降；这是 DOE 本身产生光谱
  图样编码的主要证据；
- `loss/full_total` 是否下降；不要用 warm-up 期间的 `loss/train_total` 跨步
  比较；
- `constraint/minimum_retraction_scale` 应接近1，若远小于1说明边界更新失稳；
- pixel-phase没有Zernike RMS投影；检查 `doe/wrapped_height_min/max_m` 是否位于
  单个参考相位周期内，并检查初始化记录中的 `center_intensity_gain`；
- MTF 提升是否伴随 r90 无限扩大或 129×129 capture 明显下降。

如果 rank-9 和 free-150 都无法明显改善，说明瓶颈更可能来自当前光路本身或
高度预算，而非联合训练。若 free-150 明显优于 rank-9，则下一步才把最佳 DOE
作为联合训练初始化，并保持网络与数据设置不变进行受控对照。

## 参考

- Baek et al., [Single-Shot Hyperspectral-Depth Imaging with Learned
  Diffractive Optics](https://arxiv.org/abs/2009.00463)，补充材料第6节。
- 同论文[官方CVF补充材料](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Baek_Single-Shot_Hyperspectral-Depth_Imaging_ICCV_2021_supplemental.pdf)：
  未包裹相位优化、Fresnel DOE初始化与最终物理高度包裹。
- Hazineh et al., [D-Flat](https://arxiv.org/abs/2207.14780)。
- Wang et al., [dO: A differentiable engine for Deep Lens design](https://github.com/vccimaging/DiffOptics)。
