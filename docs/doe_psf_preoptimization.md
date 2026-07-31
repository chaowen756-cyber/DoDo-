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
- 两者均以 0.6 μm pupil RMS 随机初始化，并在每次 Adam 更新后投影到同一个
  3 μm pupil-RMS 球内。
- rank-9 与 free-150 的差异因此主要是表达容量，而不是允许的总高度预算。

优化目标包括：

1. 单色点源 RGB PSF 关于 x、y、深度和波长的 Fisher A-optimality；
2. PSF MTF floor，保护 0.02–0.15 cycle/pixel 空间带宽；
3. RGB 传感器加权的相邻波长 PSF 分离；
4. RGB 传感器加权的相邻深度 PSF 分离；
5. 宽松的 r16/r24 能量护栏，防止通过无限扩散伪造区分度。

Fisher 项直接对应 Baek 等人在同一 HS-D 任务中使用的 DOE 初始化方法。
四个导数均按一个离散采样 bin 定标，避免米、纳米等单位选择任意改变
A-optimality；PSF 不做逐核归一化，因此低 capture 不能伪装成高信息量。

## 优化器依据

- Baek 等人的 Fisher DOE 初始化使用 Adam，并从 Fresnel DOE 开始；其论文
  没有给出 Fisher 阶段的独立学习率。论文给出的 `1e-4` 是约3948万参数网络
  与 DOE 相位联合训练的学习率，不能直接套到这里的9/150维系数空间。
- D-Flat 的不同相位/元表面示例使用过 `1e-3` 到 `1e-1`，进一步说明学习率
  取决于参数化尺度，而不是固定的光学常数。
- 当前实现使用 Adam、系数学习率 `1e-2`、梯度裁剪和 cosine 衰减，并在每步
  后投影到3 μm物理 RMS 球。完整16深度的20步预检中两种 DOE 均稳定下降，
  因而不再无依据地改成联合训练的 `1e-4`。
- 没有把 Fresnel 初始化设为默认：当前 rank-9 基底对离散二次高度的相对拟合
  误差约为0.866；free-150 中同一0.6 μm预算下，随机初始化 A-optimality
  `2.020e6` 优于较好符号的 Fresnel `2.218e6`。若按容量采用不同初始化还会
  混入额外变量，因此继续使用固定 seed 的随机初始化进行公平对照。

## 第一阶段命令

两张卡并行，但必须使用不同输出目录。GPU 2 运行当前 rank-9：

```bash
CUDA_VISIBLE_DEVICES=2 python /home/wenchao/autodl-tmp/scripts/preoptimize_psf_doe.py \
  --output_dir /home/wenchao/autodl-tmp/experiments/PSF卷积/DOE预优化/fisher_rank9_seed123_1000step \
  --modes rank9 \
  --seeds 123 \
  --device cuda:0 \
  --steps 1000 \
  --lr 1e-2 \
  --separation_warmup_steps 100 \
  --depth_layers 16 \
  --initial_height_rms_um 0.6 \
  --maximum_height_rms_um 3.0 \
  --fisher_weight 1.0 \
  --fisher_ridge 1e-8 \
  --fisher_loss_scale 1e-7 \
  --log_every 10 \
  --save_psf_bank
```

GPU 3 同时运行 free-150：

```bash
CUDA_VISIBLE_DEVICES=3 python /home/wenchao/autodl-tmp/scripts/preoptimize_psf_doe.py \
  --output_dir /home/wenchao/autodl-tmp/experiments/PSF卷积/DOE预优化/fisher_free150_seed123_1000step \
  --modes free150 \
  --seeds 123 \
  --device cuda:0 \
  --steps 1000 \
  --lr 1e-2 \
  --separation_warmup_steps 100 \
  --depth_layers 16 \
  --initial_height_rms_um 0.6 \
  --maximum_height_rms_um 3.0 \
  --fisher_weight 1.0 \
  --fisher_ridge 1e-8 \
  --fisher_loss_scale 1e-7 \
  --log_every 10 \
  --save_psf_bank
```

若 free-150 明显达到更好的 MTF/分离指标，再用三个 seed 复核：

```bash
CUDA_VISIBLE_DEVICES=2 python /home/wenchao/autodl-tmp/scripts/preoptimize_psf_doe.py \
  --output_dir /home/wenchao/autodl-tmp/experiments/PSF卷积/DOE预优化/free150_3seed_1000step \
  --modes free150 \
  --seeds 123 456 789 \
  --device cuda:0 \
  --steps 1000 \
  --lr 1e-2 \
  --separation_warmup_steps 100 \
  --depth_layers 16 \
  --initial_height_rms_um 0.6 \
  --maximum_height_rms_um 3.0 \
  --fisher_weight 1.0 \
  --fisher_ridge 1e-8 \
  --fisher_loss_scale 1e-7 \
  --log_every 10 \
  --save_psf_bank
```

## 输出

每个 `mode/seed_*` 目录包含：

- `best_doe.pt`：可复用的 DOE checkpoint、构造配置和最佳指标；
- `best_coefficients.npy`、`best_heightmap_m.npy`；
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
- `fisher/a_optimality_mean` 是否显著下降、最小特征值是否提高；
- `spectral/adjacent_cosine_mean` 是否显著低于当前约 0.974；
- `depth/adjacent_cosine_mean` 是否显著低于当前约 0.965；
- MTF 提升是否伴随 r90 无限扩大或 129×129 capture 明显下降。

如果 rank-9 和 free-150 都无法明显改善，说明瓶颈更可能来自当前光路本身或
高度预算，而非联合训练。若 free-150 明显优于 rank-9，则下一步才把最佳 DOE
作为联合训练初始化，并保持网络与数据设置不变进行受控对照。

## 参考

- Baek et al., [Single-Shot Hyperspectral-Depth Imaging with Learned
  Diffractive Optics](https://arxiv.org/abs/2009.00463)，补充材料第6节。
- Hazineh et al., [D-Flat](https://arxiv.org/abs/2207.14780)。
- Wang et al., [dO: A differentiable engine for Deep Lens design](https://github.com/vccimaging/DiffOptics)。
