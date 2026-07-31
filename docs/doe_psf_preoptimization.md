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

1. PSF MTF floor，保护 0.02–0.15 cycle/pixel 空间带宽；
2. RGB 传感器加权的相邻波长 PSF 分离；
3. RGB 传感器加权的相邻深度 PSF 分离；
4. 宽松的 r16/r24 能量护栏，防止通过无限扩散伪造区分度。

## 第一阶段命令

先在一张卡上对两种容量各跑一个 seed：

```bash
CUDA_VISIBLE_DEVICES=2 python /home/wenchao/autodl-tmp/scripts/preoptimize_psf_doe.py \
  --output_dir /home/wenchao/autodl-tmp/experiments/PSF卷积/DOE预优化/rank9_vs_free150_seed123_1000step \
  --modes rank9 free150 \
  --seeds 123 \
  --device cuda:0 \
  --steps 1000 \
  --lr 1e-2 \
  --separation_warmup_steps 100 \
  --depth_layers 16 \
  --initial_height_rms_um 0.6 \
  --maximum_height_rms_um 3.0 \
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
- `spectral/adjacent_cosine_mean` 是否显著低于当前约 0.974；
- `depth/adjacent_cosine_mean` 是否显著低于当前约 0.965；
- MTF 提升是否伴随 r90 无限扩大或 129×129 capture 明显下降。

如果 rank-9 和 free-150 都无法明显改善，说明瓶颈更可能来自当前光路本身或
高度预算，而非联合训练。若 free-150 明显优于 rank-9，则下一步才把最佳 DOE
作为联合训练初始化，并保持网络与数据设置不变进行受控对照。
