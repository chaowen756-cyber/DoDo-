# Number 18: Baek 增强与深度分布均衡

本分支保持 number17 的模型、loss、DOE、固定验证块和无噪声设置不变，只修改训练数据分布。

## 实现口径

- 空间增强：`scale=1.0` 概率 0.7，`scale=0.5` 概率 0.3；HS 双线性插值，米制深度和 mask 最近邻插值，同步水平/垂直翻转。
- 验证隔离：`scale=0.5` 只从 `scale_05_eligible=True` 的锚点采样，其 256×256 源窗口不与固定验证窗口并集相交。
- 深度增强：全局请求平移为 `-0.2/0/+0.2 m`，概率 `0.25/0.5/0.25`；实际平移裁剪到整块有效深度的可行区间，然后重新计算 IPS。
- 光谱增强：20% 平坦光源，80% 从论文使用的 29 种 CIE 光源随机选择，光源按 CIE 1931 photopic 响应归一化，再乘 `0.9--1.1` 曝光。
- 归一化：增强后只使用 number17 的全局 `fixed_scale=0.9685`，不进行每样本最大值归一化；若某个 CIE 光源与当前 patch 的组合超过 0.1% 裁剪率，则最多重抽 8 次固定光源，失败时回退平坦光源。TensorBoard 同时记录裁剪率、重抽次数和回退率。
- 均衡采样：每个 scene 均匀出现；scene 内根据 16-bin IPS patch 直方图加权，目标分布为 `sqrt(frequency)`，权重限制到 `[0.25, 4.0]`，每 scene 均值为 1，ESS 比例必须大于 50%。

29 种光源名称与原论文图 13 一致：A、B、C、D50、D55、D65、D93、E、9 种 LED、F1--F12。采样值来自 Colour Science 的 CIE 光源数据，并在 420--660 nm、10 nm 间隔上对齐。

参考：[Baek et al. ICCV 2021 supplemental](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Baek_Single-Shot_Hyperspectral-Depth_Imaging_ICCV_2021_supplemental.pdf)，[Colour Science illuminant dataset](https://github.com/colour-science/colour/blob/develop/colour/colorimetry/datasets/illuminants/sds.py)。

## 运行

```bash
bash scripts/run_number18_baek_balanced.sh build-index
bash scripts/run_number18_baek_balanced.sh stage-a-balanced
bash scripts/run_number18_baek_balanced.sh stage-a-augment
bash scripts/run_number18_baek_balanced.sh stage-a-combined
```

Stage A 各运行 12 epoch。比较固定验证集后，仅对最佳变体运行 Stage B：

```bash
INIT_CKPT=/path/to/stageA/joint-best-epoch=XXX.ckpt \
VARIANT=combined \
bash scripts/run_number18_baek_balanced.sh stage-b
```

Stage B 默认 30 epoch。训练命令、参数、工作区状态和 Git commit 分别写入实验的 `artifacts/command.txt`、`hparams.json`、`git_status.txt` 和 `git_commit.txt`。

## 晋级阈值

- 测试 scene 14--18 平均深度 MAE `< 0.1584 m`。
- 测试 scene 14--18 平均 masked HS PSNR `>= 29.71 dB`。
- scene 17 深度 MAE `<= 0.176 m`，masked HS PSNR `>= 28.63 dB`。
