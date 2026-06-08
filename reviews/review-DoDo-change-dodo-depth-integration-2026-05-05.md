# Review: DoDo-change dodo_depth integration

## 结论

上一轮已经完成了 `dodo_depth` 接入的大部分结构性工作，但当前实现还不能认为可验收。主要阻断点集中在实际训练路径仍使用 `models.simple_model_mamba.SimpleModelHS`，而通道适配只改到了未被导入的 `models/simple_model_hs.py`；另外 `dodo_depth_layers=None`、日志可视化和缺失 `depth_metric` fallback 都会导致 fast-dev 或物理语义失败。

## 已完成目标

- 已确认 `pytorch_optics/` 不存在，并选择 `torch_optics.forward_dodo.DepthAwareDoDoForwardModel` 作为当前新光学模型候选。
- 已在 `datasets/hyperspectral_dataset.py` 新增 `depth_metric`，并保留 `depth_map` 作为 IPS 深度监督 target。
- 已在 `snapshotdepth_hs.py` 增加 `--optical_model legacy_camera|dodo_depth`、`--measurement_channels`、`--dodo_depth_layers`、`--dodo_use_second_doe` 等 CLI 参数。
- 已在 `snapshotdepth_hs.py` 增加 `dodo_depth` 分支，使用 `DepthAwareDoDoForwardModel`，并在该分支中使用 `valid_mask` 抑制无效背景光谱。
- 已在 `nets/unet.py` 让 `DualHeadUNet` 支持动态 `in_channels` 和 `hs_out_channels`，修复普通 `simple_model_hs.py` 路径的硬编码输出通道问题。
- 已保留 `legacy_camera` 旧路径，且没有修改 `optics/`。

## Findings

### 1. `dodo_depth` 实际训练路径仍会因 3/25 通道不匹配崩溃

位置：

- `snapshotdepth_hs.py:26`
- `models/simple_model_mamba.py:42`
- `models/simple_model_mamba.py:46`
- `snapshotdepth_hs.py:415`

`snapshotdepth_hs.py` 实际导入的是 `models.simple_model_mamba.SimpleModelHS`，而上一轮通道适配改在 `models/simple_model_hs.py`。`dodo_depth` 的光学输出是 `(B, 3, H, W)`，但 `simple_model_mamba.py` 在 `preinverse=False` 时仍用 `nn.Conv2d(hs_channels, base_ch, ...)`，默认 `hs_channels=25`。因此 `--optical_model dodo_depth --no-preinverse` 第一轮 forward 会在 decoder input adapter 处报输入通道不匹配。

要求：

- 将 `measurement_channels` 支持同步到 `models/simple_model_mamba.py`。
- 确认 `MambaDualHeadUNet(out_hs_channels=hs_channels)` 输出仍为 `hs_channels`。
- 增加 forward-time shape check，错误信息明确指出 `captimgs.shape[1]` 和期望 `measurement_channels`。

### 2. `--dodo_depth_layers` 默认 `None` 会传入光学模型构造器

位置：

- `snapshotdepth_hs.py:346`
- `snapshotdepth_hs.py:351`

CLI 默认 `dodo_depth_layers=None`，但实现使用 `getattr(hparams, 'dodo_depth_layers', hparams.n_depths)`。当属性存在且值为 `None` 时，`n_depth_layers` 仍为 `None`，随后传入 `DepthAwareDoDoForwardModel(num_depth_layers=None)`，会在构造器的数值比较处崩溃。

要求：

- 使用 `n_depth_layers = hparams.dodo_depth_layers or hparams.n_depths`。
- 对 `n_depth_layers < 1` 给出清晰 `ValueError`。

### 3. `dodo_depth` 缺少 `depth_metric` 时会静默使用 IPS 深度，破坏物理语义

位置：

- `snapshotdepth_hs.py:409`
- `snapshotdepth_hs.py:411`

任务明确要求新物理光学前向使用 metric meters，不能混用未注明的 IPS depth。当前 fallback `depth_metric = depthmaps` 会把 IPS `[0,1]` 当成米制深度送入光学模型，其中大量值会落到 `[0.4,2.0]` 外并被 clamp，训练语义错误且不易发现。

要求：

- `dodo_depth` 路径下缺少 `depth_metric` 必须直接 `raise ValueError`。
- 错误信息说明 Dataset 必须返回 metric `depth_metric`。

### 4. `dodo_depth` 日志可视化会用 25 通道索引访问 3 通道 measurement

位置：

- `snapshotdepth_hs.py:662`
- `snapshotdepth_hs.py:664`

`vis_channels` 是按 `target_images.shape[1]` 计算的，默认为 `[6, 12, 18]`。但 `dodo_depth` 的 `captimgs` 只有 3 通道，`captimgs[:, vis_channels, ...]` 会直接越界。由于 `summary_track_train_every` 默认会在 step 0 触发，fast-dev run 很可能在 logging 阶段崩溃。

要求：

- 对 `captimgs` 单独计算可视化通道；3 通道时使用 `[0,1,2]`。
- 对 `target_images` / `est_images` 继续使用高光谱代表通道。
- 确保 concat 前所有可视化张量都是 3 通道。

### 5. PSF loss 仍依赖 hparams 被成功突变，缺少光学模型 guard

位置：

- `snapshotdepth_hs.py:341`
- `snapshotdepth_hs.py:343`
- `snapshotdepth_hs.py:622`
- `snapshotdepth_hs.py:623`

`dodo_depth` 构造时尝试把 `hparams.psf_loss_weight` 设为 `0.0`，但 loss 里仍是只要 `self.hparams.psf_loss_weight > 0` 就调用 `self.camera.psf_out_of_fov_energy()`。`DepthAwareDoDoForwardModel` 没有这个接口。这里不应该依赖 hparams 突变，而应显式限制 PSF loss 只在 `legacy_camera` 路径执行。

要求：

- `if self.hparams.psf_loss_weight > 0 and self.optical_model_type == 'legacy_camera': ...`
- `dodo_depth` 路径即便用户传了非零 `psf_loss_weight` 也不能调用 PSF-only API。

### 6. `dodo_depth` 缺少 128 输入尺寸硬校验

位置：

- `snapshotdepth_hs.py:345`
- `snapshotdepth_hs.py:348`

`DepthAwareDoDoForwardModel` 当前固定 128，但 `snapshotdepth_hs.py` 默认 `image_sz=512`。实现记录只说“打印提示”，实际代码没有尺寸 guard；如果用户忘记传 `--image_sz 128 --crop_width 0`，错误会在光学层内部才出现。

要求：

- `dodo_depth` 构造阶段检查 `image_sz == 128`。
- 第一轮建议同时要求 `crop_width == 0`，否则必须证明 crop 后 target/capt/est shape 一致。
- 给出清晰 `ValueError`，告诉用户应使用 `--image_sz 128 --crop_width 0`。

### 7. `optimize_optics=True` 时 dodo 光学参数 warmup 会被设为 `cnn_lr`

位置：

- `snapshotdepth_hs.py:68`
- `snapshotdepth_hs.py:69`
- `snapshotdepth_hs.py:72`
- `snapshotdepth_hs.py:78`
- `snapshotdepth_hs.py:79`

`configure_optimizers()` 在任何光学模型 `optimize_optics=True` 时都会把 camera 参数作为第一个 param group，但 `optimizer_step()` 只有 legacy camera 的第 0 组使用 `optics_lr`，dodo_depth 的第 0 组会落入 `else` 并使用 `cnn_lr` warmup。若启用 DoDo 光学优化，这会把 DOE 学习率放大到 CNN 学习率。

要求：

- 给 param group 增加明确标记，如 `name: optics/cnn`，warmup 按标记设置 lr。
- 或者当 `optical_model_type == 'dodo_depth' and optimize_optics=True` 时明确使用 `optics_lr`。
- 在 implementation notes 记录 dodo optics lr 默认/建议值。

## 测试缺口

上一轮只完成语法检查；运行时测试未执行。以下仍未完成：

- `DepthAwareDoDoForwardModel` smoke test。
- Dataset sample test，特别是 `depth_metric` 范围和几何变换一致性。
- `SnapshotDepthHS.forward(...)` 的 `legacy_camera` 和 `dodo_depth` batch smoke。
- `dodo_depth` one-step backward。
- `dodo_depth` fast-dev run。
- 旧 `torch_optics` smoke regression。

## 下一轮建议

下一轮不需要重新设计光学模型。Claude 应先修复上述阻断 bug，再在有 torch 的环境中完成必跑验证，并把结果追加到 `handoff/DoDo-change/implementation-notes.md`。
