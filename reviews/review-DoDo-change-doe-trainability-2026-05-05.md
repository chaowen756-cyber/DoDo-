# Review: DoDo-change DOE trainability check

## 结论

当前 Claude 正在执行或准备执行的命令虽然包含：

```bash
--optimize_optics --optics_lr 1e-6 --psf_loss_weight 0
```

但按当前代码默认配置，**不会真正更新 DoDo DOE 的 `zernike_coeffs`**。

原因：

- `snapshotdepth_hs.py` 构造 `DepthAwareDoDoForwardModel(...)` 时没有传 `doe_type_a`。
- `DepthAwareDoDoForwardModel` 默认 `doe_type_a="Zeros"`。
- `torch_optics.doe.DOELayer` 的 `Zeros` 分支创建：

```python
self.zernike_coeffs = nn.Parameter(coeffs, requires_grad=False)
```

因此 `--optimize_optics` 只会尝试优化 camera 参数；但默认 `doe1.zernike_coeffs.requires_grad=False`，zernike coefficients 不会被更新。

`--psf_loss_weight 0` 不是问题。它只关闭 PSF 正则，不会阻断来自重建 loss 的 DOE 梯度。真正的问题是 DOE 类型默认是 frozen `Zeros`。

## 必须修正

Claude 在任何长训练前必须：

1. 增加 `dodo_depth` 的 DOE 类型配置，例如 `--dodo_doe_type New`。
2. 将该参数传入：

```python
DepthAwareDoDoForwardModel(..., doe_type_a=hparams.dodo_doe_type, train_c=hparams.optimize_optics)
```

3. 正式训练命令必须包含：

```bash
--dodo_doe_type New
```

4. preflight 必须打印并记录：

```text
camera.doe1.zernike_coeffs.requires_grad = True
```

5. optimizer param group 中必须包含 `doe1.zernike_coeffs`。
6. backward 后必须确认 `doe1.zernike_coeffs.grad` 非 `None` 且 finite。
7. optimizer step 后必须执行 `camera.clamp_parameters_()`。

## 对当前已启动训练的处理

如果 Claude 已经用缺少 `--dodo_doe_type New` 的命令启动训练，应停止该训练，记录日志和原因，然后在完成上述修正与 preflight 后重启。否则得到的结果只能算“冻结 DOE + decoder 训练”，不符合用户要求的“嵌入 DOE 开启训练模式”。
