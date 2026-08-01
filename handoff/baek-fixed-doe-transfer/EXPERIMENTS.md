# Experiments

## baek-native-doe-psf-preflight-20260801

- 日期：2026-08-01
- 类型：真实 DOE 原生网格/PADO-compatible PSF 预检
- checkpoint/artifact root：无
- 配置：375x375 高度右/下补零到 376；8 µm、NOA61、PADO spherical source、
  integer-centered full circular aperture、50 mm PADO Fresnel linear convolution；
  当前 16 inverse-depth centers、25 wavelengths、中心 129x129 PSF。
- 指标：height max `1.023254e-6 m`；PSF `16x25x129x129` 全 finite；校准后
  kernel sum 约 1；裁剪前 capture min/mean/max
  `0.237906/0.361023/0.442250`；光学 trainable params `0`。
- 状态：通过。

## baek-native-fast-dev-run-20260801

- 设备：物理 GPU 1，batch 1，单个 train batch + 单个 validation batch。
- 结果：完整数据加载、native PSF、RGB sensing、34M 网络 forward/backward、Adam
  step、validation 和三类 checkpoint 写出均完成；camera 参数量为 0。
- artifact：`/tmp/baek_native_fastdev_20260801`（临时验证，不作为论文实验）。
- 状态：通过。

早期 `consistent_grid_v1` 的 128x128 area-downsample 预检仅作为缩放迁移备选，
不再是主实验口径。

## baek-native-psf-parity-20260801

- 参考端：实际 PADO GitHub `12c57df` API；25 wavelengths x 20 notebook depths。
- 当前 full-grid：cosine mean/min `0.974696/0.537816`；NRMSE
  mean/median/max `0.150379/0.097317/1.106150`。
- PADO scalar-source 隔离：保持当前 DOE/pupil/propagator，只替换球面波标量计算；
  NRMSE mean/max `4.79e-6/1.62e-5`，cosine 约 1。
- 根因：当前 vectorized float32 wavelength tensor 对数百万至数千万弧度球面相位
  的舍入/取模误差；远距离更明显。
- 决策：正式训练暂不启动；先修正 native spherical source，再重复 500-PSF parity。
- artifacts：`论文实验/PSF卷积/baek_native_psf_parity_20260801/`。
