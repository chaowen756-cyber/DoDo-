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
