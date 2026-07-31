# Implementation Notes

## 1. DOE-only 可行性入口

- 从 `psf卷积@06a6eb5` 新建分支 `DOE可编码性预优化实验`。
- 新入口不加载数据集和 CNN，只优化当前完整 PSF bank。
- rank-9 与 free-150 使用统一的 0.6 μm 初始化 RMS 和 3 μm 最大 RMS。
- 输出 DOE checkpoint、系数、高度、PSF、历史、摘要和可视化。
- CPU smoke 与16层短对照均完成；正式1000步实验需在可用GPU上执行。

## 2. Review 与验证

- 新增测试3项全部通过；原有 orthogonal-RMS/PSF 卷积回归37项全部通过。
- rank-9 与 free-150 最终 CLI smoke 均通过。
- `test/test_psf_regularization.py` 在当前 shell 因缺少 `pytorch_lightning`
  无法收集；本次直接调用的四个正则已由新增目标梯度测试覆盖。
- 当前 shell 的 NVIDIA 驱动不可用，因此没有在本轮冒充完成正式1000步GPU实验。

## 3. Fisher A-optimality 最后修改

- 依据同一 HS-D 任务的 Baek ICCV 2021 补充材料第6节，将单色点源
  `(x,y,depth,wavelength)` Fisher A-optimality 加入 DOE-only 目标。
- 保留 MTF 和能量护栏，防止 Fisher 优化通过牺牲空间带宽或 capture 取巧；
  原相邻光谱/深度 cosine 继续作为直接分离辅助项。
- 使用一个像素/深度层/波长层作为导数单位，并显式保留 RGB 响应与 PSF
  capture 强度。
- 16深度20步测试确认 `Adam lr=1e-2` 稳定；没有照搬联合训练的 `1e-4`。
- Fresnel 投影在当前 rank-9 基底中拟合较差，且在 free-150 中初始 Fisher
  不优，因此保留相同 seed、相同物理 RMS 的随机初始化。
- 最终新增测试4项、原有 DOE/PSF 回归37项全部通过。

## 4. Task-Fisher 深度/波长聚焦

- 上一轮结果显示 free-150 的 full Fisher 明显改善，但主要来自 x/y，直接
  光谱/深度 cosine 几乎不变。
- 本轮仍对完整 Fisher 求逆，只把 CRLB trace 权重改为 x/y 各0.1、depth与
  wavelength各1.0；这保留了 nuisance coupling，避免乐观的2×2子矩阵指标。
- 新增 full/task/weighted A-optimality、四参数 CRLB 指标，便于判断自由度到底
  花在何处。
- 完整16深度20步 free-150 preflight 稳定，task A-optimality下降约9.9%。
- 新增测试5项、原有 DOE/PSF 回归37项全部通过。

## 5. Optical-only 编码与 RMS 边界优化

- 双 seed 正式结果表明 task Fisher 改善没有转化成单色 PSF 波长形状分离；
  新增对总强度和 RGB response 不敏感的 optical-only PSF cosine 目标。
- 波长同时比较1/2/4 band offset，深度比较相邻层；sensor-weighted Fisher
  和分离指标继续计算，确保光学图样改善仍可由真实 RGB 测量观察。
- 历史同时记录 warm-up 训练目标与固定完整目标，最佳 DOE 改为始终按完整目标
  选择，消除早期日志假上升。
- 3 μm RMS 约束改为候选更新的边界切向修正加安全回缩。压力测试发现并删除了
  不正确的 Adam 一阶动量投影；修复后从2.99 μm开始的50步完整损失稳定下降。
- 新增 optical shape 尺度不变性/梯度测试和 RMS 边界测试；完整16深度30/200步
  GPU preflight 均稳定。

## 6. 逐像素包裹相位DOE

- free-150双seed各3000步仍停在相邻波长cosine约0.9898，确认瓶颈是参数化
  表达上限，不再继续增加Zernike阶数或步数。
- 依据HS-D论文补充材料的未包裹相位优化方法，新增128×128 pixel-phase模式；
  为保证宽带物理一致性，每次前向都先包裹到550 nm的单个2π高度周期。
- 新增Prop3严格离散伴随，从当前Prop1场直接构造1 m/550 nm相位共轭初始化；
  参考中心强度约提升4.6倍，不引入50 mm/6.45 μm等外部参数。
- 200步学习率对照选择Adam lr=0.1；物理包裹模式完整损失约0.797、task Fisher
  约3.66e5，明显优于free-150 3000步的约0.864/5.2e5。
- 邻近波段权重10消融收益极小且损失Fisher，正式配置保留光谱权重5与
  offset 1/2/4。
- 新增相位周期宽带一致性、相位共轭聚焦和传播伴随测试；相关套件60项通过。
