# Current Code Changes

## torch_optics/propagation.py

- 区域：`PropagationLayer.adjoint`。
- 目的：在当前离散padding/crop传播上生成严格匹配的相位共轭聚焦载波，不使用
  可能与网格不一致的连续Fresnel公式。
- 修改：新增crop零嵌入、共轭传播核和回裁组成的离散伴随算子。
- 潜在风险：FFT归一化或crop位置错误会使相位共轭符号失真。
- 验证：复数内积 `<y,Ax>=<A* y,x>` 数值测试通过；参考点中心强度提升测试。

## torch_optics/doe.py

- 区域：`DOEPixelPhaseLayer`。
- 目的：突破150阶平滑Zernike表达上限，采用HS-D论文同类逐像素未包裹相位变量。
- 修改：新增128×128相位参数；光学前向先按550 nm参考相位做2π物理高度包裹，
  再复用现有色散折射率模型；导出0到单相位周期的物理高度，同时保留未包裹变量。
- 潜在风险：逐像素高频结构可能利用离散网格，后续需结合实际制造横向分辨率判断。
- 验证：相位周期不变性、三波长宽带前向一致性和GPU短实验。

## torch_optics/forward_dodo.py

- 区域：`DepthAwareDoDoForwardModel` DOE构造。
- 目的：让PSF前向原生支持可复用的pixel-phase checkpoint。
- 修改：新增默认关闭的 `doe_parameterization=pixel_phase` 与参考波长参数；原有
  zernike默认和所有传播/卷积设置不变。
- 潜在风险：仅新模式改变state-dict参数形状，历史默认路径不受影响。
- 验证：现有DOE/PSF卷积回归与新模式构造测试。

## util/doe_preoptimization.py

- 区域：相位共轭初始化和DOE物理统计。
- 目的：从当前Prop1输入场与Prop3伴随场直接求参考点最佳相位载波，并审计包裹高度。
- 修改：新增1 m/550 nm默认参考点、seed相关0.05 rad扰动、聚焦增益记录；新增包裹
  高度范围/RMS与未包裹相位RMS指标。
- 潜在风险：单参考点初始化不代表最终多深度最优，只作为高MTF稳定起点。
- 验证：参考中心强度提升超过3倍；正式网格实测约4.6倍。

## scripts/preoptimize_psf_doe.py

- 区域：mode、CLI、初始化、优化约束和artifact。
- 目的：独立运行pixel-phase DOE可行性实验并保存可制造高度。
- 修改：新增 `pixelphase` mode和三个初始化参数；pixel phase不使用3 μm Zernike
  RMS球，而是始终限制到一个物理2π高度周期；保存wrapped height、unwrapped
  phase/height及完整checkpoint。
- 潜在风险：旧mode命令行为不变；pixelphase必须用对应模型恢复。
- 验证：16深度20/200步GPU预检，lr 0.01/0.05/0.1和邻近波段权重对照。

## tests/docs/handoff

- 区域：传播、DOE预优化测试与协作文档。
- 目的：固化物理周期、伴随、聚焦初始化、实验结论和下一步正式命令。
- 修改：新增测试并更新说明、实验台账、实现记录和调度入口。
- 潜在风险：无生产运行时影响。
- 验证：相关测试套件60项通过（提交前将再次执行）。
