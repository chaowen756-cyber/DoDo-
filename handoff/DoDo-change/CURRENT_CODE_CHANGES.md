# 当前代码修改记录：DoDo-change

> Claude 在每轮开始前重置此文件。
> 仅记录当前轮的代码修改。

## 轮次

26（DOE0 RGB vs spectral_bins 光谱编码容量对比诊断）

## 状态

已完成。4 组 sensing config 对比运行完毕，临时文件已清理。

## 代码修改清单

### scripts/compare_doe0_rgb_vs_spectral_bins.py（新增）

| 项目 | 内容 |
|------|------|
| **位置** | `/root/autodl-tmp/scripts/compare_doe0_rgb_vs_spectral_bins.py`（新增） |
| **修改目的** | 在 Zeros DOE 下对比 4 种 sensing config（rgb 3ch, spectral_bins 6ch, 9ch, 12ch）的光谱编码容量。 |
| **修改内容** | Wrapper 脚本：对每种 config 初始化 DepthAwareDoDoForwardModel，运行 Diagnostic B/C/D（复用 diagnose_measurement_spectral_capacity.py），收集指标后立即删除临时目录。输出 comparison_summary.json, comparison_table.csv, conclusion.md, command.txt。try/finally 保证临时目录清理。 |
| **潜在风险** | 无。独立诊断脚本，使用 Zeros DOE（不加载 checkpoint），不修改模型。 |
| **验证方式** | 端到端运行成功：4 组 config 全部完成，临时 _tmp_* 目录已删除，输出目录仅含 4 个文件。 |

### 未修改文件

未修改 `torch_optics/sensing.py`（已有 spectral_bins 实现满足需求）, `torch_optics/forward_dodo.py`, `snapshotdepth_hs.py`, `models/simple_model_mamba.py`, DOE, decoder, loss, propagation, depth layering。
