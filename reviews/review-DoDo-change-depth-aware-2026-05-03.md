# Review: DoDo-change depth-aware first-DOE forward model

## Findings

1. **`DepthAwareDoDoForwardModel` 缺少 `clamp_parameters_()`，训练第一 DOE 时会丢失约束入口。**
   `torch_optics/forward_dodo.py:101`-`205` 新增了 depth-aware 模型，但没有像 `DoDoForwardModel.clamp_parameters_()` 一样暴露参数投影方法。上一轮已经把 `DOELayer.clamp_parameters_()` 改成匹配 Keras `MinMaxNorm(axis=2)` 的 L2 投影；depth-aware 路径如果用于训练 `DOE_type='New'`，调用方无法通过统一接口执行这个约束，只能手动访问 `model.doe1`。这会影响“后续修改都基于第一个 DOE 处理”的训练路径。

2. **深度图尺寸没有主动校验，不满足 TASK 的输入验证要求。**
   `torch_optics/forward_dodo.py:170`-`192` 只把 `(B,H,W)` depth 转为 `(B,1,H,W)`，没有检查 batch/spatial size 是否与 spectral 一致。实际用 `(1,64,64)` depth 输入 `(1,128,128,25)` spectral 时，会在 `spectral * mask` 抛出底层 broadcast `RuntimeError`，而不是清晰的 `ValueError`。TASK 明确要求 “Validate that depth spatial size matches spectral spatial size”。

3. **depth-aware 构造参数缺少合法性校验。**
   `torch_optics/forward_dodo.py:104`-`131` 接收 `depth_min`, `depth_max`, `num_depth_layers`，但没有校验 `depth_min < depth_max`、`num_depth_layers >= 1`。例如 `num_depth_layers=0` 会构造空 `ModuleList`，forward 时 `y_sum=None` 再进入 `_normalize_once`，错误会很晚且不明确。

4. **无深度 `DoDoForwardModel` 仍然无条件使用第二 DOE，和用户“先忽略第二 DOE”的全局意图不完全一致。**
   depth-aware 新路径已通过 `use_second_doe=False` 默认绕过 DOE2；但旧的 `DoDoForwardModel.forward()` 在 `torch_optics/forward_dodo.py:92`-`96` 仍然无条件执行 `self.doe2(x)`。如果用户后续还会用 `Forward_DM_Spiral(...)` 做第一 DOE 相关实验，这条路径仍不是“第二 DOE 保留但不参与”。建议给 `DoDoForwardModel` 和 `Forward_DM_Spiral` / `Forward_DM_Spiral_Free` 增加 `use_second_doe` 参数，并按当前 change 的 first-DOE 模式默认绕过或至少显式可控。

5. **格式参数校验不完整，错误输入会被静默当成 NCHW。**
   `DepthAwareDoDoForwardModel._to_nchw()` 和 `_from_nchw()` 在 `torch_optics/forward_dodo.py:160`-`168` 只特殊处理 `"nhwc"`，其他任何字符串都会直接返回原张量。旧 `DoDoForwardModel` 已经对非法格式抛 `ValueError`；新类应保持一致。

## Completed

- 已完成 depth-aware 模型主体：`DepthAwareDoDoForwardModel` 和 `Forward_DM_Spiral_Depth` 已添加并从 `torch_optics/__init__.py` 导出。
- 已完成第二 DOE 在 depth-aware 路径默认 bypass：`use_second_doe=False` 时不调用 `doe2`，`True` 时可启用。
- 已完成固定深度分层：默认 `[0.4, 2.0]`，bin center 作为每层 `prop1` 的 `zi`。
- 已完成 hard mask、越界 clamp、每层 unnormalized sensing 后再统一归一化的融合策略。
- 本地 review smoke test 通过：`Forward_DM_Spiral`、`Forward_DM_Spiral_Free`、`Forward_DM_Spiral_Depth(num_depth_layers=4)` 都返回 `(1,128,128,3)` 且非零随机输入输出有限。

## Review Test Notes

Review 期间运行了轻量脚本：

```text
Forward_DM_Spiral(DOE_typeA='Zeros') -> (1, 128, 128, 3), finite=True, max=1.0
Forward_DM_Spiral_Free(DOE_typeA='Zeros', Nterms=150) -> (1, 128, 128, 3), finite=True, max=1.0
Forward_DM_Spiral_Depth(num_depth_layers=4) -> (1, 128, 128, 3), finite=True, max=1.0
bin edges: [0.4, 0.8, 1.2, 1.6, 2.0]
z centers: [0.6, 1.0, 1.4, 1.8]
bad depth shape currently raises low-level RuntimeError instead of explicit ValueError
```

## Required Next Step

下一轮 Claude 应做收尾修正，不需要重做物理设计：

- 给 `DepthAwareDoDoForwardModel` 增加 `clamp_parameters_()`，至少转发到 `doe1`，`use_second_doe=True` 时也可转发 `doe2`。
- 增加 depth-aware 输入校验：spectral 必须是 4D；depth 必须是 `(B,H,W)` 或 `(B,1,H,W)`；batch 与 spatial 必须一致；非法格式参数要抛 `ValueError`。
- 增加构造参数校验：`depth_min < depth_max`、`num_depth_layers >= 1`。
- 给无深度 `DoDoForwardModel` / `Forward_DM_Spiral` / `Forward_DM_Spiral_Free` 增加 `use_second_doe` 显式参数，让旧路径也能在 first-DOE 模式下绕过第二 DOE。
- 在 `handoff/DoDo-change/implementation-notes.md` 记录修复和测试结果。测试可在终端执行 `conda activate DoDo` 后运行；缺少包由 Claude 在该环境中安装。
