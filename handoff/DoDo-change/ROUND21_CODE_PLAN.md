# Round 21 代码落实方案：DoDo Intensity Sensing

## 1. Scope

本方案仅限于以下改造：

- 在 SensingLayer 中新增 opt-in 物理强度测量模式（`abs(field)^2`）
- 兼容旧 checkpoint 的 CLI/hparams 管道
- 修正 full-scene 指标合约（full PSNR、SAM、pseudo-RGB PSNR 为主指标）
- 补充低前景 tile 背景诊断
- 标注 oracle-simulation 标签

本方案不解决真正的 measurement-only 联合重建。那属于后续架构/公式变更。

## 2. 拟修改文件清单

| 文件 | 修改类型 | 是否影响默认行为 |
|------|----------|----------------|
| `torch_optics/sensing.py` | 修改 SensingLayer | 否（amplitude 保持旧行为） |
| `torch_optics/forward_dodo.py` | 传递 sensor_measurement | 否 |
| `snapshotdepth_hs.py` | 新增 CLI + 传递参数 | 否 |
| `snapshotdepth_trainer_hs.py` | 无需修改* | — |
| `infer_contect.py` | 新增 oracle_simulation 标签 + 低前景诊断 | 否 |
| `round21_lowfg_diag.py`（新增） | 低前景 tile 背景诊断脚本 | 新增 |

*`snapshotdepth_trainer_hs.py` 的 `add_model_specific_args` 已在 `snapshotdepth_hs.py` 中定义，新 CLI 参数通过现有 `parent_parser` 机制自动继承。

## 3. 各文件具体修改方案

### 3.1 `torch_optics/sensing.py`

**修改位置**：`SensingLayer.__init__()` 和 `SensingLayer.forward()`

**新增参数**：
```python
sensor_measurement: str = "amplitude"
```
- `"amplitude"`：保持当前 `torch.abs(x)` 行为（默认，兼容旧 checkpoint）
- `"intensity"`：使用 `torch.abs(x) ** 2`（物理强度测量）

**forward() 修改**：
```python
x_abs = torch.abs(x).to(torch.float32)
if self.sensor_measurement == "intensity":
    x_abs = x_abs ** 2
# 后续 rgb / spectral_bins / identity 路径不变
```

**验证方法**：按现有 unit test 逻辑做 deterministic 复振幅 tensor 验证（见 Section 4）。

**风险**：强度测量的值域范围可能增大（动态范围约 3×），需要确认归一化层（`_normalize_once` / `dodo_measurement_norm`）不会产生数值问题。`amplitude` 默认保持所有旧行为不变。

### 3.2 `torch_optics/forward_dodo.py`

**修改位置**：
- `DepthAwareDoDoForwardModel.__init__()` — 新增 `sensor_measurement: str = "amplitude"` 参数
- `self.sensing_unnorm = SensingLayer(..., sensor_measurement=sensor_measurement)`
- `Forward_DM_Spiral_Depth(...)` — 新增 `sensor_measurement="amplitude"` 参数并传递

**实现方式**：与 `measurement_norm_mode`、`sensing_mode` 完全一致的传参模式。

**验证方法**：构造固定 seed 的 `DepthAwareDoDoForwardModel`，对比 `amplitude` vs `intensity` 输出 shape、finite 状态和数值范围。

### 3.3 `snapshotdepth_hs.py`

**修改位置**：
- `add_model_specific_args()` — 新增 CLI 参数
- `__build_model()` — 读取并传递 `dodo_sensor_measurement` 给 `DepthAwareDoDoForwardModel`

**新增 CLI**：
```python
parser.add_argument('--dodo_sensor_measurement', type=str, default='amplitude',
                    choices=['amplitude', 'intensity'],
                    help='DoDo sensor measurement type')
```

**hparams 记录**：自动通过 `save_hyperparameters()` 记录，checkpoint reload 自动使用保存值。

**验证方法**：`amplitude` 默认值训练 1 epoch，确认 `metrics.json` / `hparams.json` 值正确。`intensity` 模式需通过 finite smoke 验证。

### 3.4 `infer_contect.py`

**修改位置**：
- `metrics_full_scene.csv` 输出行新增 `oracle_simulation` 标签列
- Measurement 统计输出增加 sensor_measurement 模式记录

**修改内容**：
```python
# metrics_full_scene.csv header 新增最后一列
'oracle_simulation\n'
# 每行末尾追加 'True'
```

- 低前景 tile 诊断：检查 `vm_t.mean() < 0.1` 的 tile，记录其坐标、valid_ratio、spectral_sum、captimgs stats，保存到 `skipped_tiles.csv`

**验证方法**：对 deploy 1 已有 checkpoint 跑一次 full-scene gate，确认 `metrics_full_scene.csv` 包含 `oracle_simulation=True`。

### 3.5 `round21_lowfg_diag.py`（新增诊断脚本）

**目的**：在低前景 tile（valid_ratio < 10%）上运行 Diagnostic C（背景贡献）。

**实现方式**：
- 加载 deploy 1 / deploy 16 场景
- 扫描所有 128×128 tile，找到 valid_ratio < 0.1 的 tile（至少 3 个）
- 每个 tile 并行运行：full measurement vs foreground-only measurement vs background-only measurement
- 输出：`diag_lowfg_tiles.csv`（tile 坐标、valid_ratio、bg_fraction）、`diag_lowfg_summary.json`

**验证方法**：至少 3 个低前景 tile 的背景能量占比数据。

## 4. Validation Plan

按顺序执行，前一步不通过则停止：

### Step 1: SensingLayer 单元 smoke test

```python
# 固定 deterministic 复振幅 tensor
torch.manual_seed(42)
x = torch.randn(1, 25, 128, 128, dtype=torch.complex64)
s_amp = SensingLayer(sensing_mode='rgb', sensor_measurement='amplitude')
s_int = SensingLayer(sensing_mode='rgb', sensor_measurement='intensity')
y_amp = s_amp(x)       # 与旧代码输出一致
y_int = s_int(x)        # 新行为
# 验证 y_amp.shape == y_int.shape, finite, y_int 值域 ≥ y_amp
```

通过标准：y_amp == 旧 SensingLayer 输出（精确匹配或 < 1e-6 差异），y_int finite。

### Step 2: RGB backward-compatibility test

```python
# 以默认 amplitude 模式构造 DepthAwareDoDoForwardModel
cam_new = DepthAwareDoDoForwardModel(..., sensor_measurement='amplitude')
cam_old = DepthAwareDoDoForwardModel(...)  # 无 sensor_measurement 参数
# 固定输入，比较输出
```

通过标准：cam_new 输出 == cam_old 输出（精确匹配）。

### Step 3: DoDo-depth intensity finite forward smoke

```bash
# 构造 intensity 模式 camera
python -c "
from torch_optics.forward_dodo import DepthAwareDoDoForwardModel
import torch
cam = DepthAwareDoDoForwardModel(sensor_measurement='intensity', ...)
# 随机 HS + 随机 depth
hs = torch.rand(1, 25, 128, 128)
d = torch.ones(1, 128, 128) * 1.0
out = cam(hs, d)
assert torch.isfinite(out).all(), 'Non-finite in intensity mode!'
print('Intensity mode: finite OK, shape:', out.shape)
"
```

通过标准：输出 finite、shape 正确。

### Step 4: 1-epoch preflight（仅当 Steps 1-3 全部通过后）

```bash
python snapshotdepth_trainer_hs.py \
  --experiment_name round21_intensity_preflight_v1 \
  --dodo_sensor_measurement intensity \
  --optical_model dodo_depth ... \
  --batch_sz 1 --max_epochs 1 --limit_train_batches 2 --limit_val_batches 2
```

通过标准：exit 0、nonfinite_count=0、hparams 记录 `dodo_sensor_measurement=intensity`。

### Step 5: 低前景 tile 诊断

运行 `round21_lowfg_diag.py`，收集至少 3 个低前景 tile 的背景能量数据。

## 5. Metric Contract Plan

以下变更在 Step 4 preflight 通过后生效：

| 旧指标 | 新地位 | 说明 |
|--------|--------|------|
| `hs_psnr_masked` | Secondary | 仅作为诊断参考 |
| `hs_psnr_full` | **Primary** | 完整图像 PSNR |
| `sam_hs_masked` | **Primary** | 光谱角度误差 |
| `pseudo_rgb_psnr` | **Primary** | 三波段可视化 PSNR |
| `depth_mae_m` | **Primary** | 米制深度 MAE |
| `mae_vs_median` | **Primary** | 相对常数 baseline 的改善率 |
| `oracle_simulation` | **Metadata** | 标注使用 GT depth 合成测量 |

`metrics.json` 同时保存 full/masked 两套指标。`metrics_full_scene.csv` 新增 `oracle_simulation` 列。

## 6. Risks and Rollback

| 风险 | 缓解措施 |
|------|----------|
| intensity 模式动态范围增大导致数值不稳定 | `_normalize_once` 已有 eps guard；preflight 验证 nonfinite_count |
| intensity 模式训练可能需要不同 lr | 先只跑 preflight，不直接跑长训；如 loss scale 差过大记录建议 |
| 旧 amplitude checkpoint 被误用于 intensity 推理 | checkpoint hparams 自动记录 sensor_measurement，reload 保持原值 |
| 旧实验不可比较于 intensity 模式 | 所有 intensity 模式实验使用独立 artifact root 和 experiment_name |

回滚方法：将 `--dodo_sensor_measurement` 默认值设为 `amplitude`，所有旧命令行不受影响。

## 7. Stop Condition

等待 Codex review 后再执行代码修改。
