# TASK: DoDo-change optical forward parity review

## Context

The user rewrote the original TensorFlow/Keras DoDo optical forward model under `torch_optics/`.
The goal is to verify whether the PyTorch implementation is functionally equivalent to the original code under `optics/`, especially the optical formulas.

Current code areas:

- Original TensorFlow/Keras model:
  - `optics/Forward_DM_Spiral128.py`
  - `optics/Prop_layer_v4_sizes.py`
  - `optics/DOE_layer_v4_128.py`
  - `optics/Sensing_layer.py`
- PyTorch rewrite:
  - `torch_optics/forward_dodo.py`
  - `torch_optics/propagation.py`
  - `torch_optics/doe.py`
  - `torch_optics/sensing.py`
  - `torch_optics/utils_fft.py`

## Codex Findings

### Forward chain

The PyTorch chain in `torch_optics/forward_dodo.py` matches the original high-level chain in `optics/Forward_DM_Spiral128.py`:

1. `Propagation(Mp=128, L=0.01, zi=0.06, Trai=False)`
2. first DOE, either `DOE` or `DOE_Free`
3. `Propagation(Mp=128, L=0.006, zi=0.05, Trai=False)`
4. fixed `DOE_type='Spiral'`
5. `Propagation(Mp=128, L=0.0048, zi=0.01, Trai=False)`
6. `Sensing(Ms=128, Trai=False)`

The wrapper functions `Forward_DM_Spiral` and `Forward_DM_Spiral_Free` use NHWC input/output, matching Keras. The internal PyTorch module uses NCHW and converts at the wrapper boundary.

### Propagation formula

The propagation formula is functionally aligned:

- Original transfer function:
  `H(lambda) = exp((-i * pi * lambda * z) * (fx^2 + fy^2))`
- PyTorch transfer function:
  `kernel = exp((-1j * pi * lambda * z_eff) * freq2)`

The frequency grid, `fftshift(input) -> fft2 -> multiply H -> ifft2 -> ifftshift` sequence, and omitted global phase term are consistent with the original.

Potential parity caveat:

- Original `z` is a Keras weight with `NonNeg()` constraint. PyTorch clamps `z` in forward. With `trainable_z=False` in the DoDo forward model, this is equivalent for current defaults. If standalone trainable propagation is used, optimizer behavior is not identical.

### DOE formula

The DOE phase formula is mostly aligned:

- Original `DOE` uses:
  `exp(1j * (2 * 10.0 * pi / lambda) * IdLens(lambda) * Hm)`
- PyTorch `DOELayer` uses `phase_scale_mode='legacy_doe'`, which maps to `20*pi`, matching original `DOE`.
- Original `DOE_Free` uses:
  `exp(1j * (2 * pi / lambda) * IdLens(lambda) * Hm)`
- PyTorch `DOEFreeLayer` uses `phase_scale_mode='legacy_free'`, which maps to `2*pi`, matching original `DOE_Free`.
- `IdLens(lambda)` matches:
  `1.5375 + 0.00829045 * (lambda_um ** -2) - 0.000211046 * (lambda_um ** -4) - 1`.
- The original loads `P` but does not multiply it into the active phase formula because the pupil-mask multiplication line is commented. PyTorch defaults `use_pupil_mask=False`, matching active original behavior.
- Spatial downsampling by `Mesce / Mdoe` matches the original slicing behavior.

Potential parity caveats:

- Original Keras `MinMaxNorm(min_value=-1.0, max_value=1.0, axis=2)` constrains a coefficient-vector norm, while PyTorch `clamp_parameters_()` clamps each coefficient element to `[-1, 1]`. This is not equivalent during training.
- Original `DOE_type='New'` random initialization uses Python `random.random()`. PyTorch uses `torch.uniform_`. Same distribution, different seed/source behavior.
- Original `DOE_Free` can generate and save a missing Zernike basis via `poppy`; PyTorch raises `FileNotFoundError` if the basis file is missing. The current assets include the required 150 and 200 term basis files.

### Sensing formula

The sensing formula is aligned for normal nonzero inputs:

- Original:
  `y_rgb = sum_lambda(abs(input_lambda) * sensor_rgb_lambda)`
  then `y_final = y_final / reduce_max(y_final)`
- PyTorch:
  same weighted absolute spectral sum, then global max normalization by default.

Potential parity caveat:

- PyTorch adds `eps=1e-8` to the denominator. Original TensorFlow divides by `reduce_max(y_final)` directly. This differs for zero or extremely small outputs: original can produce `NaN`, PyTorch returns finite values.

### Assets

Codex verified that these asset files are byte-identical between `optics/` and `torch_optics/assets/`:

- `Base_zernike_128x128_nopadd.mat`
- `Spiral_128x128_nopadd.mat`
- `Sensor_25_new3.mat`
- `zernike_volume1_128_Nterms_150.npy`
- `zernike_volume1_128_Nterms_200.npy`

### Local execution status

TensorFlow/Keras are not installed in the current environment, so Codex could not run a direct TensorFlow-vs-PyTorch numeric parity test.

Codex did run a PyTorch-only smoke check:

- `Forward_DM_Spiral(DOE_typeA='Zeros')` on random NHWC input returns shape `(1, 128, 128, 3)`, finite values, max `1.0`.
- `Forward_DM_Spiral_Free(DOE_typeA='Zeros', Nterms=150)` returns shape `(1, 128, 128, 3)`, finite values, max `1.0`.

## Claude Task

Do not implement code changes until the user explicitly asks for fixes.

Perform a verification-focused pass and write the result to `handoff/DoDo-change/implementation-notes.md`.

Required work:

1. Re-read the original and PyTorch files listed above.
2. Confirm or correct the Codex formula-parity findings.
3. If TensorFlow/Keras are available in your runtime, run a small numeric parity check for the following cases:
   - `Forward_DM_Spiral(DOE_typeA='Zeros')`
   - `Forward_DM_Spiral_Free(DOE_typeA='Zeros', Nterms=150)`
   - Optional: `DOE_typeA='New'` with manually synchronized coefficients, if practical.
4. If TensorFlow/Keras are not available, state that clearly and provide a formula-level verification only.
5. Document any functional differences that affect:
   - default inference parity
   - trainable DOE parity
   - trainable propagation parity
   - zero-input or near-zero sensing behavior
   - missing asset behavior

## Non-goals

- Do not modify `torch_optics/`.
- Do not modify `optics/`.
- Do not refactor model code.
- Do not change assets.
- Do not add tests unless the user explicitly asks.

## Acceptance Criteria

Claude is done when `handoff/DoDo-change/implementation-notes.md` contains:

- A clear conclusion on whether default PyTorch inference is formula-equivalent to the original active TensorFlow code.
- A list of exact behavioral differences, if any.
- Any numeric parity result that could be run, including input shape, mode, tolerance, and observed max/mean error.
- A statement that no source code was modified.

## Stop Condition

After writing `handoff/DoDo-change/implementation-notes.md`, stop and return control to Codex/user for review.

## Codex Review 2026-05-03

Review file:

- `reviews/review-DoDo-change-2026-05-03.md`

上一轮目标完成情况：

- 已完成：公式级验证、PyTorch-only smoke check、资产一致性确认、无法进行 TensorFlow/Keras 数值 parity 的环境限制说明。
- 未完成：用户原始目标是两个版本代码功能完全一致；当前仍存在若干默认或扩展行为差异，需要下一轮修正。

需要修正的问题：

1. `torch_optics/sensing.py` 默认使用 `eps=1e-8`，导致全零输入时返回 0；原版 `optics/Sensing_layer.py` 会执行 `y / reduce_max(y)` 并产生 `NaN`。如果目标是完全一致，默认行为必须对齐原版。
2. `torch_optics/doe.py` 中 `DOELayer` 的 `clamp_parameters_()` 是逐元素 clamp；原版 `DOE` 的 `MinMaxNorm(axis=2)` 对 `(1,1,N)` 系数向量做 L2 norm 约束。训练 `DOE_type='New'` 时不等价。
3. `DOEFreeLayer` 在缺失 `zernike_volume1_<Mdoe>_Nterms_<Nterms>.npy` 时直接报错；原版会用 `1e-6 * poppy.zernike.zernike_basis(...)` 自动生成并保存。
4. `handoff/DoDo-change/implementation-notes.md` 中“原版随机初始化 11 个 DOE 系数”的记录错误。源码实际初始化索引 `0..11`，共 12 个；PyTorch 也是 12 个。此处只需修正文档，不要改这段代码。

## Next Claude Task

Implement the remaining parity fixes. Do not modify `optics/`.

Required code changes:

1. In `torch_optics/sensing.py`, make the default normalization match the original active TensorFlow behavior:
   - default denominator for `normalize_mode='global'` must be exactly `torch.max(y)`;
   - default behavior must not add `1e-8`;
   - if keeping `eps` as an optional robustness parameter, default it to `0.0` so the default path remains legacy-compatible.
2. In `torch_optics/doe.py`, make `DOELayer` parameter projection match Keras `MinMaxNorm(axis=2)` for original `DOE` coefficients:
   - for `DOELayer`, project the full coefficient vector so L2 norm is at most `1.0`;
   - do not apply this vector-norm rule to fixed `Zeros` or `Spiral` behavior beyond preserving current frozen/fixed semantics;
   - `DOEFreeLayer` may keep elementwise clamp semantics because original `DOE_Free` coefficient shape is `(N,1,1)` with `axis=2`.
3. In `torch_optics/doe.py`, restore `DOEFreeLayer` missing-basis behavior:
   - if `zernike_volume1_<Mdoe>_Nterms_<n_terms>.npy` is missing, generate it with `1e-6 * poppy.zernike.zernike_basis(nterms=n_terms, npix=Mdoe, outside=0.0)`;
   - save it to the resolved assets directory if possible;
   - if `poppy` is unavailable, raise an explicit error explaining that original-compatible basis generation requires `poppy`.
4. Update `handoff/DoDo-change/implementation-notes.md`:
   - correct the false “11 coefficients” claim to “12 coefficients in both versions”;
   - document all source files changed;
   - document verification commands and results.

Required verification:

- Run PyTorch smoke tests for `Forward_DM_Spiral(DOE_typeA='Zeros')` and `Forward_DM_Spiral_Free(DOE_typeA='Zeros', Nterms=150)`.
- Run a zero-input check for `SensingLayer` showing default behavior now matches original semantics by producing `NaN` when denominator is zero.
- Run a small `DOELayer(DOE_type='New')` projection check showing a coefficient vector with norm greater than 1 is projected to norm <= 1.
- If TensorFlow/Keras are available, run a direct numeric parity check; otherwise state clearly that they are unavailable.

Stop after updating code and `handoff/DoDo-change/implementation-notes.md`.
