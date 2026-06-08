# Review: DoDo-change Round 18 Forward-Norm Probe

Date: 2026-05-08

## Scope

Reviewed:

- `handoff/DoDo-change/implementation-notes.md` Section 18
- `handoff/DoDo-change/BEST_CHECKPOINT.md`
- `torch_optics/forward_dodo.py`
- `torch_optics/sensing.py`
- `snapshotdepth_hs.py`
- `models/simple_model_mamba.py`
- `infer_contect.py`

## Findings

### 1. Round 18 candidates should not be extended as-is

Severity: High

Both 40-epoch probes failed the crop and full-scene gates:

- Candidate A crop PSNR: 18.59 dB
- Candidate B crop PSNR: 18.45 dB
- Deploy 1 SAM worsened from the R17 best 0.541 rad to about 0.75 rad.
- Deploy 1 pseudo-RGB PSNR dropped from 26.39 dB to about 11.5-11.7 dB.
- Both candidates stayed worse than constant median-depth baseline on depth.

This does not prove a 260-epoch version could never recover, because the historical R12 run improved substantially after 120 epochs. However, Round 18 gives no positive signal that forward-normalization removal or background loss is addressing the actual full-scene failure. A long run on the same configuration is not a good next use of compute.

### 2. Forward normalization is not the primary current bottleneck

Severity: High

`dodo_forward_norm=none` and the legacy normalized path produced similar failures at 40 epochs. This matches the Round 17 inference-only result where second-stage normalization overrides did not rescue the checkpoint.

The next experiment should not keep searching only normalization variants. The more plausible bottleneck is that the current optical forward collapses 25 HS bands plus depth effects into only 3 RGB sensor channels before the decoder.

### 3. Current sensing layer hard-codes a 3-channel measurement

Severity: High

`torch_optics/sensing.py::SensingLayer` always maps the 25-band field into `[R, G, B]`. `snapshotdepth_hs.py` then sets `measurement_channels=3` for DoDo-depth. This is physically meaningful for RGB sensing, but it is a severe information bottleneck for joint HS reconstruction and depth inference.

Round 19 should add controlled multi-channel sensing modes while preserving the current RGB mode as the default. This is a diagnostic first, not a final optical claim.

### 4. Existing network can likely accept more measurement channels

Severity: Medium

`SimpleModelHS` already reads `hparams.measurement_channels` and adapts its input adapter accordingly. This makes a multi-channel sensing probe relatively contained:

- extend `SensingLayer`;
- pass sensing mode/channel count through `DepthAwareDoDoForwardModel`;
- validate `measurement_channels` before constructing the decoder;
- verify full-scene inference reloads checkpoints with non-3-channel measurements.

### 5. Selected checkpoint remains crop-only best

Severity: Medium

No Round 18 checkpoint should replace the R12 depth-best selection. The R12 depth-best checkpoint remains valid only as the best filtered 128x128 crop checkpoint, not as an approved full-scene model.

## Recommendation

Round 19 should be a measurement-capacity diagnostic:

1. Add opt-in multi-channel sensing modes with default behavior unchanged.
2. Run a production-like 8-channel spectral-bin candidate.
3. Run a 25-channel identity/oracle upper-bound candidate.
4. Evaluate both on crop validation and full-scene deploy 1/deploy 16 gates.
5. Use the result to decide whether the dominant failure is the 3-channel optical measurement bottleneck or the decoder/training formulation.

Do not run a 120/260 epoch continuation in Round 19 unless the short multi-channel candidate shows clear full-scene improvement.
