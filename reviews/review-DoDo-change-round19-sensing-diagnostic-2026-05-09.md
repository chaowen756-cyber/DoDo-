# Review: DoDo-change Round 19 Multi-Channel Sensing Diagnostic

Date: 2026-05-09

## Scope

Reviewed:

- `handoff/DoDo-change/implementation-notes.md` Section 19
- `handoff/DoDo-change/BEST_CHECKPOINT.md`
- `handoff/DoDo-change/CURRENT_CODE_CHANGES.md`
- `handoff/DoDo-change/EXPERIMENTS.md`
- `torch_optics/sensing.py`
- `torch_optics/forward_dodo.py`
- `snapshotdepth_hs.py`
- `infer_contect.py`

## Findings

### 1. Round 19 did not produce a promotable model

Severity: High

Both Round 19 candidates failed the crop and full-scene gates:

- 8-channel `spectral_bins`: crop PSNR 14.20 dB, deploy 1 SAM 0.856 rad, deploy 1 pseudo-RGB PSNR 17.59 dB.
- 25-channel `identity`: crop PSNR 18.69 dB, deploy 1 SAM 0.635 rad, deploy 1 pseudo-RGB PSNR 15.40 dB.
- Both remain worse than the R17/R12 260-epoch baseline on full-scene quality.
- Neither replaces the selected R12 depth-best crop checkpoint.

### 2. The 40-epoch probe is now the dominant confounder

Severity: High

Round 19 does not prove that multi-channel sensing is ineffective. It proves that this family of from-scratch 40-epoch probes is too undertrained to answer the question.

Historical reference matters:

- R12 needed roughly 120+ epochs to reach the mid-20 dB PSNR range.
- R12 needed around 220+ epochs to reach the best depth-MAE checkpoint.
- R18 and R19 40-epoch candidates all stayed below 20 dB crop PSNR.

Therefore, more short from-scratch probes are low value. The next useful test must either run at a convergence-relevant scale or stop architecture probing.

### 3. 25-channel identity is diagnostic only and should not be extended yet

Severity: Medium

The 25-channel identity candidate is not a physically valid final optical design. However, it is the upper-bound diagnostic for the sensing-channel bottleneck. If identity sensing cannot recover toward historical 120-epoch crop PSNR after additional training, then continuing the weaker 8-channel candidate is unlikely to be useful.

That said, the R12/R17 full-scene failure means the forward contract itself is suspect. Round 20 should not extend identity or 8-channel training until the forward model is audited.

### 4. New collaboration ledger contract was not satisfied

Severity: Medium

`CURRENT_CODE_CHANGES.md` and `EXPERIMENTS.md` are still templates and do not contain Round 19 code-change or experiment entries. This breaks the new review workflow because Codex cannot use those files as the primary index for code review and experiment traceability.

Round 20 must backfill `EXPERIMENTS.md` for Round 19 and must maintain `CURRENT_CODE_CHANGES.md` for Round 20. If Round 20 has no code changes, the file must explicitly say `本轮无代码修改`.

### 5. More training on the same forward contract is not the right next move

Severity: High

The R12/R17 260-epoch checkpoint already showed the central problem: filtered crop metrics can look good while full-scene HS/depth outputs are visually and metrically unusable. Extending Round 19 candidates only tests whether they can reproduce the same flawed crop-optimized behavior.

Before spending more long-training compute, the optical forward contract itself must be audited. The most suspicious design points are:

- sensor measurement uses `torch.abs(field)` rather than physical intensity `abs(field) ** 2`;
- a natural scene is propagated as one coherent complex field, which can introduce interference cross-terms inappropriate for incoherent scene radiance;
- full-scene inference synthesizes measurement from GT HS and GT depth, so it is an oracle simulation contract, not measurement-only deployment;
- invalid/background depth may be clamped into a valid depth plane and contribute measurement energy;
- per-patch normalization and masked metrics can hide full-scene spectral collapse.

## Recommendation

Round 20 should not train. It should:

1. Backfill the experiment ledger for Round 19.
2. Build a focused optical-forward audit document.
3. Run non-training diagnostics for amplitude-vs-intensity sensing, coherent additivity, background/invalid-depth contribution, normalization effects, and GT-depth oracle dependency.
4. Decide which forward-model assumption must be corrected before any new training.
5. Keep the R12 depth-best checkpoint as crop-only best; full-scene deployment remains not approved.
