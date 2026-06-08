# Review: DoDo-change Round 20 Optical Forward Audit

Date: 2026-05-09

## Scope

Reviewed:

- `handoff/DoDo-change/OPTICAL_FORWARD_AUDIT.md`
- `handoff/DoDo-change/implementation-notes.md` Section 20
- `handoff/DoDo-change/CURRENT_CODE_CHANGES.md`
- `handoff/DoDo-change/EXPERIMENTS.md`
- `round20_forward_diagnostics.py`
- `round20_forward_diag_CE.py`
- `torch_optics/sensing.py`
- `torch_optics/forward_dodo.py`
- `infer_contect.py`

## Findings

### 1. The next code target should be intensity sensing, not more training

Severity: High

Round 20 confirmed that current `SensingLayer` integrates `abs(field)`, while a physical sensor should measure field intensity, approximately `abs(field) ** 2`. The diagnostic reports a post-normalization correlation of 0.927 and about 3x larger dynamic range for intensity measurements.

This is enough to justify a focused code-change plan for an opt-in intensity-sensing forward path. It should not silently change existing checkpoint behavior.

### 2. GT-depth oracle dependency remains a blocking interpretation issue

Severity: High

Full-scene inference still uses GT depth to synthesize the optical measurement. That means current DoDo-depth full-scene evaluation is an oracle simulation, not measurement-only deployment. Fixing intensity sensing may improve physical consistency, but it does not solve the deployment contract.

Next documentation and metrics must explicitly label oracle-simulation results. Do not claim real measurement-only joint HS/depth reconstruction until the input contract is redesigned.

### 3. Masked PSNR is not acceptable as the main full-scene success metric

Severity: High

Round 20 quantified the deploy 1 masked-vs-full PSNR gap as +8.45 dB. This explains why a 30 dB masked PSNR can coexist with poor visual quality and high SAM.

Future gates must prioritize full PSNR, SAM, pseudo-RGB PSNR, and depth-vs-baseline metrics. Masked PSNR may remain a secondary diagnostic but must not be the primary promotion criterion.

### 4. Diagnostic C did not test the intended low-foreground failure mode

Severity: Medium

The background/invalid-depth diagnostic used a crop with `valid_ratio=100%`, so it cannot answer whether background or invalid depth dominates low-foreground full-scene tiles. The conclusion should be treated as incomplete, not negative evidence.

Before training on any corrected forward, run a targeted low-valid-ratio tile diagnostic.

### 5. Code-change ledger needs correction

Severity: Medium

`CURRENT_CODE_CHANGES.md` says "本轮无对现有文件的代码修改" but also records newly added diagnostic scripts. Adding scripts is still a code change for review purposes. The ledger should distinguish "no production code changes" from "diagnostic scripts added".

Also, Section 20 references `/root/autodl-tmp/round20_forward_diagnostics.py`, while the current workspace contains `round20_forward_diagnostics.py` and `round20_forward_diag_CE.py`. Round 21 should standardize the canonical script paths in the notes/ledger.

## Recommendation

Round 21 must be a plan-only implementation-design round:

1. Do not modify production code yet.
2. Write a concrete code implementation plan for opt-in intensity sensing and metric-gate corrections.
3. Include exact files, functions/classes, proposed CLI args, compatibility behavior, tests, and risks.
4. Stop after writing the plan and ledger corrections.
5. Wait for Codex review before any code implementation.
