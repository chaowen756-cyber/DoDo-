# Review: DOE optical-only shape coding

## Scope

Reviewed the optical-only PSF separation objective, fixed full-loss logging,
best-checkpoint selection, and pupil-RMS constrained Adam update on branch
`DOE可编码性预优化实验`.

## Findings

- No blocking correctness issue remains.
- Optical spectral/depth signatures are built from individually L2-normalized
  monochromatic PSFs and contain no sensor response, so per-band intensity
  scaling cannot satisfy the new coding target.
- The sensor-weighted Fisher still uses the full RGB observation and full
  x/y/depth/wavelength inverse; optical-only metrics do not replace the real
  measurement objective.
- `loss/full_total` is independent of warm-up and is used for checkpoint
  selection. `loss/train_total` remains the actual scheduled backward loss.
- Current `New` DOELayer/DOEFreeLayer height maps are linear homogeneous sums
  of trainable basis coefficients. Therefore radial coefficient retraction is
  a valid strict RMS safeguard for both rank-9 and free-150.
- RMS boundary correction projects only the realized Adam candidate step. An
  attempted first-moment projection was rejected after a pressure test exposed
  inconsistent Adam first/second moments and was removed before commit.

## Verification

- `black --check`: passed for all three modified Python files.
- `python -m py_compile`: passed.
- DOE preoptimization and existing DOE/PSF convolution regressions: 44 passed.
- GPU 16-depth preflight, 30 and 200 steps: fixed full loss decreased.
- GPU boundary pressure test from 2.99 um RMS, 50 steps: full loss
  `0.9074 -> 0.8918`; 49 tangent corrections; minimum safety retraction scale
  `0.9895`; no constraint violation.
- `git diff --check`: passed.

## Residual risks and experiment gate

- Adjacent-band optical cosine remains close to one in short preflight; the
  1000-step experiment is deliberately a feasibility test, not evidence that
  joint reconstruction will succeed.
- The stricter optical spectral term dominates the initial composite loss by
  design. MTF, capture, r90 and sensor-weighted Fisher must be checked together
  before a checkpoint is used for joint training.
- A tangent step requires a small second-order radial retraction on an active
  spherical constraint. Retraction count alone is not a failure; an anomalously
  small `constraint/minimum_retraction_scale` is the relevant instability flag.

## Decision

Approved for the two controlled free-150, 1000-step, seed-123/456 DOE-only
experiments. Do not proceed to CNN joint training until both summaries are
inspected.
