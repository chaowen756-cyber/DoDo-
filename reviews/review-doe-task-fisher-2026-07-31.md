# Review: DOE Task-Weighted Fisher Objective

## Decision

Accepted for a controlled free-150 two-seed feasibility run.

## Correctness

The code does not delete x/y rows and columns from the Fisher matrix. It solves
the full x/y/depth/wavelength inverse and applies weights only to the resulting
CRLB diagonal. Therefore spatial uncertainty and cross-parameter correlations
remain nuisance variables in the depth/wavelength estimation bound.

The weights are normalized to a four-parameter trace scale. `(1,1,1,1)` exactly
recovers the previous full A-optimality; `(0,0,1,1)` is twice the depth plus
wavelength CRLB from the same complete inverse. Full and task metrics are both
saved, so the new run remains comparable to the completed all-parameter run.

## Scope

No propagation, PSF normalization, MTF target, direct separation loss, energy
guard, DOE parameterization or physical RMS budget is changed. The next run
therefore isolates Fisher task weighting.

## Preflight

The 16-depth/25-wavelength free-150 20-step CPU run is finite and stable at
`lr=1e-2`; task A-optimality decreases from `6.635e5` to `5.979e5` while mean
MTF@0.05 rises from `0.0320` to `0.0332`.

- `pytest -q test/test_doe_preoptimization.py`: 5 passed.
- existing orthogonal-RMS and PSF-convolution regressions: 37 passed.
- Python compilation and `git diff --check`: passed.
