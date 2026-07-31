# Review: DOE PSF Preoptimization

## Scope reviewed

- Shared physical height/RMS helpers in `torch_optics/doe.py`.
- DOE-only objective and checkpoint loader in `util/doe_preoptimization.py`.
- rank-9/free-150 comparison CLI in `scripts/preoptimize_psf_doe.py`.
- tests, experiment documentation and handoff records.

## Findings

No blocking correctness issue found.

The CLI does not construct a dataset or reconstruction network. All model
parameters are frozen before only `doe1.zernike_coeffs` is re-enabled. Both
bases use the same measured pupil-height initialization and post-update maximum
RMS, so the comparison does not silently give free-150 a larger total height
budget.

The optical forward remains the existing 128-grid `consistent_grid_v1` path:
Prop1 point source with pad2, no Prop2/second DOE, intensity sensor, 16 soft
diopter layers, full-field normalization and 129 PSF crop. This change does not
claim to validate alternative real-camera geometry or scaled Fresnel sampling.

The objective measures normalized PSF MTF, sensor-visible adjacent-wavelength
and adjacent-depth similarity, plus a loose energy-spread guard. The guard is
important because separation alone could reward unbounded PSF spreading.

## Verification

- `pytest -q test/test_doe_preoptimization.py`: 3 passed.
- `PYTHONPATH=. pytest -q test/test_doe_orthogonal_rms.py test/test_dodo_psf_convolution.py`:
  37 passed.
- free-150 final one-step CPU CLI smoke: passed and wrote all required artifacts.
- `git diff --check` and Python compilation: passed.
- `test/test_psf_regularization.py`: not collected in the current shell because
  `pytorch_lightning` is not installed; this is an environment limitation.

## Experimental boundary

The 16-depth, 20-step CPU preflight showed a small improvement and a somewhat
faster free-150 trend, but it is not decisive. No GPU is visible in the current
execution environment, so acceptance here means the experiment implementation
is ready—not that a sufficiently informative DOE has already been found.

## Decision

Accepted for the requested DOE-only feasibility experiment. Run the documented
1000-step GPU comparison before changing joint training or promoting a DOE.
