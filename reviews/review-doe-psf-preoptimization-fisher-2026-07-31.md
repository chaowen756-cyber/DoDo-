# Review: Fisher DOE PSF Preoptimization

## Conclusion

Accepted as the final objective update before the two-GPU feasibility run.

## Evidence

The exact HS-D task paper by Baek et al. uses A-optimal Fisher information of
a monochromatic point-source RGB PSF over x, y, depth and wavelength to obtain
the DOE initialization. The previous adjacent-cosine-only objective did not
measure whether these four source parameters are jointly identifiable.

The implementation uses finite differences in one-bin coordinates, includes
the RGB response in the wavelength derivative, and retains full-field PSF
capture. A ridge-stabilized 4x4 solve is used instead of a raw matrix inverse.
MTF and energy guards remain active, so minimizing A-optimality cannot silently
replace spatial bandwidth or throughput.

## Numerical review

- `pytest -q test/test_doe_preoptimization.py`: 4 passed.
- existing orthogonal-RMS and PSF-convolution regressions: 37 passed.
- Fisher loss is finite and differentiable on synthetic PSFs.
- Doubling signal amplitude lowers A-optimality as expected; no hidden per-PSF
  normalization removes signal strength.
- At `lr=1e-2`, full 16-depth 20-step rank-9 and free-150 runs both descend
  stably. free-150 lowers A-optimality by about 18.5% in the short run.
- The short run does not establish final DOE feasibility; 1000 GPU steps and
  at least one repeated seed are still required.

## Learning-rate decision

Keep Adam at `1e-2` for coefficient-only preoptimization. Baek reports Adam but
does not publish a separate Fisher-stage learning rate; its reported `1e-4`
belongs to joint DOE-phase/network training and is not on the same parameter
scale. The current choice is backed by the local full-bank stability test and
the hard physical RMS projection.
