# Next Action

## Owner

Implementation / verification

## Current Change

Baek pretrained fixed-height DOE native PSF parity.

## Status

Do not start the 12-epoch network training yet. An independent comparison using
the actual PADO API over all 25 wavelengths x 20 notebook depths found material
PSF-shape discrepancies in the current vectorized spherical source. Full-grid
NRMSE is `0.1504` on average and `1.1062` worst-case.

The DOE, pupil and Fresnel propagation are not the cause: feeding the current
chain a PADO scalar-semantics spherical source reduces full-grid NRMSE to
`4.79e-6` mean and `1.62e-5` max.

## Execute

1. Change only `doe_native_grid_v1` spherical-source construction so it is
   numerically equivalent to PADO's per-wavelength Python-float calculation.
2. Re-run `scripts/compare_baek_native_psfs.py` against actual PADO.
3. Require all 500 PSFs to remain near the isolated baseline (target max NRMSE
   around `2e-5`, cosine near 1) before reconsidering network training.

## Preserve

- Keep the 376x376/8 µm/NOA61/50 mm DOE-native geometry.
- Do not change legacy or consistent-grid optical paths.
- Do not use capture fraction alone as the parity criterion; it stayed nearly
  equal even when PSF morphology differed.
