# Next Action

## Owner

User / experiment runner

## Current Change

Pixel-wise wrapped-phase DOE feasibility experiment

## Status

The free-150 Zernike parameterization failed consistently after two 3000-step
runs. Branch `DOE可编码性预优化实验` now provides a 128x128 pixel-wise
unwrapped phase variable whose optical forward is always converted to a
single-period wrapped physical height at 550 nm. Initialization uses the exact
discrete Prop1/Prop3 operators and their adjoint to focus the approximately
1 m, 550 nm reference point without external camera parameters.

## Execute

Run pixelphase seed 123 on GPU 2 and seed 456 on GPU 3 for 1000 steps with
Adam lr 0.1, optical spectral offsets 1/2/4 and the balanced weights in the
final handoff commands. Use separate output directories.

## Decision

- Compare against the free-150 3000-step ceiling, not only against the new
  random initialization.
- Primary success metrics remain optical adjacent/offset-2/offset-4 cosine,
  task Fisher depth/wavelength CRLB and MTF p10.
- The exported `best_heightmap_m.npy` is the physical wrapped height actually
  used by the wideband forward; do not evaluate only the unwrapped phase.
- If both seeds preserve the strong 200-step improvement, use the better
  pixelphase checkpoint for the next controlled joint-CNN experiment.
- If MTF p10 remains near 0.011 despite better coding, treat propagation/sensor
  sampling—not DOE capacity—as the next bottleneck.

## Required Artifact

- `comparison.json`, `summary.json`, `history.jsonl`
- `best_doe.pt`, `best_psf_bank.pt`, PSF montage
- wrapped physical height and unwrapped phase artifacts

## Stop Condition

Stop after the two 1000-step pixelphase runs and inspect both summaries before
changing the propagation grid or starting joint reconstruction training.
