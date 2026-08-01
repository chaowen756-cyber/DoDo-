# Next Action

## Owner

User / experiment runner

## Current Change

Baek pretrained fixed-height DOE transfer on native optical sampling.

## Status

Branch `Baek预训练DOE冻结联合训练` loads the Baek 375x375 physical height,
right/bottom pads it to 376x376 without interpolation, and uses its 8 µm,
NOA61, PADO spherical-source/circular-aperture/Fresnel, 50 mm convention. The
DOE and all optics have zero trainable parameters. Real-file 16x25 PSF
preflight passed.

## Execute

Run `bash scripts/run_baek_fixed_doe_joint.sh`. Defaults: physical GPUs 1 and
3, 12 epochs, frozen optics, joint HS/depth network training. Use `DRY_RUN=1`
to inspect the expanded command first.

## Decision

- Treat native 376x376 as the primary experiment; retain 128x128 area-resampled
  consistent-grid only as an optional later ablation.
- Keep halo64/129x129 for the first controlled run. Capture fraction is logged;
  a larger optical halo is a separate follow-up experiment.
- Use joint-best by validation loss as the primary result; depth-best and
  HS-best remain auxiliary ceilings.

## Required Artifact

- `artifacts/command.txt`, `hparams.json`, `metrics.json`
- joint/depth/HS best checkpoints
- loss history and quicklooks

## Stop Condition

Stop after the 12-epoch run and compare validation/full-scene metrics before
changing PSF support, reconstruction architecture, or loss weights.
