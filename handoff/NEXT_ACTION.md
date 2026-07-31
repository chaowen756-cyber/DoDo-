# Next Action

## Owner

User / experiment runner

## Current Change

DOE optical-only PSF coding feasibility experiment

## Status

Branch `DOE可编码性预优化实验` now separates DOE-created monochromatic PSF
shape coding from information already present in RGB spectral response. The
sensor-weighted task Fisher remains in the objective. Training logs both the
warm-up objective and a fixed full objective, and the 3 um RMS constraint uses
a tangent candidate correction plus a safety retraction.

## Execute

Run free-150 seed 123 on GPU 2 and seed 456 on GPU 3 with identical optical
shape weights and different output directories. Use the complete commands in
the final handoff/chat. Do not start joint CNN training from these checkpoints
before the two optical-only feasibility summaries are compared.

## Decision

- Primary evidence of DOE spectral coding is improvement in
  `optical_spectral/adjacent_cosine_mean` and offset-2/4 cosine, not sensor
  spectral cosine or wavelength Fisher alone.
- Require `loss/full_total` to improve; ignore the expected warm-up rise in
  `loss/train_total`.
- Reject a run if MTF collapses, crop capture falls, or the minimum RMS
  retraction scale is anomalously far below one.
- Only initialize a joint CNN experiment if both seeds show consistent optical
  shape improvement with acceptable MTF/energy.

## Required Artifact

- `comparison.json`
- each run's `summary.json`, `best_doe.pt`, height map and PSF montage
- `best_psf_bank.pt`

## Stop Condition

Stop after the two 1000-step DOE-only runs and inspect their optical-only,
sensor-weighted, Fisher, MTF, energy and RMS-constraint metrics together.
