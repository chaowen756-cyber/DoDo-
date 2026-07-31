# Next Action

## Owner

User / experiment runner

## Current Change

DOE PSF preoptimization feasibility experiment

## Status

Branch `DOE可编码性预优化实验` contains an independently reviewed DOE-only
search. It preserves the current `consistent_grid_v1` optical forward and
compares rank-9 with free-150 under the same physical pupil-height RMS budget.
The final objective includes task-weighted Fisher A-optimality. It keeps x/y as
nuisance variables but weights their CRLB at 0.1 versus 1.0 for depth and
wavelength, in addition to MTF, direct separation, and energy guards.

## Execute

Run free-150 seed 123 on GPU 2 and seed 456 on GPU 3 with separate output
directories, using the full commands in the final handoff/chat. Do not start
joint CNN training before both runs finish.

## Decision

- If neither mode materially raises MTF and lowers spectral/depth cosine without
  losing crop energy, treat the current optical grid/propagation contract or the
  3 um RMS budget as the limiting factor.
- If free-150 is consistently better, repeat free-150 with seeds 123/456/789,
  then use its best DOE checkpoint as the only changed initialization in a
  controlled joint-training experiment.
- If rank-9 is already comparable, retain rank-9 and investigate the joint
  objective/decoder instead of increasing DOE capacity.

## Required Artifact

- `comparison.json`
- each run's `summary.json`, `best_doe.pt`, height map and PSF montage
- optional `best_psf_bank.pt` for offline inspection

## Stop Condition

Stop after the DOE-only comparison and inspect its metrics. Do not infer full
reconstruction success from this experiment alone.
