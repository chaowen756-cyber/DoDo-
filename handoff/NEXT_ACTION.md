# Next Action

## Owner

User / experiment runner

## Current Change

DOE PSF preoptimization feasibility experiment

## Status

Branch `DOE可编码性预优化实验` contains an independently reviewed DOE-only
search. It preserves the current `consistent_grid_v1` optical forward and
compares rank-9 with free-150 under the same physical pupil-height RMS budget.
The final objective includes the task-matched Fisher A-optimality from Baek et
al. in addition to MTF, direct separation, and energy guards.

## Execute

Run rank-9 on GPU 2 and free-150 on GPU 3 with separate output directories,
using the full commands in the final handoff/chat and the Fisher parameters in
`docs/doe_psf_preoptimization.md`. Do not start joint CNN training before both
runs finish.

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
