# Next Action

## Owner

Pending user scope / Claude after Codex plan

## Role

Round 24 Spectral Modification Planning

## Current Change

DoDo-change / next spectral-side modification

## Context

Round 23 intensity sensing and the LowFG measurement-energy diagnostic repair are accepted by Codex review.

Accepted review files:

- `reviews/review-DoDo-change-round23-intensity-sensing-2026-05-09.md`
- `reviews/review-DoDo-change-round23-repair-lowfg-2026-05-09.md`

Current status:

- `sensor_measurement={amplitude,intensity}` core implementation is accepted.
- Soft diopter regression remains passing per Claude report.
- LowFG diagnostic now computes true DoDo measurement energy and confirms pure-background lowfg tiles generate finite measurement energy.
- No checkpoint is promoted.
- Full-scene deployment remains not approved.
- Do not run long training until the next spectral-side change is specified and reviewed.

## Awaiting User Scope

Before Claude implements Round 24, define the spectral modification target precisely. The next prompt should specify which spectral-side change is intended, for example:

- spectral loss / SAM loss changes;
- spectral metric contract changes;
- spectral sensing/channel response changes;
- decoder spectral-head architecture changes;
- dataset spectral normalization or band selection changes;
- full-scene spectral artifact mitigation.

## Required Before Implementation

Codex should create a patch plan for the chosen spectral modification before Claude writes code.

The plan must include:

- files/classes/functions to modify;
- explicit in-scope and out-of-scope items;
- expected tests/smokes;
- whether any training is allowed;
- artifact cleanup expectations.

## Stop Condition

No active Claude implementation task until the user provides the spectral modification scope and Codex writes/approves a plan.
