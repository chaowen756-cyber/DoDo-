# Review: DoDo-change Round 22 Soft Diopter Runtime Validation

Date: 2026-05-09

## Scope

Reviewed Claude's reported runtime validation results and inspected the resulting repository state for:

- `torch_optics/forward_dodo.py`
- `test/test_soft_diopter_depth_layering.py`
- `handoff/DoDo-change/CURRENT_CODE_CHANGES.md`
- `handoff/DoDo-change/EXPERIMENTS.md`
- `handoff/DoDo-change/implementation-notes.md`
- `handoff/DoDo-change/BEST_CHECKPOINT.md`

## Result

Accepted. Round 22 soft diopter depth layering validation is complete.

Claude reported and documented:

- `pytest -q test/test_soft_diopter_depth_layering.py`: PASS, 8/8.
- `python -m py_compile torch_optics/forward_dodo.py snapshotdepth_hs.py test/test_soft_diopter_depth_layering.py`: PASS.
- Forward smoke across `hard_depth`, `hard_meter`, `soft_diopter`: PASS, shape `(1,3,128,128)`, finite outputs, `valid_mask` accepted.
- Soft-diopter backward smoke with trainable `doe1`: PASS, finite nonzero grad norm `0.675`.
- Tiny one-batch train/val preflight: PASS, exit 0, nonfinite 0, DOE grad norm `9.89`, `hparams.json` records `depth_layering_mode=soft_diopter`.

## Code Finding

### 1. Runtime unpack bug found and fixed

Severity: Medium, resolved.

`DepthAwareDoDoForwardModel.forward()` originally unpacked `self.diopter_binner(...)` as a 3-tuple unconditionally. `SoftDiopterBinner.forward()` returns:

- `(weights, z_centers)` when `return_debug=False`;
- `(weights, z_centers, debug)` when `return_debug=True`.

Claude fixed this with conditional unpacking in `torch_optics/forward_dodo.py` soft-diopter branch. This is the correct minimal fix and does not change the optical path.

## Acceptance Notes

- No DOE, decoder, `SensingLayer`, loss, optical regularizer, CFA/CCA, DOE2, PSF loss, or occlusion-aware propagation changes were made in this validation round.
- `hard_depth` and `hard_meter` remain supported.
- `soft_diopter` is opt-in via `depth_layering_mode=soft_diopter`.
- Default remains `hard_depth`.
- No checkpoint was promoted; existing crop-only best checkpoint remains unchanged.

## Next Action

Proceed to Round 23: implement the already-approved opt-in intensity sensing task from `ROUND21_CODE_PLAN.md`, with the amendments from `review-DoDo-change-round21-code-plan-2026-05-09.md`.

Round 23 should keep soft diopter tests as regression coverage because it will touch `torch_optics/forward_dodo.py` again.
