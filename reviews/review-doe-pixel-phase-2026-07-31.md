# Review: pixel-wise wrapped-phase DOE feasibility mode

## Scope

Reviewed the discrete propagation adjoint, pixel-phase DOE forward, phase
conjugate initialization, physical height export, checkpoint path and
preoptimization integration on branch `DOE可编码性预优化实验`.

## Design basis

- Baek et al. optimize an unwrapped DOE phase initialized from a Fresnel DOE,
  then wrap it and convert it to physical height. The new mode follows that
  high-capacity phase parameterization.
- This implementation is stricter for broadband validity: the single-period
  wrapped physical height participates in every forward pass, rather than
  being applied only after optimization.
- The focusing carrier is computed from the exact discrete Prop1 input and
  Prop3 adjoint. It does not assume 50 mm focal length, 6.45 um sensor pitch or
  any other parameters absent from the current forward.

## Correctness findings

- `PropagationLayer.adjoint` satisfies the complex inner-product identity for
  the exact padded/cropped operator.
- Adding integer 2π periods to the trainable phase leaves the wrapped height
  and all tested wavelengths unchanged.
- `best_heightmap_m.npy` and checkpoint `doe_heightmap_m` contain the wrapped
  physical height actually used by PSF generation. Unwrapped phase/height are
  clearly separated into additional artifacts.
- Legacy rank9/free150 construction remains the default; pixel phase is only
  selected explicitly.
- The phase-conjugate carrier raises the reference center intensity by roughly
  4.6x on the full 16-depth model and about 5.0x in final CLI smoke.

## Empirical selection

- Pixel phase, 200 steps: lr 0.01/0.05/0.1 produced full losses approximately
  0.827/0.786/0.779 in the theoretical unwrapped comparison; lr 0.1 was stable.
- Physical wrap-aware lr 0.1 reached full loss 0.797 and task Fisher 3.66e5 in
  200 steps, materially better than free150 after 3000 steps (about 0.864 and
  5.2e5).
- Restricting offsets to 1/2 and doubling optical spectral weight produced only
  about 1.3e-4 extra adjacent-cosine improvement and worse Fisher, so the
  balanced offsets 1/2/4 and weight 5 are retained.

## Verification

- Python compilation passed.
- Related propagation, DOE, orthogonal-RMS and PSF-convolution suites: 60
  passed.
- Final one-step GPU CLI smoke produced all checkpoint, wrapped height,
  unwrapped phase, PSF-bank and visualization artifacts.
- `git diff --check` passed.

## Residual risk and gate

- The wrapped map contains near-Nyquist phase structure. It is valid on the
  current 78.125 um DOE sampling grid but must later be checked against actual
  fabrication lateral resolution.
- MTF p10 remained near 0.011 in 200-step tests even while coding/Fisher
  improved. If this persists for two 1000-step seeds, changing DOE capacity is
  no longer the priority; propagation-to-sensor sampling must be audited next.
- Approval is limited to two DOE-only pixelphase runs. Joint CNN training still
  requires a separate result review.

## Decision

Approved for the controlled seed-123/456, 1000-step pixelphase feasibility
experiments with Adam lr 0.1.
