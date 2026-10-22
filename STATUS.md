# STATUS — multilayer LS (MLS) / entropy-barrier MLS

Goal: extend LS to a multilayer structure, raise sample quality on CIFAR-10 /
CelebA / MNIST, and keep cost independent of image size so the same code can
eventually denoise full-resolution ImageNet.

## Current best result (2026-10-22, commit below, 500 samples, 5 IS splits, FID vs full train set)

**The multilayer layer is dominated on both metrics. The win is the per-step
window schedule — and its *shape*, not just its overall width.**

Best config is now a **ramped multiplier, 3.4 at t=0 down to 1.6 at t=1**, which
loads the extra window width onto the low-noise end and flattens the base
schedule. Against uniform ×2 at matched memorization:

| dataset | schedule | NN-RMS | IS | FID | copies |
|---|---|---|---|---|---|
| CIFAR-10 | uniform ×2 (s0/s1) | 0.270/0.274 | 6.29/6.07 | 93.96/93.04 | 0.000 |
| CIFAR-10 | **ramp 3.4→1.6** (s0/s1) | 0.267/0.271 | 6.44/6.31 | **83.27/86.08** | 0.002/0.000 |
| CelebA | uniform ×2 | 0.302 | 2.344 | 79.37 | 0.000 |
| CelebA | **ramp 3.4→1.6** | 0.301 | 2.344 | **70.02** | 0.000 |
| MNIST | uniform ×2 | 0.240 | 1.918 | 17.03 | 0.078 |
| MNIST | ramp 3.4→1.6 | 0.191 | 1.949 | 16.59 | 0.250 |

CIFAR-10 and CelebA both gain ~9 FID at the same distance to the training set
and the same (zero) copy fraction, against a ~1 FID seed spread. MNIST shows no
effect — it sits on the uniform frontier and its whole FID range is 19.0→15.8
against a real-image floor of 12.2, so it is weak evidence either way.
The up-ramp is the control and fails (1.4→2.6: FID 89.8 at NN-RMS 0.227 vs ≈77
for the uniform curve there), so the tilt direction matters and runs *against*
the coarse-to-fine intuition.

**But flattening is not the mechanism.** The limit of the down-ramp — a flat
schedule, same window at every step (`--flat_k`) — lands *on* the uniform
frontier, not the ramp's: k=17 gives IS 5.384 at NN-RMS 0.268 (ramp 6.437 at
0.267, uniform ≈6.32 there) and k=23 gives IS 6.862 at NN-RMS 0.237 (uniform
≈6.86 there). So the coarse-to-fine ordering still earns its keep; the down-ramp
is a partial reshaping of it, not its removal. FID for these two is queued
(job 48915101) but the IS gaps are ~5× the seed spread.

Earlier chain, still standing: EBMLS is dominated everywhere (barrier 1.0 beats
the LS baseline on IS, 3.30 vs 2.78, but is worse on FID, 200.7 vs 188.7);
uniform ×2 beats ×1 on all three datasets with zero copies; ×2 on half the
schedule is strongly sub-additive.

Notebook `notebooks/mls_entropy_barrier_results.ipynb`, slides
`notebooks/2026-10-21_mls_entropy_barrier_slides.pdf`, data in
`results/mls_inception/2026-10-2{1,1b,1c,2}/`.

## Running jobs

- 48915101 `add_fid.py` for the two flat-schedule runs. Pending on `sganguli`
  for ~50 min (Priority); it only merges FID into their metrics.json, so the
  notebook/slides are complete without it — rerun the builders once it lands.

## Open problems

- **Blocked on hshunt: the layer-2 posterior.** hshunt said the implemented
  posterior is not the correct one for a training-set image and is re-deriving
  it. Today's code conditions on `x_hat_1` (not the noisy `x`), per-pixel
  Gaussian with `sigma^2 = temperature * b_t^2/a_t^2`, no explicit prior.
  Changing the conditioning variable or the variance also changes the entropies
  the barrier thresholds on, so every MLS/EBMLS number would need rerunning
  (~15 min per CIFAR config). All LS schedule results are unaffected.
- **Layer 2 replaces rather than refines.** Its weights sum to 1 per pixel, so
  `x_hat_1 + delta` is exactly the layer-2 posterior mean over whole training
  images. An additive residual that cannot represent a single training image on
  its own is the change worth trying before abandoning the architecture.
- **Barrier wording.** Entropy falls with k, so "largest k *below* the barrier"
  is degenerate. Implemented as largest k whose entropy is still >= barrier.
- OOM at large windows: cost is genbatch × scorebatch × k² × pixels with the
  unfold materialized in full, so batch sizes are hand-tuned per multiplier.
  (Down-ramps help: they shrink the widest windows.)
- Pre-existing, not mine: `tests/test_channel_theory.py::...boundary_returns_nan`
  fails and that file hangs on its last test. Needs a semantics call from hshunt.
- Cosmetic: I named `results/mls_inception/2026-10-22/` a day ahead of the
  system clock (which reads 2026-10-21). Left as-is rather than moving data; the
  builders glob `2026-10-2*` so both are picked up.

## Next steps

1. Fit the per-step schedule directly against FID under a zero-copy constraint
   instead of searching two-parameter ramps — the ramp family is clearly not the
   end of this, since every down-ramp tried beat uniform while its flat limit
   did not, so the optimum is somewhere between the two shapes.
2. Retry the architecture with an additive residual layer 2, after the corrected
   posterior lands.
3. ImageNet: candidate windows already cost the same at any resolution, but the
   per-step pass over the whole training set still scales with N. Needs a
   candidate subset or ANN search.
