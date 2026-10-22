# STATUS — multilayer LS (MLS) / entropy-barrier MLS

Goal: extend LS to a multilayer structure, raise sample quality on CIFAR-10 /
CelebA / MNIST, and keep cost independent of image size so the same code can
eventually denoise full-resolution ImageNet.

## Current best (commit 25794ab, 500 samples, 5 IS splits, FID vs full train set)

**The multilayer layer is dominated on both metrics; the win is the per-step
window schedule — and its *shape*, not just its overall width.** Best config is a
**ramped multiplier, ×3.4 at t=0 down to ×1.6 at t=1**, which loads window width
onto the low-noise end. Against uniform ×2 at matched memorization:

| dataset | schedule | NN-RMS | IS | FID | copies |
|---|---|---|---|---|---|
| CIFAR-10 | uniform ×2 (s0/s1) | 0.270/0.274 | 6.29/6.07 | 94.0/93.0 | 0.000 |
| CIFAR-10 | **ramp 3.4→1.6** (s0/s1) | 0.267/0.271 | 6.44/6.31 | **83.3/86.1** | ~0.000 |
| CelebA | uniform ×2 | 0.302 | 2.344 | 79.4 | 0.000 |
| CelebA | **ramp 3.4→1.6** | 0.301 | 2.344 | **70.0** | 0.000 |
| MNIST | ramp vs uniform ×2 | 0.191 vs 0.240 | 1.95/1.92 | 16.6/17.0 | 0.25/0.08 |

~9 FID at the same distance to the training set and the same (zero) copy
fraction, against a ~1 FID seed spread. MNIST shows no effect (it sits on the
uniform frontier; its whole FID range is 19.0→15.8 vs a 12.2 real-image floor).

Both controls fail, which brackets the mechanism: the **up-ramp** is worse than
uniform (1.4→2.6: FID 89.8 at NN-RMS 0.227 vs ≈77 for uniform there), so the
tilt direction matters and runs *against* the coarse-to-fine intuition; but a
**flat schedule** (`--flat_k`, same window every step) sits on the *uniform*
frontier, not the ramp's (k=17: IS 5.38 at 0.268 vs the ramp's 6.44 at 0.267;
k=23: 6.86 at 0.237 vs uniform's ≈6.86). So the ordering still earns its keep —
the down-ramp is a partial reshaping, not a removal, and the optimum is between.

Earlier, still standing: EBMLS is dominated everywhere (barrier 1.0 beats the LS
baseline on IS, 3.30 vs 2.78, but is worse on FID, 200.7 vs 188.7); uniform ×2
beats ×1 on all three datasets with zero copies.

Notebook `notebooks/mls_entropy_barrier_results.ipynb`, slides
`notebooks/2026-10-21_mls_entropy_barrier_slides.pdf` (sent to OneDrive), data in
`results/mls_inception/2026-10-2{1,1b,1c,2}/`.

## Running jobs

- 48915101 `add_fid.py` for the two flat-schedule runs. Pending on `sganguli`
  ~50 min (Priority). It only merges FID into their metrics.json — the notebook
  and slides are complete without it; rerun the builders once it lands.

## Open problems

- **Blocked on hshunt: the layer-2 posterior.** You said the implemented
  posterior is not the correct one for a training-set image and are re-deriving
  it. Today's code conditions on `x_hat_1` (not the noisy `x`), per-pixel
  Gaussian, `sigma^2 = temperature * b_t^2/a_t^2`, no explicit prior. Changing
  the conditioning variable or variance also changes the entropies the barrier
  thresholds on, so every MLS/EBMLS number needs rerunning (~15 min per CIFAR
  config). All LS schedule results are unaffected.
- **Layer 2 replaces rather than refines.** Its weights sum to 1 per pixel, so
  `x_hat_1 + delta` is exactly the layer-2 posterior mean over whole training
  images. An additive residual that cannot represent a single training image on
  its own is the change worth trying before abandoning the architecture.
- **Barrier wording.** Entropy falls with k, so "largest k *below* the barrier"
  is degenerate; implemented as largest k whose entropy is still >= barrier.
- OOM at large windows: cost is genbatch × scorebatch × k² × pixels with the
  unfold materialized in full, so batch sizes are hand-tuned per multiplier.
- Pre-existing, not mine: `tests/test_channel_theory.py::...boundary_returns_nan`
  fails and that file hangs on its last test. Needs a semantics call from you.
- Cosmetic: `results/mls_inception/2026-10-22/` is named a day ahead of the
  system clock. Left as-is; the builders glob `2026-10-2*`.

## Next steps

1. Fit the per-step schedule directly against FID under a zero-copy constraint
   rather than searching two-parameter ramps — the optimum is between the ramp
   and flat shapes, so the family is too coarse to find it.
2. Retry the architecture with an additive residual layer 2, once the corrected
   posterior lands.
3. ImageNet: candidate windows already cost the same at any resolution, but the
   per-step pass over the whole training set still scales with N. Needs a
   candidate subset or ANN search.
