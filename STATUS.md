# STATUS — multilayer LS (MLS) / entropy-barrier MLS

Goal: extend LS to a multilayer structure, raise sample quality on CIFAR-10 /
CelebA / MNIST, and keep cost independent of image size so the same code can
eventually denoise full-resolution ImageNet.

## Current best result (2026-10-21, seed 0, 500 samples, 5 IS splits, FID vs full train set)

**The multilayer layer is dominated on both metrics. The win is window width,
and it generalizes to all three datasets.** At every matched memorization level
plain LS with wider windows beats EBMLS, so the second layer buys quality only
by memorizing, and less efficiently than widening layer 1.

| dataset | metric | real | LS x1 | **LS x2** | best EBMLS at ~0 copies |
|---|---|---|---|---|---|
| CIFAR-10 | IS / FID | 8.75 / 54.9 | 2.78 / 188.7 | **6.29 / 94.0** | 3.30 / 200.7 (b=1) |
| CelebA | IS / FID | 3.31 / 32.8 | 2.39 / 128.3 | **2.34 / 79.4** | 1.64 / 128.0 (b=1) |
| MNIST | IS / FID | 1.92 / 12.2 | 1.86 / 19.0 | **1.92 / 17.0** | 1.89 / 15.6 (b=1, 46% copies) |

x2 is the last zero-copy point on CIFAR-10/CelebA and the usable operating
point. Past it the gain is memorization: x2.5 copies 15%, x3 copies 51% (and
only then reaches the real-image IS). CIFAR-10 reproduced across two seeds
(2.777/2.857 at x1, 6.286/6.073 at x2).

**FID changed a conclusion.** IS is flat on CelebA (real images only score 3.31)
and I had called it a failure to generalize; FID shows a 38% improvement. IS
also *over*rated EBMLS: barrier 1.0 beats the LS baseline on IS (3.30 vs 2.78)
but is worse on FID (200.7 vs 188.7).

**The widening must be coherent.** x2 on only the low-noise half of the schedule
gives IS 4.00, only the high-noise half 3.06, baseline 2.78 — but both halves
together 6.29. +1.22 and +0.28 separately, +3.51 jointly.

Notebook `notebooks/mls_entropy_barrier_results.ipynb`, slides
`notebooks/2026-10-21_mls_entropy_barrier_slides.pdf`, data in
`results/mls_inception/2026-10-2{1,1b,1c,2}/`.

## Running jobs

None.

## Open problems

- **Blocked on hshunt: the layer-2 posterior.** hshunt said the implemented
  posterior is not the correct one for a training-set image and is re-deriving
  it. What the code does today: conditions on `x_hat_1` (not the noisy `x`),
  per-pixel Gaussian with `sigma^2 = b_t^2/a_t^2 * temperature`, no explicit
  prior. Changing the conditioning variable or the variance also changes the
  entropies the barrier thresholds on, so every MLS/EBMLS number above would
  need rerunning (~15 min per CIFAR config). Do not touch this math first.
- **Layer 2 replaces rather than refines.** Its weights sum to 1 per pixel, so
  `x_hat_1 + delta` is exactly the layer-2 posterior mean over whole training
  images. That is why widening it walks straight into copying. An additive
  residual that cannot represent a single training image on its own is the
  change worth trying before abandoning the architecture.
- **Barrier wording.** Entropy falls with k, so "largest k *below* the barrier"
  is degenerate. Implemented as largest k whose entropy is still >= barrier.
- OOM at large windows: cost is genbatch x scorebatch x k^2 x pixels with the
  unfold materialized in full. Chunking inside `LocalScoreModule` would remove
  the per-multiplier batch-size hand-tuning.
- Pre-existing, not mine: `tests/test_channel_theory.py::...boundary_returns_nan`
  fails and that file hangs on its last test. Needs a semantics call from hshunt.

## Next steps

1. Optimize the per-step window schedule directly (the thing that actually
   moves the metrics) instead of adding layers — e.g. fit the multiplier per
   noise level against FID at a fixed zero-copy constraint.
2. Retry the architecture with an additive residual layer 2, after the
   corrected posterior lands.
3. ImageNet: candidate windows already cost the same at any resolution, but the
   per-step pass over the whole training set still scales with N. Needs a
   candidate subset or ANN search.
