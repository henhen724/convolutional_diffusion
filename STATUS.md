# STATUS — multilayer LS (MLS) / entropy-barrier MLS

Goal: extend LS to a multilayer structure, raise Inception Score on CIFAR-10 /
CelebA / MNIST, and keep cost independent of image size so the same code can
eventually denoise full-resolution ImageNet.

## Current best result (2026-10-21, commit 2aeebca, seed 0, 500 samples, 5 IS splits)

**The multilayer layer is dominated. The win is window width.** Both EBMLS and
plain LS climb toward the real-image IS as they drift onto the training set; at
every matched memorization level plain LS is higher, so the second layer buys IS
only by memorizing, and less efficiently than just widening layer 1.

CIFAR-10 (real images = 8.753, previous baseline LS = 2.777):

| config | IS | NN-RMS | copies |
|---|---|---|---|
| LS (baseline scales) | 2.777 / 2.857 (s0/s1) | 0.342 | 0.000 |
| **LS windows x2** | **6.286 / 6.073 (s0/s1)** | 0.270 | **0.000** |
| LS windows x2.25 | 6.803 | 0.240 | 0.028 |
| LS windows x2.5 | 7.780 | 0.191 | 0.146 |
| LS windows x3 | 8.401 | 0.045 | 0.514 |
| EBMLS barrier 1.0 | 3.298 | 0.139 | 0.004 |
| EBMLS barrier 0.25 | 7.040 | 0.078 | 0.196 |

x2 is the last zero-copy point and the usable operating point: **IS 2.78 -> 6.29,
~15x the seed spread, no sample within the copy threshold**. Past it the gain is
just memorization (x3 reaches the real-image score by copying half the samples).
EBMLS loses at both ends: 3.30 vs 6.29 at ~0 copies, 7.04 vs 7.78 at ~15-20%.
CelebA agrees so far (LS 2.393 vs EBMLS barrier 1.0 1.644). MNIST is
uninformative — real images only score 1.915 and everything is within the spread.

Notebook `notebooks/mls_entropy_barrier_results.ipynb`, slides
`results/mls_inception/2026-10-21_mls_slides.pdf`, data in
`results/mls_inception/2026-10-21{,b,c}/`.

## Running jobs

- 48831340 celeba EBMLS barriers 2.0 / 4.0 (barrier 1.0 already in).
- 48836840 mnist LS x2 / x3 (first attempt OOM'd; rerun at genbatch 10 / scorebatch 64).
- 48840424 celeba LS x1.5 / x2 — the one that matters: does the window-width win
  generalize off CIFAR-10? 64x64 at genbatch 5 / scorebatch 48, so it is slow.

## Open problems

- **Layer 2 replaces rather than refines.** Its weights sum to 1 per pixel, so
  `x_hat_1 + delta` is exactly the layer-2 posterior mean over whole training
  images; layer 1 survives only through the weights. That is what the spec
  describes, and it is why widening layer 2 walks straight into copying. A
  genuinely additive residual (one that cannot represent a single training image
  on its own) is the change worth trying before giving up on the architecture.
- **Barrier wording.** Entropy falls with k, so "largest k *below* the barrier" is
  degenerate. Implemented as largest k whose entropy is still >= barrier. One-line
  flip if the other reading was meant.
- OOM at large windows: cost is genbatch x scorebatch x k^2 x pixels and the
  unfold is materialized in full. Chunking it inside `LocalScoreModule` would
  remove the need to hand-tune batch sizes per multiplier.
- Pre-existing, not mine: `tests/test_channel_theory.py::...boundary_returns_nan`
  fails (code returns a one-sided difference, test asserts NaN — both arrived in
  55f499c, so they contradicted each other from the start); that file also hangs
  on its last test. Both need a semantics call from hshunt.

## Next steps

1. Land 48840424 — if CelebA x1.5/x2 also beats its baseline, the window-width
   result generalizes and the scales file should be revisited generally.
2. Sweep the multiplier per-step rather than uniformly: the gain plausibly comes
   only from the high-noise steps, where the current scales are narrowest.
3. Retry the architecture with an additive residual layer 2 before abandoning it.
4. ImageNet: candidate windows already cost the same at any resolution, but the
   per-step pass over the whole training set still scales with N. Needs a
   candidate subset or ANN search.
