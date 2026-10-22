# STATUS — multilayer LS (MLS) / entropy-barrier MLS

Goal: multilayer LS, better samples on CIFAR-10 / CelebA / MNIST, cost independent of image size (ImageNet eventually).

## Current best (commit 7625f61, 500 samples, 5 IS splits, FID vs full train set)
Record: `notebooks/mls_entropy_barrier_results.ipynb` + slides PDF (OneDrive); data in `results/mls_inception/2026-10-2{1,1b,1c,1d,1e,1f,2}/`.

**The multilayer layer is dominated on both metrics; the win is the per-step window
schedule — its *shape*, not just width.** Best: multiplier ramped **×3.4 at t=0 down
to ×1.6 at t=1**, vs uniform ×2 at matched memorization (ramp/uniform):

| dataset | NN-RMS | IS | FID | copies |
|---|---|---|---|---|
| CIFAR-10 s0/s1 | 0.267/0.270, 0.271/0.274 | 6.44/6.29, 6.31/6.07 | **83.3/94.0, 86.1/93.0** | ~0 |
| CelebA | 0.301/0.302 | 2.34/2.34 | **70.0/79.4** | 0.000/0.000 |

~9 FID at matched distance and zero copies, vs a ~1 FID seed spread. MNIST is flat
(16.6/17.0, real floor 12.2); EBMLS stays dominated (IS 3.30 vs 2.78, FID 201/189).
Four shape controls bound the mechanism and all fail (detail in the notebook): up-
ramp, flat and single-ended pushes lose to a balanced down-ramp at matched
memorization — the schedule is one object, its ends move together.

**(level, compression) is the right parameterization.** `--scale_gamma` compresses
the base schedule's range about its geometric mean (γ=1 ≡ uniform, γ=0 ≡ flat), so
level and shape are separable. At the champion's level (m=2.5, `2026-10-21f/`)
γ=0.90/0.75/0.60/0.45 gives FID 73.0/80.4/85.5/88.1 at NN-RMS 0.231/0.248/0.262/
0.275, copies 5.8%/1.2%/0.4%/0.2%. γ=0.60 reproduces ramp 3.4→1.6's schedule (11/
31/18.9 vs 11/31/18.5 for k0/kmax/kgeo) and lands on it (85.5 at 0.262 vs 83.3 at
0.267), so the hand-tuned ramp is a point in this family — but the γ curve traces
the *same* FID-vs-NN-RMS frontier, not a better one, and **nothing displaces ramp
3.4→1.6 at zero copies.** Compressed and tilted shapes sit ≈7-10 FID below uniform
and flat at every matched distance; that gap, not which family draws it, is it.

## Running jobs

- 48948935 → `2026-10-22/cifar10_LSgamma0.90_m*` (FID follows as 48948937; move to
  `2026-10-22b/` once it lands — `2026-10-22/` is the mis-dated ramp dir). The
  *level* axis at the best compression (γ=0.90, m=2.35/2.20/2.05), the other half of
  the 2-D grid: γ0.90/m2.5 hit FID 73.0 but at 5.8% copies, so dropping the level
  should walk that back toward zero. Hypothesis: m≈2.2 lands near NN-RMS 0.26 at
  FID < 83.3, beating the champion; if FID instead rises onto the ramp curve, the
  frontier is one object in both knobs and shape-fitting is finished.

## Open problems

- **Blocked on hshunt: the layer-2 posterior.** You said the implemented one is wrong
  for a training-set image. Today's code conditions on `x_hat_1` (not the noisy `x`),
  per-pixel Gaussian, `sigma^2 = temperature * b_t^2/a_t^2`, no prior. It also moves
  the entropies the barrier thresholds on: every MLS/EBMLS number needs rerunning.
- **Layer 2 replaces rather than refines**: weights sum to 1 per pixel, so `x_hat_1 + delta` *is* the layer-2 posterior mean. Try an additive residual.
- Barrier wording: entropy falls with k, so "largest k *below* the barrier" is degenerate; implemented as largest k >= barrier.
- Pre-existing, not mine: `test_channel_theory.py::...boundary_returns_nan` fails and that file hangs on its last test — a semantics call from you.

## Next steps

1. Close the 2-D grid with 48948935, then stop fitting shapes: four families land on
   one frontier and only its level moves you along it.
2. Additive residual layer 2, once the corrected posterior lands.
3. ImageNet: candidates cost O(1) in resolution, but the per-step pass over the
   training set scales with N (needs ANN search).
