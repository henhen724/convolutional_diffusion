# STATUS — multilayer LS (MLS) / entropy-barrier MLS

Goal: multilayer LS, better samples on CIFAR-10 / CelebA / MNIST, cost
independent of image size (for ImageNet eventually).

## Current best (commit 25794ab, 500 samples, 5 IS splits, FID vs full train set)

**The multilayer layer is dominated on both metrics; the win is the per-step
window schedule — its *shape*, not just its width.** Best config: a multiplier
ramped **×3.4 at t=0 down to ×1.6 at t=1**, loading width onto the low-noise end.
Against uniform ×2 at matched memorization:

| dataset | NN-RMS ramp/unif | IS ramp/unif | FID ramp/unif | copies |
|---|---|---|---|---|
| CIFAR-10 s0 | 0.267/0.270 | 6.44/6.29 | **83.3/94.0** | 0.002/0.000 |
| CIFAR-10 s1 | 0.271/0.274 | 6.31/6.07 | **86.1/93.0** | 0.000/0.000 |
| CelebA | 0.301/0.302 | 2.34/2.34 | **70.0/79.4** | 0.000/0.000 |
| MNIST | 0.191/0.240 | 1.95/1.92 | 16.6/17.0 | 0.25/0.08 |

~9 FID at matched training-set distance and zero copies, vs a ~1 FID seed spread.
MNIST is flat (uniform frontier; FID range 19.0→15.8 against a 12.2 real floor).
EBMLS stays dominated: barrier 1.0 wins on IS (3.30 vs 2.78) but not FID (201/189).

Both controls fail, bracketing the mechanism. **Up-ramp** is worse than uniform
(1.4→2.6: FID 89.8 at NN-RMS 0.227 vs ≈77), so tilt *direction* matters and runs
against the coarse-to-fine intuition. But a **flat schedule** (`--flat_k`) sits on
the *uniform* frontier, not the ramp's (k=17: IS 5.38 at 0.268 vs the ramp's 6.44;
k=23: 6.86 at 0.237). The ordering still earns its keep; the optimum is between.

`notebooks/mls_entropy_barrier_results.ipynb` + slides PDF (sent to OneDrive); data in `results/mls_inception/2026-10-2{1,1b,1c,1d,2}/`.

## Running jobs

- 48915101 `add_fid.py`, the two flat-schedule runs, pending on `sganguli`; merges FID into their metrics.json only — rerun the builders once it lands.
- 48921219 `--scale_gamma` scan, CIFAR-10 seed 0, gamma 0.75/0.60/0.45/0.25 at m=2.0 → `2026-10-21d/`. Gamma compresses the base schedule's dynamic range around its geometric mean: gamma=1 *is* uniform x2, gamma=0 is flat at the same level, so it interpolates the two shapes the controls bracketed. Hypothesis: FID at matched NN-RMS is best at intermediate gamma, beating uniform x2 (94.0 at NN-RMS 0.270).

## Open problems

- **Blocked on hshunt: the layer-2 posterior.** You said the implemented one is
  wrong for a training-set image. Today's code conditions on `x_hat_1` (not the
  noisy `x`), per-pixel Gaussian, `sigma^2 = temperature * b_t^2/a_t^2`, no
  prior. Changing it also moves the entropies the barrier thresholds on, so
  every MLS/EBMLS number needs rerunning (LS results are unaffected).
- **Layer 2 replaces rather than refines**: weights sum to 1 per pixel, so
  `x_hat_1 + delta` *is* the layer-2 posterior mean. Try an additive residual.
- Barrier wording (entropy falls with k, so "largest k *below* the barrier" is
  degenerate; implemented as largest k >= barrier); OOM at large windows forces
  per-multiplier batch sizes; `2026-10-22/` is a day ahead of the clock.
- Pre-existing, not mine: `tests/test_channel_theory.py::...boundary_returns_nan`
  fails and that file hangs on its last test — a semantics call from you.

## Next steps

1. Once 48921219 lands: trace the winning gamma's own level frontier (vary m at
   fixed gamma), then fit the per-step schedule directly against FID.
2. Additive residual layer 2, once the corrected posterior lands.
3. ImageNet: candidates cost the same at any resolution, but the per-step pass
   over the training set still scales with N (needs ANN search).
