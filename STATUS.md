# STATUS — multilayer LS (MLS) / entropy-barrier MLS

Goal: multilayer LS, better samples on CIFAR-10 / CelebA / MNIST, cost independent of image size (ImageNet eventually).

## Current best (commit f5f1733, 500 samples, 5 IS splits, FID vs full train set)
Record: `notebooks/mls_entropy_barrier_results.ipynb` + slides PDF (OneDrive); data in `results/mls_inception/2026-10-2{1,1b,1c,1d,1e,1f,2,2b,2c}/`.

**The multilayer layer is dominated on both metrics. Two things beat it: the window
schedule's *shape*, and — new today — the sampler's *step count*.** The champion
schedule ramps the multiplier **×3.4 at t=0 down to ×1.6 at t=1** (vs uniform ×2 at
matched memorization, ramp/uniform):

| dataset | NN-RMS | IS | FID | copies |
|---|---|---|---|---|
| CIFAR-10 s0/s1 | 0.267/0.270, 0.271/0.274 | 6.44/6.29, 6.31/6.07 | **83.3/94.0, 86.1/93.0** | ~0 |
| CelebA | 0.301/0.302 | 2.34/2.34 | **70.0/79.4** | 0.000/0.000 |

~9 FID at matched distance, zero copies, vs a ~2.8 FID seed spread. MNIST is flat
(16.6/17.0, floor 12.2) and EBMLS dominated (IS 3.30 vs 2.78, FID 201/189).
**Six shape controls now bound the mechanism and all fail** (detail in the notebook):
up-ramp, flat, single-ended pushes, and both axes of the (level, compression)
parameterization. Each trades memorization for quality along one frontier.

**Step count is the first knob that leaves that frontier** (`2026-10-22c/`). `--nsteps`
resamples the window-vs-t curve onto more steps, holding the window at each noise level
fixed, and improves IS, FID *and* median training-set distance together:

| steps | NN-RMS | IS | FID | copies | gen s |
|---|---|---|---|---|---|
| 20 (champion) | 0.267 | 6.44±0.31 | 83.3 | 0.002 | 311 |
| 40 | 0.279 | 6.31±0.27 | 82.3 | 0.010 | 1180 |
| **60** | **0.280** | **6.75±0.34** | **78.8** | 0.016 | 1798 |

FID is monotone in step count; IS is not (40 dips) but is highest at 60. Not yet a
claim: +0.31 IS is one split-spread and the copy tail rises 0.2%→1.6%, so a seed
replicate is running. Costs 5.8× generation time for 3× the steps.

## Running jobs

- 49006220 → `2026-10-22d/` (FID follows as 49006221): seed-1 replicate at 60 steps to
  decide whether IS 6.75 > 6.44 survives, plus 100 steps at seed 0 to test whether FID
  keeps falling. Hypothesis: FID continues down and IS stays ≥6.6, since FID moved
  monotonically and only IS was noisy.

## Open problems

- **Blocked on hshunt: the layer-2 posterior.** You said the implemented one is wrong
  for a training-set image. Today's code conditions on `x_hat_1` (not the noisy `x`),
  per-pixel Gaussian, `sigma^2 = temperature * b_t^2/a_t^2`, no prior. It also moves
  the entropies the barrier thresholds on: every MLS/EBMLS number needs rerunning.
- **Layer 2 replaces rather than refines**: weights sum to 1 per pixel, so `x_hat_1 + delta` *is* the layer-2 posterior mean. Try an additive residual.
- With more steps the median sample moves *away* from the training set while the copy tail grows: most get less memorized, a few get more. Unexplained.
- Pre-existing, not mine: `test_channel_theory.py::...boundary_returns_nan` fails and that file hangs on its last test — a semantics call from you.

## Next steps

1. Settle the step-count win with 49006220, then re-tune the window schedule at the better step count — the champion ramp was fitted at 20 steps and need not still win.
2. Additive residual layer 2, once the corrected posterior lands.
3. ImageNet: candidates cost O(1) in resolution, but the per-step pass over the train set scales with N (ANN search), and more steps multiply that cost.
