# STATUS — multilayer LS (MLS) / entropy-barrier MLS

Goal: multilayer LS, better samples on CIFAR-10 / CelebA / MNIST, cost independent of image size (ImageNet eventually).

## Current best (commit 6688b2d, 500 samples, 5 IS splits, FID vs full train set)
Record: `notebooks/mls_entropy_barrier_results.ipynb` (59 runs) + `notebooks/2026-10-22_mls_entropy_barrier_slides.pdf`, also on OneDrive; data in `results/mls_inception/2026-10-2{1,1b-1f,2,2b-2d}/`.

**Best CIFAR-10: IS 6.75, FID 78.8 — the champion window schedule run at 60 sampler
steps.** The multilayer layer is dominated on both metrics; two things beat it, the
schedule's *shape* and the *step count*. The shape ramps the multiplier **×3.4 at t=0
down to ×1.6 at t=1** (below, at 20 steps, vs uniform ×2 at matched memorization):

| dataset | NN-RMS | IS | FID | copies |
|---|---|---|---|---|
| CIFAR-10 s0/s1 | 0.267/0.270, 0.271/0.274 | 6.44/6.29, 6.31/6.07 | **83.3/94.0, 86.1/93.0** | ~0 |
| CelebA | 0.301/0.302 | 2.34/2.34 | **70.0/79.4** | 0.000/0.000 |

~9 FID at matched distance, zero copies, vs a ~2.8 FID seed spread. MNIST is flat
(16.6/17.0, floor 12.2) and EBMLS dominated (IS 3.30 vs 2.78, FID 201/189).
**Six shape controls now bound the mechanism and all fail** (detail in the notebook):
up-ramp, flat, single-ended pushes, and both axes of the (level, compression)
parameterization. Each trades memorization for quality along one frontier.

**Step count is the first knob that leaves that frontier, and today's win**
(`2026-10-22c,d/`). `--nsteps` resamples the window-vs-t curve onto more steps, holding
the window at each noise level fixed. Seed-matched, 60 steps beats the 20-step champion
on IS, FID *and* median training-set distance at once:

| steps | seed | NN-RMS | IS | FID | copies |
|---|---|---|---|---|---|
| 20 (champion) | 0/1 | 0.267/0.270 | 6.44/6.31 | 83.3/86.1 | 0.002/~0 |
| 40 | 0 | 0.279 | 6.31±0.27 | 82.3 | 0.010 |
| **60** | **0/1** | **0.280/0.284** | **6.75/6.75** | **78.8/80.0** | 0.016/0.008 |
| 100 | 0 | 0.286 | 6.63±0.62 | 78.9 | 0.014 |

The IS gain replicates: both 60-step seeds beat the better 20-step seed, +0.37
seed-matched against a 0.13 seed spread. It saturates at 60 — 100 steps leaves FID flat
and triples the IS split spread — and costs 5.8× the generation time of 20 steps.

## Running jobs

- None.

## Open problems

- **Blocked on hshunt: the layer-2 posterior.** You said the implemented one is wrong
  for a training-set image. Today's code conditions on `x_hat_1` (not the noisy `x`),
  per-pixel Gaussian, `sigma^2 = temperature * b_t^2/a_t^2`, no prior. It also moves
  the entropies the barrier thresholds on: every MLS/EBMLS number needs rerunning.
- **Layer 2 replaces rather than refines**: weights sum to 1 per pixel, so `x_hat_1 + delta` *is* the layer-2 posterior mean. Try an additive residual.
- With more steps the median sample moves *away* from the training set while the copy tail grows (0.2%→1.6% at seed 0, but 0.8% at seed 1, so noisy at 500 samples). Unexplained.
- Pre-existing, not mine: `test_channel_theory.py::...boundary_returns_nan` fails and that file hangs on its last test — a semantics call from you.

## Next steps

1. Re-tune the window schedule at 60 steps — the ramp was fitted at 20 and need not still win; the shape-vs-step interaction is untested.
2. Additive residual layer 2, once the corrected posterior lands.
3. ImageNet: candidates cost O(1) in resolution, but the per-step pass over the train set scales with N (ANN search), and 60 steps multiplies that by 3.
