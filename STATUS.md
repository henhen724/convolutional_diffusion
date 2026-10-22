# STATUS — multilayer LS (MLS) / entropy-barrier MLS

Goal: multilayer LS, better samples on CIFAR-10 / CelebA / MNIST, cost independent of image size (ImageNet eventually).

## Current best (commit 7625f61, 500 samples, 5 IS splits, FID vs full train set)

**The multilayer layer is dominated on both metrics; the win is the per-step window
schedule — its *shape*, not just its width.** Best config: a multiplier ramped **×3.4
at t=0 down to ×1.6 at t=1**, vs uniform ×2 at matched memorization (ramp/uniform):

| dataset | NN-RMS | IS | FID | copies |
|---|---|---|---|---|
| CIFAR-10 s0/s1 | 0.267/0.270, 0.271/0.274 | 6.44/6.29, 6.31/6.07 | **83.3/94.0, 86.1/93.0** | ~0 |
| CelebA | 0.301/0.302 | 2.34/2.34 | **70.0/79.4** | 0.000/0.000 |

~9 FID at matched training-set distance and zero copies, vs a ~1 FID seed spread.
MNIST is flat (16.6/17.0, on the uniform frontier, and uninformative throughout: its
whole FID range is 19.0→15.8 against a 12.2 real floor).
EBMLS stays dominated: barrier 1.0 wins on IS (3.30 vs 2.78) but not FID (201/189).

Both controls fail, bracketing the mechanism. **Up-ramp** is worse than uniform
(1.4→2.6: FID 89.8 at NN-RMS 0.227 vs ≈77), so tilt *direction* matters and runs
against the coarse-to-fine intuition; a **flat schedule** (`--flat_k`) is no better
than uniform, worse at the zero-copy point (k=17: FID 99.0 at NN-RMS 0.268 vs 94.0
uniform, 83.3 ramp), so the coarse-to-fine ordering earns its keep.

**Compressing the schedule's range is a second, FID-only win** (`2026-10-21d/`).
`--scale_gamma` raises the base windows to a power γ about their geometric mean
(γ=1 ≡ uniform ×2, γ=0 ≡ flat at the same level). At m=2.0, γ=0.75/0.60/0.45/0.25
give FID 101.8/103.6/105.3/110.1 at NN-RMS 0.288–0.291, zero copies, vs ≈121 for
the uniform curve at 0.289 — ≈430 FID per unit NN-RMS against uniform's ≈1400.
**IS reads it as negative (5.17→4.94); I called it that before the FID landed.**

`notebooks/mls_entropy_barrier_results.ipynb` + slides PDF (OneDrive); data in `results/mls_inception/2026-10-2{1,1b,1c,1d,1e,2}/`.

## Running jobs

- 48924943 → `2026-10-21e/`: with the schedule's peak window held fixed (m1=1.6), push only the low-noise end — ramps 4.5→1.6 and 6.0→1.6, plus 3.4→2.0 as the control that moves the peak instead. Hypothesis from the kgeo/kmax table: memorization tracks the *peak* window while the low-noise end buys quality almost free, so 4.5/6.0 should hold NN-RMS ≈0.27 at zero copies with lower FID, and 3.4→2.0 should memorize more. FID follows as 48924944.

## Open problems

- **Blocked on hshunt: the layer-2 posterior.** You said the implemented one is
  wrong for a training-set image. Today's code conditions on `x_hat_1` (not the
  noisy `x`), per-pixel Gaussian, `sigma^2 = temperature * b_t^2/a_t^2`, no prior.
  It also moves the entropies the barrier thresholds on, so every MLS/EBMLS number
  needs rerunning (LS results are unaffected).
- **Layer 2 replaces rather than refines**: weights sum to 1 per pixel, so
  `x_hat_1 + delta` *is* the layer-2 posterior mean. Try an additive residual.
- Barrier wording (entropy falls with k, so "largest k *below* the barrier" is
  degenerate; implemented as largest k >= barrier); `2026-10-22/` is a day ahead.
- Pre-existing, not mine: `test_channel_theory.py::...boundary_returns_nan` fails and that file hangs on its last test — a semantics call from you.

## Next steps

1. Rebuild notebook + slides once `2026-10-21e` has FID, and send the deck.
2. Then fit the per-step schedule directly against FID under a zero-copy limit.
3. Additive residual layer 2, once the corrected posterior lands.
4. ImageNet: candidates cost the same at any resolution, but the per-step pass
   over the training set still scales with N (needs ANN search).
