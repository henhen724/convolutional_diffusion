# STATUS — multilayer LS (MLS) / entropy-barrier MLS

Goal: extend LS to a multilayer structure, raise sample quality on CIFAR-10 /
CelebA / MNIST, keep cost independent of image size (for ImageNet eventually).

## Current best (commit 25794ab, 500 samples, 5 IS splits, FID vs full train set)

**The multilayer layer is dominated on both metrics; the win is the per-step
window schedule — its *shape*, not just its width.** Best config: a ramped
multiplier **×3.4 at t=0 down to ×1.6 at t=1**, loading width onto the low-noise
end. Against uniform ×2 at matched memorization (NN-RMS and copy fraction):

| dataset | NN-RMS ramp/unif | IS ramp/unif | FID ramp/unif | copies |
|---|---|---|---|---|
| CIFAR-10 s0 | 0.267/0.270 | 6.44/6.29 | **83.3/94.0** | 0.002/0.000 |
| CIFAR-10 s1 | 0.271/0.274 | 6.31/6.07 | **86.1/93.0** | 0.000/0.000 |
| CelebA | 0.301/0.302 | 2.34/2.34 | **70.0/79.4** | 0.000/0.000 |
| MNIST | 0.191/0.240 | 1.95/1.92 | 16.6/17.0 | 0.25/0.08 |

~9 FID at matched distance to the training set and zero copies, vs a ~1 FID seed
spread. MNIST shows no effect (on the uniform frontier; its whole FID range is
19.0→15.8 against a 12.2 real-image floor).

Both controls fail, which brackets the mechanism. **Up-ramp** is worse than
uniform (1.4→2.6: FID 89.8 at NN-RMS 0.227 vs ≈77 for uniform), so tilt
*direction* matters and runs against the coarse-to-fine intuition. But a **flat
schedule** (`--flat_k`) sits on the *uniform* frontier, not the ramp's (k=17: IS
5.38 at 0.268 vs the ramp's 6.44 at 0.267; k=23: 6.86 at 0.237 ≈ uniform's
6.86). The ordering still earns its keep: the down-ramp partially reshapes it
rather than removing it, and the optimum lies between the two shapes.

Still standing from earlier: EBMLS is dominated everywhere (barrier 1.0 beats
the LS baseline on IS, 3.30 vs 2.78, but is worse on FID, 200.7 vs 188.7).

Notebook `notebooks/mls_entropy_barrier_results.ipynb`, slides
`notebooks/2026-10-21_mls_entropy_barrier_slides.pdf` (sent to OneDrive), data in
`results/mls_inception/2026-10-2{1,1b,1c,2}/`.

## Running jobs

- 48915101 `add_fid.py` for the two flat-schedule runs — pending on `sganguli`
  ~50 min. Only merges FID into their metrics.json; notebook and slides are
  complete without it. Rerun the builders once it lands.

## Open problems

- **Blocked on hshunt: the layer-2 posterior.** You said the implemented one is
  not the correct posterior for a training-set image. Today's code conditions on
  `x_hat_1` (not the noisy `x`), per-pixel Gaussian, `sigma^2 = temperature *
  b_t^2/a_t^2`, no explicit prior. Changing the conditioning variable or the
  variance also changes the entropies the barrier thresholds on, so every
  MLS/EBMLS number needs rerunning (~15 min/CIFAR config). LS results unaffected.
- **Layer 2 replaces rather than refines**: its weights sum to 1 per pixel, so
  `x_hat_1 + delta` is exactly the layer-2 posterior mean over whole training
  images. An additive residual that cannot represent a single training image on
  its own is worth trying before abandoning the architecture.
- **Barrier wording**: entropy falls with k, so "largest k *below* the barrier"
  is degenerate; implemented as largest k whose entropy is still >= barrier.
- OOM at large windows (genbatch × scorebatch × k² × pixels, unfold
  materialized in full) means batch sizes are hand-tuned per multiplier.
- Pre-existing, not mine: `tests/test_channel_theory.py::...boundary_returns_nan`
  fails and that file hangs on its last test — needs a semantics call from you.
- Cosmetic: `results/mls_inception/2026-10-22/` is named a day ahead of the
  clock. Left as-is; builders glob `2026-10-2*`.

## Next steps

1. Fit the per-step schedule directly against FID under a zero-copy constraint;
   the two-parameter ramp family is too coarse to reach the optimum.
2. Additive residual layer 2, once the corrected posterior lands.
3. ImageNet: candidates already cost the same at any resolution, but the
   per-step pass over the training set still scales with N (needs ANN search).
