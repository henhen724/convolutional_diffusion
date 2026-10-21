# LOG

Append-only. Newest at the bottom. At most a few lines per session; STATUS.md
holds the live state.

- 2026-10-21 Built `MultilayerLocalScoreModule` + `EntropyBarrierMLSModule` (layer 2 sees only layer 1's estimate, adds the posterior-mean error over training images in a larger window chosen by an entropy floor); fixed three batch>1 broadcasting bugs in `LocalScoreModule.forward` that made mixed-t batches NaN; 17 tests in `tests/test_multilayer_ls.py`.
- 2026-10-21 Swept the entropy barrier on CIFAR-10/MNIST/CelebA against an IS + nearest-neighbour-memorization eval (`scripts/mls_inception.py`). Negative result: EBMLS is dominated by plain LS with wider windows at every matched memorization level, because layer 2 as specified replaces rather than refines layer 1's estimate.
- 2026-10-21 Positive result from the control: uniformly scaling the per-step LS windows x2 takes CIFAR-10 IS from 2.78 to 6.29 with zero training-set copies, reproduced across two seeds; x2.5 and x3 gain more only by copying 15% and 51% of samples. Notebook `notebooks/mls_entropy_barrier_results.ipynb`, slides `results/mls_inception/2026-10-21_mls_slides.pdf`.
