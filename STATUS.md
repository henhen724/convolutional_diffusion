# STATUS

Two threads. **RHM sampling complexity** is active; **MLS** is parked (see LOG.md, and
`notebooks/mls_entropy_barrier_results.ipynb` + `2026-10-22_mls_entropy_barrier_slides.pdf`).

## A. RHM sampling complexity (active, started 2026-11-02)

Question: how many training grids `P` before the LS machine generates RHM samples that are
**both novel and fully grammatical**, and how does that `P*` scale with `(v, s, L, m)`?

New, all validated by `tests/test_rhm_sampling.py` (14 pass):
- `src/utils/rhm_grammar.py` — exact parser. Bottom-up feasible-symbol bitmasks give
  grammaticality, `parse_height`, per-level parse fractions and exact novelty.
- `src/utils/rhm_sampling.py` — `PatchCountDiscreteScoreMachine`, identical posterior to
  `LocalDiscreteScoreMachine` (max|diff| 0.0) but costed by *distinct* patches, so `P` can
  go to 1e5. Options `block_align` (phase-aware vs translation-invariant) and `pad_mode`
  (`outside` tells it where the grid edge is) isolate two separate assumptions.
- `discrete_score.py` gained `reveal_one`: the existing sampler reveals sites in parallel
  from their marginals, costing ~`n^2/2T` independence errors per sample (~12% of grids at
  `n=64, T=256`). One site per step is exact *and* 4x cheaper. Used everywhere below.

**Established so far.** Counting is exact: distinct aligned `s^j` blocks
`A_j = v·m^((s^(2j)-1)/(s^2-1))`, measured 12/6/972/96 vs predicted for four `(v,m,L)`.
Total grids `v·m^((s^(2L)-1)/(s^2-1))` (4e10 at `v=4,m=3,L=3`), so **novelty is free and
grammaticality is the binding constraint**. A window of side `k` centred on a site reaches
only `k//2`, so it can enforce a block of side `b` only when `k >= 2b-1`: for `L=3`, `k=3`
covers the 2x2 leaf blocks, `k=7` the 4x4, `k=15` everything (pure memorization, verified
gram 1.000 / novel 0.000). Predicts grammaticality decaying like `f^U` with
`U = (s^(2(L-j))-1)/(s^2-1)` unenforced rules and `f ≈ m/v^(s^2-1)`.

## Running jobs

- **50859790** (CPU, 8 h) → `results/rhm_sampling/2026-11-02/counts_{m,v,k7}.json`:
  `N_k(P)`, distinct windows vs P, to 3e5. Cheap proxy for `P*`; tests `A_j` scaling in v, m.
- **50859792** (GPU, 10 h) → same dir, `A1`–`D`: generation sweep, grammatical/novel/success
  and per-level parse fraction vs P, over k, block_align, pad_mode, then v, m, L one at a time.

## Open problems

- **The repo's LS machine may have no finite `P*`.** It pools patches over all offsets, but
  the RHM is block-aligned, so a window cannot tell its phase. At `v=4,L=2,m=3,k=3` with the
  grammar's *whole* support in training (P=2000 vs 972 grids) it scores gram 0.000 with only
  22% of aligned 2x2 blocks valid — a ceiling, not a data limit. Phase-awareness lifts 50%-mask
  token accuracy 0.613→0.714. Job A2/A3 prices this properly.
- `k=5` is *worse* than `k=3` at P=2000 (token acc 0.40 vs 0.82) and gets worse with *more*
  context — the data-starvation signature, 1e5 distinct 5x5 patches from 2000 grids.
- **Blocked on hshunt (MLS):** the corrected layer-2 posterior. Nothing MLS runs until it lands.
- Pre-existing, not mine: `test_channel_theory.py::...boundary_returns_nan` fails, file hangs.

## Next steps

1. Read `P*` off both jobs; fit `P*(k, v, m, L)` and check it against the `f^U` prediction.
2. Results notebook + slide PDF, with example grids and the tokenization spelled out.
3. MLS: monotonized 20-step schedule control, to separate finer integration from smoother shape.
