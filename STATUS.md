# STATUS

**RHM sampling complexity** is answered and delivered (was due 12:00 PT Fri 11/6). Running: **51141860**
(s3 v4m2), **51143385** (s3 v2m3, 64G after 51141861 OOM'd), **51142878** (pooling: 40 cells + phase test).
**MLS** is parked on hshunt's corrected layer-2 posterior (LOG.md, `mls_entropy_barrier_results.ipynb`).

## A. The answer
`P` = training samples before the LS machine emits output **both novel and fully grammatical**.
> **`P* = exp(Theta(n))`, `n` = sites. At the U=1 window in 2D, `P* = 0.21 |R_L|`.**
Measured, not derived; the `C(k) = f0^U` ceiling stays retracted. 30 bracketed 2D peaks collapse
to median 0.197 while `|R_L|` spans 49 152x; in 1D `ln P*` grows 0.374/site over n=4..16 but
`P* = Theta(|R_L|)` fails (`b = 0.689 +- 0.045`). Root needs `k >= U1 = 2s^(L-1)-1`.

## B. The open question: why x* = P*/|R_L| differs by dimension
One instrument, n=16: **1D 0.083 (k=U1=15), 2D 0.159 (25 cells), 1.92x apart.** Three explanations,
**all three refuted**: (1) coverage -- sign backwards, `x*` *rises* with it (Spearman +1.00); (2)
table saturation `y* = P* n/N_k` -- 4.4x across dimensions, 61x across k within 1D; (3)
posterior-entropy budget -- `x* = exp(H_tot)/|R_L|` gives 1D 0.68 / 2D 337.6, **496x apart**.
**What (3) did establish -- why success peaks at k=11, not U1=15:** H_tot falls 11.82 -> 8.21 nats
over k=3..17 at n=16 as grammaticality rises 0.06 -> 0.78 and novelty falls 1.00 -> 0.52.

## C. hshunt's patch-size criterion: s=2 said no, s=3 says *half* yes
s=2 is a degenerate board -- `U1 = side - 1`, so the only odd rung in `[U1, side]` is U1 at *every*
depth -- and there the criterion lost to fixed `k=U1` 12/15. s=3 L=2 (9x9, rungs 5/7/9) is the
first fair test. **8 of 12 cells scored** (`scripts/rhm_adaptive_s3_score.py`):
- **Adaptive beats the best fixed k 8/8, median ratio 50.4x.** Fixed optimum is `k=7`, not `U1=5`,
  in 7/8. `k(t)` falls 9.0 -> 5.8..7.5 monotonically, span growing with `v` (1.5 / 2.9 / 3.2).
- **But the best floor is `h_min=0` in 8/8, and success falls monotonically in `h_min` in 8/8**
  (median .181 / .103 / .101 / .057 at h = 0 / .1 / .5 / 1.0). Accept is `n_match > 0 and
  h >= h_min` (`rhm_sampling.py:502`), and in `match` mode `h = log(effective matching patches)`,
  so `h_min` is a support-*diversity* floor and 0 is the bare support-*existence* test.
- Trace at v2m2 rs0, P=300, 64 gens: adaptive .406/.625/.781 (succ/gram/novel) vs fixed k=7
  .063/.094/.969. The win is **grammaticality, 6.7x**, bought with a little novelty.
> **The length scale is what matters -- "largest window that matched any training patch at all"
> is worth 50x over any fixed window. The entropy floor on top of it only subtracts.**

## D. Pooling costs the sampler, not the support
`gram_phase/gram_pooled = pool_cost x (eff_phase/eff_pooled)`, `eff = gram/ceil`, `pool_cost =
lv_pooled/lv_aligned` exact from enumeration (`rhm_straddle.py:165`). Over 10 matched cells:
**3.46x = 1.05x (support) x 3.18x (sampler)**; peak-success ratio 4.00x over 45 L=2 cells. Ceilings
nearly equal (0.68 vs 0.76) but the pooled sampler reaches 0.22 of its own ceiling, per-phase 0.72.
`scripts/rhm_pooling_cost.py`. 51142878 widens this 10 -> 50 cells and tests phase-count scaling.

## Open problems
- The 1D/2D offset has no mechanism after three refutations. No fourth without a measurement.
- 2D peaks scatter 4.8x per cell though population medians reproduce to 1.2x: raise n_gen first.
  2D L=4 needs k=15, out of reach; one L=3 curve unbracketed (v3m2 rs0, `x* >= 0.477`).
- **Blocked on hshunt (MLS):** the corrected layer-2 posterior. Nothing MLS runs until it lands.

## Next steps
1. When 51141860/51143385 land, rescore all 12 s=3 cells, then build the fair-test deck + notebook
   (traces from `rhm_adaptive_trace.py --k_fixed`, tokenization page) and upload.
2. When 51142878 lands, rerun `rhm_pooling_cost.py`: does the sampler term grow 4 -> 9 phases?
3. Does a hard count threshold `n_match >= c` beat `c = 1`? `h_min` tests only the soft version.

## Discipline notes
- Analysis runs in `srun -p sganguli -t 10 -c 2 --mem=4G`, never a login node; there
  `ml devel math python/3.12.1` gives `python3` only -- bare `python` is still 2.7.5.
- `--max_patches` does **not** bound memory: `run_cell` builds the table, then tests it
  (`:64`). Cap the P ladder. Footprint tracks patch count: 1.6 GB at 75k, >8G at 733k.
