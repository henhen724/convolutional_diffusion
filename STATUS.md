# STATUS

**RHM sampling complexity** is answered and delivered (was due 12:00 PT Fri 11/6).
Running: **51141860** (s3 cell v4m2, 2.5h) and **51141861** (s3 cell v2m3, 8h) -- the last 4 of 12.
**MLS** is parked on hshunt's corrected layer-2 posterior (LOG.md, `notebooks/mls_entropy_barrier_results.ipynb`).

## A. The answer
`P` = training samples before the LS machine emits output **both novel and fully grammatical**.
> **`P* = exp(Theta(n))`, `n` = sites. At the U=1 window in 2D, `P* = 0.21 |R_L|`.**

Measured, not derived; the `C(k) = f0^U` ceiling stays retracted. 30 bracketed 2D peaks collapse
to median 0.197 while `|R_L|` spans 49 152x; in 1D `ln P*` grows 0.374/site over n=4..16 but
`P* = Theta(|R_L|)` fails (`b = 0.689 +- 0.045`). Root needs `k >= U1 = 2s^(L-1)-1`.

## B. The open question: why x* = P*/|R_L| differs by dimension
One instrument, n=16: **1D 0.083 (k=U1=15), 2D 0.159 (25 cells), 1.92x apart.** Three explanations,
**all three refuted**: (1) coverage -- sign backwards, `x*` *rises* with it (Spearman +1.00), and
matching on it leaves 5.1x where matching on `n` leaves 2.3x; (2) table saturation `y* = P* n/N_k`
-- 4.4x across dimensions, 61x across k within 1D; (3) posterior-entropy budget --
`x* = exp(H_tot)/|R_L|` gives 1D 0.68 / 2D 337.6, **496x apart, 258x worse**.
**What (3) did establish -- why success peaks at k=11, not U1=15:** H_tot falls 11.82 -> 8.21 nats
over k=3..17 at n=16 as grammaticality rises 0.06 -> 0.78 and novelty falls 1.00 -> 0.52; copies
are the low-entropy trajectories (-1.75 nats in 1D, 84% of 781 runs; -7.42 in 2D, 100%).

## C. hshunt's patch-size criterion: s=2 said no, s=3 says *half* yes
"Largest `k` above an entropy floor." s=2 is a degenerate board -- `U1 = side - 1`, so the only odd
rung in `[U1, side]` is U1 at *every* depth -- and there the criterion lost to fixed `k=U1` 12/15.
s=3 L=2 (9x9, rungs 5/7/9) is the first fair test. **8 of 12 cells scored (51106614, timed out):**
- **Adaptive beats the best fixed k 8/8, median ratio 50.4x.**
- Fixed optimum is `k=7`, not `U1=5`, in 7/8. `k(t)` falls 9.0 -> 5.8..7.5 monotonically, span
  growing with `v` (1.5 / 2.9 / 3.2 at v=2/3/4): the schedule really uses the middle rung.
- **But the best floor is `h_min=0` in 8/8, and success falls monotonically in `h_min` in 8/8**
  (median .181 / .103 / .101 / .057 at h = 0 / .1 / .5 / 1.0). Accept is `n_match > 0 and
  h >= h_min` (`rhm_sampling.py:502`), and in `match` mode `h = log(effective matching patches)`,
  so `h_min` is a support-*diversity* floor and 0 is the bare support-*existence* test.
> **The length scale is what matters -- "largest window that matched any training patch at all"
> is worth 50x over any fixed window. The entropy floor on top of it only subtracts.**

## Open problems
- The 1D/2D offset has no mechanism after three refutations. No fourth without a measurement.
- 2D peaks scatter 4.8x per cell though population medians reproduce to 1.2x: raise n_gen first.
- 2D L=4 needs k=15, out of reach; one L=3 curve unbracketed (v3m2 rs0, `x* >= 0.477`). The int64
  overflow at `k >= 7` is in `rhm_straddle.py:window_codes` (`a_j` support only, not the sampler).
- Pooled-vs-per-phase gap (3-4x) far exceeds their support gap (`pool_cost = lv_pooled/lv_aligned`,
  1.04-1.38x, `rhm_straddle.py:165`): pooling costs the *sampler*, not the support. Page unwritten.
- **Blocked on hshunt (MLS):** the corrected layer-2 posterior. Nothing MLS runs until it lands.

## Next steps
1. When 51141860/51141861 land, rerun `scripts/rhm_adaptive_s3_score.py` over all 12 cells, then
   build the fair-test deck + notebook (traces + tokenization page) and upload.
2. The pooling-costs-the-sampler page: join peak success at `block_align` 1 vs 2 against
   `pool_cost` per matched (cell, k). Pooled has 2205 L=2 rows, per-phase 1611, k in {3,5,7}.
3. Open question the s=3 result raises: if support-existence is the whole rule, does a *count*
   threshold (`n_match >= c`) beat `c = 1`? h_min already tests the soft version and says no.

## Discipline notes
- Analysis runs in `srun -p sganguli -t 10 -c 2 --mem=4G`, never a login node; there
  `ml devel math python/3.12.1` gives `python3` only -- bare `python` is still 2.7.5.
- `rhm_entropy_analysis.py`'s `geo()` returns the **median**, not a geometric mean (1.5-1.7x).
- s=3 L=2 adaptive costs ~2900-5500 s per cell, 1.6 GB: one cell per job, 8G, not four in one.
