# STATUS

**RHM sampling complexity** is answered and delivered (due 12:00 PT Fri 11/6; re-reported
2026-11-05 with the measurement record below). Both jobs **COMPLETED**: 51143385 (s3 P ladder,
5:41:05) and 51175489 (`rhm_k_content.py`, 6:51). **MLS** is parked on hshunt's corrected
layer-2 posterior (LOG.md, `mls_entropy_barrier_results.ipynb`).

## A. The answer
`P` = training samples before the LS machine emits output **both novel and fully grammatical**.
> **`P* = exp(Theta(n))`, `n` = sites. At the U=1 window in 2D, `P* = 0.21 |R_L|`.**
Measured, not derived; the `C(k) = f0^U` ceiling stays retracted. 30 bracketed 2D peaks collapse to median 0.197 while `|R_L|` spans 49 152x; in 1D `ln P*` grows 0.374/site over n=4..16 but `P* = Theta(|R_L|)` fails (`b=0.689+-0.045`). Root needs `k >= U1 = 2s^(L-1)-1`.

## B. Open question: why x* = P*/|R_L| differs by dimension
n=16: **1D 0.083 (k=U1=15), 2D 0.159 (25 cells), 1.92x apart.** Three explanations **all refuted**: coverage (sign backwards, Spearman +1.00); saturation `y* = P* n/N_k` (4.4x across dimensions, 61x across k within 1D); entropy budget `exp(H_tot)/|R_L|` (1D 0.68 / 2D 337.6, **496x**). The third did explain k=11 beating U1=15: H_tot falls 11.82 -> 8.21 nats over k=3..17.

## C. hshunt's patch-size criterion: the length scale is everything, the floors are nothing
s=2 is degenerate (`U1 = side - 1`, one odd rung in `[U1, side]` at every depth). s=3 L=2 (9x9,
rungs 5/7/9) is the first fair test; **all 12 cells** (`rhm_patch_size_2026-11-04.ipynb`):
- **Adaptive beats the best fixed k 12/12**, median ratio 11.8x, median delta gram +0.208. Fixed
  optimum is `k=7`, not `U1=5`, in 10/12. `k(t)` falls 9.0 -> 5.8..7.5, span growing with `v`.
- **Both floors refuted.** Best `h_min` is 0 in 12/12 (success monotone down in all 10 scored before the last cells landed: median .181/.103/.101/.049 at h=0/.1/.5/1). The count rival `n_min` loses the same way (v2m2 rs0 P=300: .414 -> .148 as `n_min` 1 -> 10, gram -0.40 for novel +0.14). Both only ever move a site to a *shorter* window, which is less grammatical.
> **"Largest window matching any training patch at all" is the whole rule; nothing added to it helps.**

## D. The patch size is per-site, not a schedule -- answers "how are you setting it over time?"
`rhm_k_schedule.py`: four matched arms, control schedule derived from the adaptive arm's *own* trace so it cannot be rigged. **9 cells, `rhm_ksched_score.py`: adaptive beats the bare schedule 9/9 and the best fixed k 9/9** (median 8.2x at v2, ~65x at v3/v4). The schedule recovers a median **15%** of adaptive success at the *same window budget* (median mean-k gap 0.08); capping the criterion to its own schedule is free at v2, costs 8-30% at v3/v4.
> **Time alone cannot set the patch size. What matters is which *site* gets which size.**
`rhm_k_locality.py` says what the site signal is *not*: chosen k regressed on the site's offset within its level-1 block gives **R^2 = 0.000 in 18/18 runs**, adding nothing jointly with step either. Step alone explains 0.09-0.21 (v2), 0.22-0.36 (v3), 0.30-0.41 (v4), rising with `v`, falling with `P`.
The local **revealed-neighbour count** (how many of the 9x9 around a site are fixed) beats step in 18/18 and subsumes it (adding step gains .005-.013), so the schedule is a *consequence* of that count. It caps at .47: the rest turns on *which* tokens were revealed, not how many.
> **Position 0%, time <=41%, local reveal count <=47%, remainder >=53% genuinely content-dependent.**
That residual is **grammar, mostly across block boundaries** (`rhm_k_crossblock.py`, 18/18): over
(nrev, n_words, n_sym_local), the block's admissible-symbol count *after every other block
constrains it through the root* adds **median +0.196 R^2** (+.116..+.294) net of a permutation null
of +0.000; within-block control <=+0.002; root tightens a block at 46-71% of observations, rising
with `v`. **It is one bit, not a count** -- a bare "no root production survives every block at once"
flag reproduces the gain to +0.000 in 18/18, so the straddling window runs the *consistency check*
hshunt described rather than accumulating evidence.
**The bit survives the exact posterior** (`rhm_k_bp.py`, 51201708, 18/18): redone with
`belief_propagation` instead of survivor counts, the graded `h_root = H(block symbol | all observed
leaves)` adds **+0.000** over (nrev, n_words, h_local, dead) in every cell, net −0.000 -- while the
bit itself re-earns **+0.197 [+0.116..+0.294]**, reproducing the count path's +0.196 to three
decimals from an independent computation. BP deadness equals the enumeration's `n_sym_root == 0` at
**1.0000 in 18/18**. The feature had plenty to grade: 6-9 bins over 33-218 distinct values, with the
zero atom (root pins the symbol) at 63-80% of alive observations.
> **The Bayes-optimal posterior adds nothing over the one consistency bit** -- a much harder
> statement than the coarse count adding nothing, and the direction I flagged as the stronger one.
Unregistered diagnostic, same pass: `h_leaf`, the exact posterior entropy at the *site's own token*
-- the quantity hshunt named for the criterion -- explains **0.416 [0.303..0.470]** of `k` alone but
also adds **+0.000** over the baseline, 18/18. So the criterion's behaviour is *consistent with*
tracking posterior entropy while that entropy carries no information independent of reveal count,
block word count and the bit. Unregistered: the sign on `k` **flips with `v`**
(root-tightened site gets a shorter window at v=2, 6/6; longer at v=4, 6/6). No mechanism claimed.
The caveat I raised from this against the 51153821 prediction was **wrong**: sched/adaptive does
fall with `v` (.212 -> .140, non-monotone, 3 seeds/v). R^2 in k is not decision quality -- the
schedule can explain more of k at high `v` and still pick wrong at the sites that cost the most.

## E. Pooling costs the sampler, not the support, and not via phase count
`gram_phase/gram_pooled = pool_cost x (eff ratio)`, `pool_cost` exact from enumeration (`rhm_straddle.py:165`). **50 cells: 3.51x = 1.06x (support) x 3.21x (sampler)**; the pooled sampler reaches 0.28 of its own ceiling, per-phase 0.89. The registered prediction that the ratio grows with phases mixed is **refuted**: s=2 (4 phases) 4.04x over 86 cells, s=3 (9 phases) 3.20x over 14.

## Open problems
- The 1D/2D offset has no mechanism after three refutations. No fourth without a measurement.
- 2D peaks scatter 4.8x per cell though medians reproduce to 1.2x (raise n_gen first); 2D L=4 needs
  k=15, out of reach; one L=3 curve unbracketed (v3m2 rs0, `x* >= 0.477`).
- **Blocked on hshunt (MLS):** the corrected layer-2 posterior. Nothing MLS runs until it lands.

## Next steps
1. **Done.** C+D deck re-rendered on 51143385's completed ladder and re-uploaded 2026-11-05
   (9 pages, cross-block page added). Counts are now derived from the plotted rows, not literals.
   **Correction (2026-11-06):** my note that the notebook "still shows the pre-completion numbers"
   was wrong, and so was the 12.342 / +0.210 I recorded for the re-score. Recomputed from the 693
   raw rows: median ratio **11.7509**, median delta gram **+0.2083**, 12 cells. The notebook's
   `11.8x` (`%.1f`) and the deck's `12x` (`%.0f`) are the same number -- there was never a
   discrepancy. Notebook re-executed anyway; outputs unchanged. Section C was right all along.
2. **Done (51175489), 18/18 cells**: `n_words` (level-1 words still compatible with a site's own block) adds **median +0.107 R^2 over `nrev`**, clearing the registered .05 in **18/18**; block-fill control <= .006 everywhere. So the >=53% residual *is* grammatical ambiguity at the block scale, and it is not fill level wearing a grammar label.
   But the registered "gain grows with `v`" is **refuted**: it falls monotonically, .125 / .102 / .096 at v=2/3/4 (6 cells each). The stated mechanism was already refuted by the smoke cell (mean k *rises* with `n_words`, confounded with fill), which is why the gain is scored over `nrev`.
3. **Done (51197626 / 51198000 / 51198269), 18/18**: cross-block result in D. 51198269 adds only
   the `frac_multi` substrate column that licenses the "one bit" wording.
4. **Done (51201708), 18/18**: BP refinement in D. Registered 1 (`h_root` adds >=0.02) is
   **refuted 0/18** -- the null I had flagged as the stronger outcome; 2 (BP vs enumeration >99%)
   holds at **1.0000**; 3 (substrate) holds at 6-9 bins. Smoke testing caught 3 failing for the
   wrong reason first: `h_root` is exactly 0 on 71-80% of alive sites, so plain quantile cuts all
   landed on that atom and the graded tail got 2 bins. `qbin` now bins the atom separately. The
   gain was +0.000 before and after, so that fixed the objection, not the answer.
   Deck re-rendered to 10 pages (BP page 6, traces now 7) and re-uploaded 2026-11-05.
5. Open, cheapest next: the cross-block bit is measured only at `s=3 L=2`, where one root production
   couples 9 blocks. Whether it stays one bit when the root is further away (`L=3`) is untested, and
   is the first place a graded version could appear.

## Discipline notes
- Analysis runs in `srun -p sganguli -t 10 -c 2 --mem=4G`, never a login node; there `ml devel math
  python/3.12.1` gives `python3` only -- bare `python` is still 2.7.5.
- `--max_patches` does **not** bound memory (`run_cell` builds the table, then tests it, `:64`), so cap the P ladder. Footprint tracks patch count: 1.6 GB at 75k, >8G at 733k.
- `rhm_k_schedule.py` absolute numbers sit ~5% off `rhm_sampling_complexity.py` at the same cell (different train/sampler seed convention). Arms within one script are matched; across scripts they are not.
