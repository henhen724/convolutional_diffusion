# STATUS

**RHM sampling complexity** is answered and delivered (due 12:00 PT Fri 11/6; re-reported
2026-11-05 with the measurement record below). Both jobs **COMPLETED**: 51143385 (s3 P ladder,
5:41:05) and 51175489 (`rhm_k_content.py`, 6:51). **MLS** is parked on hshunt's corrected
layer-2 posterior (LOG.md, `mls_entropy_barrier_results.ipynb`).

## A. The answer
`P` = training samples before the LS machine emits output **both novel and fully grammatical**.
> **`P* = exp(Theta(n))`, `n` = sites. At the U=1 window in 2D, `P* = 0.21 |R_L|`.**
Measured, not derived; the `C(k) = f0^U` ceiling stays retracted. 30 bracketed 2D peaks collapse to median 0.197 while `|R_L|` spans 49 152x; in 1D `ln P*` grows 0.374/site over n=4..16 but `P* = Theta(|R_L|)` fails (`b=0.689+-0.045`). Root needs `k >= U1 = 2s^(L-1)-1`.
**Estimator-robust (2026-11-05):** swapping the 3-point peak fit for an inverse-variance fit over the whole top of each curve moves the 2D exponent to `b=0.927+-0.033` and the median to 0.177, overlapping CIs -- the scatter it removes was all estimator noise (Open problems). **Not yet depth-robust at fixed `(v,m)`:** next step 6.

## B. Open question: why x* = P*/|R_L| differs by dimension
**Scope label added 2026-11-05.** This section used to read "n=16: 1D 0.083, 2D 0.159, 1.92x apart" with no statement of *which* peak population those came from, and there are three in the repo. All agree on the direction; the gap's size depends on the population, so the label matters:
> | population | 1D n=16, k=U1=15 | 2D n=16, k=U1=3 | gap |
> |---|---|---|---|
> | published peaks (`peaks.json` + `peak_summary.json`) | **0.0498** (23 cells) | **0.1925** (25) | **3.87x** |
> | `rhm_collapse_variable.py` (keeps only saturated `N_k`) | 0.0599 (13) | 0.1925 (25) | 3.21x |
> | `rhm_entropy_budget`'s own re-run, `BR1` -- what this section was quoting | 0.0828 (6) | 0.1592 (25) | 1.92x |

The **published population is the one to quote**, and 3.87x is the honest headline: the other two are subsets or a separate sweep. The collapse script drops 10 of the 23 1D cells to get a saturated `N_k` (2D untouched). The entropy-budget notebook ran its own peaks and says so -- paired on the same 25 2D cells it sits at 0.83x of published (median x* 0.1592 vs 0.1925, quartiles 0.35/0.61/1.50, only 48% within a factor of 2), and its 1D side is 6 bracketed cells against 23. For the before/after in refutation 3 below, measuring both halves on `BR1` is the right call; it just is not the population number.
Every n=16 peak in both dimensions already sits at `k=U1`, so the old "(k=U1=15)" qualifier was not doing scoping work. **The published gap is larger than this section recorded, so scoping `P* = 0.21 |R_L|` to 2D is better supported than section A claimed, not worse.** Pooled over n: 1D 0.0834 (110 cells), 2D 0.1966 (30), 2.36x -- that 2D 0.1966 is section A's 0.197.
**Correction to commit `3a0bcc7`:** that commit claimed 0.159 "reproduces under no statistic" and that 0.083 was the pooled-over-all-n 1D median mislabeled as n=16. Both wrong -- I had searched only the published population. The pair is `BR1`'s, exactly as tabulated above; pooled-over-n 1D being 0.0834 against `BR1`'s 0.0828 is a coincidence that is what misled me. What survives from that commit: the published-population figures, and that my earlier flag against `rhm_collapse_variable.py`'s docstring (0.050 / 0.193 / 3.9x) was **backwards -- the docstring was right**, and it was correctly left unedited. Working files: `results/rhm_sampling/2026-11-05/xstar_{scope,stats}.txt`.
Three explanations **all refuted**: coverage (sign backwards, Spearman +1.00); saturation `y* = P* n/N_k` (4.4x across dimensions, 61x across k within 1D); entropy budget `exp(H_tot)/|R_L|` (1D 0.68 / 2D 337.6, **496x**). The third did explain k=11 beating U1=15: H_tot falls 11.82 -> 8.21 nats over k=3..17.

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
constrains it* (at `L=2` that is "through the root"; `L=3` narrows it to the enclosing parent, see
the depth result below) adds **median +0.196 R^2** (+.116..+.294) net of a permutation null
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
**How far does that consistency reach? To the parent, not the root** (`rhm_k_depth.py`, array
51214941, 18/18 cells at s=3 **L=3**, 17.0M observations per cell). Every result above was read at
`L=2`, where a block's only parent *is* the root, so "consistent with the other blocks" and
"consistent through the root" are the same sentence. `L=3` splits them: the parent is a 9x9 region,
the root couples nine of those. **Near** (dead at the 9x9 parent) adds **median +0.017 R^2**
(+.013..+.034, net +.016); **far** (dead at the root, *on top of* near) adds **+0.0005**
(+.0002..+.0018), **37x smaller and 0/18 over the 0.02 bar**. Both cuts built post-probe to give the
far bit its best shot agree: **+0.0029 among blocks the parent left alive** (2.54M observations
where `dead_root` still splits ~80/20, so this is a null with range, not a null from a pinned
feature) and **+0.0011 on the first schedule third**, 0/18 each. Graded-vs-bit replicates at the new
depth: the exact BP posterior adds **+0.000 in 18/18**. Machinery checks pass -- `hp-hr` spans
0.69..1.39 nats (near and far genuinely separate), parent and root disagree on 9.6-19.1% of
observations in 18/18.
> **The straddling window's consistency check is real but LOCAL: it reaches the site's own enclosing
> block, and the `L=2` "+0.196 one bit" was that near coupling read at the one depth where near and
> far coincide.**
Registered 1 (near >= 0.02) is **refuted on the absolute bar, 5/18**, though near is 9.7% of the
residual against an `L=3` baseline of 0.82 (`L=2`'s was 0.38). **Standing caveat, stated before the
run:** 729 sites puts `P* = exp(Theta(n))` astronomically past `P <= 1000`, so the sampler is
off-grammar early -- `dead_root` is 95/100/100% by schedule third and 95/97/98% at v=2/3/4. Item 7
therefore bought almost no range and item 6 is the stronger of the two robustness cuts. The far
coupling is being read in a deeply under-trained regime; that is a limit of this board, not an
excuse for the null.

## E. Pooling costs the sampler, not the support, and not via phase count
`gram_phase/gram_pooled = pool_cost x (eff ratio)`, `pool_cost` exact from enumeration (`rhm_straddle.py:165`). **50 cells: 3.51x = 1.06x (support) x 3.21x (sampler)**; the pooled sampler reaches 0.28 of its own ceiling, per-phase 0.89. The registered prediction that the ratio grows with phases mixed is **refuted**: s=2 (4 phases) 4.04x over 86 cells, s=3 (9 phases) 3.20x over 14.

## Open problems
- The 1D/2D offset has no mechanism after three refutations. No fourth without a measurement.
- ~~2D peaks scatter 4.8x per cell though medians reproduce to 1.2x (raise n_gen first)~~
  **Answered 2026-11-05 (`scripts/rhm_peak_noise.py`, commit `76b6001`): the scatter is the peak
  estimator, not the grammar.** Resampling each measured curve from `Binomial(n_gen, p_hat)` gives
  the estimator's spread on a *fixed* grammar; log-variances subtract, and over 30 U=1 2D curves
  nothing is left -- observed 1.47x = estimator 1.59x, grammar **1.00x** (floored; the subtraction
  went negative, which is sampling noise in the comparison, and with 2-3 seeds per cell the
  observed spread is itself poorly determined, so this is "undetectable at this power", not zero).
  Cause: `peak_of` fits a parabola through the top **three** points, so at n_gen=4096, where
  success~0.03 has SE ~0.003 and the top is a statistical tie, one tied point drags the window.
  An inverse-variance fit over every point within 2x of the max is quieter (1.26x vs 1.50x), tracks
  the same peak (displacement 1.22x) and removes A's per-cell biases (v3m2 rs1 **0.130**, v3m3 rs1
  **2.294**). The headline does not depend on the choice: `b = 0.965+-0.041` (A, med x* 0.197) vs
  `0.927+-0.033` (B, 0.177), overlapping CIs. Still open from this bullet: 2D L=4 needs k=15, out of
  reach; one L=3 curve unbracketed (v3m2 rs0, `x* >= 0.477`).
- **2D depth-stability at fixed `(v,m)` is untested** -- see next step 6. The 0.21 headline leans on
  L=2 pooled 0.1925 (nine combos) against L=3 pooled 0.2075 (two), and those combos are
  **disjoint** while x* varies 7.6x across (v,m) medians. 1D, paired within (v,m), has x* *falling*
  with depth in **15/15** combos (median ~2.8x, up to 51x), which is exactly why `P* = Theta(|R_L|)`
  fails there. If 2D does the same, section A needs a depth-dependent statement.
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
5. **Done (array 51214941, all 3 tasks COMPLETED exit 0; 2:11 / 2:07 / 2:28, MaxRSS 12.6/15.1/17.5 G
   against 24 G), 18/18 cells.** Result and reading are in section D. Scores:
   **1 near >=0.02 refuted 5/18** (median +0.0171, net +0.0157); **2 far >=0.02 null 0/18** (median
   +0.0005); **3 graded adds nothing confirmed 18/18** (+0.0000); **6 far among parent-alive null
   0/18** (+0.0029 over 2.54M obs); **7 far on the first third null 0/18** (+0.0011); checks **4**
   (`hp-hr` 0.693..1.386, not 0) and **5** (parent/root disagree 9.6-19.1%) both pass 18/18.
   Residual fractions: near 0.097, far 0.0030, far_pa 0.0089, far_early 0.0054. Deck re-rendered to
   11 pages (depth page 7, traces 8) and uploaded 2026-11-05.
   Design record, unchanged, since it was fixed before the run:
   At `L=2` a block's only parent IS the root, so "consistent with the
   other blocks" and "consistent through the root" are one sentence; `L=3` splits them into a near
   (within the 9x9 parent) and a far (through the root) coupling. Registered items 1-7 are in the
   script docstring. **Machinery validated at `L=2`:** `hp-hr 0.0e+00` exactly, parent/root disagree
   0.0%, and `+near +0.218` reproduces the known cross-block bit from independent code (check 4).
   **Sizing probes, all committed, which changed the design before the run:**
   - `dead_root` saturates at `L=3` and nothing buys it range: 98/96/96% at `P=50/300/1000`, and
     *worse* with `v` (96% at v=2, 99% at v=4; parent-alive 22% -> 11%). It is already 91% dead in
     the **first third** of the reveal schedule. Structural, not a knob: 729 sites puts
     `P* = exp(Theta(n))` astronomically beyond `P <= 1000`, so the sampler goes off-grammar almost
     immediately and the global constraint is violated nearly everywhere. **This is the standing
     caveat on any L=3 null** -- the far coupling is being read in the deeply under-trained regime.
   - Hence items **6** (far bit among blocks their parent has not killed, ~82/18 there) and **7**
     (far bit on the first schedule third), added post-probe and labelled, to give the far coupling
     its best shot before any null is claimed. Gains also reported as a fraction of residual, since
     the `L=3` baseline is **0.82** vs `L=2`'s 0.38 -- an absolute 0.02 is a far stiffer bar here.
   - Footprint measured before raising `--mem`: 28.2 GB peak at `P=3000` against a 32 G cap, RSS
     linear in patch count, so `P=1000` ~9.4 GB and `P=300` ~3 GB; job asks 24 G. `P=3000` is
     deliberately not run -- near-OOM and it buys nothing.
   - `n_gen 64`, not `L=2`'s 512: `27x27` is 9x the sites, so `64 x 729` matches `512 x 81` on
     `n_obs`. Matching n_obs is the point; matching n_gen would be a 9x overspend.
   The one smoke cell (v2 P=300 `n_gen` 16) predicted the full run closely -- near +0.019 net
   +0.017, far +0.002, item 6 +0.0083, item 7 +0.0047, same ordering -- so the 18 cells tightened
   the estimate rather than changing the verdict. Rows written incrementally; results in
   `results/rhm_sampling/2026-11-05/kdepth_v{2,3,4}_m2_L3.json` (committed with `git add -f`).
6. **In flight: 51244089** (`scripts/rhm_depth2d.sbatch`, CPU, 3 h cap). Measures the 2D `L=2 m=2`
   peaks at **v=2 and v=3**, which have never been run -- m=2 at L=2 exists only for v=4 -- so that
   L=2 can be paired against the existing `L3peak_v{2,3}_m2_k7.json` **at fixed `(v,m)`** instead of
   by pooling over disjoint combos. 25-point ladder at ~1.26x spacing over `P=2..1000` (`|R_L|` is
   64/96/128, so the whole peak is below 1000), `rule_seeds 8`, `n_gen 16384` -- which is also the
   "raise n_gen first" step, and 4x the published L=2 setting. **v=4 is the control:** it is the one
   m=2 L=2 cell that already exists (n_gen 8192, 2 seeds), so agreement there shows the denser
   ladder and extra seeds do not move the answer by themselves. Exact `|R_L|` is recomputed for
   seeds 3-7 rather than assumed flat, since x* is a ratio to it.
   **Registered before the run:** (1) v=4 reproduces the published x* within the bootstrap band;
   (2) if 2D behaves like 1D, x*(L=3) < x*(L=2) at both v=2 and v=3. **Standing caveat:** the L=3
   side stays at its published `n_gen=1024` on a 3.16x-spaced ladder, with per-cell bootstrap
   spread 1.2-3.5x over 3 seeds, so only an effect of roughly the 1D size (~2.8x) is resolvable
   here; a null will be reported as "no effect resolvable at this power", not as depth-stability.
   **First paired estimate** (`scripts/rhm_depth2d_analysis.py`, 7 of 8 L=2 seeds against the
   published 3-seed L=3 side): `x*(L=2)/x*(L=3)` = **1.07x [0.52, 2.34]** under A, **1.32x
   [0.66, 2.56]** under B. That covers both 1 and 1D's ~2.8x, so it decides nothing yet -- and the
   binding constraint is the **seed count**, not `n_gen`: per-cell bootstrap spread at L=3 is only
   1.11-1.17x on the well-measured cells.
7. **In flight: 51244750** (`scripts/rhm_depth2d_L3seeds.sbatch`, array 0-1, 1 H100 each, 10 h cap).
   Takes L=3 m=2 to **10 seeds at v=2** and 8 at v=3 at the published settings, so they pool with
   `L3peak_v*_m2_k7.json` instead of becoming a fourth population. Expected to shrink the interval
   by ~`sqrt(10/3)` = 1.8x, enough to separate no-effect from the 1D-sized one. v=3's ladder now
   runs to `P=6e6` rather than stopping at 3e6: `|R_L|` is 6.29e6 for two of its three published
   seeds, so the old ladder reached only `P/|R_L| = 0.48` and never got past the peak, which is why
   rs0 is unbracketed under both estimators. Enabled by a new `--rule_seed_start` (default 0,
   every old invocation unchanged), verified against values 51244089 produced independently.
   **Measurement fix folded in:** `L3k7_v*_m*.json` (n_gen 128) and `L3peak_v*_k7.json` (n_gen 1024)
   both cover `P = 1e4, 3e4, 1e5`, so `rhm_peak_analysis.curves` hands every L=3 curve three
   adjacent duplicate points at 8x different precision -- a zigzag right where a 3-point parabola
   is weakest. The new analysis pools them into one binomial.
   **Incidental, and it mattered:** `|R_L|` is **not** flat across rule seeds even at L=2 --
   v=2 m=2 is 64/64/64 for seeds 0-2 but 38/44/44/40 for seeds 3-6. Recomputing it per seed rather
   than assuming is load bearing, since x* is a ratio to it.

## Discipline notes
- Analysis runs in `srun -p sganguli -t 10 -c 2 --mem=4G`, never a login node; there `ml devel math
  python/3.12.1` gives `python3` only -- bare `python` is still 2.7.5.
- `--max_patches` does **not** bound memory (`run_cell` builds the table, then tests it, `:64`), so cap the P ladder. Footprint tracks patch count: 1.6 GB at 75k, >8G at 733k.
- `rhm_k_schedule.py` absolute numbers sit ~5% off `rhm_sampling_complexity.py` at the same cell (different train/sampler seed convention). Arms within one script are matched; across scripts they are not.
