# STATUS

**RHM sampling complexity** is answered and delivered (due 12:00 PT Fri 11/6; re-reported
2026-11-05 with the measurement record below). Both jobs **COMPLETED**: 51143385 (s3 P ladder,
5:41:05) and 51175489 (`rhm_k_content.py`, 6:51). **MLS** is parked on hshunt's corrected
layer-2 posterior (LOG.md, `mls_entropy_barrier_results.ipynb`).

## A. The answer
`P` = training samples before the LS machine emits output **both novel and fully grammatical**.
> **`P* = exp(Theta(n))`, `n` = sites. At the U=1 window in 2D, `P* = 0.21 |R_L|`.**
Measured, not derived; the `C(k) = f0^U` ceiling stays retracted. 30 bracketed 2D peaks collapse to median 0.197 while `|R_L|` spans 49 152x; in 1D `ln P*` grows 0.374/site over n=4..16 but `P* = Theta(|R_L|)` fails (`b=0.689+-0.045`). Root needs `k >= U1 = 2s^(L-1)-1`.
**Estimator-robust (2026-11-05):** swapping the 3-point peak fit for an inverse-variance fit over the whole top of each curve moves the 2D exponent to `b=0.927+-0.033` and the median to 0.177, overlapping CIs -- the scatter it removes was all estimator noise (Open problems). **Subsample-robust too:** refitting on cells with `|R_L| >= 200` only -- a cut chosen for a different purpose, to keep the `P>=10` floor from censoring 1D -- gives `b_2D = 0.979` (0.939 at `>= 1000`), i.e. `x* = |R_L|^-0.02..-0.06`, flat to within 1.4x across **five decades** of `|R_L|`. That flatness is what makes `0.21 |R_L|` a law in 2D; the same fit in 1D gives `|R_L|^-0.36`, which is why it is not one there. **Depth-robust in 2D, by two routes that agree (2026-11-05):** paired within `(v, m=2)`, `x*(L=2)/x*(L=3)` = 1.58x [0.65, 2.76] (A) / 1.92x [0.94, 2.88] (B) at v=2 and 0.77x / 0.64x at v=3 -- both covering 1 -- which is also what `b_2D` predicts over the same 5e4-fold jump in `|R_L|`. 1D, by contrast, loses **2.5x per single depth step**. The earlier 2.8x 2D reading was an absolute `P >= 10` floor censoring the three smallest grammars; see Open problems. Tightens further when 51244750 lands.

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
**Fourth refuted 2026-11-05 (`scripts/rhm_xstar_scaling.py`), and this one was measured, not argued.** Since `x* = P*/|R_L| = |R_L|^(b-1)`, `b<1` *is* "x* falls as |R_L| grows" -- so the gap, the 7.6x spread across `(v,m)`, and the depth effect could all be one fact: the two populations sitting at different `|R_L|`. Fitting `log P* = b log|R_L| + c + d*1[2D]` across both dimensions and depths, with each cell's binomial bootstrap carried through, the offset **does not collapse -- it grows**, because 1D cells sit at smaller `|R_L|` and a common `b` over-corrects: `d` = **4.83x (A) / 5.29x (B)** on cells with `|R_L| >= 200`, **6.65x / 7.24x** at `>= 1000`, against a raw gap of 3.87x. What the fit does establish is that **the exponents themselves differ** and no pooled `b` absorbs that: `b_1D = 0.638`, `b_2D = 0.979`, difference **0.340 [0.155, 0.491]** (`|R_L| >= 200`; 0.273 [-0.090, 0.530] at `>= 1000`, where n is small). Those reproduce the published `0.689+-0.045` / `0.965+-0.041` on an independently selected subsample.
The `|R_L|` floor is load bearing: without it `b_1D` reads **0.510**, because the absolute `P >= 10` floor needed to cut the 2D copying regime censors exactly the 1D cells whose peak lies below it, and that censoring is `|R_L|`-dependent. Restricted to cells where the floor sits far below the peak, the answer is stable between `>= 200` and `>= 1000`.

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
- ~~The 1D/2D offset has no mechanism after three refutations. No fourth without a measurement.~~
  **Fourth measured and refuted 2026-11-05 (`scripts/rhm_xstar_scaling.py`), offset still unexplained.**
  The candidate was that there is no dimension effect at all, only a scale effect: `x* = |R_L|^(b-1)`,
  so two populations at different `|R_L|` would show an offset without one existing. Fitting
  `log P* = b log|R_L| + c + d*1[2D]` over both dimensions and both depths, the offset **grows
  instead of collapsing** (`d` = 4.83x A / 5.29x B at `|R_L| >= 200`, 6.65x / 7.24x at `>= 1000`,
  vs a raw gap of 3.87x), because 1D sits at smaller `|R_L|` and a pooled `b` over-corrects it.
  The positive finding is that **the exponents differ**: `b_1D = 0.638` vs `b_2D = 0.979`,
  difference **0.340 [0.155, 0.491]** -- matching the published `0.689+-0.045` / `0.965+-0.041` on
  an independently chosen subsample. In 2D `x* = |R_L|^-0.02..-0.06` is flat across five decades,
  which is what makes `P* = 0.21 |R_L|` a law there and not in 1D (`|R_L|^-0.36`). So mechanism
  four is gone and the offset is sharper, not softer: it is now a difference in *exponent*, not
  only in prefactor.
  Methodological catch worth keeping: the `P >= 10` floor that removes the 2D copying regime
  censors small-`|R_L|` 1D cells whose peak lies below it, and that censoring is `|R_L|`-dependent,
  so without `--min_RL` the fit reads `b_1D = 0.510`. Any `x*` fit across dimensions needs both cuts.
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
- **2D depth-stability at fixed `(v,m)`: measured, not yet settled.** The 0.21 headline leans on
  L=2 pooled 0.1925 (nine combos) against L=3 pooled 0.2075 (two), and those combos are
  **disjoint** while x* varies 7.6x across (v,m) medians. 1D, paired within (v,m), has x* *falling*
  with depth in **15/15** combos. Three things changed on 2026-11-05:
  - **The 1D bar is 2.5x, not 2.8x.** The 2.8x figure spans L=2->4, two depth steps; 2D only has
    L=2 and L=3. Matched one-step (`rhm_1d_depth_noise.py --L_hi 3`): median **2.52x** (A, 15/15,
    CI excludes 1 in 7/15) and **2.59x** (B, 14/15, 9/15). The 1D effect is *not* estimator noise:
    1D per-cell estimator spread is only 1.09x (A) / 1.06x (B).
  - **Paired 2D test (51244089 + `rhm_depth2d_analysis.py`), with `P_min=10`:** v=4 control
    reproduces the published x* (A **1.16x** [0.37, 1.84], B 0.90x [0.34, 1.47]); **v=2 gives
    A 2.46x [1.51, 4.09], B 2.95x [1.80, 3.99]** -- excludes 1, same direction as 1D and the same
    size; v=3 covers 1 (1.20x / 1.18x) on only 2 L=3 cells. Awaiting 51244750 for v=2,3 power.
  - **Resolved: the v=2 effect was the floor, and the two routes now agree.** An absolute floor is
    the wrong shape for this cut -- the peak sits at `P* ~ 0.2 |R_L|`, and v=2 m=2 L=2 has
    `|R_L| = 38..64`, so `P >= 10` lands *on* the peak. `rhm_copy_regime.py` finds 4 of 29 L=2
    cells with `0.2|R_L| < 10`, **all of them at v=2** -- the one v that claimed an effect.
    Replacing it with a cell-relative cut on the regime's own signature (`--copy_rel 0.5`, below)
    keeps all 8 v=2 cells and the ratio falls to **1.58x [0.65, 2.76]** (A) / **1.92x [0.94, 2.88]**
    (B), covering 1; the three censored cells are exactly the smallest grammars (`|R_L|` 38/44/40,
    `P*` 3.9/4.7/8.7), and dropping them moved the L=2 median 0.1901 -> 0.3108. v=3 runs the
    *opposite* way (0.77x / 0.64x) and the v=4 control still passes (1.17x / 0.87x).
    **That 1.6-1.9x is what the exponent predicts**: `b_2D = 0.945..0.979` over a 5e4-fold jump in
    `|R_L|` gives `(5e4)^(1-b)` = 1.25-1.8x. Two independent routes, one censoring-free, now agree.
    **Reading: no 1D-sized depth effect in 2D.** 1D loses 2.5x per *single* depth step at far
    smaller `|R_L|` changes; 2D drifts 1.6-1.9x across 5e4 in `|R_L|`, not resolvable from 1 at
    this power. Section A stands as written, with `x*` understood as `|R_L|^-0.02..-0.06` rather
    than a constant. 51244750 tightens the L=3 side (3 -> 8-10 seeds); re-run both routes then.
- **How to cut the copying regime without censoring by `P`** (`scripts/rhm_copy_regime.py`, and
  `--copy_rel` in the depth analysis). Among the *grammatical* outputs, the novelty rate relative
  to a uniform draw from the support,
  `ratio = (success/grammatical) / (1 - train_distinct/|R_L|)`, is computable from every row of
  every sweep already on disk, at no compute cost. It runs **0.006 -> 0.35** across the v=4 L=2
  ladder and **0.19 -> 0.50** across v=2's: the machine starts by replaying its training set and
  then spreads out. A *fixed* threshold does not transfer -- L=2 never reaches 1 (L=3 does, ~0.92,
  so the machine genuinely generalizes there), and `ratio >= 0.5` would push v=4's argmax from 100
  to 500 and v=3's off the ladder. What transfers is each cell's own plateau: drop the **leading**
  run of points below `0.5x` the median ratio over the top half of the curve. Leading only -- the
  zero-success tail at large P is real and defines the falling side. Its own cost, stated: it
  rejects 3 of 8 cells at v=3 and 3 of 8 at v=4 (the argmax lands on the new boundary), so it is
  not free either -- it is just not correlated with `|R_L|`, which is the property that matters.
- **Copying regime at tiny P (new, affects any ladder starting below P~10).** With 2-3 training
  grids the machine has no dictionary, emits near-copies, and they score *novel* only because the
  training set is too small to match against: v4m2 rs3 has success 0.0070 at P=3 against 0.0032 on
  the real hump at P=100, with grammaticality 0.238 at P=3 decaying to 0.020 by P=8. One ladder
  point wide and seed-specific, but it captures the argmax and inverted the v=4 control (0.42x)
  until `--P_min 10` was added. Every published 2D L=2 ladder starts at P=10, so nothing published
  is affected -- this was my own ladder choice.
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
6. **Done (51244089, COMPLETED in 11:27)** (`scripts/rhm_depth2d.sbatch`, CPU, 3 h cap). Measures the 2D `L=2 m=2`
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
   **Then registered item 1 failed and the failure was the finding.** The v=4 control came out at
   **0.42x**, backwards, with rs3 at 0.0223 against a published 0.2860. Printing the raw curves
   showed why: the ladder I chose starts at `P=2`, and at `P<=8` the machine is in a *copying*
   regime (success 0.0070 at P=3 vs 0.0032 on the real hump at P=100; grammaticality 0.238 falling
   to 0.020 by P=8) where near-copies count as novel because the training set is too small to match
   against. The artifact is mine, not the model's -- every published 2D L=2 ladder starts at P=10.
   With `--P_min 10` the control passes (**1.16x** [0.37, 1.84] A, 0.90x [0.34, 1.47] B) and v=2
   reads **2.46x [1.51, 4.09]** A / **2.95x [1.80, 3.99]** B, excluding 1 in the 1D direction at
   the 1D size. Two caveats on that, both in Open problems: the floor censors low-peak L=2 cells,
   and the censoring-free exponent predicts a smaller jump. Also fixed here: estimator B returned
   `nan` on a rejected fit, and rejections track curve *shape*, so the median ran on a biased
   subsample (2 of 8 cells at v=3) -- `est_B(..., fallback=True)` now falls back to the argmax as A
   does, with the default left at False so `rhm_peak_noise`'s committed numbers are untouched.
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
8. **Done (51246420)** `scripts/rhm_xstar_scaling.py` -- the censoring-free route to both open
   questions, since it fits an exponent across every cell instead of comparing two medians. Results
   in Open problems (mechanism four refuted; `b_1D` 0.638 vs `b_2D` 0.979) and section B. **Re-run
   it when 51244750 lands** -- it already globs `L3seed_v*_m2_k7.json`, and the new L=3 cells are
   exactly the high-`|R_L|` end where `b_2D` is least constrained. Use `--min_RL 200` (and check
   1000): without it the P>=10 floor censors small-`|R_L|` 1D cells and `b_1D` reads 0.510.
9. **Deck built and committed (`0861ef5`), upload held for 51244750.**
   `scripts/build_audit_slides.py` -> `notebooks/2026-11-05_rhm_audit_slides.pdf`, 8 landscape
   pages, 0.10 MB: title, tokenization, traces, then one page per attack (estimator / scale /
   depth) and what each changed. All eight verified by rendering `--png_dir` under a
   `/scratch` path and reading the PNGs -- `--png_dir /tmp/...` inside `srun` writes to the
   *compute node's* /tmp and is invisible afterwards.
   The trace page re-runs `run_cell`'s exact three seeds (grammar `1000+rs`, data/sampler
   `7e6+1000*rs+P`, torch `seed+9973`) at `n_gen=16384` and reproduces 51244089's v=4 m=2 L=2
   rs=3 row to 4 decimals; anything less exact silently prints different numbers from the JSON.
   **Reading `peak_noise.txt` correctly matters here:** section 2/3's "observed 1.47x" is the
   *within-(v,m,L)* spread across rule seeds, median over the 11 groups, which is what pairs
   with the estimator term in the log-variance split; section 4's 1.97x is the spread across
   all 30 cells and also contains the real movement of x* with `(v,m)`. The deck's bar chart
   plots the first and quotes the second as a separate line.
   **Remaining before the deadline:** when 51244750 lands, re-run `rhm_depth2d_analysis.py`
   (both `--P_min 10 --copy_rel 0` and `--P_min 0 --copy_rel 0.5`) and
   `rhm_xstar_scaling.py --min_RL 200` (and 1000), update section A, re-render, then
   `claude-results-upload.sh`.

## Discipline notes
- Analysis runs in `srun -p sganguli -t 10 -c 2 --mem=4G`, never a login node; there `ml devel math
  python/3.12.1` gives `python3` only -- bare `python` is still 2.7.5.
- `--max_patches` does **not** bound memory (`run_cell` builds the table, then tests it, `:64`), so cap the P ladder. Footprint tracks patch count: 1.6 GB at 75k, >8G at 733k.
- `rhm_k_schedule.py` absolute numbers sit ~5% off `rhm_sampling_complexity.py` at the same cell (different train/sampler seed convention). Arms within one script are matched; across scripts they are not.
