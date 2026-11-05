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
`rhm_k_schedule.py`: four matched arms, control schedule derived from the adaptive arm's *own* trace
so it cannot be rigged. **9 cells, `rhm_ksched_score.py`: adaptive beats the bare schedule 9/9 and
the best fixed k 9/9** (median 8.2x at v2, ~65x at v3/v4). The schedule recovers a median **15%** of
adaptive success while spending the *same window budget* -- median mean-k gap 0.08. Capping the
criterion to its own schedule is free at v2 but costs 8-30% at v3/v4 (within 20% in 5/9).
> **Time alone cannot set the patch size. What matters is which *site* gets which size.**
`rhm_k_locality.py` says what the site signal is *not*: chosen k regressed on the site's offset within its level-1 block gives **R^2 = 0.000 in 18/18 runs**, adding nothing jointly with step either. Step alone explains 0.09-0.21 (v2), 0.22-0.36 (v3), 0.30-0.41 (v4), rising with `v`, falling with `P`.
The local **revealed-neighbour count** (how many of the 9x9 around a site are fixed) beats step in
18/18 and subsumes it -- adding step on top gains only .005-.013 -- so the schedule is a *consequence*
of that count. But it caps at .47: the rest turns on *which* tokens were revealed, not how many.
> **Position 0%, time <=41%, local reveal count <=47%, remainder >=53% genuinely content-dependent.**
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
1. C+D deck + notebook **delivered** 2026-11-04 (`build_s3_slides.py`, `rhm_patch_size_2026-11-04.ipynb`, uploaded); re-render both on 51143385's now-completed ladder.
2. **Done (51175489), 18/18 cells**: `n_words` (level-1 words still compatible with a site's own block) adds **median +0.107 R^2 over `nrev`**, clearing the registered .05 in **18/18**; block-fill control <= .006 everywhere. So the >=53% residual *is* grammatical ambiguity at the block scale, and it is not fill level wearing a grammar label.
   But the registered "gain grows with `v`" is **refuted**: it falls monotonically, .125 / .102 / .096 at v=2/3/4 (6 cells each). The stated mechanism was already refuted by the smoke cell (mean k *rises* with `n_words`, confounded with fill), which is why the gain is scored over `nrev`.
3. Next on the residual: take it **across block boundaries**, where `n_words` by construction says nothing -- that is also where hshunt's straddling-consistency question lives.

## Discipline notes
- Analysis runs in `srun -p sganguli -t 10 -c 2 --mem=4G`, never a login node; there `ml devel math
  python/3.12.1` gives `python3` only -- bare `python` is still 2.7.5.
- `--max_patches` does **not** bound memory (`run_cell` builds the table, then tests it, `:64`), so cap the P ladder. Footprint tracks patch count: 1.6 GB at 75k, >8G at 733k.
- `rhm_k_schedule.py` absolute numbers sit ~5% off `rhm_sampling_complexity.py` at the same cell (different train/sampler seed convention). Arms within one script are matched; across scripts they are not.
