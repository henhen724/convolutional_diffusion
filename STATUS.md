# STATUS

**RHM sampling complexity** is answered (due 12:00 PT Fri 11/6); **MLS** is parked
(LOG.md, `notebooks/mls_entropy_barrier_results.ipynb`).

## A. RHM sampling complexity -- the answer, 2026-11-03

`P` = training grids before the LS machine emits grids **both novel and fully grammatical**.

> **`P* = exp(Theta(n))`, `n = s^(2L)` sites. In 2D, `P* = 0.21 |R_L|`.**

Measured, not derived; the 11-02 `C(k) = f0^U` ceiling stays retracted. Write-up in
`notebooks/rhm_sampling_complexity_2026-11-03.ipynb` + 7-page deck, committed and sent.
1. **Collapse (2D).** 30 bracketed peaks (L 2-3, v 2-8, m 2-6, 3 seeds): `|R_L|` spans
   49 152x and `P*` 63 851x, `P*/|R_L|` only 11.9x (median 0.197, geo-sd 1.95x). Free fit
   `|R_L|^0.965 +- 0.041`. Rules drawn *with replacement*, so seeds differ 20x in `|R_L|`.
2. **Window bound.** Root needs `k >= 2 s^(L-1) - 1`. At L=3, k=5 the table saturates at
   exactly `N_5 = 92080` yet grammaticality never tops 0.7%; k=7 reaches 25-52%.
3. **Mechanism.** `P(novel|gram)` fits `exp(-0.725x - 0.868)`, `x = P/|R_L|`, 488 cells,
   vs `exp(-x)` for uniform. Median peak success only 0.022: the law locates the optimum,
   the optimum is not good.
4. **Scope, after C:** `0.21` is a *2D* constant, not universal. `exp(Theta(n))` is.

## B. Straddling windows see two levels up, 2026-11-03

Ensemble `E_j` tiles the grid with independent draws from `R_j`; `a_j` = fraction whose
every window lies in the saturated window set `W`; `a_L = 1` checks saturation.
Job 50996974, 160 cells, 146 saturated. Deck sent. **At L=3, k=5 (`j*=1`) all 16 saturated
grammars see two levels up**, median `a_2 = 0.10` -- hshunt's conjecture, and far bigger
than the "small signal" expected. Strength falls monotonically in m (0.77 -> 0.11, m=2..6).
Validated against exact enumeration on 75 L=2 cells (<=1.2% rel err; `|W|` reproduces `N_k`).
Caveat: `a_j` is the support an infinite-data machine defines, not finite-`P` output.

## C. 1D depth probe -- the 0.21 is 2D-specific, 2026-11-03

Third depth for `exp(Theta(n))`: in 1D the root-reaching window `k = 2s^(L-1)-1 = d-1` is
15 *sites* at L=4, so n = 4, 8, 16 are all reachable. `scripts/rhm_1d_depth.py`,
`rhm_1d_analysis.py`; 113 cells x 19 P, 110 bracketed. Same machine semantics as 2D.

- **`exp(Theta(n))` confirmed at a third depth**: `ln P*` grows 0.374/site over n=4..16.
- **But `P* = Theta(|R_L|)` does not hold in 1D**: `b = 0.708 +- 0.028` pooled, and
  `0.689 +- 0.045` after dropping ladder-floor cells (`P* < 8`) and n=4 -- 6.8 sd below 1.
  2D was `0.965 +- 0.041`.
- **Matched-n control (the decisive one).** 1D L=4 and 2D L=2 both have n=16:
  `x* = 0.050` vs `0.193`, **3.9x apart**. 2D is flat in n (0.193 at n=16, 0.207 at n=64,
  1.08x over 4x in n), so the 1D fall is *not* an n effect -- it is dimension.
- Likely mechanism, not yet tested: at n=16 one U=1 window covers 15/16 = 94% of a 1D
  string but only 9/16 = 56% of a 2D grid, so the 1D machine is near whole-string lookup.
  Consistent with its median peak success 0.49 vs 2D's 0.022.
- **Retracted in passing:** a joint `x* ~ n^-0.154` fit across both dimensions. It averages
  a flat 2D series with a falling 1D one and means nothing.

## Running jobs

- None. 50935971, 50976107, 50996974 COMPLETED. 51009204 OOM'd *after* finishing all 113
  cells, inside `exact_RL` on cells it was about to skip; data intact, cap patched.

## Open problems

- 2D L=4 needs k=15, out of table reach; `exp(Theta(n))` in 2D still rests on two depths
  (1D now supplies three). One L=3 curve unbracketed (v3m2 rs0, `x* >= 0.477`).
- `k >= 7` unreachable for `a_j`: a base-(v+1) code of `k*k` digits overflows int64.
- Pooled-vs-per-phase measured gap (3-4x) far exceeds their support gap (1.04-1.38x):
  pooling costs the *sampler*, not the support. Worth a page of its own.
- **Blocked on hshunt (MLS):** the corrected layer-2 posterior. Nothing MLS runs until it lands.

## Next steps

1. **Test C's coverage hypothesis** -- the one open "why". Sweep `k` *below* U=1 in 2D and
   see whether `x*` falls as coverage `k^2/n` rises. If it does, 1D and 2D are one law in
   coverage, not two in dimension, and the 0.21 becomes a coverage-specific constant.
2. The pooling-costs-the-sampler page (open problem 3), measurement already in hand.

Done 11-03: C's notebook (`notebooks/rhm_1d_depth_2026-11-03.ipynb`) and 8-page deck, both
committed (`59b8491`) and on OneDrive.

## Discipline notes

- Analysis runs in an allocation (`srun -p sganguli -t 15 -c 2 --mem=16G ...`), never on
  the login node -- a numpy sweep over `results/` was SIGKILLed there on 11-03.
- Measurement footprint is flat in depth (host RSS 0.99/1.02/1.06 GB at L=2/3/4; GPU peak
  0.11/0.33/1.17 GB). Memory risk lives in `exact_RL`, which scales with nominal `v*m^M`.
