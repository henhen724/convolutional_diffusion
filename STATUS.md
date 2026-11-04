# STATUS

**RHM sampling complexity** is answered (due 12:00 PT Fri 11/6); **MLS** is parked
(LOG.md, `notebooks/mls_entropy_barrier_results.ipynb`).

## A. RHM sampling complexity -- the answer

`P` = training samples before the LS machine emits output **both novel and fully grammatical**.

> **`P* = exp(Theta(n))`, `n` = sites. In 2D specifically, `P* = 0.21 |R_L|`.**

Measured, not derived; the 11-02 `C(k) = f0^U` ceiling stays retracted. Deck + notebook
`rhm_sampling_complexity_2026-11-03.ipynb` sent. Full detail in LOG 11-03.
1. **2D collapse.** 30 bracketed peaks: `|R_L|` spans 49 152x, `P*` 63 851x, the ratio only
   11.9x (median 0.197). Free fit `|R_L|^0.965 +- 0.041`. Rules drawn *with replacement*.
2. **Window bound.** Root needs `k >= 2s^(L-1)-1`. At L=3, k=5 the table saturates at exactly
   `N_5 = 92080` yet grammaticality never tops 0.7%; k=7 reaches 25-52%.
3. **Mechanism.** `P(novel|gram) ~ exp(-0.725x - 0.868)` vs `exp(-x)` for uniform. Median peak
   success only 0.022 -- the law locates the optimum; the optimum is not good.
4. **Scope (from C):** the `0.21` is a 2D constant. Only `exp(Theta(n))` is general.

## B. Straddling windows see two levels up

Ensemble `E_j` tiles the grid with independent draws from `R_j`; `a_j` = fraction whose every
window lies in the saturated set `W`. **At L=3, k=5 (`j*=1`) all 16 saturated grammars see two
levels up**, median `a_2 = 0.10` -- hshunt's conjecture, far bigger than the expected "small
signal". Strength falls monotonically in m (0.77 -> 0.11, m=2..6). Validated against exact
enumeration on 75 L=2 cells (<=1.2% rel err). Deck sent; detail in LOG 11-03.
Caveat: `a_j` is the support an infinite-data machine defines, not finite-`P` output.

## C. 1D depth probe -- the 0.21 is 2D-specific

In 1D the root-reaching window `k = d-1` is 15 *sites* at L=4, so n = 4, 8, 16 are all reachable.
113 cells x 19 P, 110 bracketed; same machine semantics as 2D. Notebook + 8-page deck sent.

- **`exp(Theta(n))` confirmed at a third depth**: `ln P*` grows 0.374/site over n=4..16.
- **But `P* = Theta(|R_L|)` fails in 1D**: `b = 0.689 +- 0.045` robust (ladder-floor cells and
  n=4 dropped), 6.8 sd below 1. 2D was `0.965 +- 0.041`.
- **Matched-n control (decisive).** 1D L=4 and 2D L=2 both have n=16: `x*` = 0.050 vs 0.193,
  **3.9x apart**, while 2D is flat in n (1.08x over a 4x change). So not an n effect.
- Hypothesis, untested: at n=16 a U=1 window covers 94% of a 1D string but 56% of a 2D grid,
  so the 1D machine is near whole-string lookup. Fits its peak success 0.49 vs 2D's 0.022.
- **Retracted:** a joint `x* ~ n^-0.154` fit across dimensions -- it averages a flat 2D series
  against a falling 1D one.

## Running jobs

- None. 50935971, 50976107, 50996974 COMPLETED. 51009204 OOM'd *after* finishing all 113 cells,
  inside `exact_RL` on cells it was about to skip; data intact, cap patched.

## Open problems

- 2D L=4 needs k=15, out of table reach; 2D itself still rests on two depths (1D gives three).
  One L=3 curve unbracketed (v3m2 rs0, `x* >= 0.477`).
- `k >= 7` unreachable for `a_j`: a base-(v+1) code of `k*k` digits overflows int64.
- Pooled-vs-per-phase gap (3-4x) far exceeds their support gap (1.04-1.38x): pooling costs the
  *sampler*, not the support. Worth a page.
- **Blocked on hshunt (MLS):** the corrected layer-2 posterior. Nothing MLS runs until it lands.

## Next steps

1. **Test C's coverage hypothesis** -- the one open "why". Sweep `k` *below* U=1 in 2D and see
   whether `x*` falls as coverage `k^2/n` rises. If it does, 1D and 2D are one law in coverage
   rather than two in dimension, and 0.21 becomes a coverage-specific constant.
2. The pooling-costs-the-sampler page, measurement already in hand.

## Discipline notes

- Analysis runs in an allocation (`srun -p sganguli -t 15 -c 2 --mem=16G ...`), never on a login
  node -- a numpy sweep over `results/` was SIGKILLed there on 11-03.
- Measurement footprint is flat in depth (host RSS 0.99/1.02/1.06 GB at L=2/3/4). Memory risk
  lives in `exact_RL`, which scales with nominal `v*m^M`, not with the measurement.
