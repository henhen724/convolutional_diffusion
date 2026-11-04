# STATUS

**RHM sampling complexity** is answered (due 12:00 PT Fri 11/6); **MLS** is parked
(LOG.md, `notebooks/mls_entropy_barrier_results.ipynb`).

## A. RHM sampling complexity -- the answer, 2026-11-03

How many training grids `P` before the LS machine emits grids that are **both novel and
fully grammatical**, and how does `P*` scale?

> **`P* = 0.21 |R_L| = Theta(v m^((n-1)/(s^2-1))) = exp(Theta(n))`**, `n = s^(2L)` sites.

Measured, not derived; the 11-02 `C(k) = f0^U` ceiling stays retracted. Full write-up in
`notebooks/rhm_sampling_complexity_2026-11-03.ipynb` and the 7-page deck (both committed).

1. **Collapse.** 30 bracketed peaks, L in {2,3}, v in {2,3,4,6,8}, m in {2..6}, 3 seeds.
   `|R_L|` spans 49 152x and `P*` 63 851x, but `P*/|R_L|` only 11.9x (median 0.197,
   geo-sd 1.95x). Free fit `|R_L|^0.965 +- 0.041`; `b=1` fits no worse. Rules are drawn
   *with replacement*, so same-(v,m) seeds differ 20x in `|R_L|` -- hence a collapse.
2. **Window bound.** Root needs `k >= 2 s^(L-1) - 1`. At L=3, k=5 the table saturates at
   exactly `N_5 = 92080` yet grammaticality never tops 0.7%; k=7 reaches 25-52%.
3. **Mechanism.** `P(novel|gram)` fits `exp(-0.725x - 0.868)` in `x = P/|R_L|` over 488
   cells vs `exp(-x)` for uniform: only 42% of grammatical outputs are novel as x -> 0.
4. **Peak height, honestly.** Median peak success 0.022 (0.001-0.342): the law locates
   the optimum, the optimum is not good.
5. **Straddling (hshunt's question) -- confirmed, see B.**

## B. Straddling windows see two levels up, 2026-11-03

`j* = floor(log_s((k+1)/2))` is the deepest level a `k x k` window contains. Ensemble
`E_j` tiles the grid with independent uniform draws from `R_j` (coherent to level `j`,
scrambled above); `a_j` = fraction whose every window is in the saturated window set `W`.
`a_L = 1` is the saturation check and holds exactly.

- **L=3, v2 m2, k=5** (`j* = 1`): `a_1 = 0.003`, **`a_2 = 0.57`**, `a_3 = 1.000`. A 5x5
  window, which only contains 2x2 blocks, rejects 43% of grids already coherent through
  level 2 -- it polices the *root*, two levels above `j*`. Exactly hshunt's conjecture.
  `|W| = 92080` reproduces the exact `N_5`, so `W` is provably complete.
- **L=3, v2 m2, k=3**: `a_1 = a_2 = 1.000`. No straddle power at L=3 though plenty at
  L=2 -- the signal dies as the grammar above `j*` gets richer.
- Validated against the exact L=2 enumeration (`rhm_straddle.py`): Monte-Carlo `a_{j*}`
  matches `lv_pooled/lv_blocks` to 0.01-0.35%. No symbol is unreachable in any of the
  150 swept grammars, so `E_j` blocks all really occur (the script now enforces this).

## Running jobs

- **50996974** `rhm_straddle_levels.sbatch` -- `a_j` profiles, L=2 (v2,v3, m2-6) with the
  exact cross-check and L=3 (v2-4, m2-3), k in {3,5}, 5 seeds. ~2-4 h.
- 50935971, 50976107 COMPLETED. Deck is on OneDrive.

## Open problems

- L=4 would need k=15; the table is out of reach, so `exp(Theta(n))` rests on two depths
  plus the exact `N_k = Theta(|R_L|)` counts. A cheaper probe would be a 1D RHM at L=4-5.
- One L=3 curve is unbracketed (v3m2 rs0, still climbing at P=3e6, `x* >= 0.477`).
- `k >= 7` is unreachable for `a_j`: a base-(v+1) code of `k*k` digits overflows int64.
- Measured grammaticality sits well below `R_L/lv_pooled` and the pooled-vs-per-phase gap
  (3-4x) is far larger than their support gap (1.04-1.38x), so pooling costs the *sampler*,
  not the support. Worth a page of its own.
- **Blocked on hshunt (MLS):** the corrected layer-2 posterior. Nothing MLS runs until it lands.

## Next steps

1. When 50996974 lands: notebook + 2 deck pages on the `a_j` profile vs (v, m, k, L).
2. Then the 1D RHM depth probe at L=4-5, as a cheap third point for `exp(Theta(n))`.
