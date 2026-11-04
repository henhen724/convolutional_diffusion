# STATUS

**RHM sampling complexity** is answered (due 12:00 PT Fri 11/6); **MLS** is parked
(LOG.md, `notebooks/mls_entropy_barrier_results.ipynb`).

## A. RHM sampling complexity -- the answer, 2026-11-03

`P` = training grids before the LS machine emits grids **both novel and fully grammatical**.

> **`P* = 0.21 |R_L| = Theta(v m^((n-1)/(s^2-1))) = exp(Theta(n))`**, `n = s^(2L)` sites.

Measured, not derived; the 11-02 `C(k) = f0^U` ceiling stays retracted. Write-up in
`notebooks/rhm_sampling_complexity_2026-11-03.ipynb` + 7-page deck, committed and sent.
1. **Collapse.** 30 bracketed peaks (L 2-3, v 2-8, m 2-6, 3 seeds): `|R_L|` spans 49 152x
   and `P*` 63 851x, `P*/|R_L|` only 11.9x (median 0.197, geo-sd 1.95x). Free fit
   `|R_L|^0.965 +- 0.041`. Rules drawn *with replacement*, so seeds differ 20x in `|R_L|`.
2. **Window bound.** Root needs `k >= 2 s^(L-1) - 1`. At L=3, k=5 the table saturates at
   exactly `N_5 = 92080` yet grammaticality never tops 0.7%; k=7 reaches 25-52%.
3. **Mechanism.** `P(novel|gram)` fits `exp(-0.725x - 0.868)`, `x = P/|R_L|`, 488 cells,
   vs `exp(-x)` for uniform: only 42% of grammatical outputs are novel as x -> 0.
   Median peak success is only 0.022 (0.001-0.342): the law locates the optimum, the
   optimum is not good.

## B. Straddling windows see two levels up, 2026-11-03

`j* = floor(log_s((k+1)/2))` is the deepest level a `k x k` window contains. Ensemble
`E_j` tiles the grid with independent uniform draws from `R_j` (coherent to level `j`,
scrambled above); `a_j` = fraction whose every window is in the saturated window set `W`;
`a_L = 1` is the saturation check. Job 50996974 (00:01:57), 160 cells, 146 saturated (the
14 that are not are all L=3 k=5 m=3). Deck `straddle_levels_slides.pdf`.

- **L=3, k=5** (`j* = 1`): **all 16 saturated grammars see two levels up.** Median
  `a_2 = 0.10` (v2m2 0.65, v3m2 0.10, v4m2 0.03): a 5x5 window containing only 2x2 blocks
  rejects ~90% of grids *already coherent through level 2* -- it polices the root.
  hshunt's conjecture, and not the "small signal" expected.
- **L=3, k=3**: 19/30 see two levels up, 4 one, 7 none. v2m2 `a_1` ranges 0.392-1.000 over
  5 seeds -- the same (v,m,k) reaches the root on one grammar, stops at the blocks on another.
- **Strength falls monotonically in m** (`1 - a_{L-1}`, L=2 k=3): 0.77, 0.58, 0.38, 0.30,
  0.11 for m=2..6; k=5 decays far slower (0.80 -> 0.58). More synonyms = more legal
  boundary pairs = fewer detectable scrambles.
- Validated on 75 L=2 cells against exact enumeration: `a_{j*}` matches
  `lv_pooled/lv_blocks` to <=1.2% rel, `|W|` reproduces the exact `N_k`, `a_L = 1.0000`
  everywhere, no symbol unreachable in any of the 150 grammars.
- Caveat: `a_j` is the *support* an infinite-data machine defines, not what a finite-`P`
  sampler emits (A.4 is lower).

## Running jobs

- None. 50935971, 50976107, 50996974 all COMPLETED; both decks are on OneDrive.

## Open problems

- L=4 needs k=15, out of table reach, so `exp(Theta(n))` rests on two depths plus the
  exact `N_k` counts. One L=3 curve is unbracketed (v3m2 rs0, `x* >= 0.477`).
- `k >= 7` is unreachable for `a_j`: a base-(v+1) code of `k*k` digits overflows int64.
- Pooled-vs-per-phase measured gap (3-4x) far exceeds their support gap (1.04-1.38x):
  pooling costs the *sampler*, not the support. Worth a page of its own.
- **Blocked on hshunt (MLS):** the corrected layer-2 posterior. Nothing MLS runs until it lands.

## Next steps

1. 1D RHM depth probe at L=4-5, as a cheap third point for `exp(Theta(n))`. 1D needs only
   `k >= 2 s^(L-1) - 1` in one dimension, so the int64 code survives to deeper `L`.
2. The pooling-costs-the-sampler page (open problem 3), which is a measurement already in
   hand and only needs writing up.
