# STATUS

**RHM sampling complexity** is active; **MLS** is parked (LOG.md, `notebooks/mls_entropy_barrier_results.ipynb`).

## A. RHM sampling complexity -- ANSWERED (2026-11-02)

How many training grids `P` before the LS machine emits grids that are **both novel and
fully grammatical**, and how does `P*` scale with `(v, s, L, m, k)`?

**There is no finite `P*`:** grammaticality and novelty come from the same mechanism --
copying a training grid -- so they never hold together. Where grammaticality is `Theta(1)`,

    P*  =  v m^(Theta(n / (s^2 - 1)))  =  exp(Theta(n)),   n = s^(2L) sites

and there novelty is identically 0. At fixed `k`, `ln(1/success) = Theta(n / k^2)`. The LS
machine gets **no** benefit from the hierarchy: same cost as memorizing the distribution.

Two legs. **(1) Ceiling, derived.** A `k x k` window reaches `k//2`, so it polices aligned
blocks only up to side `s^j*`, `j* = floor(log_s((k+1)/2))`. With unlimited data it samples
the locally-valid set, of which the grammar is `C(k) = |R_L| / |R_j*|^(s^(2(L-j*)))`,
collapsing to `f0^U` with `f0 = m/v^(s^2-1)`, `U = (s^(2(L-j*))-1)/(s^2-1)`.
`rhm_ceiling_exact.py` checks it by enumeration -- over 150 enumerable L=2 cases it equals
the closed form in exactly the 70 with no word collision and differs in exactly the 80 with
one -- and by MC (0.047195 vs exact 0.046875). **(2) Cost, measured.** Filling the table is
coupon-collecting `N_k` distinct windows: `N_3` is polynomial (`~v^4.4 m^2.8` at L=3;
13 602 at v=4,m=3,L=3, +2%/decade), `N_5`/`N_7` nowhere near (5.3e6, 1.6e6, +5.4x/+8.9x).

**Measured** (jobs 50859790 / 50861030; `succ = gram + novel - 1` identically):
- L=3 k=3: `gram=0` at every `P` 30..1e5 **but one** (1 grid in 1024 at P=3e4, the
  resolution floor, gone again at 1e5) while the patch table runs out (11 890, +7%/decade).
  2x2 blocks 0.578->0.995, 4x4 stuck at 0.125: it learns the level it reaches and stalls
  forever on the one above. k=5 is *worse* (same `j*`, bigger table); k=7 the only nonzero
  gram short of the full window (0.021, *falling* with P); k=15=2d-1 gives gram 0.82-0.87,
  novel 0.13-0.18, succ 0.000-0.002.
- Exponents at L=2 k=3: `succ_max` tracks the exact ceiling (ratios 0.75-1.43 over 10
  (v,m)); slope in v **-3.19** (predicted -3), in m +1.57 (predicted +1). Depth law with
  `f0` fixed: succ_max 0.432 (U=1) -> 0.0172 (U=5), predicted 0.375 -> 0.0074. The ceiling
  depends on `U` alone, not on `L` and `k` separately: L=4 k=7 and L=3 k=3 both give U=5
  and MC-agree at two of three v=2 seeds (0.0073 vs 0.0078, 0.0032 vs 0.0031; closed
  0.00742) -- job 50883925, which also fixed the `j_max` OOM that lost L=4 in 50861297.
- `P*` at L=2: peaks succ=0.052 at P*=500 (m=3, |R_L|=972) and 0.118 at P*=3000 (m=5,
  |R_L|=12500), then collapses as novelty drains (0.91->0.82). At k=7=2d-1, succ=1.6e-4
  at P=10 (1 grid in 6144) and exactly 0 at the other 16 P.
- Geometry: 2x2 parse 0.995 phase-aware vs 0.348 pooled vs 0.241 pooled+zero-pad (repo's `LocalDiscreteScoreMachine`); use `block_align=s`, `pad_mode=outside`.

## Running jobs

- None. 50882105 / 50882792 / 50884796 (`rhm_report.sbatch`) and 50883925
  (`rhm_ceiling_L4.sbatch`) all COMPLETED; deck and notebook are current on disk.

## Open problems

- Measured ceiling sits a bounded ~1.2-2.7x above the independent-blocks prediction
  (overlapping windows carry cross-block correlation). Renormalizes `f0`, not `U`, so the asymptotics stand, but unquantified.
- **Blocked on hshunt (MLS):** the corrected layer-2 posterior. Nothing MLS runs until it lands.

## Next steps

1. **The experiment this points at:** a hierarchy of 3x3 windows (a second machine on
   inferred level-1 symbols) polices scale `s^2` for two tables of size `|R_1|`: polynomial, not exponential. RHM-side MLS, and falsifiable.
