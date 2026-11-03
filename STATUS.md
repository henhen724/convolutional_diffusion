# STATUS

**RHM sampling complexity** is active; **MLS** is parked (LOG.md, `notebooks/mls_entropy_barrier_results.ipynb`).

## A. RHM sampling complexity -- 2026-11-02 ANSWER RETRACTED (2026-11-03)

How many training grids `P` before the LS machine emits grids that are **both novel and
fully grammatical**, and how does `P*` scale with `(v, s, L, m, k)`?

The 2026-11-02 answer -- no finite `P*`, `P* = v m^(Theta(n/(s^2-1))) = exp(Theta(n))` --
rested on a ceiling `C(k) = |R_L| / |R_j*|^n_blocks = f0^U` that **assumes a window
straddling a block boundary constrains nothing**. hshunt flagged the assumption; exact
enumeration disproves it. The deck and notebook pushed on 2026-11-02, and the copy on
OneDrive, carry the retracted claim and must be re-issued.

Two errors, both found 2026-11-03, both from hshunt:

1. **Alignment.** Every primary generation sweep used `--block_align 2`, which hands the
   machine the phase of the RHM block grid. The real LS machine is translation invariant
   (`block_align 1`, one pooled table). Every `P*` in the 11-02 data is the privileged
   machine's.
2. **Straddling windows carry signal.** `scripts/rhm_straddle.py` enumerates all `v^16`
   grids at `L=2` and *counts* each machine's locally-valid set instead of assuming it.
   The independent-blocks count reproduces `|R_j*|^4` exactly (pipeline check), and
   against it straddling shrinks the locally-valid set by **1.6x-4.0x** at `k=3`. At
   `v=2, m=2, seed 1` the strictly-local `k=3` machine has locally-valid set `= R_L`
   exactly, **ceiling 1.0**, where `f0` predicted 0.25 -- a local window that is both
   grammatical and capable of novelty, which the retracted law said cannot exist.
   Pooling costs little at the ceiling (`pool cost` 1.05-1.38x), so the large pooled-vs-
   aligned gap seen at `L=3` (2x2 parse 0.348 vs 0.995) is a finite-data effect, not a
   ceiling effect. (`k=5` at `L=2` is the memorization regime -- a 5x5 window centred at
   (1,1) covers the whole 4x4 grid -- so its ceiling of 1 is trivial.)

**Still standing:** `succ = gram + novel - 1` identically; `N_k` saturation is the cost
leg and it is what binds at depth -- pooled `N_7` at `L=3` is 2.5e6-1.2e7 distinct
windows at `P=3e5` and still climbing (job 50895001), while `N_3` at `L=2` saturates by
`P~1e3` (274-1890).

## Running jobs

- **50903090** `rhm_pstar_fine.sbatch` -- `P*` at `L=2`, 25-point ladder, 5 seeds, 9
  grammars, `--block_align 1 2` so the real machine and the privileged control share
  seeds and ladder. -> `results/rhm_sampling/2026-11-03/P_v*.json`
- **50903249** `rhm_straddle.sbatch` -- exact locally-valid sets, `v=2` (k=3,5) and `v=3`
  (k=3), m=2..6, 5 seeds. -> `straddle_v{2,3}.json`
- **50903583** `rhm_adaptive_k.sbatch` -- k chosen per site per step by an entropy floor
  (hshunt's criterion), `h_mode` token vs match, `h_min` 0/0.1/0.5/1.0, L=2 and L=3.
  -> `adapt_L{2,3}_*.json`

## Open problems

- Does the straddle gain survive depth? It is measured only at `L=2`, where the two
  blocks a window straddles are siblings under the root. `L=3` is not enumerable; needs
  the saturated-table route at small `v`.
- `k(t)`: no run before 2026-11-03 varied the window along the trajectory at all.
- **Blocked on hshunt (MLS):** the corrected layer-2 posterior. Nothing MLS runs until it lands.
