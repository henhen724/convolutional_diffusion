# STATUS

**RHM sampling complexity** is active (due 12:00 PT Fri 11/6); **MLS** is parked
(LOG.md, `notebooks/mls_entropy_barrier_results.ipynb`).

## A. RHM sampling complexity -- where the answer stands 2026-11-03

How many training grids `P` before the LS machine emits grids that are **both novel and
fully grammatical**, and how does `P*` scale with `(v, s, L, m, k)`?

The 2026-11-02 answer (`P* = exp(Theta(n))`) was retracted 2026-11-03: its ceiling
`C(k) = f0^U` assumed a straddling window constrains nothing, which exact enumeration
disproved. **The exponential is returning by the cost leg**, and the pieces are now
measured rather than argued:

1. **Window bound.** A `k x k` window resolves levels up to `j* = floor(log_s((k+1)/2))`,
   so reaching the root needs `k >= 2 s^(L-1) - 1` (the "U=1" window: k=3 at L=2, k=7 at
   L=3, k=15 at L=4). Confirmed both ways: every L=3 run at `k<=5` has grammaticality
   0.000 at every `P`, and L=4 at `k<=5` never parses above two levels from the leaves.
   At L=3 **k=7 is the first window where grammaticality rises with P at all**
   (0.008 -> 0.234 for v3m2 rs1 over P = 1e2 -> 1e5, job 50930718).
2. **Cost at that window.** Exact enumeration over the whole grammar (`nk_exact_L3.json`)
   gives `N_k = Theta(|R_L|)`: `N_3/|R_2| = 4.3-6.0`, `N_7/|R_3| = 1.6-6.1`. Coupon
   collection needs `P_sat ~ N_k ln N_k / n`, i.e. as much data as the grammar has grids.
3. **Novelty needs the opposite.** `P << |R_L|`, else the training set *is* the grammar
   (at L=2, `train_distinct = 486 = |R_L|` by P=1e4, with `gram + novel = 1.000`,
   `succ = 0`). The two requirements collide, so success peaks at `P* = Theta(|R_L|)`
   and `|R_L| = v m^((n-1)/(s^2-1))` makes that `exp(Theta(n))`.
4. **`locally_valid` ~ `grammatical` at k=7** (0.234/0.219, 0.117/0.102, 0.055/0.055), so
   at the U=1 window staying inside the training patch table is very nearly equivalent to
   being grammatical -- the failures are the sampler leaving the table, not the table
   admitting junk.
5. **Straddling is real (hshunt's question).** `R_1 = q_1 / q_2^(s^2)` is 1 when the four
   children of a level-1 node parse independently. Measured median `R_1`: **0.4 at k=3,
   1.7 at k=5, 2.1 at k=7**, and 6-14 at v3m2. Once the window spans a block boundary the
   sibling subtrees become positively correlated -- the parent parses far more often than
   independent children allow. (This conflates window size with straddling; a
   block-aligned-only control would separate them and does not exist yet.)

## Running jobs

- **50935971** `rhm_L3_peak.sbatch` -- the deliverable. k=7 at L=3, `P` to 4e6, v2m2
  (3 seeds) and v3m2 (3 seeds), plus a k=5 control. Ladders chosen so `P/|R_L|` crosses 1
  using the *exact* per-seed `|R_L|` (v2m2: 4194304/4194304/2057216; v3m2:
  6291456/301440/6291456). If the success peak lands at the same `P/|R_L|` across seeds
  whose `|R_L|` differ 20x, `P* = Theta(|R_L|)` is established. -> `L3peak_*.json`

## Open problems

- Peak *height*: success never exceeded 0.078 so far. Is the peak O(1) or decaying in n?
- `q_l` depth-independence is not clean: at k=3 one level above the leaves, L=4 gives
  0.62-0.67 but L=3 gives ~0.51. Rule draws differ between depths, so this may be seed
  spread; not resolved, and not on the critical path.
- Deck/notebook/OneDrive copy still carry the retracted 11-02 law; re-issue once, Thursday.
- **Blocked on hshunt (MLS):** the corrected layer-2 posterior. Nothing MLS runs until it lands.

## Infrastructure note

`55ae0d1` made P=4e6 feasible: chunk before the unfold (reshaping a strided unfold view
copied ~100 GB), pack window rows into int64 words before `torch.unique`, bytes keys for
novelty, sorted-lookup membership in `locally_valid_fraction`. All four verified exact
against what they replaced. The `sites=` path in the GEMM matcher (one site per step
instead of all n) is also verified behaviour-preserving.
