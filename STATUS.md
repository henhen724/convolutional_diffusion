# STATUS

**RHM sampling complexity** is answered and delivered (was due 12:00 PT Fri 11/6). No jobs of mine running.
**MLS** is parked on hshunt's corrected layer-2 posterior (LOG.md, `notebooks/mls_entropy_barrier_results.ipynb`).

## A. The answer

`P` = training samples before the LS machine emits output **both novel and fully grammatical**.

> **`P* = exp(Theta(n))`, `n` = sites. At the U=1 window in 2D, `P* = 0.21 |R_L|`.**

Measured, not derived; the `C(k) = f0^U` ceiling stays retracted. 30 bracketed 2D peaks collapse
to median 0.197 while `|R_L|` spans 49 152x; in 1D `ln P*` grows 0.374/site over n=4..16 but
`P* = Theta(|R_L|)` fails (`b = 0.689 +- 0.045`). Root needs `k >= U1 = 2s^(L-1)-1`.

## B. The open question: why x* = P*/|R_L| differs by dimension

Head-to-head on one instrument at n=16: **1D 0.083 (k=U1=15), 2D 0.159 (25 cells), 1.92x apart.**
Three explanations tested, **all three refuted**:
1. **Coverage -- sign backwards** (51058208, deck sent 11-04): `x*` *rises* with coverage
   (Spearman +1.00), and matching on it leaves 5.1x where matching on `n` leaves 2.3x.
2. **Table saturation `y* = P* n/N_k`** -- 4.4x across dimensions, 61x across k within 1D.
3. **Posterior-entropy budget -- refuted 11-04** (51073161, 51076386; notebook + deck sent).
   `x* = exp(H_tot)/|R_L|` gives 1D 0.68 / 2D 337.6 = **496x apart, 258x worse**; novelty law off
   2.06x/597x; peak 32.6x below the `H_tot = ln P` crossing; pooled `ln P* = 0.100 H + 4.18`.

**What (3) did establish -- why success peaks at k=11, not U1=15.** H_tot falls 11.82 -> 8.21 nats
over k=3..17 at n=16 as grammaticality rises 0.06 -> 0.78 and novelty falls 1.00 -> 0.52; copies are
the low-entropy trajectories (-1.75 nats in 1D, 84% of 781 runs; -7.42 in 2D, 100%).

## C. hshunt's patch-size criterion: answered 11-04, deck + notebook sent
"Largest `k` whose posterior entropy stays above a floor", scored on novel-and-grammatical for the
first time (re-read of 50903583, then 51098252). **Direction confirmed:** `k(t)` shrinks
monotonically along every trajectory, produced by the floor, not imposed. **Works as an
anti-copier:** `h_min` 0 -> 1.0 lifts novelty 0.460 -> **0.997** (median/21 cells) while
grammaticality 0.595 -> 0.006. **The choice set was the bug:** forbidding `k < U1` makes the best
floor a *binding* one in **15/15** cells (was 0/21) and beats the old machine 15/15 (ratio 1.140).
**But it still loses to fixed `k = U1 = 3` in 12/15** (ratio 0.775; the 3 wins are +0.003, +0.000,
+0.014). Registered prediction half confirmed; the null it named is the answer: **at L=2, U1
already is the schedule.** `notebooks/rhm_adaptive_window_2026-11-04.ipynb`.

## Open problems
- The 1D/2D offset has no mechanism after three refutations. No fourth without a measurement.
- 2D peaks scatter 4.8x per cell (2048 gens, 3-point parabola on a doubling ladder) though
  population medians reproduce to 1.2x: raise n_gen before quoting a single cell.
- 2D L=4 needs k=15, out of table reach, so 2D rests on two depths; one L=3 curve unbracketed
  (v3m2 rs0, `x* >= 0.477`); `k >= 7` overflows the int64 window code for `a_j`.
- Pooled-vs-per-phase gap (3-4x) far exceeds their support gap (1.04-1.38x): pooling costs the
  *sampler*, not the support. Measurement in hand, page unwritten.
- **Blocked on hshunt (MLS):** the corrected layer-2 posterior. Nothing MLS runs until it lands.

## Next steps
1. Fix the int64 overflow in the `a_j` window code so `k = 7` runs: prerequisite for testing a
   *schedule* at L=3 (`U1 = 7`), where it has somewhere to go. L=2 gives it only rungs 3 and 5.
2. The pooling-costs-the-sampler page.

## Discipline notes
- Analysis runs in an allocation (`srun -p sganguli -t 15 -c 2 --mem=8G`), never a login node.
- `rhm_entropy_analysis.py`'s `geo()` returns the **median**, not a geometric mean (1.5-1.7x).
- Footprint flat in depth (~1 GB host); risk lives in `exact_RL`. sganguli CPU-starved: few cpus.
