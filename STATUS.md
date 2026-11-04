# STATUS

**RHM sampling complexity** is answered and delivered (was due 12:00 PT Fri 11/6).
**MLS** is parked on hshunt's corrected layer-2 posterior (LOG.md, `notebooks/mls_entropy_barrier_results.ipynb`).

## A. The answer

`P` = training samples before the LS machine emits output **both novel and fully grammatical**.

> **`P* = exp(Theta(n))`, `n` = sites. At the U=1 window in 2D, `P* = 0.21 |R_L|`.**

Measured, not derived; the `C(k) = f0^U` ceiling stays retracted. 30 bracketed 2D peaks
collapse to median 0.197 while `|R_L|` spans 49 152x; in 1D `ln P*` grows 0.374/site over
n=4..16 but `P* = Theta(|R_L|)` fails (`b = 0.689 +- 0.045`), so only `exp(Theta(n))` is general.
Root needs `k >= 2s^(L-1)-1`; straddling windows do see two levels up (hshunt's conjecture).

## B. The open question: why x* = P*/|R_L| differs by dimension

Head-to-head on one instrument at n=16: **1D 0.083 (k=U1=15), 2D 0.159 (25 cells), 1.92x apart.**
Three explanations tested, **all three refuted**:
1. **Coverage -- sign backwards** (51058208, deck sent 11-04). `x*` *rises* with coverage
   (Spearman +1.00); matching on it leaves 5.1x where matching on `n` leaves 2.3x.
2. **Table saturation `y* = P* n/N_k`** -- 4.4x across dimensions and 61x across k within 1D.
3. **Posterior-entropy budget -- refuted 11-04** (51073161, 51076386; notebook + deck sent).
   `x* = exp(H_tot)/|R_L|` gives 1D 0.68 / 2D 337.6 = **496x apart, 258x worse**. Novelty law
   off 2.06x / 597x; peak 32.6x below the `H_tot = ln P` crossing; pooled
   `ln P* = 0.100 H + 4.18` (residual 5.5x) where the claim needs slope 1.

**What (3) did establish -- why success peaks at k=11, not U1=15.** H_tot falls 11.82 -> 8.21
nats over k=3..17 at n=16 (no feature at U1) as grammaticality rises 0.06 -> 0.78 and novelty
falls 1.00 -> 0.52; copies are the low-entropy trajectories, -1.75 nats in 1D (84% of 781 runs)
and -7.42 in 2D (100%). The window is bought with the currency novelty needs: block structure
says where a window *can* be grammatical, entropy says where it *should* sit. Aside, no theory:
`exp(H)/|R_L|` = 0.68 in 1D vs 338 in 2D, so 2D is grammar-limited and 1D novelty-limited.

## Running jobs — none

## Open problems

- The 1D/2D offset has no mechanism after three refutations. No fourth without a measurement.
- 2D peak estimates scatter 4.8x per cell (2048 gens, 3-point parabola on a doubling ladder),
  though population medians reproduce to 1.2x: raise n_gen before quoting a single cell.
- 2D L=4 needs k=15, out of table reach; 2D rests on two depths. One L=3 curve unbracketed
  (v3m2 rs0, `x* >= 0.477`). `k >= 7` overflows the int64 window code for `a_j`.
- Pooled-vs-per-phase gap (3-4x) far exceeds their support gap (1.04-1.38x): pooling costs the
  *sampler*. Measurement in hand, page unwritten.
- **Blocked on hshunt (MLS):** the corrected layer-2 posterior. Nothing MLS runs until it lands.

## Next steps

1. Score `AdaptiveWindowDiscreteScoreMachine` (`src/utils/rhm_sampling.py`; picks k per site
   and step off an `h_min` floor) against novel-and-grammatical -- so far only scored on
   grammaticality. hshunt's standing patch-size question; B(3) says the floor is the right knob.
2. The pooling-costs-the-sampler page.

## Discipline notes

- Analysis runs in an allocation (`srun -p sganguli -t 15 -c 2 --mem=16G`), never a login node.
- `rhm_entropy_analysis.py`'s `geo()` returns the **median**, not a geometric mean (1.5-1.7x).
- Footprint flat in depth (~1 GB host); risk lives in `exact_RL`. sganguli CPU-starved: few cpus.
