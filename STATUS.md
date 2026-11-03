# STATUS

**RHM sampling complexity** is answered (due 12:00 PT Fri 11/6); **MLS** is parked
(LOG.md, `notebooks/mls_entropy_barrier_results.ipynb`).

## A. RHM sampling complexity -- the answer, 2026-11-03

How many training grids `P` before the LS machine emits grids that are **both novel and
fully grammatical**, and how does `P*` scale?

> **`P* = 0.21 |R_L| = Theta(v m^((n-1)/(s^2-1))) = exp(Theta(n))`**, `n = s^(2L)` sites.

Measured, not derived. The 11-02 ceiling argument (`C(k) = f0^U`) stays retracted; the
asymptotic answer survived it. Notebook:
`notebooks/rhm_sampling_complexity_2026-11-03.ipynb` (executed, commit recorded inside).

1. **Collapse (the evidence).** 30 bracketed success peaks, L in {2,3}, v in {2,3,4,6,8},
   m in {2..6}, 3 rule seeds each. `|R_L|` spans 49 152x and `P*` spans 63 851x, but
   `P*/|R_L|` spans only 11.9x: median 0.197, geo-mean 0.212, geo-sd 1.95x. Free fit
   `P* ~ |R_L|^0.965 +- 0.041` (1 is 0.8 sd away); constrained `b=1` fits no worse.
   L=2 median 0.193 (n=25) vs L=3 median 0.207 (n=5), agreeing to 7% while n goes 16 -> 64.
   Rules are drawn *with replacement*, so seeds at the same (v,m) differ up to 20x in
   `|R_L|` -- that is what makes this a collapse and not a one-parameter rescaling.
2. **Window bound (the control).** Reaching the root needs `k >= 2 s^(L-1) - 1`. At L=3,
   k=5 the table saturates at **exactly `N_5 = 92080`** (provably complete) yet
   grammaticality never exceeds 0.7% from P=1e4 to 4e6; k=7 on the same grammars reaches
   25-52%. Below the U=1 window no `P` helps, so all curves above are at `k = U1`.
3. **Mechanism.** success = grammatical x P(novel|grammatical). The first saturates at a
   window-set ceiling; the second fits `exp(-0.725 x - 0.868)` in `x = P/|R_L|` over 488
   cells, where uniform sampling would give `exp(-x)`. The intercept means only **42%** of
   grammatical outputs are novel even as x -> 0: a P-independent memorization drag.
4. **Peak height, honestly.** Median peak success **0.022**, range 0.001-0.342. The law
   locates the optimum; the optimum is not good. Peak decays with v and m.
5. **Straddling (hshunt's question).** `R_1 = lvl[1]/lvl[2]^4` median 0.4 at k=3, 1.7 at
   k=5, 2.1 at k=7 (6-14 at v3m2): once the window spans a block boundary, sibling
   subtrees are positively correlated. Caveat: this conflates window size with straddling;
   the block-aligned-only control that separates them does not exist yet
   (`block_align=s` is *not* it -- a per-phase table gives the machine more information).

## Running jobs

- **50976107** `rhm_deck.sbatch` -- rebuilds `peak_summary.json`/`collapse.png` and the
  slide PDF (`results/rhm_sampling/2026-11-03/rhm_sampling_complexity_slides.pdf`, plus
  per-page PNGs). GPU only for the trace page. Upload to OneDrive once it lands.
- 50935971 (L=3 peak hunt + k=5 control) COMPLETED 02:33.

## Open problems

- L=4 would need k=15; the table is out of reach, so `exp(Theta(n))` rests on two depths
  plus the exact `N_k = Theta(|R_L|)` counts. A cheaper probe would be a 1D RHM at L=4-5.
- One L=3 curve is unbracketed (v3m2 rs0, still climbing at P=3e6, `x* >= 0.477`).
- The straddling control (same window, block-aligned tables only) is still unbuilt.
- **Blocked on hshunt (MLS):** the corrected layer-2 posterior. Nothing MLS runs until it lands.

## Next steps

1. Fold 50976107's PDF into `claude-results-upload.sh` and push everything to origin main.
2. If time before Friday: the straddling control, then the 1D depth probe.
