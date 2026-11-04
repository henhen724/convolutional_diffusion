"""Is the 1D/2D gap in ``x* = P*/|R_L|`` a *coverage* effect?  Sweep k at fixed n.

The 1D depth probe left exactly one "why" open.  At matched n=16 the collapse
ratio is 0.050 in 1D against 0.193 in 2D -- 3.9x apart -- and the standing
hypothesis was coverage: a U=1 window spans 15/16 = 94% of a 1D string but only
9/16 = 56% of a 2D grid, so the 1D machine is nearly whole-string lookup.

That hypothesis is already strained by data in hand.  Across depth at U=1,
2D coverage *rises* (9/16 = 0.56 at L=2 to 49/64 = 0.77 at L=3) while its x* is
flat (0.193 -> 0.207); 1D coverage rises (0.75 -> 0.94) and its x* falls 3x.
Same direction of coverage, opposite response.  But the two coverage ranges
barely overlap, so a law that is flat below ~0.75 and falls above it still fits
both, and the single overlapping pair is suggestive the other way: 1D L=2
(c=0.75, x*=0.148) against 2D L=3 (c=0.766, x*=0.207) is 1.4x apart where
matched *n* was 3.9x apart.  Coverage cannot be dismissed from the depth sweeps.

So vary it directly, at **fixed n, fixed dimension, fixed grammar** -- the one
lever the depth sweeps could not pull.  In 1D at L=4 (n=16 sites) k runs 3..17
and nominal coverage k/d runs 0.19..1.0, which brackets 2D's entire range.  The
test is sharp because k=9 has coverage 9/16, *exactly* 2D L=2's:

  coverage drives x*   -> x* rises smoothly as k falls, reaching ~0.19 at k=9
  seeing the root does -> x* flat for k >= U1 = d-1, success collapsing below
                          it with no locatable peak, and coverage is a red herring

2D already votes for the second: at L=3, k=5 the window table saturates at
exactly N_5 = 92080 and grammaticality never tops 0.7%.  Running the same test
in 2D would therefore return empty cells rather than a curve, which is why this
is done in 1D.

Note the law *must* turn over at the top: at k = d+1 the window is the whole
string from every centre, the machine is pure lookup, and novelty goes to zero.
A monotone reading of "more coverage is better" is wrong on its face; what the
sweep measures is where the turn happens and how sharp it is.

Everything but k is the depth probe's code, imported rather than copied, so a
difference between this and ``rhm_1d_depth`` is the window and nothing else.

Usage:
  python scripts/rhm_1d_coverage.py --L 3 4 --v 2 3 --m 2 3 4 \
      --rule_seeds 3 --out results/rhm_1d/2026-11-04/coverage.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.rhm_1d_depth import RHM1D, exact_RL, run_cell


def coverage(k, d):
    """Nominal and effective window coverage of a d-site string.

    ``nominal`` is k/d, the number the 1D-vs-2D comparison was first quoted in.
    ``effective`` is the mean over centres of the *in-grid* fraction a window
    sees, which is strictly smaller because border windows hang off the edge --
    and it is the one the hypothesis is really about, since an off-grid site
    carries no information about the sample beyond where the edge is.
    """
    h = (k - 1) // 2
    i = np.arange(d)
    ingrid = np.minimum(d - 1, i + h) - np.maximum(0, i - h) + 1
    return k / d, float(ingrid.mean()) / d


def k_values(d, s, L, k_max_extra=1):
    """Odd window sizes from 3 up to just past whole-string.

    ``U1 = 2 s^(L-1) - 1`` is the smallest window that contains a complete
    top-level child block, i.e. the smallest that can see the root; in 1D with
    s=2 that is d-1.  Going one rung past d-1 reaches d+1, where every centre
    sees the entire string and the machine degenerates to lookup -- the
    turnover point, included deliberately as the control on "more is better".
    """
    U1 = 2 * s ** (L - 1) - 1
    top = d + 2 * k_max_extra - 1
    return [k for k in range(3, top + 1, 2)], U1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v', type=int, nargs='+', default=[2, 3])
    ap.add_argument('--s', type=int, default=2)
    ap.add_argument('--L', type=int, nargs='+', default=[3, 4])
    ap.add_argument('--m', type=int, nargs='+', default=[2, 3, 4])
    ap.add_argument('--rule_seeds', type=int, default=3)
    ap.add_argument('--n_gen', type=int, default=2048)
    ap.add_argument('--gen_batch', type=int, default=1024)
    ap.add_argument('--alpha_smooth', type=float, default=1e-3)
    ap.add_argument('--R_max', type=float, default=1e5)
    # the depth probe had no floor and paid for it: with x* ~ 0.05-0.2 a grammar
    # of |R_L| = 49 peaks at P* = 2-8, on the bottom rungs of a 2x ladder, and
    # such cells had to be dropped afterwards as unable to report a slope.  Here
    # the whole measurement *is* where the peak sits, so exclude them up front
    # rather than spend GPU time on peaks that cannot move.  |R_L| >= 200 puts
    # P* at 10 or more even at the smallest x* the depth probe saw.
    ap.add_argument('--R_min', type=float, default=200)
    ap.add_argument('--max_patches', type=int, default=3_000_000,
                    help='skip (cell, k) whose window table exceeds this')
    ap.add_argument('--P_min', type=int, default=1)
    ap.add_argument('--P_max', type=int, default=262144)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ladder = [p for p in (args.P_min * 2 ** i for i in range(40)) if p <= args.P_max]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for L in args.L:
        for v in args.v:
            for m in args.m:
                for rs in range(args.rule_seeds):
                    model = RHM1D(v, args.s, L, m, seed=1000 + rs)
                    d = model.d
                    nominal = v * m ** ((d - 1) // (args.s - 1))
                    R = None if nominal > 1000 * args.R_max else exact_RL(model)
                    if R is None or R > args.R_max or R < args.R_min:
                        print('skip  v%d m%d L%d rs%d  |R_L|=%s (nominal %d) outside '
                              '[%g, %g]' % (v, m, L, rs, R, nominal,
                                            args.R_min, args.R_max), flush=True)
                        continue
                    ks, U1 = k_values(d, args.s, L)
                    print('cell  v%d m%d L%d rs%d  n=%d  U1=%d  |R_L|=%d  k in %s'
                          % (v, m, L, rs, d, U1, R, ks), flush=True)
                    for k in ks:
                        cov_nom, cov_eff = coverage(k, d)
                        for P in ladder:
                            r = run_cell(model, P, k, args.n_gen, device,
                                         1234 + 7 * rs, args.alpha_smooth, args.gen_batch)
                            if r['n_distinct_patches'] > args.max_patches:
                                print('   k=%-3d P=%-7d SKIP, table %d > max_patches'
                                      % (k, P, r['n_distinct_patches']), flush=True)
                                break
                            r.update(R_L=int(R), U1=U1, sees_root=bool(k >= U1),
                                     cov_nom=round(cov_nom, 4), cov_eff=round(cov_eff, 4),
                                     n=d)
                            rows.append(r)
                            print('   k=%-3d(c=%.2f%s) P=%-7d gram %.4f  novel %.4f  '
                                  'succ %.4f  |table|=%-7d  gpu %.2fG  [%.0fs]'
                                  % (k, cov_nom, '*' if k >= U1 else ' ', P,
                                     r['grammatical'], r['novel'], r['success'],
                                     r['n_distinct_patches'], r['gpu_peak_gb'],
                                     r['seconds']), flush=True)
                            json.dump(rows, open(out, 'w'), indent=1)
    json.dump(rows, open(out, 'w'), indent=1)
    print('wrote %s (%d rows)' % (out, len(rows)))


if __name__ == '__main__':
    main()
