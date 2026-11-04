"""Is the window optimum -- and the 1D/2D gap in x* -- a posterior-entropy budget?

Tests the prediction registered in ``rhm_entropy_window``: the sampler escapes a
training set of P strings only if the entropy it injects along a generation
exceeds ln P, so the effective support is ``S_eff = exp(H_tot)``, novelty dies
at ``P ~ S_eff``, and

    x* = P*/|R_L|  =  exp(H_tot) / |R_L|          -- no fitted constant.

Five tests, in order of how hard they are to pass:

  1. per-sample.  Within a single run, do the copies come from lower-entropy
     trajectories than the novel samples?  This is the weakest test -- it only
     needs the sign -- but it is also the one that says the quantity is
     mechanistically the right one rather than a correlate.
  2. the novelty law.  Is ``novel = exp(-P/S_eff)`` row by row?  Plotted as
     ``-ln(novel)`` against ``P/S_eff`` this is the identity line, with no
     freedom anywhere.
  3. the crossing.  Does the success peak sit at the P where H_tot = ln P?
  4. the k-optimum.  Does the rule pick k=11 over U1=15 at n=16?
  5. the gap.  Does ``exp(H_tot)/|R_L|`` reproduce x* in *both* dimensions --
     0.083 in 1D and 0.193 in 2D -- from one formula?

Test 5 is the point.  A law that only works within a dimension explains the
k-optimum and leaves the original question open.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.rhm_peak_analysis import peak_of


def geo(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if not len(x):
        return float('nan'), float('nan')
    return float(np.median(x)), (float(np.exp(np.log(x).std(ddof=1))) if len(x) > 1 else 1.0)


def at_Pstar(cs, Pstar, field):
    """Log-P linear interpolation of ``field`` at the refined peak position.

    P* comes from a parabola through three ladder points and generally falls
    between rungs, so reading H_tot off the nearest rung would quantise it by
    up to a factor of 2 in P -- which is the whole quantity being tested.
    """
    P = np.log([c['P'] for c in cs])
    y = np.array([c[field] for c in cs], float)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return float('nan')
    return float(np.interp(np.log(Pstar), P[ok], y[ok]))


def cells_of(rows, dim):
    """Group rows into (cell, k) curves sorted by P, with |R_L| and n attached."""
    out, meta = {}, {}
    for r in rows:
        key = (dim, r['v'], r['m'], r['s'], r['L'], r['rule_seed'], r['k'])
        out.setdefault(key, []).append(r)
        meta[key] = r
    for k in out:
        out[k].sort(key=lambda r: r['P'])
    return out, meta


def crossing(cs):
    """Smallest P on the ladder where H_tot(P) drops below ln P, interpolated.

    H_tot is not monotone in P -- more training data both sharpens the posterior
    (less entropy) and fills in windows that were running on the prior (more) --
    so this takes the first crossing rather than solving globally.
    """
    P = np.array([c['P'] for c in cs], float)
    g = np.array([c['H_tot'] - c['ln_P'] for c in cs], float)
    for i in range(1, len(g)):
        if g[i - 1] > 0 >= g[i]:
            f = g[i - 1] / (g[i - 1] - g[i])
            return float(np.exp(np.log(P[i - 1]) + f * (np.log(P[i]) - np.log(P[i - 1]))))
    return float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--e1d', default='results/rhm_entropy/2026-11-04/entropy_1d.json')
    ap.add_argument('--e2d', default='results/rhm_entropy/2026-11-04/entropy_2d.json')
    ap.add_argument('--out', default='results/rhm_entropy/2026-11-04/entropy_peaks.json')
    args = ap.parse_args()

    rows1 = json.load(open(ROOT / args.e1d))
    p2 = ROOT / args.e2d
    rows2 = json.load(open(p2)) if p2.exists() else []
    print('%d 1D rows, %d 2D rows' % (len(rows1), len(rows2)))

    cells, meta = {}, {}
    for rows, dim in ((rows1, 1), (rows2, 2)):
        c, m = cells_of(rows, dim)
        cells.update(c)
        meta.update(m)

    peaks = []
    for key, cs in sorted(cells.items()):
        if len(cs) < 3:
            continue
        mt = meta[key]
        p = peak_of(cs, key, {key: mt['R_L']})
        H = at_Pstar(cs, p['P_star'], 'H_tot')
        p.update(dim=key[0], v=key[1], m=key[2], s=key[3], L=key[4], rule_seed=key[5],
                 k=key[6], n=mt['n'], U1=mt.get('U1'), H_at_peak=H,
                 S_eff=float(np.exp(H)) if np.isfinite(H) else float('nan'),
                 x_pred=float(np.exp(H)) / mt['R_L'] if np.isfinite(H) else float('nan'),
                 P_cross=crossing(cs))
        peaks.append(p)
    br = [p for p in peaks if p['bracketed']]
    print('%d (cell, k) curves, %d bracketed  (1D %d, 2D %d)'
          % (len(peaks), len(br), sum(p['dim'] == 1 for p in br),
             sum(p['dim'] == 2 for p in br)))

    print('\n--- 1. per-sample: do copies come from lower-entropy trajectories? ---')
    for dim, rows in ((1, rows1), (2, rows2)):
        sel = [r for r in rows if np.isfinite(r['H_novel']) and np.isfinite(r['H_copy'])
               and 0.02 < r['novel'] < 0.98]
        if not sel:
            continue
        d = [r['H_novel'] - r['H_copy'] for r in sel]
        print('  %dD  %4d runs with both classes present:  H_novel - H_copy = '
              '%+.2f nats (median), positive in %.0f%% of runs'
              % (dim, len(sel), float(np.median(d)), 100 * np.mean(np.array(d) > 0)))

    print('\n--- 2. the novelty law  novel = exp(-P / exp(H_tot)) ---')
    # no free parameter: -ln(novel) should equal P/S_eff.  The ratio of the two
    # is the whole test, and it is reported as a spread rather than a p-value
    # because the claim is order-of-magnitude bookkeeping, not a fit.
    for dim, rows in ((1, rows1), (2, rows2)):
        sel = [r for r in rows if 0.02 < r['novel'] < 0.98 and np.isfinite(r['H_tot'])]
        if not sel:
            continue
        obs = np.array([-np.log(r['novel']) for r in sel])
        pred = np.array([r['P'] * np.exp(-r['H_tot']) for r in sel])
        mr, sr = geo(obs / pred)
        print('  %dD  %4d runs in the informative band:  observed/predicted = '
              '%.2f  (geo sd %.2fx)' % (dim, len(sel), mr, sr))

    print('\n--- 3. does the success peak sit at the H_tot = ln P crossing? ---')
    for dim in (1, 2):
        sel = [p for p in br if p['dim'] == dim and np.isfinite(p['P_cross'])]
        if not sel:
            print('  %dD  no bracketed peak had a crossing on the ladder' % dim)
            continue
        mr, sr = geo([p['P_cross'] / p['P_star'] for p in sel])
        print('  %dD  %3d peaks:  P_cross / P* = %.2f  (geo sd %.2fx)' % (dim, len(sel), mr, sr))

    print('\n--- 4. the k-optimum at fixed n: measured against the entropy rule ---')
    # the rule picks the largest k whose crossing is still above the grammar's
    # requirement, which in practice is the k maximising min(S_eff, grammar) --
    # here reported as the k maximising S_eff-weighted success, against the
    # measured argmax.  Printed per n so k=11 vs U1=15 is readable directly.
    for n in sorted({p['n'] for p in br if p['dim'] == 1}):
        sel = [p for p in br if p['dim'] == 1 and p['n'] == n]
        ks = sorted({p['k'] for p in sel})
        if len(ks) < 3:
            continue
        H = [float(np.median([p['H_at_peak'] for p in sel if p['k'] == k])) for k in ks]
        S = [float(np.median([p['succ_max'] for p in sel if p['k'] == k])) for k in ks]
        X = [float(np.median([p['ratio'] for p in sel if p['k'] == k])) for k in ks]
        Xp = [float(np.median([p['x_pred'] for p in sel if p['k'] == k])) for k in ks]
        u1 = sel[0]['U1']
        print('  n=%-3d U1=%-3d k      =%s' % (n, u1, ['%7d' % k for k in ks]))
        print('              H_tot  =%s' % ['%7.2f' % v for v in H])
        print('              succ   =%s' % ['%7.3f' % v for v in S])
        print('              x* meas=%s' % ['%7.3f' % v for v in X])
        print('              x* pred=%s' % ['%7.3f' % v for v in Xp])
        print('        measured best k = %d; H_tot falls %.2f nats from k=%d to k=%d'
              % (ks[int(np.argmax(S))], H[0] - H[-1], ks[0], ks[-1]))

    print('\n--- 5. THE GAP.  x* before (U=1) and after (entropy formula) ---')
    # "before" is the convention both dimensions were published at; "after" is
    # the same bracketed peaks with x* replaced by exp(H_tot)/|R_L|.  If the
    # budget is the mechanism, the 1D and 2D columns come together.
    print('  %-6s %-6s %-6s %-12s %-12s %s'
          % ('dim', 'n', 'cells', 'x* measured', 'x* predicted', 'pred/meas'))
    rowsout = []
    for dim in (1, 2):
        for n in sorted({p['n'] for p in br if p['dim'] == dim}):
            sel = [p for p in br if p['dim'] == dim and p['n'] == n
                   and p['k'] == p['U1']] if dim == 1 else \
                  [p for p in br if p['dim'] == dim and p['n'] == n]
            sel = [p for p in sel if np.isfinite(p['x_pred'])]
            if not sel:
                continue
            mx, _ = geo([p['ratio'] for p in sel])
            mp, _ = geo([p['x_pred'] for p in sel])
            rowsout.append((dim, n, mx, mp))
            print('  %-6s %-6d %-6d %-12.4f %-12.4f %.2fx'
                  % ('%dD' % dim, n, len(sel), mx, mp, mp / mx))
    for n in sorted({r[1] for r in rowsout}):
        a = [r for r in rowsout if r[1] == n and r[0] == 1]
        b = [r for r in rowsout if r[1] == n and r[0] == 2]
        if a and b:
            gm = max(a[0][2], b[0][2]) / min(a[0][2], b[0][2])
            gp = max(a[0][3], b[0][3]) / min(a[0][3], b[0][3])
            print('  at matched n=%d:  measured gap %.2fx  ->  predicted gap %.2fx'
                  % (n, gm, gp))

    print('\n--- 6. the weaker claim: is H the collapse variable at all? ---')
    # Tests 2-3 fix the coefficient at 1 and the intercept at 0.  Even if that
    # fails, entropy may still be *the* variable the dimensions share: fit
    # ln P* = a H + b pooled, then refit with a per-dimension intercept and see
    # whether the dimension term is still needed.  If it drops to nothing, the
    # 1D/2D gap is an entropy effect with a coefficient that is not 1; if it
    # survives, entropy joins coverage and table size on the refuted pile.
    sel = [p for p in br if np.isfinite(p['H_at_peak']) and p['P_star'] > 0]
    if len(sel) > 4:
        H = np.array([p['H_at_peak'] for p in sel])
        lP = np.array([np.log(p['P_star']) for p in sel])
        d2 = np.array([p['dim'] == 2 for p in sel], float)
        a, b = np.polyfit(H, lP, 1)
        r1 = lP - (a * H + b)
        A = np.stack([H, d2, np.ones_like(H)], 1)
        coef, *_ = np.linalg.lstsq(A, lP, rcond=None)
        r2 = lP - A @ coef
        print('  pooled      ln P* = %+.3f H %+.3f           residual %.2fx'
              % (a, b, np.exp(r1.std(ddof=1))))
        print('  + dim term  ln P* = %+.3f H %+.3f*[2D] %+.3f  residual %.2fx'
              % (coef[0], coef[1], coef[2], np.exp(r2.std(ddof=1))))
        print('  the 2D offset is %+.2f nats-equivalent (%.2fx in P*); it is what '
              'entropy would have to absorb to count as the collapse variable'
              % (coef[1], np.exp(abs(coef[1]))))
        for dim in (1, 2):
            m = np.array([p['dim'] == dim for p in sel])
            if m.sum():
                print('    %dD mean residual about the pooled line: %+.2f (%d peaks)'
                      % (dim, float(r1[m].mean()), int(m.sum())))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(peaks, open(out, 'w'), indent=1)
    print('\nwrote %s (%d curves)' % (out, len(peaks)))


if __name__ == '__main__':
    main()
