"""Does window coverage set x* = P*/|R_L|?  The k-sweep at fixed n, analysed.

The depth probes could never answer this because k and n moved together: every
cell was measured at the root-reaching window U1 = 2 s^(L-1) - 1, so coverage
was a function of depth and the two could not be separated.  ``rhm_1d_coverage``
holds n, the dimension and the grammar fixed and moves k alone, which is the
only way to put a number on coverage's effect.

The prediction registered before the run:

  coverage drives x*   -> x* rises smoothly as k falls, reaching ~0.19 at k=9,
                          since 1D L=4 k=9 has coverage 9/16 -- *exactly* 2D
                          L=2's, where x* is measured at 0.193
  seeing the root does -> x* flat for k >= U1, success collapsing below it,
                          and coverage is a red herring

Peaks are located with the depth probe's ``peak_of`` (log-P parabola through the
top three ladder points, reported only when the argmax is interior), so every
x* here is comparable with the 1D and 2D numbers already published.
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

# the published 2D numbers this is being tested against (STATUS section A/C)
TWO_D = {16: dict(x=0.193, cov=9 / 16), 64: dict(x=0.207, cov=49 / 64)}


def geo(x):
    x = np.asarray(x, float)
    return float(np.median(x)), (float(np.exp(np.log(x).std(ddof=1))) if len(x) > 1 else 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep', default='results/rhm_1d/2026-11-04/coverage.json')
    ap.add_argument('--out', default='results/rhm_1d/2026-11-04/coverage_peaks.json')
    # "grammatical" only means something when grammatical strings are rare.  At
    # n=8 with v=2 there are only 2^8 = 256 strings in all, so the --R_min 200
    # floor admits grammars covering 78% of them and a machine that emits noise
    # scores 0.78.  Cells denser than this are reported but excluded from the
    # headline comparison.
    ap.add_argument('--max_chance', type=float, default=0.05,
                    help='exclude cells where |R_L| / v^n exceeds this')
    args = ap.parse_args()

    rows = json.load(open(ROOT / args.sweep))
    cells, meta = {}, {}
    for r in rows:
        key = (r['v'], r['m'], r['s'], r['L'], r['rule_seed'], r['k'])
        cells.setdefault(key, []).append(r)
        meta[key] = r
    for k in cells:
        cells[k].sort(key=lambda r: r['P'])

    peaks = []
    for key, cs in sorted(cells.items()):
        if len(cs) < 3:
            continue
        mt = meta[key]
        p = peak_of(cs, key, {key: mt['R_L']})
        p.update(v=key[0], m=key[1], s=key[2], L=key[3], rule_seed=key[4], k=key[5],
                 n=mt['n'], U1=mt['U1'], sees_root=mt['sees_root'],
                 cov_nom=mt['cov_nom'], cov_eff=mt['cov_eff'],
                 succ_at_top=cs[-1]['success'],
                 gram_best=max(c['grammatical'] for c in cs),
                 succ_best=max(c['success'] for c in cs),
                 chance=mt['R_L'] / float(key[0]) ** mt['n'])
        peaks.append(p)

    print('%d (cell, k) curves; %d bracketed' % (len(peaks), sum(p['bracketed'] for p in peaks)))
    dense = sorted({(p['v'], p['m'], p['n'], round(p['chance'], 3))
                    for p in peaks if p['chance'] > args.max_chance})
    if dense:
        print('cells denser than chance %.0f%% (v, m, n, |R_L|/v^n), excluded from '
              'the headline: %s' % (100 * args.max_chance, dense))
    ok = [p for p in peaks if p['chance'] <= args.max_chance]
    print('%d of %d curves survive the density cut' % (len(ok), len(peaks)))

    print('\n--- 1. x* against k, at fixed n (the whole point) ---')
    print('%-5s %-4s %-7s %-7s %-6s %-7s %-11s %-9s %-10s %-10s %s'
          % ('n', 'k', 'cov', 'cov_eff', 'root?', 'cells', 'median x*', 'geo sd',
             'med succ', 'med gram', 'chance'))
    for n in sorted({p['n'] for p in peaks}):
        for k in sorted({p['k'] for p in peaks if p['n'] == n}):
            sel = [p for p in peaks if p['n'] == n and p['k'] == k and p['bracketed']]
            allk = [p for p in peaks if p['n'] == n and p['k'] == k]
            if not allk:
                continue
            r0 = allk[0]
            ch = float(np.median([p['chance'] for p in allk]))
            if sel:
                mx, sx = geo([p['ratio'] for p in sel])
                sm = np.median([p['succ_max'] for p in sel])
                gm = np.median([p['gram_best'] for p in sel])
                print('%-5d %-4d %-7.3f %-7.3f %-6s %-7s %-11.4f %-9.2fx %-10.3f %-10.3f %.4f'
                      % (n, k, r0['cov_nom'], r0['cov_eff'],
                         'yes' if r0['sees_root'] else 'no',
                         '%d/%d' % (len(sel), len(allk)), mx, sx, sm, gm, ch))
            else:
                gm = np.median([p['gram_best'] for p in allk])
                sb = np.median([p['succ_best'] for p in allk])
                print('%-5d %-4d %-7.3f %-7.3f %-6s %-7s %-11s %-9s %-10.3f %-10.3f %.4f'
                      % (n, k, r0['cov_nom'], r0['cov_eff'],
                         'yes' if r0['sees_root'] else 'no',
                         '0/%d' % len(allk), 'unbracketed', '--', sb, gm, ch))
        print()

    print('--- 2. the registered prediction, at matched coverage ---')
    # 1D L=4 k=9 has cov 9/16, the same as 2D L=2 at its U=1 window.  If
    # coverage is what separates the dimensions these two must agree.
    for n, tgt in TWO_D.items():
        sel = [p for p in ok if p['n'] == n and p['bracketed']
               and abs(p['cov_nom'] - tgt['cov']) < 1e-6]
        if not sel:
            continue
        mx, _ = geo([p['ratio'] for p in sel])
        print('  1D n=%d at cov %.3f (k=%d, %d cells): x* = %.4f'
              % (n, tgt['cov'], sel[0]['k'], len(sel), mx))
        print('  2D n=%d at cov %.3f              : x* = %.4f   -> %.2fx apart'
              % (n, tgt['cov'], tgt['x'], max(mx, tgt['x']) / min(mx, tgt['x'])))
        print('  (coverage hypothesis predicts these agree; matched *n* alone '
              'was 3.2x apart)')

    print('\n--- 3. is x* monotone in coverage at fixed n? ---')
    # a coverage law needs x* to be a function of coverage alone within a depth.
    # Spearman on (cov, x*) says whether it even trends; the spread of x* across
    # k says whether coverage matters at all next to the 3.2x it must explain.
    for n in sorted({p['n'] for p in ok}):
        sel = [p for p in ok if p['n'] == n and p['bracketed']]
        by_k = {}
        for p in sel:
            by_k.setdefault(p['k'], []).append(p['ratio'])
        if len(by_k) < 3:
            continue
        ks = sorted(by_k)
        med = [float(np.median(by_k[k])) for k in ks]
        rk = np.argsort(np.argsort(ks)).astype(float)
        rx = np.argsort(np.argsort(med)).astype(float)
        rho = float(np.corrcoef(rk, rx)[0, 1])
        print('  n=%-3d k=%s' % (n, ks))
        print('        x*=%s' % ['%.3f' % v for v in med])
        print('        spread across k %.2fx, Spearman(cov, x*) = %+.2f'
              % (max(med) / min(med), rho))

    print('\n--- 4. what k actually buys: grammaticality and peak success ---')
    # the mechanism check.  If the root-reaching window is what matters, the
    # jump shows up here, at k = U1, and not gradually with coverage.
    for n in sorted({p['n'] for p in ok}):
        sel = [p for p in ok if p['n'] == n]
        ks = sorted({p['k'] for p in sel})
        g = [float(np.median([p['gram_best'] for p in sel if p['k'] == k])) for k in ks]
        s = [float(np.median([p['succ_best'] for p in sel if p['k'] == k])) for k in ks]
        u1 = sel[0]['U1']
        br = [p for p in sel if p['bracketed']]
        ga = [float(np.median([p['gram_at'] for p in br if p['k'] == k])) for k in ks]
        na = [float(np.median([p['novel_at'] for p in br if p['k'] == k])) for k in ks]
        sa = [float(np.median([p['succ_max'] for p in br if p['k'] == k])) for k in ks]
        print('  n=%-3d U1=%-3d k        =%s' % (n, u1, ['%5d' % k for k in ks]))
        print('              gram_best=%s' % ['%5.2f' % v for v in g])
        print('              succ_best=%s' % ['%5.2f' % v for v in s])
        print('       at the peak: gram=%s' % ['%5.2f' % v for v in ga])
        print('                   novel=%s' % ['%5.2f' % v for v in na])
        print('                    succ=%s' % ['%5.2f' % v for v in sa])
        best = int(np.argmax(sa))
        iu1 = ks.index(u1) if u1 in ks else None
        if iu1 is not None:
            print('       best k = %d (succ %.3f) vs U1 = %d (succ %.3f): %+.0f%% '
                  'relative' % (ks[best], sa[best], u1, sa[iu1],
                                100 * (sa[best] / sa[iu1] - 1)))

    print('\n--- 5. is the table-size law right *within* a dimension? ---')
    # x* rises 24x across k at fixed n, and so does the window table.  So test
    # y* = P* n / N_k here, where N_k is this cell's own saturated table.  The
    # cross-dimension test said y* separates 1D from 2D by 4.4x; if it is
    # nonetheless flat in k within 1D, then table size is the right bookkeeping
    # inside a dimension and the 1D/2D offset is a separate, geometric constant.
    sat = {}
    for key, cs in cells.items():
        last = [c['n_distinct_patches'] for c in cs[-2:]]
        if len(last) == 2 and last[1] > 0 and (last[1] - last[0]) / last[1] <= 0.02:
            sat[key] = float(last[1])
    print('  %d of %d tables saturated by the top of the ladder' % (len(sat), len(cells)))
    for n in sorted({p['n'] for p in ok}):
        ks, xs, ys = [], [], []
        for k in sorted({p['k'] for p in ok if p['n'] == n}):
            sel = [p for p in ok if p['n'] == n and p['k'] == k and p['bracketed']
                   and (p['v'], p['m'], p['s'], p['L'], p['rule_seed'], p['k']) in sat]
            if not sel:
                continue
            ks.append(k)
            xs.append(float(np.median([p['ratio'] for p in sel])))
            ys.append(float(np.median(
                [p['P_star'] * p['n'] / sat[(p['v'], p['m'], p['s'], p['L'],
                                             p['rule_seed'], p['k'])] for p in sel])))
        if len(ks) < 3:
            continue
        print('  n=%-3d k =%s' % (n, ['%6d' % k for k in ks]))
        print('        x*=%s   spread %.1fx' % (['%6.3f' % v for v in xs],
                                                max(xs) / min(xs)))
        print('        y*=%s   spread %.1fx' % (['%6.3f' % v for v in ys],
                                                max(ys) / min(ys)))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(peaks, open(out, 'w'), indent=1)
    print('\nwrote %s (%d curves)' % (out, len(peaks)))


if __name__ == '__main__':
    main()
