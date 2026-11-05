"""x* is not a constant with a dimension offset -- it is |R_L|^(b-1), b < 1.

Three separate findings in STATUS are plausibly one fact, and this tests that:

  * section B's unexplained 1D/2D gap in x* = P*/|R_L| (3.87x at n=16, three
    refuted mechanisms, "no fourth without a measurement");
  * x* falling 7.6x across (v,m) medians, monotonically in m at fixed v
    (0.565/0.332/0.201/0.144/0.121 for m=2..6 at v=4);
  * x* falling with depth -- 15/15 (v,m) combos in 1D, median 2.5x per step,
    and 2.5-3.0x at 2D v=2 m=2 once the copying regime is cut.

All three move x* in the direction that |R_L| moves: m raises |R_L| at fixed v,
depth raises it enormously, and the 1D and 2D populations sit at different
|R_L|.  The free-exponent fits already say the same thing from the other side --
b = 0.965+-0.041 in 2D, 0.689+-0.045 in 1D -- because

    x* = P* / |R_L| = |R_L|^(b-1) * const,

so b < 1 IS "x* falls as |R_L| grows", and a constant x* is exactly b = 1.
Under that reading the "dimension gap" and the "depth effect" are not separate
phenomena needing separate mechanisms; they are two ways of sampling different
|R_L| at the same nominal n.

The test: fit log P* = b log|R_L| + c + d*1[2D] over both dimensions at once.
If the gap is a scale effect, d collapses toward 0 and the raw 3.87x is
absorbed by b.  If d survives at the size of the raw gap, dimension is a real
offset and section B's open question stands.  A common-b fit is also compared
against one with a separate b per dimension, since 0.689 vs 0.965 is itself a
difference the pooled model would be papering over.

Caveats kept in view, not swept:
  * P* is a peak estimate with real noise, so this propagates each cell's
    binomial bootstrap through the fit rather than treating P* as exact.
  * 1D L=2 cells are degenerate: x* = 1.63 at v2m2, i.e. the peak sits at MORE
    training grids than there are distinct grammatical grids. --min_RL drops
    the cells too small for x* to mean anything, and the fit is reported with
    and without them.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from rhm_peak_noise import U1, est_A, est_B  # noqa: E402


def collect_2d(dirs, P_min):
    RL, rows = {}, defaultdict(list)
    for d in dirs:
        for f in glob.glob(str(ROOT / d / 'nk_exact_L*.json')):
            for r in json.load(open(f)):
                RL[(r['v'], r['m'], r['s'], r['L'], r['rule_seed'])] = r['R_L']
    pats = ['P_v*_m*.json', 'L3peak_v*_k7.json', 'L3k7_v*_m*.json',
            'depth2d_L2_v*_m2.json', 'L3seed_v*_m2_k7.json']
    for d in dirs:
        for pat in pats:
            for f in glob.glob(str(ROOT / d / pat)):
                for r in json.load(open(f)):
                    if 'success' not in r or r.get('block_align', 1) != 1:
                        continue
                    if not isinstance(r.get('k'), int):
                        continue
                    if r['k'] != U1(r['s'], r['L']):
                        continue
                    key = (r['v'], r['m'], r['s'], r['L'], r['rule_seed'])
                    if key in RL:
                        rows[key].append(r)
    out = []
    for key, rs in rows.items():
        by_P = defaultdict(lambda: [0.0, 0.0])
        for r in rs:                       # pool duplicate ladder points
            n = float(r.get('n_gen', 4096))
            by_P[r['P']][0] += r['success'] * n
            by_P[r['P']][1] += n
        P = np.array(sorted(by_P), float)
        h = np.array([by_P[p][0] for p in sorted(by_P)], float)
        n = np.array([by_P[p][1] for p in sorted(by_P)], float)
        keep = P >= P_min
        if keep.sum() >= 4:
            out.append((2, key, P[keep], (h / n)[keep], n[keep], float(RL[key])))
    return out


def collect_1d(sweep, P_min):
    rows = defaultdict(list)
    for r in json.load(open(ROOT / sweep)):
        if r.get('block_align', 1) != 1 or r['k'] != U1(r['s'], r['L']):
            continue
        rows[(r['v'], r['m'], r['s'], r['L'], r['rule_seed'])].append(r)
    out = []
    for key, rs in rows.items():
        rs = sorted(rs, key=lambda r: r['P'])
        P = np.array([r['P'] for r in rs], float)
        S = np.array([r['success'] for r in rs], float)
        n = np.array([float(r['n_samples']) for r in rs])
        keep = P >= P_min
        if keep.sum() >= 4:
            out.append((1, key, P[keep], S[keep], n[keep], float(rs[0]['R_L'])))
    return out


def fit(logR, logP, dim2, common_b=True):
    """Least squares for log P* = b logR + c + d*1[2D] (or a separate b)."""
    if common_b:
        X = np.column_stack([logR, np.ones_like(logR), dim2])
    else:
        X = np.column_stack([logR, np.ones_like(logR), dim2, logR * dim2])
    if len(logR) <= X.shape[1]:
        return None
    coef, *_ = np.linalg.lstsq(X, logP, rcond=None)
    return coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+',
                    default=['results/rhm_sampling/2026-11-03',
                             'results/rhm_sampling/2026-11-05'])
    ap.add_argument('--sweep', default='results/rhm_1d/2026-11-03/sweep.json')
    ap.add_argument('--P_min', type=float, default=10)
    ap.add_argument('--min_RL', type=float, default=0,
                    help='drop cells whose grammar is too small for x* to mean '
                         'anything (1D v2m2 L=2 has x* = 1.63, i.e. P* exceeds '
                         'the entire support)')
    ap.add_argument('--boot', type=int, default=300)
    ap.add_argument('--frac', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    cells = collect_2d(a.dirs, a.P_min) + collect_1d(a.sweep, a.P_min)
    cells = [c for c in cells if c[5] >= a.min_RL]
    rng = np.random.default_rng(a.seed)

    recs = []
    for dim, key, P, S, N, R in cells:
        ng = float(np.median(N))
        pa = est_A(P, S, ng)
        pb = est_B(P, S, ng, a.frac, fallback=True)
        if not np.isfinite(pa):
            continue
        ba = np.empty(a.boot)
        bb = np.empty(a.boot)
        for j in range(a.boot):
            Sb = rng.binomial(N.astype(int), np.clip(S, 0, 1)) / N
            ba[j] = est_A(P, Sb, ng)
            bb[j] = est_B(P, Sb, ng, a.frac, fallback=True)
        recs.append(dict(dim=dim, v=key[0], m=key[1], s=key[2], L=key[3],
                         rule_seed=key[4], R_L=R, P_A=float(pa),
                         P_B=float(pb) if np.isfinite(pb) else float('nan'),
                         x_A=float(pa) / R,
                         x_B=(float(pb) / R) if np.isfinite(pb) else float('nan'),
                         rep_A=ba, rep_B=bb))

    for tag, dsel in (('1D', 1), ('2D', 2)):
        sub = [r for r in recs if r['dim'] == dsel]
        print('%s cells: %3d   |R_L| %.3g .. %.3g   depths %s'
              % (tag, len(sub),
                 min(r['R_L'] for r in sub), max(r['R_L'] for r in sub),
                 sorted({r['L'] for r in sub})))
    print()

    def run(which):
        ok = [r for r in recs
              if np.isfinite(r['P_A' if which == 'A' else 'P_B'])]
        logR = np.log([r['R_L'] for r in ok])
        logP = np.log([r['P_A' if which == 'A' else 'P_B'] for r in ok])
        d2 = np.array([1.0 if r['dim'] == 2 else 0.0 for r in ok])
        reps = [r['rep_A' if which == 'A' else 'rep_B'] for r in ok]

        base_c = fit(logR, logP, d2, True)
        base_s = fit(logR, logP, d2, False)
        bc, bs = [], []
        for _ in range(a.boot):
            idx = rng.integers(0, len(ok), len(ok))
            lp = []
            for i in idx:
                f = reps[i][np.isfinite(reps[i])]
                lp.append(np.log(f[rng.integers(0, len(f))]) if len(f)
                          else logP[i])
            lp = np.array(lp)
            cc = fit(logR[idx], lp, d2[idx], True)
            ss = fit(logR[idx], lp, d2[idx], False)
            if cc is not None:
                bc.append(cc)
            if ss is not None:
                bs.append(ss)
        return ok, base_c, base_s, np.array(bc), np.array(bs), logR, logP, d2

    ci = lambda col: (np.percentile(col, 2.5), np.percentile(col, 97.5))
    results = {}
    for which in ('A', 'B'):
        ok, cc, ss, bc, bs, logR, logP, d2 = run(which)
        print('===== estimator %s   (%d cells) =====' % (which, len(ok)))

        for tag, dsel in (('1D only', 1), ('2D only', 2)):
            sel = d2 == (1.0 if dsel == 2 else 0.0)
            X = np.column_stack([logR[sel], np.ones(sel.sum())])
            co, *_ = np.linalg.lstsq(X, logP[sel], rcond=None)
            print('  %-9s b = %.3f    (x* = |R_L|^%.3f, so x* %s with |R_L|)'
                  % (tag, co[0], co[0] - 1,
                     'falls' if co[0] < 1 else 'rises'))

        lo, hi = ci(bc[:, 0])
        print('  pooled, common b:  b = %.3f  [%.3f, %.3f]' % (cc[0], lo, hi))
        lo, hi = ci(bc[:, 2])
        print('                     2D offset d = %.3f  [%.3f, %.3f]'
              '  -> %.2fx in x*' % (cc[2], lo, hi, np.exp(cc[2])))
        print('                     raw 1D/2D gap in median x* is 3.87x'
              ' (= d of 1.353)')
        lo, hi = ci(bs[:, 3])
        print('  separate b:        b_1D = %.3f   b_2D = %.3f'
              '   difference %.3f  [%.3f, %.3f]'
              % (ss[0], ss[0] + ss[3], ss[3], lo, hi))
        results[which] = dict(
            n=len(ok), b_common=float(cc[0]), d=float(cc[2]),
            d_ci=[float(x) for x in ci(bc[:, 2])],
            b_ci=[float(x) for x in ci(bc[:, 0])],
            b_1d=float(ss[0]), b_2d=float(ss[0] + ss[3]),
            db_ci=[float(x) for x in ci(bs[:, 3])])
        print()

    print('--- reading ---')
    dA, dB = results['A']['d'], results['B']['d']
    shrink = [np.exp(d) / 3.87 for d in (dA, dB)]
    print('  Raw 1D/2D gap 3.87x. After allowing x* to depend on |R_L| with a')
    print('  single exponent, the residual dimension offset is %.2fx (A) and'
          % np.exp(dA))
    print('  %.2fx (B) -- %.0f%% and %.0f%% of the raw gap.'
          % (np.exp(dB), 100 * shrink[0], 100 * shrink[1]))
    sigA = results['A']['d_ci'][0] > 0 or results['A']['d_ci'][1] < 0
    if not sigA:
        print('  The offset is no longer distinguishable from zero: the gap is a')
        print('  scale effect, and "x* differs by dimension" was x* depending on')
        print('  |R_L| while the two populations sat at different |R_L|.')
    else:
        print('  The offset survives, so dimension is a real effect on top of the')
        print('  scale dependence and section B\'s open question stands.')
    print('  Separate exponents still differ (b_1D %.3f vs b_2D %.3f under A),'
          % (results['A']['b_1d'], results['A']['b_2d']))
    print('  which is the part a single pooled b cannot absorb.')

    if a.out:
        p = ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(fits=results,
                       cells=[{k: v for k, v in r.items()
                               if not k.startswith('rep_')} for r in recs]),
                  open(p, 'w'), indent=1)
        print('\nwrote %s' % p)


if __name__ == '__main__':
    main()
