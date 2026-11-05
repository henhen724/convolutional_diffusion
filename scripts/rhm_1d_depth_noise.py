"""Does 1D's depth effect in x* survive the peak estimator, as the 2D scatter did not?

`rhm_peak_noise.py` showed that the whole across-seed scatter in 2D x* is noise
in the peak estimator, and that the published 3-point parabola has per-cell
bootstrap biases up to 2.3x.  The 1D "x* falls with depth in 15/15 (v,m)"
finding is the reason `P* = Theta(|R_L|)` is scoped to 2D, and it was measured
with the same estimator on curves that are *sparser* than the 2D ones.  So it
has to face the same test before it is leaned on any further.

A 15/15 count is not itself evidence against noise, either: the estimator's
bias depends on where the peak sits on the ladder, and P* grows with depth by
construction, so a bias that drifts along the ladder would produce a consistent
direction across all 15 combos without any real depth dependence.  What
separates the two is whether the effect survives an estimator with different
bias (B) and whether the per-combo intervals clear 1 once estimator noise is
propagated.

Everything here is the 1D sweep (`results/rhm_1d/2026-11-03/sweep.json`), U=1
windows only (k = 2s^(L-1) - 1), block_align 1, n_samples 2048 throughout, and
|R_L| is carried in the rows.  Unlike the 2D L=3 files there are no duplicate
ladder points to pool.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from rhm_peak_noise import U1, est_A, est_B, geo_sd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', default='results/rhm_1d/2026-11-03/sweep.json')
    ap.add_argument('--boot', type=int, default=400)
    ap.add_argument('--frac', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--L_hi', type=int, default=None,
                    help='deeper side of the comparison; default is each '
                         "combo's deepest run (L=3 or 4). Set 3 for the "
                         'ONE-step ratio, which is what 2D can be compared '
                         'against -- 2D only has L=2 and L=3, so a 1D ratio '
                         'that spans two steps overstates the gap.')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    rows = json.load(open(ROOT / a.sweep))
    cur = defaultdict(list)
    for r in rows:
        if r.get('block_align', 1) != 1:
            continue
        if r['k'] != U1(r['s'], r['L']):
            continue
        cur[(r['v'], r['m'], r['L'], r['rule_seed'])].append(r)

    rng = np.random.default_rng(a.seed)
    recs, reps = [], {}
    for key in sorted(cur):
        v, m, L, rs = key
        cs = sorted(cur[key], key=lambda r: r['P'])
        P = np.array([c['P'] for c in cs], float)
        S = np.array([c['success'] for c in cs], float)
        ng = float(cs[0]['n_samples'])
        R = float(cs[0]['R_L'])
        pa, pb = est_A(P, S, ng), est_B(P, S, ng, a.frac)
        ba = np.empty(a.boot)
        bb = np.empty(a.boot)
        for j in range(a.boot):
            Sb = rng.binomial(int(ng), np.clip(S, 0, 1)) / ng
            ba[j] = est_A(P, Sb, ng)
            bb[j] = est_B(P, Sb, ng, a.frac)
        recs.append(dict(v=v, m=m, L=L, rule_seed=rs, R_L=R, n_P=len(P),
                         x_A=(pa / R) if np.isfinite(pa) else float('nan'),
                         x_B=(pb / R) if np.isfinite(pb) else float('nan'),
                         boot_sd_A=geo_sd(ba), boot_sd_B=geo_sd(bb)))
        reps[key] = (ba / R, bb / R)

    print('1D U=1 curves scored: %d' % len(recs))
    print('median per-cell estimator noise:  A %.2fx   B %.2fx'
          % (np.nanmedian([r['boot_sd_A'] for r in recs]),
             np.nanmedian([r['boot_sd_B'] for r in recs])))

    def med(v, m, L, w):
        xs = [r['x_A'] if w == 'A' else r['x_B'] for r in recs
              if r['v'] == v and r['m'] == m and r['L'] == L]
        xs = [x for x in xs if np.isfinite(x)]
        return (float(np.median(xs)), len(xs)) if xs else (float('nan'), 0)

    def ratio_ci(v, m, Llo, Lhi, w, nb):
        """Nested bootstrap on median x*(Llo) / median x*(Lhi)."""
        pick = lambda L: [reps[k][0 if w == 'A' else 1]
                          for k in reps if k[:3] == (v, m, L)]
        A_, B_ = pick(Llo), pick(Lhi)
        if not A_ or not B_:
            return (np.nan,) * 3
        draws = []
        for _ in range(nb):
            vals = []
            for cells in (A_, B_):
                idx = rng.integers(0, len(cells), len(cells))
                got = []
                for i in idx:
                    f = cells[i][np.isfinite(cells[i])]
                    if len(f):
                        got.append(f[rng.integers(0, len(f))])
                vals.append(np.median(got) if got else np.nan)
            if np.isfinite(vals[0]) and np.isfinite(vals[1]) and vals[1] > 0:
                draws.append(vals[0] / vals[1])
        if len(draws) < 20:
            return (np.nan,) * 3
        d = np.array(draws)
        return (float(np.median(d)), float(np.percentile(d, 2.5)),
                float(np.percentile(d, 97.5)))

    combos = sorted({(r['v'], r['m']) for r in recs})
    print()
    print('--- x* against depth, within each (v, m) ---')
    print('ratio = x*(L=2) / x*(L=max); > 1 means x* FALLS with depth (the 1D claim)')
    print('%-8s %-18s %-18s %-24s %-24s'
          % ('(v,m)', 'A: L=2 / L=max', 'B: L=2 / L=max',
             'A ratio [95% CI]', 'B ratio [95% CI]'))
    tally = {'A': [0, 0], 'B': [0, 0]}   # [falls, CI excludes 1 upward]
    out_rows = []
    for v, m in combos:
        Ls = sorted({r['L'] for r in recs if r['v'] == v and r['m'] == m})
        lo = Ls[0]
        hi = a.L_hi if (a.L_hi is not None and a.L_hi in Ls) else Ls[-1]
        if hi == lo:
            continue
        cells = {}
        for w in ('A', 'B'):
            a2, n2 = med(v, m, lo, w)
            a3, n3 = med(v, m, hi, w)
            r, cl, ch = ratio_ci(v, m, lo, hi, w, a.boot)
            cells[w] = (a2, n2, a3, n3, r, cl, ch)
            if np.isfinite(r) and r > 1:
                tally[w][0] += 1
            if np.isfinite(cl) and cl > 1:
                tally[w][1] += 1
        fa, fb = cells['A'], cells['B']
        fmt = lambda t: '%.4f/%.4f' % (t[0], t[2])
        cif = lambda t: ('%.2fx [%.2f, %.2f]' % (t[4], t[5], t[6])
                         if np.isfinite(t[4]) else '-')
        print('v%dm%-5d %-18s %-18s %-24s %-24s'
              % (v, m, fmt(fa), fmt(fb), cif(fa), cif(fb)))
        out_rows.append(dict(v=v, m=m, L_lo=lo, L_hi=hi,
                             A=dict(zip(('x_lo', 'n_lo', 'x_hi', 'n_hi',
                                         'ratio', 'ci_lo', 'ci_hi'), fa)),
                             B=dict(zip(('x_lo', 'n_lo', 'x_hi', 'n_hi',
                                         'ratio', 'ci_lo', 'ci_hi'), fb))))

    n = len(combos)
    print()
    print('--- reading ---')
    for w in ('A', 'B'):
        print('  %s: x* falls with depth in %d/%d combos; CI excludes 1 in %d/%d'
              % (w, tally[w][0], n, tally[w][1], n))
    if tally['B'][0] >= n - 1 and tally['B'][1] >= n // 2:
        print('  The 1D depth effect is NOT an estimator artifact: it holds under')
        print('  an estimator with different bias, and most per-combo intervals')
        print('  clear 1 with estimator noise propagated.')
    elif tally['B'][0] < n - 2:
        print('  The direction does not survive estimator B. The 15/15 count was')
        print('  reading A\'s ladder-dependent bias, not the grammar, and the 1D')
        print('  half of the dimension story needs reopening.')
    else:
        print('  Direction survives but the intervals mostly cover 1: consistent')
        print('  with a real effect, not established per combo at this power.')

    if a.out:
        p = ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(cells=recs, combos=out_rows, tally=tally),
                  open(p, 'w'), indent=1)
        print('\nwrote %s' % p)


if __name__ == '__main__':
    main()
