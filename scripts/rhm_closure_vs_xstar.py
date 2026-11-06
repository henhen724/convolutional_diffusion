"""Does the straddling check set the sampling complexity?

Section E2 established that ``closure`` -- the share of the independent-blocks
coupon slack that the real LS machine's straddling windows remove -- is near 1
when synonyms are few and falls to 0 as they multiply.  ``rhm_closure_mc.py``
then put a closure on every cell of the measured peak population, validated
against 75 exact enumerations.  So the mechanism question can finally be asked
directly: is ``x* = P*/|R_L|`` set by how much the straddling windows see?

Mechanism: a machine whose locally-valid set is tighter (closure -> 1) has
fewer ungrammatical configurations available to it at any P, so it should
start emitting grammatical output at a smaller P, and the novel-and-
grammatical peak should sit at a smaller x*.  Hence x* falls as closure rises.

WHY THIS POPULATION CAN TEST IT.  closure is not a relabelling of the design,
because the two ladders in the data move it in opposite directions:

    v = 4, m = 2..6   (14 cells)   closure FALLS  0.96 -> 0.53
    m = 3, v = 2..8   (14 cells)   closure RISES  0.41 -> 1.00

so "x* tracks closure" is a different claim from "x* tracks v" or "x* tracks
m", and the two ladders disagree about which.

REGISTERED BEFORE FITTING (2026-11-06)
  1. Pooled over all cells, the slope of log x* on closure is NEGATIVE, with a
     bootstrap CI excluding 0, for both peak estimators.
  2. The crossing test, and the one that decides it: that slope is negative in
     BOTH ladders taken separately.  If closure only works pooled, it is doing
     the job of v or m and nothing more.
  3. closure adds >= 0.05 R^2 over (log v, log m, log |R_L|) jointly, against a
     permutation null built by shuffling closure within each ladder.

If 1 holds but 2 fails, the honest reading is that this is confounded and the
mechanism is not established.

Usage:
    python3 scripts/rhm_closure_vs_xstar.py --out results/.../closure_xstar.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(20261106)
KEY = ('v', 's', 'L', 'm', 'k', 'rule_seed')


def load(peaks, clo):
    by = {tuple(r[f] for f in KEY): r for r in json.load(open(ROOT / clo))}
    rows = []
    for p in json.load(open(ROOT / peaks)):
        c = by.get(tuple(p[f] for f in KEY))
        if c is None:
            continue
        for w in 'AB':
            x = p.get('x_%s' % w)
            if x is None or not np.isfinite(x) or x <= 0:
                continue
            rows.append(dict(est=w, x=float(x), closure=c['closure_mc'],
                             v=p['v'], m=p['m'], L=p['L'], k=p['k'],
                             R_L=p['R_L'], rule_seed=p['rule_seed']))
    return rows


def slope_ci(x, y, n_boot=4000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or len(np.unique(x)) < 2:
        return np.nan, (np.nan, np.nan), len(x)
    b = np.polyfit(x, y, 1)[0]
    boot = []
    for _ in range(n_boot):
        i = RNG.integers(0, len(x), len(x))
        if len(np.unique(x[i])) < 2:
            continue
        boot.append(np.polyfit(x[i], y[i], 1)[0])
    if len(boot) < 100:
        return b, (np.nan, np.nan), len(x)
    return b, (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))), len(x)


def r2(X, y):
    X = np.column_stack([np.ones(len(y))] + list(X))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    tot = ((y - y.mean()) ** 2).sum()
    return 1 - (resid ** 2).sum() / tot if tot > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--peaks', default='results/rhm_sampling/2026-11-05/peak_noise.json')
    ap.add_argument('--closure', default='results/rhm_sampling/2026-11-05/closure_peakpop.json')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    rows = load(a.peaks, a.closure)
    out = {'fits': {}, 'n_rows': len(rows)}
    print('%d (cell, estimator) observations with both x* and closure\n'
          % len(rows))

    print('--- registered 1: pooled slope of log x* on closure ---')
    print('%-10s %6s %9s %-22s %s' % ('estimator', 'cells', 'slope', '[95% CI]', 'verdict'))
    for w in 'AB':
        sel = [r for r in rows if r['est'] == w]
        b, (lo, hi), n = slope_ci([r['closure'] for r in sel],
                                  np.log([r['x'] for r in sel]))
        v = ('x* FALLS as closure rises' if np.isfinite(hi) and hi < 0 else
             'x* rises as closure rises' if np.isfinite(lo) and lo > 0 else 'covers 0')
        print('%-10s %6d %9.3f  [%6.3f, %6.3f]   %s' % (w, n, b, lo, hi, v))
        out['fits']['pooled_%s' % w] = dict(slope=b, ci=[lo, hi], n=n)
    print()

    ladders = {'v=4, m ladder': lambda r: r['v'] == 4 and r['L'] == 2,
               'm=3, v ladder': lambda r: r['m'] == 3 and r['L'] == 2}
    print('--- registered 2 (the crossing test): same slope, each ladder alone ---')
    print('%-16s %-4s %6s %9s %-22s %9s %s'
          % ('ladder', 'est', 'cells', 'slope', '[95% CI]', 'clo range', 'verdict'))
    for name, f in ladders.items():
        for w in 'AB':
            sel = [r for r in rows if r['est'] == w and f(r)]
            if len(sel) < 3:
                continue
            cl = [r['closure'] for r in sel]
            b, (lo, hi), n = slope_ci(cl, np.log([r['x'] for r in sel]))
            v = ('negative' if np.isfinite(hi) and hi < 0 else
                 'positive' if np.isfinite(lo) and lo > 0 else 'covers 0')
            print('%-16s %-4s %6d %9.3f  [%6.3f, %6.3f]   %.2f-%.2f  %s'
                  % (name, w, n, b, lo, hi, min(cl), max(cl), v))
            out['fits']['%s_%s' % (name.split(',')[0].replace('=', ''), w)] = dict(
                slope=b, ci=[lo, hi], n=n, ladder=name)
    print()

    print('--- registered 3: does closure add over the design variables? ---')
    print('%-10s %8s %8s %8s %s' % ('estimator', 'R2 base', 'R2 +clo', 'dR2', 'perm null p95'))
    for w in 'AB':
        sel = [r for r in rows if r['est'] == w]
        y = np.log([r['x'] for r in sel])
        base = [np.log([r['v'] for r in sel]), np.log([r['m'] for r in sel]),
                np.log([r['R_L'] for r in sel])]
        cl = np.array([r['closure'] for r in sel])
        r_base, r_full = r2(base, y), r2(base + [cl], y)
        # shuffle closure within each ladder, so the null keeps whatever
        # closure shares with v and m and destroys only its own information
        lab = np.array(['v4' if r['v'] == 4 and r['L'] == 2 else
                        'm3' if r['m'] == 3 and r['L'] == 2 else 'other' for r in sel])
        null = []
        for _ in range(2000):
            sh = cl.copy()
            for g in np.unique(lab):
                i = np.flatnonzero(lab == g)
                sh[i] = RNG.permutation(sh[i])
            null.append(r2(base + [sh], y) - r_base)
        p95 = float(np.percentile(null, 95))
        print('%-10s %8.3f %8.3f %8.3f   %.3f%s'
              % (w, r_base, r_full, r_full - r_base, p95,
                 '   <- clears' if r_full - r_base > max(p95, 0.05) else '   <- does not clear'))
        out['fits']['dR2_%s' % w] = dict(base=r_base, full=r_full,
                                         d=r_full - r_base, null_p95=p95)
    print()

    print('--- reading ---')
    pooled = [out['fits']['pooled_%s' % w] for w in 'AB']
    cross = [v for k, v in out['fits'].items() if 'ladder' in str(v.get('ladder', ''))]
    pooled_neg = all(np.isfinite(f['ci'][1]) and f['ci'][1] < 0 for f in pooled)
    cross_neg = cross and all(np.isfinite(f['ci'][1]) and f['ci'][1] < 0 for f in cross)
    if pooled_neg and cross_neg:
        print('Registered 1 and 2 both hold: x* falls as closure rises, pooled and')
        print('in each ladder separately, though v and m move closure oppositely')
        print('along them.  The straddling check is a mechanism for the (v, m)')
        print('spread in x*, not a relabelling of the design.')
    elif pooled_neg:
        print('Registered 1 holds but 2 does not: the pooled slope is carried by')
        print('one ladder, so closure cannot be separated from v and m here.')
        print('Confounded -- no mechanism established.')
    else:
        adds = [out['fits']['dR2_%s' % w] for w in 'AB']
        collinear = all(f['d'] <= max(f['null_p95'], 0.05) for f in adds)
        print('Registered 1 fails, and in the opposite direction: the pooled slope')
        print('is POSITIVE (%.2f A / %.2f B), i.e. x* is larger where the machine'
              % (pooled[0]['slope'], pooled[1]['slope']))
        print('sees MORE.  That alone would be a finding, but registered 3 says it')
        if collinear:
            print('is not one: closure adds only %.3f / %.3f R^2 over (log v, log m,'
                  % (adds[0]['d'], adds[1]['d']))
            print('log |R_L|), inside the permutation null.  So the positive slope is')
            print('collinearity -- closure is a proxy for the design variables, not an')
            print('independent driver.  The straddling check does NOT set the sampling')
            print('complexity, in either direction.  Mechanism refuted.')
        else:
            print('closure does add R^2 over the design (%.3f / %.3f), so the positive'
                  % (adds[0]['d'], adds[1]['d']))
            print('direction is real and needs an explanation -- it is the reverse of')
            print('the registered mechanism, not a null.')

    if a.out:
        o = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
        o.parent.mkdir(parents=True, exist_ok=True)
        out['rows'] = rows
        json.dump(out, open(o, 'w'), indent=1)
        print('\nwrote %s' % o)


if __name__ == '__main__':
    main()
