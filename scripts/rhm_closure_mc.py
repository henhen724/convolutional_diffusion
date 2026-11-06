"""Closure by Monte Carlo, so the straddling signal can be read at any v.

``rhm_straddle.py`` gets the locally-valid counts by enumerating every one of
``v ** (s**L)**2`` grids.  That caps out at v=3 (43M grids); v=4 is 4.3e9 and
the peak population runs to v=8.  So the exact route cannot reach the cells
whose ``x*`` we actually measured, and the question "does the straddling check
set the sampling complexity?" cannot be asked.

It can be asked by importance sampling instead.  Everything but one count is
already cheap:

    lv_blocks = |R_j*| ** n_blocks        analytic (the enumeration confirms it)
    |R_L|                                 small -- 1944 at v=8 m=3, 31104 at v=4 m=6

so the only hard term is ``lv_pooled``, and it does not need the whole grid
space: draw grids uniformly from the coupon set (each aligned ``s**j*`` block
independently uniform over ``R_j*``) and measure the fraction whose every k x k
window lies in the pooled window set.  That estimates

    p = |lv_pooled AND lv_blocks| / lv_blocks

with a binomial error, and

    closure = log(1/p) / log(lv_blocks / |R_L|)

is the same scale-free quantity as in rhm_straddle_synonyms.py: the share of
the independent-blocks coupon slack the straddling windows remove.

The estimator assumes lv_pooled is contained in lv_blocks, which is not
guaranteed a priori -- a pooled window may match a training window from
another phase.  ``--validate`` settles it empirically rather than by argument:
it reruns the 75 cells rhm_straddle.py enumerated exactly and compares.  If the
containment fails anywhere the MC closure is biased and the main mode's numbers
should not be used.

Usage:
    python3 scripts/rhm_closure_mc.py --validate --n_mc 2000000
    python3 scripts/rhm_closure_mc.py --cells_from results/.../peak_noise.json \
        --n_mc 4000000 --out results/.../closure_peakpop.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.random_hierarchy import RandomHierarchyModel2D
from scripts.rhm_ceiling_exact import renderings, j_star
from scripts.rhm_straddle import grammatical_grids, window_codes


def member(codes, table):
    i = torch.searchsorted(table, codes.reshape(-1)).clamp(max=len(table) - 1)
    return (table[i] == codes.reshape(-1)).reshape(codes.shape)


def closure_cell(v, s, L, m, k, rs, n_mc, chunk, device, gen):
    """Monte Carlo closure for one (v, s, L, m, k, rule seed) cell."""
    d = s ** L
    model = RandomHierarchyModel2D(v=v, s=s, L=L, m=m, seed=1000 + rs)
    gram = grammatical_grids(model)
    R_L = gram.shape[0]

    js = j_star(k, s)
    b, nb = s ** js, d // (s ** js)
    Rj = sorted(set().union(*renderings(model, js).values()))
    blocks = torch.as_tensor(
        np.stack([np.frombuffer(x, dtype=np.int64).reshape(b, b) for x in Rj]),
        dtype=torch.int64, device=device)                        # (R_j, b, b)
    lv_blocks = float(len(Rj)) ** (nb * nb)

    # |R_L| reaches 6.3M at the L=3 cells and k=7 gives 64 windows each, so the
    # code array alone would be 3 GB in one shot; fold it down incrementally.
    pool = torch.empty(0, dtype=torch.int64, device=device)
    gstep = max(1, int(4e7 // (k * k)))
    for lo in range(0, R_L, gstep):
        c = window_codes(gram[lo:lo + gstep], k, v, device).reshape(-1)
        pool = torch.unique(torch.cat([pool, torch.unique(c)]))

    hits = 0
    done = 0
    while done < n_mc:
        n = min(chunk, n_mc - done)
        pick = torch.randint(len(Rj), (n, nb, nb), device=device, generator=gen)
        g = blocks[pick]                                          # (n, nb, nb, b, b)
        g = g.permute(0, 1, 3, 2, 4).reshape(n, d, d)
        c = window_codes(g, k, v, device)
        hits += int(member(c, pool).all(-1).all(-1).sum())
        done += n

    p = hits / n_mc
    # Wilson interval: at m=2 the hit rate runs to ~0.5, at m=6 it is 1.0, and
    # a normal interval would run past 1 there and report a closure below 0.
    z = 1.959963985
    den = 1 + z * z / n_mc
    cen = (p + z * z / (2 * n_mc)) / den
    half = z * np.sqrt(p * (1 - p) / n_mc + z * z / (4 * n_mc ** 2)) / den
    slack = np.log(lv_blocks) - np.log(R_L)

    def clo(q):
        if q <= 0 or slack <= 0:
            return float('nan')
        return float(-np.log(q) / slack)

    return dict(v=v, s=s, L=L, m=m, k=k, rule_seed=rs, R_L=int(R_L), R_j=len(Rj),
                n_blocks=nb * nb, lv_blocks=lv_blocks, n_mc=int(n_mc), hits=int(hits),
                p_hat=p, lv_pooled_mc=p * lv_blocks,
                closure_mc=clo(p),
                closure_lo=clo(min(cen + half, 1.0)), closure_hi=clo(max(cen - half, 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--cells_from', default=None)
    ap.add_argument('--n_mc', type=int, default=2_000_000)
    ap.add_argument('--chunk', type=int, default=200_000)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = torch.Generator(device=device); gen.manual_seed(20261106)
    print('device %s, n_mc %d' % (device, a.n_mc), flush=True)

    rows = []
    if a.validate:
        exact = []
        for f in sorted(glob.glob(str(ROOT / 'results/rhm_sampling/*/straddle_v*.json'))):
            exact += [r for r in json.load(open(f)) if 'lv_pooled' in r]
        print('--- validation against %d exactly enumerated cells ---' % len(exact))
        print('%-22s %12s %12s %8s %9s %9s'
              % ('cell', 'lv_pooled', 'MC estimate', 'ratio', 'closure', 'MC'))
        worst = 0.0
        for e in exact:
            r = closure_cell(e['v'], e['s'], e['L'], e['m'], e['k'], e['rule_seed'],
                             a.n_mc, a.chunk, device, gen)
            # exact closure, same definition, from the enumerated counts
            r['lv_pooled_exact'] = e['lv_pooled']
            slack = np.log(e['lv_blocks']) - np.log(e['R_L'])
            r['closure_exact'] = ((np.log(e['lv_blocks']) - np.log(e['lv_pooled'])) / slack
                                  if slack > 0 else float('nan'))
            ratio = r['lv_pooled_mc'] / e['lv_pooled']
            r['ratio_mc_exact'] = ratio
            worst = max(worst, abs(np.log(ratio)))
            print('%-22s %12d %12.1f %8.4f %9.3f %9.3f'
                  % ('v%d m%d k%d rs%d' % (e['v'], e['m'], e['k'], e['rule_seed']),
                     e['lv_pooled'], r['lv_pooled_mc'], ratio,
                     r['closure_exact'], r['closure_mc']), flush=True)
            rows.append(r)
        # 3 SE on the worst-case hit rate, as the bar the agreement has to clear
        bar = 3 * np.sqrt(0.25 / a.n_mc) / 0.25
        print('\nworst |log(MC/exact)| = %.5f;  3 SE at the least favourable rate = %.5f'
              % (worst, bar))
        print('%s' % ('PASS -- lv_pooled is contained in lv_blocks and the estimator is '
                      'unbiased.' if worst < bar else
                      'FAIL -- containment does not hold; MC closure is biased, do not use it.'))
    else:
        if not a.cells_from:
            raise SystemExit('need --validate or --cells_from')
        src = json.load(open(ROOT / a.cells_from))
        src = src['cells'] if isinstance(src, dict) else src
        seen, cells = set(), []
        for r in src:
            key = (r['v'], r['s'], r['L'], r['m'], r['k'], r['rule_seed'])
            if key not in seen:
                seen.add(key); cells.append(key)
        print('--- %d cells from %s ---' % (len(cells), a.cells_from))
        print('%-24s %9s %9s %12s %9s %-16s'
              % ('cell', '|R_L|', '|R_j*|', 'lv_blocks', 'closure', '[95% CI]'))
        for (v, s, L, m, k, rs) in cells:
            t = time.time()
            r = closure_cell(v, s, L, m, k, rs, a.n_mc, a.chunk, device, gen)
            r['seconds'] = time.time() - t
            print('%-24s %9d %9d %12.3g %9.3f  [%.3f, %.3f]   %.0fs'
                  % ('v%d s%d L%d m%d k%d rs%d' % (v, s, L, m, k, rs),
                     r['R_L'], r['R_j'], r['lv_blocks'], r['closure_mc'],
                     r['closure_lo'], r['closure_hi'], r['seconds']), flush=True)
            rows.append(r)
            if a.out:
                o = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
                o.parent.mkdir(parents=True, exist_ok=True)
                json.dump(rows, open(o, 'w'), indent=1)

    if a.out:
        o = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
        o.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(o, 'w'), indent=1)
        print('\nwrote %s' % o)


if __name__ == '__main__':
    main()
