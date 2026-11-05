"""Separate the copying regime from the scaling peak WITHOUT an absolute P cut.

`rhm_depth2d_analysis.py --P_min 10` exists because 51244089's ladder runs down
to P=2 and finds a spike there that is not the peak the scaling law is about.
But an absolute floor is the wrong instrument: x* ~ 0.2, so a cell's peak sits
at P* ~ 0.2 |R_L|, and v=2 m=2 L=2 has |R_L| = 38..64, i.e. P* ~ 8..13.  The
floor lands ON the peak for exactly the cells the depth test needs, censors
them, and the censoring is |R_L|-dependent -- which is the same defect that
made `b_1D` read 0.510 in `rhm_xstar_scaling.py` until `--min_RL` was added.

The regime has a direct signature that does not mention P at all.  Among the
GRAMMATICAL outputs, what fraction are novel?

    nov_g    = success / grammatical          (measured)
    chance   = 1 - train_distinct / |R_L|     (a uniform draw from the grammar)
    ratio    = nov_g / chance

A machine that has learned the grammar spreads its grammatical mass over the
whole support, so ratio ~ 1.  A machine that is replaying its training set
concentrates on the few grids it was given: v4m2 rs3 at P=3 has grammatical
0.238 but success 0.0070, so 97% of its grammatical output is a training grid
when chance says 2% -- ratio 0.03.  Both numbers are already in every row of
every sweep, so this reclassifies the published data too, at no compute cost.

This script only measures the signature and reports where the cut would fall.
The cut itself goes into `rhm_depth2d_analysis.py --copy_min`.
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

from rhm_peak_noise import U1  # noqa: E402


def load_RL(paths):
    out = {}
    for p in paths:
        for f in glob.glob(str(p)):
            for r in json.load(open(f)):
                out[(r['v'], r['m'], r['s'], r['L'], r['rule_seed'])] = r['R_L']
    return out


def rows_at_U1(specs):
    """[(tag, glob)] -> {(tag, v, m, s, L, rs): {P: pooled row}} at k = U1."""
    grouped = defaultdict(lambda: defaultdict(list))
    for tag, pat in specs:
        for f in glob.glob(str(ROOT / pat)):
            for r in json.load(open(f)):
                if 'success' not in r or 'grammatical' not in r:
                    continue
                if r.get('block_align', 1) != 1 or not isinstance(r.get('k'), int):
                    continue
                if r['k'] != U1(r['s'], r['L']):
                    continue
                grouped[(tag, r['v'], r['m'], r['s'], r['L'], r['rule_seed'])][r['P']].append(r)

    out = {}
    for key, by_P in grouped.items():
        cell = {}
        for P, rs in by_P.items():
            n = sum(float(r.get('n_gen', r.get('n_samples', 4096))) for r in rs)
            succ = sum(r['success'] * float(r.get('n_gen', r.get('n_samples', 4096)))
                       for r in rs)
            gram = sum(r['grammatical'] * float(r.get('n_gen', r.get('n_samples', 4096)))
                       for r in rs)
            cell[P] = dict(n=n, succ=succ, gram=gram,
                           train_distinct=max(r.get('train_distinct', P) for r in rs))
        out[key] = cell
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', default='results/rhm_sampling/2026-11-03')
    ap.add_argument('--new', default='results/rhm_sampling/2026-11-05')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    RL = load_RL([ROOT / a.old / 'nk_exact_L*.json', ROOT / a.new / 'nk_exact_L*.json'])
    cells = rows_at_U1([
        ('L2new', a.new + '/depth2d_L2_v*_m2.json'),
        ('L2pub', a.old + '/P_v*_m2.json'),
        ('L3pub', a.old + '/L3peak_v*_k7.json'),
        ('L3pub', a.old + '/L3k7_v*_m*.json'),
        ('L3pub', a.new + '/L3seed_v*_m2_k7.json'),
    ])
    cells = {k: v for k, v in cells.items() if k[2] == 2 and k[1:6] in RL}
    print('cells: %d   (m=2, k=U1, block_align=1)' % len(cells))

    # ---- 1. the signature, pooled over seeds so one table shows the shape ----
    print()
    print('--- 1. novelty among grammatical outputs, pooled over rule seeds ---')
    print('ratio = (success/grammatical) / (1 - train_distinct/|R_L|).')
    print('ratio ~ 1 means grammatical output is spread over the support like a')
    print('uniform draw; ratio << 1 means it is concentrated on the training set.')
    rows_out = []
    for tag, L in (('L2new', 2), ('L2pub', 2), ('L3pub', 3)):
        for v in (2, 3, 4):
            sel = {k: c for k, c in cells.items() if k[0] == tag and k[1] == v and k[4] == L}
            if not sel:
                continue
            print()
            print('  %s  v=%d  L=%d   (%d seeds, |R_L| %s)'
                  % (tag, v, L, len(sel),
                     '/'.join(str(RL[k[1:6]]) for k in sorted(sel))))
            print('  %10s %9s %9s %9s %9s %8s'
                  % ('P', 'gram', 'succ', 'nov|gram', 'chance', 'ratio'))
            allP = sorted({P for c in sel.values() for P in c})
            for P in allP:
                n = g = s = 0.0
                ch = []
                for k, c in sel.items():
                    if P not in c:
                        continue
                    r = c[P]
                    n += r['n']
                    g += r['gram']
                    s += r['succ']
                    ch.append(1.0 - min(r['train_distinct'], RL[k[1:6]]) / RL[k[1:6]])
                if g <= 0:
                    continue
                nov_g = s / g
                chance = float(np.mean(ch))
                ratio = nov_g / chance if chance > 0 else np.nan
                print('  %10g %9.4f %9.4f %9.4f %9.4f %8.3f'
                      % (P, g / n, s / n, nov_g, chance, ratio))
                rows_out.append(dict(tag=tag, v=v, L=L, P=float(P), n=n,
                                     gram=g / n, succ=s / n, nov_g=nov_g,
                                     chance=chance, ratio=ratio))

    # ---- 2. where would a cut land, cell by cell ----
    print()
    print('--- 2. per cell: the argmax with no cut, and under candidate cuts ---')
    print('%-7s %3s %3s %4s %8s %8s %10s %10s %10s'
          % ('src', 'v', 'L', 'rs', '|R_L|', '0.2|R_L|', 'argmax raw',
             'argmax P>=10', 'argmax r>=.5'))
    per_cell = []
    for key in sorted(cells):
        tag, v, m, s_, L, rs = key
        c = cells[key]
        R = RL[key[1:6]]
        Ps = np.array(sorted(c), float)
        succ = np.array([c[P]['succ'] / c[P]['n'] for P in sorted(c)])
        gram = np.array([c[P]['gram'] / c[P]['n'] for P in sorted(c)])
        td = np.array([min(c[P]['train_distinct'], R) for P in sorted(c)], float)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = (succ / np.where(gram > 0, gram, np.nan)) / (1.0 - td / R)

        def amax(mask):
            if not mask.any():
                return float('nan')
            idx = np.where(mask)[0]
            return float(Ps[idx[int(np.argmax(succ[idx]))]])

        raw = amax(np.ones_like(Ps, bool))
        p10 = amax(Ps >= 10)
        rcut = amax(np.nan_to_num(ratio, nan=0.0) >= 0.5)
        print('%-7s %3d %3d %4d %8d %8.1f %10g %10g %10g'
              % (tag, v, L, rs, R, 0.2 * R, raw, p10, rcut))
        per_cell.append(dict(tag=tag, v=v, L=L, rule_seed=rs, R_L=R,
                             argmax_raw=raw, argmax_P10=p10, argmax_ratio=rcut))

    print()
    print('--- reading ---')
    bad10 = [r for r in per_cell if r['L'] == 2 and np.isfinite(r['argmax_raw'])
             and 0.2 * r['R_L'] < 10]
    print('L=2 cells whose expected peak 0.2|R_L| sits BELOW the absolute P>=10')
    print('floor: %d of %d -- for these the floor is not a nuisance cut, it'
          % (len(bad10), sum(1 for r in per_cell if r['L'] == 2)))
    print('removes the peak itself.')
    moved = [r for r in per_cell
             if np.isfinite(r['argmax_raw']) and np.isfinite(r['argmax_ratio'])
             and r['argmax_ratio'] != r['argmax_raw']]
    print('cells where the copy cut moves the argmax at all: %d of %d'
          % (len(moved), len(per_cell)))

    if a.out:
        p = ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(pooled=rows_out, cells=per_cell), open(p, 'w'), indent=1)
        print('\nwrote %s' % p)


if __name__ == '__main__':
    main()
