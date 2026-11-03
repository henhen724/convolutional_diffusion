#!/usr/bin/env python
"""How many distinct ``k x k`` windows does the RHM produce, and how fast?

The LS machine's posterior is an empirical conditional indexed by the contents
of a ``k x k`` window. It can only be right once the training set has filled in
that table, so the machine's sampling complexity is governed by

    N_k = number of distinct k x k windows the grammar can emit,

and by the coupon-collector time to see them all. ``N_k`` is far cheaper to
measure than generation -- it needs no sampler and no scoring -- so this script
pushes ``P`` several orders of magnitude further than
``rhm_sampling_complexity.py`` can, and gives the scaling law its backbone.

The prediction to test. An *aligned* block of side ``s**j`` is the leaf
rendering of a node ``j`` levels above the leaves, and the number of distinct
renderings is

    A_j = v * m**((s**(2j) - 1) / (s**2 - 1))

(one of ``v`` symbols on top, then ``m`` choices at each of
``1 + s^2 + ... + s^(2(j-1))`` nodes below it). A ``k x k`` window is not
aligned, so ``N_k`` interpolates between ``A_j`` values, but it must scale the
same way: linearly in ``v`` and as a power of ``m`` set by how many rule
choices fit inside the window.

Usage:
  python scripts/rhm_patch_counts.py --v 4 --s 2 --L 3 --m 2 3 4 5 --k 3 5 \
      --P 100 1000 10000 100000 --out results/rhm_sampling/counts.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.random_hierarchy import RandomHierarchyModel2D


def _pack(windows: np.ndarray, base: int) -> np.ndarray:
    """Pack (N, k*k) small ints into the fewest int64 columns that hold them."""
    n, digits = windows.shape
    per_col = 1
    while base ** (per_col + 1) < 2 ** 62:
        per_col += 1
    cols = []
    for start in range(0, digits, per_col):
        chunk = windows[:, start:start + per_col].astype(np.int64)
        weights = base ** np.arange(chunk.shape[1], dtype=np.int64)
        cols.append(chunk @ weights)
    return np.stack(cols, axis=1)


def distinct_window_curve(model, k, p_values, pad_mode, chunk=2000, seed=0):
    """N_k after each P in ``p_values``, streaming so large P stays in memory."""
    seen = set()
    curve, drawn = [], 0
    base = model.v + 1 if pad_mode == 'outside' else model.v
    for target in sorted(p_values):
        while drawn < target:
            take = min(chunk, target - drawn)
            grids, _ = model.sample(take, seed=seed + drawn)
            if pad_mode == 'outside':
                d = k // 2
                grids = np.pad(grids, ((0, 0), (d, k - 1 - d), (d, k - 1 - d)),
                               constant_values=model.v)
            w = np.lib.stride_tricks.sliding_window_view(grids, (k, k), axis=(1, 2))
            w = w.reshape(-1, k * k)
            packed = _pack(w, base)
            seen.update(map(tuple, np.unique(packed, axis=0).tolist()))
            drawn += take
        curve.append((target, len(seen)))
    return curve


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--v', type=int, nargs='+', default=[4])
    p.add_argument('--s', type=int, default=2)
    p.add_argument('--L', type=int, nargs='+', default=[3])
    p.add_argument('--m', type=int, nargs='+', default=[3])
    p.add_argument('--k', type=int, nargs='+', default=[3, 5])
    p.add_argument('--P', type=int, nargs='+', required=True)
    p.add_argument('--pad_mode', choices=['zeros', 'outside'], default='outside')
    p.add_argument('--rule_seeds', type=int, default=3)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in args.v:
        for L in args.L:
            for m in args.m:
                for rs in range(args.rule_seeds):
                    model = RandomHierarchyModel2D(v=v, s=args.s, L=L, m=m, seed=1000 + rs)
                    for k in args.k:
                        if k > model.grid_size and args.pad_mode == 'zeros':
                            continue
                        t0 = time.time()
                        curve = distinct_window_curve(model, k, args.P, args.pad_mode,
                                                      seed=50_000 * (rs + 1))
                        rows.append(dict(v=v, s=args.s, L=L, m=m, k=k, rule_seed=rs,
                                         pad_mode=args.pad_mode, curve=curve,
                                         seconds=round(time.time() - t0, 1)))
                        print(f"v={v} s={args.s} L={L} m={m} k={k} rs={rs}: "
                              f"N_k={[c[1] for c in curve]} at P={[c[0] for c in curve]} "
                              f"({rows[-1]['seconds']}s)", flush=True)
                        json.dump(rows, open(out, 'w'), indent=1)
    json.dump(rows, open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
