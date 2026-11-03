#!/usr/bin/env python
"""Exact grammaticality ceiling of a k x k local-score machine on the RHM.

A ``k x k`` window centred on a site reaches only ``k // 2``, so it contains the
aligned block of side ``b`` through that site only when ``k >= 2b - 1``. Let

    j* = floor(log_s((k + 1) / 2))

be the deepest level the window can police. Everything the machine can check is
"every aligned ``s**j*`` block is a legal leaf rendering of *some* symbol"; the
rules above ``j*`` are invisible to it. So however much data it is given, the
best it can do is sample from the *locally valid* set

    LV = { grids whose every aligned s**j* block lies in R_j* }

where ``R_j`` is the set of distinct leaf renderings of an aligned ``s**j``
block. The grammar's own support is ``R_L``, so the ceiling on grammaticality is

    C(k) = |R_L| / |R_j*| ** (s ** (2 * (L - j*)))                        (1)

This script computes (1) exactly, by enumerating ``R_j`` level by level with the
actual (collision-prone) production rules, and checks it against the closed form

    C(k) = f0 ** U,  f0 = m / v**(s**2 - 1),  U = (s**(2*(L-j*)) - 1)/(s**2 - 1)

which (1) reduces to when no two drawn words coincide. The closed form is what
the measured plateaus in ``rhm_sampling_complexity.py`` are compared against;
this is its derivation, run as code so the algebra is checked rather than
asserted.

``R_j`` is enumerated in full, so keep ``v * m**((s**(2j)-1)/(s**2-1))`` small:
``L <= 2`` at ``v, m <= 8``, or ``L = 3`` only at ``m = 1``.

Usage:
  python scripts/rhm_ceiling_exact.py --v 2 3 4 6 8 --m 2 3 4 5 --L 2 --k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.random_hierarchy import RandomHierarchyModel2D


def renderings(model, j):
    """{symbol: set of distinct leaf renderings} for an aligned s**j block.

    Level indexing follows ``model.rules``: ``rules[-1]`` is the rule applied
    last, i.e. the one that emits leaf tokens, so a depth-``j`` subtree uses the
    final ``j`` rule levels.
    """
    s = model.s
    # depth 0: the block is a single site holding the symbol itself
    cur = {a: {np.array([[a]], dtype=np.int64).tobytes()} for a in range(model.v)}
    size = 1
    for depth in range(1, j + 1):
        rules = model.rules[model.L - depth]
        nxt = {}
        for a in range(model.v):
            out = set()
            for w in range(model.m):
                word = rules[a, w]  # (s, s) of child symbols
                blocks = [[sorted(cur[int(word[di, dj])]) for dj in range(s)]
                          for di in range(s)]
                # cartesian product over the s*s child slots
                acc = [np.zeros((size * s, size * s), dtype=np.int64)]
                for di in range(s):
                    for dj in range(s):
                        nxt_acc = []
                        for grid in acc:
                            for b in blocks[di][dj]:
                                g = grid.copy()
                                g[di * size:(di + 1) * size,
                                  dj * size:(dj + 1) * size] = \
                                    np.frombuffer(b, dtype=np.int64).reshape(size, size)
                                nxt_acc.append(g)
                        acc = nxt_acc
                out.update(g.tobytes() for g in acc)
            nxt[a] = out
        cur, size = nxt, size * s
    return cur


def distinct_renderings(model, j):
    """|R_j|: distinct leaf renderings of an aligned s**j block, any symbol."""
    per_symbol = renderings(model, j)
    return len(set().union(*per_symbol.values()))


def j_star(k, s):
    j = 0
    while 2 * s ** (j + 1) - 1 <= k:
        j += 1
    return j


def ceiling_mc(model, j, n_draw, seed=0):
    """Ceiling by Monte Carlo, for depths where ``R_L`` cannot be enumerated.

    ``R_L`` is a subset of the locally valid set, so the exact ratio (1) is just
    ``P(grammatical)`` when each aligned ``s**j`` block is drawn uniformly and
    independently from ``R_j`` -- which needs only ``R_j``, not ``R_L``.
    """
    from src.utils.rhm_grammar import is_grammatical

    pool = np.stack([np.frombuffer(b, dtype=np.int64).reshape(model.s ** j, model.s ** j)
                     for b in sorted(set().union(*renderings(model, j).values()))])
    nb = model.grid_size // model.s ** j
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < n_draw:
        take = min(4096, n_draw - done)
        idx = rng.integers(0, len(pool), size=(take, nb, nb))
        grids = pool[idx].transpose(0, 1, 3, 2, 4).reshape(take, model.grid_size, model.grid_size)
        hits += int(np.asarray(is_grammatical(model, grids)).sum())
        done += take
    return hits / n_draw, len(pool)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--v', type=int, nargs='+', default=[2, 3, 4, 6, 8])
    p.add_argument('--s', type=int, default=2)
    p.add_argument('--L', type=int, nargs='+', default=[2])
    p.add_argument('--m', type=int, nargs='+', default=[2, 3, 4, 5])
    p.add_argument('--k', type=int, nargs='+', default=[3])
    p.add_argument('--rule_seeds', type=int, default=3)
    p.add_argument('--mc', type=int, default=0,
                   help='draws for the Monte Carlo ceiling; needed when R_L is too big')
    p.add_argument('--enumerate_top', action='store_true',
                   help='also enumerate R_L for the exact ratio (feasible only for small L)')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    s, q = args.s, args.s ** 2 - 1
    rows = []
    for v in args.v:
        for L in args.L:
            for m in args.m:
                for rs in range(args.rule_seeds):
                    model = RandomHierarchyModel2D(v=v, s=s, L=L, m=m, seed=1000 + rs)
                    # R_L is enumerable only at small depth; R_j for j < L always is.
                    # j* == L is the full-window case: the ceiling is exactly 1
                    # (and novelty exactly 0), so nothing needs enumerating.
                    j_max = min(L - 1, max(min(j_star(k, s), L) for k in args.k))
                    rj = {j: distinct_renderings(model, j) for j in range(j_max + 1)}
                    if args.enumerate_top and L not in rj:
                        rj[L] = distinct_renderings(model, L)
                    for k in args.k:
                        j = min(j_star(k, s), L)
                        nblocks = s ** (2 * (L - j))
                        U = (s ** (2 * (L - j)) - 1) // q
                        closed = (m / v ** q) ** U
                        if j >= L:  # full window: samples are training grids verbatim
                            exact, mc, pool = 1.0, 1.0, rj.get(L)
                        else:
                            exact = rj[L] / rj[j] ** nblocks if L in rj else None
                            mc, pool = (ceiling_mc(model, j, args.mc, seed=77 + rs)
                                        if args.mc else (None, None))
                        ideal = {jj: v * m ** ((s ** (2 * jj) - 1) // q) for jj in rj}
                        rows.append(dict(v=v, s=s, L=L, m=m, k=k, rule_seed=rs, j_star=j,
                                         U=U, R=rj, R_ideal=ideal, pool=pool,
                                         ceiling_exact=exact, ceiling_closed=closed,
                                         ceiling_mc=mc, mc_draws=args.mc,
                                         collisions=any(rj[jj] != ideal[jj] for jj in rj)))
                        both = (f"exact={exact:.6g} " if exact is not None else "") + \
                               (f"mc={mc:.6g} " if mc is not None else "")
                        print(f"v={v} s={s} L={L} m={m} k={k} rs={rs}: j*={j} U={U} "
                              f"|R|={[rj[jj] for jj in sorted(rj)]} "
                              f"(ideal {[ideal[jj] for jj in sorted(ideal)]}) "
                              f"ceiling {both}closed={closed:.6g}", flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(out, 'w'), indent=1)
        print('wrote', out)


if __name__ == '__main__':
    main()
