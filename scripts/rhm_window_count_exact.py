"""Saturated window-table size ``N_k``, counted exactly over the whole grammar.

With the ceiling at ``U = 1`` now measured to be ``Theta(1)`` rather than
``f0^U`` (``rhm_straddle.py``), the sampling complexity is set by the other leg:
how much data it takes to fill the machine's window table. That is coupon
collection over ``N_k``, the number of *distinct* ``k x k`` windows the grammar
can produce -- so the scaling law needs ``N_k`` itself, saturated.

Sampling cannot supply it at depth: the pooled ``N_7`` at ``L=3`` is still
climbing at ``P = 3e5`` (2.5e6 to 1.2e7 distinct windows, job 50895001). But the
grammar is finite, so ``N_k`` can be counted exactly wherever ``|R_L| = v m^((s^(2L)-1)/(s^2-1))``
is enumerable: ``v=2, m=2, L=3`` is 4.2e6 grids and ``v=3, m=2, L=3`` is 6.3e6.

Enumeration is by odometer over the word choice at every internal node, built
with the same block assembly as ``RandomHierarchyModel2D.sample`` so the two
cannot drift apart. Windows are packed into a pair of int64 keys -- base
``v + 1`` over ``k*k`` digits overflows one word at ``k=7`` -- and deduplicated
by sorting, chunked so the pass fits in memory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.random_hierarchy import RandomHierarchyModel2D


def enumerate_grammar(model, lo, hi, device):
    """Grids ``lo..hi`` of the grammar, in odometer order over the word choices.

    Index layout, from the root down: the root symbol varies slowest, then one
    base-``m`` digit per internal node, level by level. Every grid the grammar
    can emit appears exactly once, though not necessarily distinctly -- rules
    are drawn with replacement, so two word choices can render alike.
    """
    v, s, m, L = model.v, model.s, model.m, model.L
    idx = torch.arange(lo, hi, device=device, dtype=torch.int64)
    n = hi - lo
    # nodes per level: 1, s^2, s^4, ... the last level's words emit the leaves
    counts = [(s ** (2 * l)) for l in range(L)]
    total_nodes = sum(counts)
    digits = torch.empty(n, total_nodes, dtype=torch.int64, device=device)
    for p in range(total_nodes):
        digits[:, p] = idx % m
        idx = idx // m
    root = idx % v

    grid = root[:, None, None]
    at = 0
    for l in range(L):
        rule = torch.as_tensor(model.rules[l], device=device)  # (v, m, s, s)
        _, h, w = grid.shape
        wi = digits[:, at:at + h * w].reshape(n, h, w)
        at += h * w
        words = rule[grid, wi]                                  # (n, h, w, s, s)
        grid = words.permute(0, 1, 3, 2, 4).reshape(n, h * s, w * s)
    return grid


def pack_windows(grids, k, v, device):
    """(N * npos, 2) int64 keys, one per k x k window, framed in the marker ``v``."""
    n, d, _ = grids.shape
    pad = k // 2
    framed = torch.full((n, d + k - 1, d + k - 1), v, dtype=torch.int64, device=device)
    framed[:, pad:pad + d, pad:pad + d] = grids
    w = framed.unfold(1, k, 1).unfold(2, k, 1).reshape(-1, k * k)
    # base v+1 in two halves: one int64 cannot hold (v+1)**49
    half = (k * k + 1) // 2
    base = v + 1
    p_lo = base ** torch.arange(half, device=device, dtype=torch.int64)
    p_hi = base ** torch.arange(k * k - half, device=device, dtype=torch.int64)
    return torch.stack([(w[:, :half] * p_lo).sum(1), (w[:, half:] * p_hi).sum(1)], dim=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v', type=int, nargs='+', default=[2, 3])
    ap.add_argument('--s', type=int, default=2)
    ap.add_argument('--L', type=int, default=3)
    ap.add_argument('--m', type=int, nargs='+', default=[2])
    ap.add_argument('--k', type=int, nargs='+', default=[3, 5, 7])
    ap.add_argument('--rule_seeds', type=int, default=3)
    ap.add_argument('--rule_seed_start', type=int, default=0,
                   help='first rule seed; --rule_seeds stays the count. Matches '
                        'rhm_sampling_complexity so the two stay in step when a '
                        'published sweep is extended. Default 0 is unchanged.')
    ap.add_argument('--max_grids', type=float, default=2e7)
    ap.add_argument('--chunk', type=int, default=1 << 16)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for v in args.v:
        for m in args.m:
            n_nodes = sum((args.s ** (2 * l)) for l in range(args.L))
            total = v * m ** n_nodes
            if total > args.max_grids:
                print(f'v={v} m={m} L={args.L}: {total:.3g} grids > --max_grids, skipped',
                      flush=True)
                continue
            for rs in range(args.rule_seed_start, args.rule_seed_start + args.rule_seeds):
                model = RandomHierarchyModel2D(v=v, s=args.s, L=args.L, m=m, seed=1000 + rs)
                for k in args.k:
                    t0 = time.time()
                    seen = torch.zeros(0, 2, dtype=torch.int64, device=device)
                    gram = set()
                    for lo in range(0, total, args.chunk):
                        g = enumerate_grammar(model, lo, min(lo + args.chunk, total), device)
                        gram.update(map(bytes, g.to(torch.int8).cpu().numpy()))
                        keys = pack_windows(g, k, v, device)
                        # merge into the running set each chunk: keeping every
                        # window instance would be |R_L| * d^2 rows at once
                        seen = torch.unique(torch.cat([seen, keys], 0), dim=0)
                    r = dict(v=v, s=args.s, L=args.L, m=m, k=k, rule_seed=rs,
                             n_grids=int(total), R_L=len(gram), N_k=int(seen.shape[0]),
                             seconds=round(time.time() - t0, 1))
                    rows.append(r)
                    print(f"v={v} m={m} L={args.L} k={k} rs={rs}: |R_L|={r['R_L']} "
                          f"(of {total} word choices) N_k={r['N_k']} ({r['seconds']}s)",
                          flush=True)
                    json.dump(rows, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
