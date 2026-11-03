"""Exact locally-valid set of a k x k LS machine, by enumerating every grid.

The ceiling used so far, ``C(k) = |R_L| / |R_j*| ** n_blocks``, counts a grid as
locally valid when each *aligned* ``s**j*`` block is a legal rendering. That
treats the blocks as independent coupons, which is an assumption, not a fact: a
window that straddles a block boundary sees the right edge of one block and the
left edge of the next, and if the synonym rules make such boundary pairs rare
the window can tell that the two blocks descend from different parent words.
Such a grid is locally invalid even though every one of its aligned blocks is
legal, so the real locally-valid set is smaller and the real ceiling higher.

This script measures the real one. At ``L = 2`` the grid has ``(s**2)**2 = 16``
sites, so the whole space ``v**16`` can be enumerated and each grid tested
against the machine's actual window set -- no formula, no sampling. Three sets
are counted on the same enumeration:

* ``blocks``  -- every aligned ``s**j*`` block is in ``R_j*``. This is the
  independent-blocks assumption, and the count must come out at exactly
  ``|R_j*| ** n_blocks``; it is kept as a check that the pipeline is right.
* ``aligned`` -- every ``k x k`` window is in the training window set *for its
  own phase*, i.e. what ``block_align = s`` enforces.
* ``pooled``  -- every ``k x k`` window is in the pooled window set, i.e. what
  the translation-invariant machine (``block_align = 1``, the real LS machine)
  enforces.

``blocks >= aligned`` measures what the straddling windows buy; ``pooled >=
aligned`` measures what pooling the phases costs. The ceiling of each machine
with unlimited data is ``|R_L| / |locally valid|``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.random_hierarchy import RandomHierarchyModel2D
from scripts.rhm_ceiling_exact import renderings, j_star


def grammatical_grids(model):
    """(|R_L|, d, d) int8 array of every grid the grammar can emit."""
    d = model.s ** model.L
    allg = sorted(set().union(*renderings(model, model.L).values()))
    return np.stack([np.frombuffer(b, dtype=np.int64).reshape(d, d) for b in allg]).astype(np.int8)


def window_codes(grids, k, v, device):
    """(N, d, d) int64 code of the k x k window predicting each site, base v+1.

    Grids are framed in the off-grid marker ``v`` first (``pad_mode='outside'``),
    so border sites get a window too and it records where the edge is.
    """
    g = torch.as_tensor(grids, dtype=torch.int64, device=device)
    n, d, _ = g.shape
    pad = k // 2
    framed = torch.full((n, d + k - 1, d + k - 1), v, dtype=torch.int64, device=device)
    framed[:, pad:pad + d, pad:pad + d] = g
    # unfold gives one window per predicted site, since the frame is k//2 wide
    w = framed.unfold(1, k, 1).unfold(2, k, 1).reshape(n, d, d, k * k)
    powers = (v + 1) ** torch.arange(k * k, device=device, dtype=torch.int64)
    return (w * powers).sum(-1)


def enumerate_chunk(lo, hi, v, d, device):
    """(hi - lo, d, d) int8 array of grids lo..hi in base-v odometer order."""
    idx = torch.arange(lo, hi, device=device, dtype=torch.int64)
    digits = torch.empty(hi - lo, d * d, dtype=torch.int64, device=device)
    for p in range(d * d):
        digits[:, p] = idx % v
        idx = idx // v
    return digits.reshape(-1, d, d)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v', type=int, nargs='+', default=[2, 3])
    ap.add_argument('--s', type=int, default=2)
    ap.add_argument('--L', type=int, default=2)
    ap.add_argument('--m', type=int, nargs='+', default=[2, 3, 4])
    ap.add_argument('--k', type=int, nargs='+', default=[3, 5])
    ap.add_argument('--rule_seeds', type=int, default=3)
    ap.add_argument('--max_enum', type=float, default=2e8,
                    help='skip a grammar whose grid space exceeds this')
    ap.add_argument('--chunk', type=int, default=1 << 21)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = args.s ** args.L
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for v in args.v:
        total = v ** (d * d)
        if total > args.max_enum:
            print(f'v={v}: {total:.3g} grids > --max_enum, skipped', flush=True)
            continue
        for m in args.m:
            for rs in range(args.rule_seeds):
                model = RandomHierarchyModel2D(v=v, s=args.s, L=args.L, m=m, seed=1000 + rs)
                gram = grammatical_grids(model)
                RL = gram.shape[0]

                for k in args.k:
                    js = j_star(k, args.s)
                    b = args.s ** js
                    nb = d // b
                    Rj = sorted(set().union(*renderings(model, js).values()))
                    Rj_set = torch.zeros(v ** (b * b), dtype=torch.bool, device=device)
                    bpow = v ** torch.arange(b * b, device=device, dtype=torch.int64)
                    for byt in Rj:
                        blk = torch.as_tensor(
                            np.frombuffer(byt, dtype=np.int64).reshape(b * b).copy(),
                            device=device)
                        Rj_set[int((blk * bpow).sum())] = True

                    # The machine's window set, pooled and split by phase, held as
                    # sorted codes rather than a dense boolean table: (v+1)**(k*k)
                    # is 4**25 already at v=3, k=5, far past what can be allocated.
                    gc = window_codes(gram, k, v, device)
                    pool = torch.unique(gc.reshape(-1))
                    rr = torch.arange(d, device=device)
                    pid = ((rr % args.s)[:, None] * args.s + (rr % args.s)[None, :])
                    phase = [torch.unique(gc[:, pid == p].reshape(-1))
                             for p in range(args.s * args.s)]

                    def member(codes, table):
                        i = torch.searchsorted(table, codes.reshape(-1)).clamp(max=len(table) - 1)
                        return (table[i] == codes.reshape(-1)).reshape(codes.shape)

                    n_blocks = n_align = n_pool = 0
                    for lo in range(0, total, args.chunk):
                        hi = min(lo + args.chunk, total)
                        g = enumerate_chunk(lo, hi, v, d, device)

                        blk = g.reshape(-1, nb, b, nb, b).permute(0, 1, 3, 2, 4).reshape(-1, nb, nb, b * b)
                        ok_b = Rj_set[(blk * bpow).sum(-1)].all(-1).all(-1)
                        n_blocks += int(ok_b.sum())

                        c = window_codes(g, k, v, device)
                        n_pool += int(member(c, pool).all(-1).all(-1).sum())
                        ok_a = torch.ones(c.shape[0], dtype=torch.bool, device=device)
                        for p in range(args.s * args.s):
                            ok_a &= member(c[:, pid == p], phase[p]).all(-1)
                        n_align += int(ok_a.sum())

                    pred = len(Rj) ** (nb * nb)
                    r = dict(v=v, s=args.s, L=args.L, m=m, k=k, rule_seed=rs, j_star=js,
                             R_L=RL, R_j=len(Rj), n_grids=int(total),
                             lv_blocks=n_blocks, lv_blocks_pred=pred,
                             lv_aligned=n_align, lv_pooled=n_pool,
                             ceil_blocks=RL / pred,
                             ceil_aligned=RL / n_align if n_align else float('nan'),
                             ceil_pooled=RL / n_pool if n_pool else float('nan'),
                             straddle_gain=pred / n_align if n_align else float('nan'),
                             pool_cost=n_pool / n_align if n_align else float('nan'))
                    rows.append(r)
                    print(f"v={v} m={m} k={k} rs={rs}: |R_L|={RL} |R_j*|={len(Rj)} "
                          f"LV blocks={n_blocks} (pred {pred}) aligned={n_align} "
                          f"pooled={n_pool} | straddle gain x{r['straddle_gain']:.3f}, "
                          f"pool cost x{r['pool_cost']:.3f}", flush=True)
                    json.dump(rows, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
