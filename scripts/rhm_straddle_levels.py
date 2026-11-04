"""How many levels up the tree can a straddling window see?

A ``k x k`` window fully contains the aligned block of side ``b`` only when
``k >= 2b - 1``, so it can certify levels up to ``j* = floor(log_s((k+1)/2))``
by itself.  Everything above ``j*`` is invisible to a *block-aligned* reading of
the same window.  But a window centred near a block boundary straddles two
sibling blocks, and if the synonym rules make some boundary pairs impossible it
can tell that those two blocks do not descend from a common parent -- which is
information about level ``j* + 1``, ``j* + 2``, ... that the block-aligned
reading does not have.  ``rhm_straddle.py`` shows at ``L = 2`` that this signal
is real; the question here is how far up it carries, which needs ``L >= 3`` and
so cannot be answered by enumerating grids.

It can be answered without enumeration.  For each level ``j`` in
``{j*, ..., L}`` define the ensemble ``E_j``: tile the grid with *independent*
uniform draws from ``R_j``, the distinct leaf renderings of an aligned ``s**j``
block.  A grid from ``E_j`` is internally coherent up to level ``j`` and
scrambled above it; ``E_L`` is the grammar itself.  Let

    a_j = P( every k x k window of a grid drawn from E_j lies in W )

with ``W`` the machine's saturated pooled window set.  Then

* ``a_L = 1`` by construction -- a check that ``W`` really is saturated;
* ``a_{j*} < 1``  means the windows reject grids whose every aligned ``s**j*``
  block is legal, i.e. the straddle signal exists at all (one level up);
* ``a_{j*+1} < 1`` means they still reject grids that are already coherent one
  level higher: the window sees **two** levels up.

The profile ``a_{j*}, a_{j*+1}, ..., a_L = 1`` is how far the boundary signal
carries, and ``1 - a_j`` is its strength at each level.

At ``L = 2`` this reproduces the exact enumeration in ``rhm_straddle.py``:
``a_{j*}`` must come out at ``lv_pooled / lv_blocks``.  Pass ``--check`` with
that script's JSON and the comparison is run and printed.

Usage:
  python scripts/rhm_straddle_levels.py --v 2 3 --m 2 3 4 --L 3 --k 3 5 \
      --out results/.../straddle_levels_L3.json
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
from scripts.rhm_ceiling_exact import renderings, j_star
from scripts.rhm_straddle import window_codes


def reachable(model, depth):
    """Symbols that can occur ``depth`` levels below the root, which is uniform.

    ``renderings`` keys on every symbol in the alphabet, but a symbol the rules
    never produce renders blocks that no grammatical grid contains.  Tiling with
    those would make ``E_j`` grids rejected for a trivial reason -- an impossible
    block -- rather than for the boundary incoherence this script is measuring.
    (For the grammars swept here no symbol is ever unreachable, so this only
    guards the general case.)
    """
    cur = set(range(model.v))
    for rule in model.rules[:depth]:
        cur = set(int(x) for a in cur for x in np.unique(rule[a]))
    return cur


def block_set(model, j):
    """(|R_j|, b, b) int64 array of the distinct renderings of an aligned block."""
    b = model.s ** j
    per_symbol = renderings(model, j)
    live = reachable(model, model.L - j)
    words = sorted(set().union(*(per_symbol[a] for a in sorted(live))))
    return np.stack([np.frombuffer(w, dtype=np.int64).reshape(b, b) for w in words])


def tile(blocks, nb, n, device, generator):
    """(n, nb*b, nb*b) grids, each an independent uniform tiling of ``blocks``."""
    nw, b, _ = blocks.shape
    idx = torch.randint(nw, (n, nb, nb), device=device, generator=generator)
    g = blocks[idx]                                    # (n, nb, nb, b, b)
    return g.permute(0, 1, 3, 2, 4).reshape(n, nb * b, nb * b)


def window_set(model, k, device, n_sample, chunk, patience=8, seed=7):
    """Sorted unique window codes over grammatical grids, grown until it stops.

    Returns ``(codes, n_used, stalled)``.  Sampling stops early once ``patience``
    consecutive chunks add no new window, which is what makes a k=3 cell cost a
    few seconds and leaves the budget for the k=5 cells that actually need it.
    ``stalled`` records whether it stopped that way rather than running out of
    ``n_sample``; either way the honest test of saturation is ``a_L == 1``,
    since an incomplete ``W`` rejects grammatical grids.
    """
    seen, done, quiet = None, 0, 0
    while done < n_sample:
        take = min(chunk, n_sample - done)
        grids, _ = model.sample(take, seed=seed + done)
        c = window_codes(np.asarray(grids), k, model.v, device).reshape(-1)
        before = 0 if seen is None else seen.numel()
        seen = c.unique() if seen is None else torch.cat([seen, c]).unique()
        done += take
        quiet = quiet + 1 if seen.numel() == before else 0
        if quiet >= patience:
            return seen, done, True
    return seen, done, False


def accept(grids_iter, k, v, table, device):
    """Fraction of grids whose every window is in ``table``, and of windows that are."""
    ok = tot = win_ok = win_tot = 0
    for g in grids_iter:
        c = window_codes(g, k, v, device)
        i = torch.searchsorted(table, c.reshape(-1)).clamp(max=table.numel() - 1)
        hit = (table[i] == c.reshape(-1)).reshape(c.shape)
        ok += int(hit.all(-1).all(-1).sum())
        tot += c.shape[0]
        win_ok += int(hit.sum())
        win_tot += hit.numel()
    return ok / tot, win_ok / win_tot, tot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v', type=int, nargs='+', default=[2, 3])
    ap.add_argument('--s', type=int, default=2)
    ap.add_argument('--L', type=int, default=3)
    ap.add_argument('--m', type=int, nargs='+', default=[2, 3, 4])
    ap.add_argument('--k', type=int, nargs='+', default=[3, 5])
    ap.add_argument('--rule_seeds', type=int, default=5)
    ap.add_argument('--n_test', type=int, default=200000,
                    help='grids drawn per ensemble E_j')
    ap.add_argument('--n_wset', type=int, default=400000,
                    help='grammatical grids used to build the window set W')
    ap.add_argument('--chunk', type=int, default=50000)
    ap.add_argument('--wset_patience', type=int, default=8,
                    help='stop growing W after this many chunks add nothing')
    ap.add_argument('--max_block_words', type=float, default=3e5,
                    help='skip a level whose R_j is larger than this to enumerate')
    ap.add_argument('--check', type=str, default=None,
                    help='rhm_straddle.py JSON to cross-check a_{j*} against')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = torch.Generator(device=device).manual_seed(20261103)
    # window_codes packs a k*k window base (v+1) into one int64; past 2**63 the
    # codes collide silently and every acceptance rate below would be garbage.
    for v in args.v:
        for k in args.k:
            if (k * k - 1) * np.log(v + 1) >= 63 * np.log(2):
                raise SystemExit('v=%d k=%d: base-%d code of %d digits overflows int64'
                                 % (v, k, v + 1, k * k))
    d = args.s ** args.L
    ref = {}
    if args.check:
        for r in json.load(open(args.check)):
            ref[(r['v'], r['m'], r['k'], r['rule_seed'])] = r

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for v in args.v:
        for m in args.m:
            for rs in range(args.rule_seeds):
                model = RandomHierarchyModel2D(v=v, s=args.s, L=args.L, m=m, seed=1000 + rs)
                for k in args.k:
                    t0 = time.time()
                    js = j_star(k, args.s)
                    W, n_used, stalled = window_set(model, k, device, args.n_wset,
                                                    args.chunk, args.wset_patience)
                    nf = int(W.numel())

                    a, aw, skipped = {}, {}, []
                    for j in range(js, args.L + 1):
                        if j == args.L:
                            def it():
                                for lo in range(0, args.n_test, args.chunk):
                                    n = min(args.chunk, args.n_test - lo)
                                    g, _ = model.sample(n, seed=90000 + lo)
                                    yield torch.as_tensor(np.asarray(g), dtype=torch.int64,
                                                          device=device)
                        else:
                            nwords = v * m ** ((args.s ** (2 * j) - 1) // (args.s ** 2 - 1))
                            if nwords > args.max_block_words:
                                skipped.append(j)
                                continue
                            blocks = torch.as_tensor(block_set(model, j), device=device)
                            nb = d // (args.s ** j)

                            def it(blocks=blocks, nb=nb):
                                for lo in range(0, args.n_test, args.chunk):
                                    n = min(args.chunk, args.n_test - lo)
                                    yield tile(blocks, nb, n, device, gen)
                        a[j], aw[j], _ = accept(it(), k, v, W, device)

                    # a_L < 1 can only mean W is missing windows the grammar uses,
                    # so it is the saturation test; the stall is only a hint.
                    sat = a.get(args.L, 0.0) >= 0.9999
                    r = dict(v=v, s=args.s, L=args.L, m=m, k=k, rule_seed=rs, j_star=js,
                             n_test=args.n_test, n_wset_used=n_used, wset_stalled=stalled,
                             W_size=nf, W_saturated=bool(sat),
                             accept={str(j): a[j] for j in a},
                             accept_window={str(j): aw[j] for j in aw},
                             levels_skipped=skipped, seconds=round(time.time() - t0, 1))

                    g = ref.get((v, m, k, rs))
                    if g and js in a:
                        r['exact_a_jstar'] = g['lv_pooled'] / g['lv_blocks']
                        r['check_rel_err'] = abs(a[js] - r['exact_a_jstar']) / r['exact_a_jstar']

                    rows.append(r)
                    prof = '  '.join('a_%d=%.4f' % (j, a[j]) for j in sorted(a))
                    chk = ('  [exact %.4f, rel err %.3g]'
                           % (r['exact_a_jstar'], r['check_rel_err'])) if 'exact_a_jstar' in r else ''
                    print('v%d m%d k%d rs%d  j*=%d  |W|=%-8d (%.1fM grids)%s  %s%s  [%.0fs]'
                          % (v, m, k, rs, js, nf, n_used / 1e6,
                             '' if sat else ' UNSATURATED', prof, chk, r['seconds']),
                          flush=True)
                    json.dump(rows, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
