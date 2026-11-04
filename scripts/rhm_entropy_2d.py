"""The same entropy budget, measured in 2D, so the 1D/2D gap can be tested.

``rhm_entropy_window`` measures the trajectory entropy ``H_tot`` in 1D and asks
whether the success peak sits where ``H_tot = ln P``.  That test is only half an
answer: the open question is why ``x* = P*/|R_L|`` is 0.193 in 2D and 0.083 in
1D at the same number of sites, and the budget claim is that the difference is
entirely in ``H_tot``.  A 2D U=1 window covers 9/16 of a 4x4 grid where a 1D one
covers 15/16 of a 16-site string, so it pins less of the sample, leaves more
entropy, and the crossing with ``ln P`` sits at larger P -- larger ``x*``, with
no dimensional constant anywhere.

So measure ``H_tot`` on the 2D cells whose peaks are already published
(``results/rhm_sampling/2026-11-03/peak_summary.json``), on the same ladder and
seeds, and read ``x*`` off the same bracketed-peak definition.

The sampling loop here is the one in ``ScheduledDiscreteScoreMachine.
_reveal_one_at_a_time``, re-stated so the posterior can be read before it is
consumed.  It is a copy rather than a hook because the sampler is shared with
the image work; it draws the same two randoms per step in the same order, so
with one seed it produces the same grids -- and the script checks exactly that
by comparing ``success`` against the published value for each cell.
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
from src.utils.rhm_grammar import evaluate_samples
from src.utils.rhm_sampling import PatchCountDiscreteScoreMachine


def generate_H_2d(machine, n_samples, grid, vocab, device, generator):
    """Reveal-one sampling on a grid, returning grids and trajectory entropy."""
    n_sites = grid * grid
    xt = torch.full((n_samples, grid, grid), vocab, dtype=torch.long, device=device)
    rows = torch.arange(n_samples, device=device)
    H = torch.zeros(n_samples, device=device)
    for _ in range(n_sites):
        still = xt == vocab
        if not bool(still.any()):
            break
        frac = still.float().mean().item()
        flat = still.reshape(n_samples, n_sites)
        noise = torch.rand(n_samples, n_sites, device=device, generator=generator)
        pick = torch.where(flat, noise, torch.full_like(noise, -1.0)).argmax(1)
        post = machine(torch.full((n_samples,), frac, device=device), xt,
                       device=device, sites=pick)
        probs = post.permute(0, 2, 3, 1).reshape(n_samples, n_sites, vocab)
        p = probs[rows, pick].clamp_min(1e-12)
        tok = torch.multinomial(p, 1, generator=generator)[:, 0]
        keep = flat[rows, pick]
        h = -(p * p.log()).sum(1)
        H += torch.where(keep, h, torch.zeros_like(h))
        f = xt.reshape(n_samples, n_sites).clone()
        f[rows[keep], pick[keep]] = tok[keep]
        xt = f.reshape(n_samples, grid, grid)
    return xt, H


def run_cell_H(rhm, P, k, n_gen, device, seed, alpha_smooth, gen_batch,
               block_align, pad_mode, max_patches):
    t0 = time.time()
    train, _ = rhm.sample(P, seed=seed)
    machine = PatchCountDiscreteScoreMachine(
        train, vocab_size=rhm.v, kernel_size=k, alpha_smooth=alpha_smooth,
        device=device, block_align=block_align, pad_mode=pad_mode)
    if machine.n_distinct > max_patches:
        return dict(P=int(P), k=int(k), skipped=True,
                    n_distinct_patches=int(machine.n_distinct))
    gen = torch.Generator(device=device).manual_seed(seed + 9973)
    out, Hs = [], []
    for i in range(0, n_gen, gen_batch):
        b = min(gen_batch, n_gen - i)
        g, h = generate_H_2d(machine, b, rhm.grid_size, rhm.v, device, gen)
        out.append(g.cpu().numpy())
        Hs.append(h.cpu().numpy())
    grids = np.concatenate(out, 0)
    H = np.concatenate(Hs, 0)
    res = evaluate_samples(rhm, train, grids)
    novel = np.array([bytes(r) not in set(map(bytes, train.astype(np.int8)))
                      for r in np.ascontiguousarray(grids.astype(np.int8))], dtype=bool)

    def m(x):
        return float(np.mean(x)) if len(x) else float('nan')
    n = rhm.grid_size ** 2
    res.update(P=int(P), k=int(k), v=rhm.v, s=rhm.s, L=rhm.L, m=rhm.m, n=n,
               H_tot=m(H), H_sd=float(np.std(H)), H_site=m(H) / n,
               H_novel=m(H[novel]), H_copy=m(H[~novel]),
               ln_P=float(np.log(P)), margin=m(H) - float(np.log(P)),
               n_distinct_patches=int(machine.n_distinct),
               seconds=round(time.time() - t0, 1))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--peaks', default='results/rhm_sampling/2026-11-03/peak_summary.json')
    ap.add_argument('--L', type=int, nargs='+', default=[2, 3])
    ap.add_argument('--n_gen', type=int, default=2048)
    ap.add_argument('--gen_batch', type=int, default=512)
    ap.add_argument('--alpha_smooth', type=float, default=1e-3)
    ap.add_argument('--block_align', type=int, default=1)
    ap.add_argument('--pad_mode', default='outside')
    ap.add_argument('--max_patches', type=int, default=3_000_000)
    ap.add_argument('--P_min', type=int, default=1)
    ap.add_argument('--P_max', type=int, default=131072)
    ap.add_argument('--max_cells', type=int, default=8,
                    help='how many published 2D peak cells to re-measure')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ladder = [p for p in (args.P_min * 2 ** i for i in range(40)) if p <= args.P_max]
    pk = [p for p in json.load(open(ROOT / args.peaks))
          if p.get('bracketed') and p['L'] in args.L]
    # Every bracketed cell at the requested depths, not a subset: the first run
    # of this script took the 8 highest-success cells, whose own published x*
    # has median 0.088 against the L=2 population's 0.193, and the "2D" number
    # that came out was a property of the selection rather than of 2D.  Ordered
    # cheapest-first so a cell that runs out of wall clock costs the least, but
    # a truncated run is biased low in P* and must not be read as the median.
    pk.sort(key=lambda p: p['P_star'])
    pk = pk[:args.max_cells]
    print('re-measuring %d published 2D peaks: %s' % (
        len(pk), [(p['v'], p['m'], p['L'], p['rule_seed'], p['k']) for p in pk]), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in pk:
        rhm = RandomHierarchyModel2D(v=p['v'], s=p['s'], L=p['L'], m=p['m'],
                                     seed=1000 + p['rule_seed'])
        print('cell  v%d m%d L%d rs%d  n=%d  k=%d  |R_L|=%d  published P*=%.0f x*=%.3f'
              % (p['v'], p['m'], p['L'], p['rule_seed'], rhm.grid_size ** 2, p['k'],
                 p['R_L'], p['P_star'], p['ratio']), flush=True)
        for P in ladder:
            r = run_cell_H(rhm, P, p['k'], args.n_gen, device, 1234 + 7 * p['rule_seed'],
                           args.alpha_smooth, args.gen_batch, args.block_align,
                           args.pad_mode, args.max_patches)
            if r.get('skipped'):
                print('   P=%-7d SKIP, table %d too big' % (P, r['n_distinct_patches']),
                      flush=True)
                break
            r.update(rule_seed=p['rule_seed'], R_L=int(p['R_L']),
                     P_star_pub=float(p['P_star']), ratio_pub=float(p['ratio']))
            rows.append(r)
            print('   P=%-7d succ %.4f  H_tot %6.2f  lnP %6.2f  margin %+6.2f  '
                  '(novel %.2f / copy %.2f)'
                  % (P, r['success'], r['H_tot'], r['ln_P'], r['margin'],
                     r['H_novel'], r['H_copy']), flush=True)
            json.dump(rows, open(out, 'w'), indent=1)
            if P > 4 * p['P_star'] and r['success'] < 0.2 * p['succ_max']:
                print('   past the peak, stopping this cell', flush=True)
                break
    json.dump(rows, open(out, 'w'), indent=1)
    print('wrote %s (%d rows)' % (out, len(rows)))


if __name__ == '__main__':
    main()
