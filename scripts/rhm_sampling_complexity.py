#!/usr/bin/env python
"""Sampling complexity of the local score machine on the Random Hierarchy Model.

hshunt's success criterion: the LS machine is sampling the RHM successfully when
the grids it generates are **not in the training set** and **obey every grammar
rule**. This sweeps the training-set size ``P`` and reports both, so the smallest
``P`` at which they hold together can be read off and its scaling in the RHM
parameters ``(v, s, L, m)`` -- and in the machine's window ``k`` -- measured.

Each (P, k) cell draws a fresh training set from *fixed* production rules, builds
the patch-count machine from it, generates ``n_gen`` grids by reverse erasure
diffusion, and scores them with ``src.utils.rhm_grammar``. Rule randomness is
averaged over ``--rule_seeds`` independent rule draws, since a single random
grammar can be unusually easy or hard.

Usage:
  python scripts/rhm_sampling_complexity.py --v 4 --L 3 --m 3 --k 3 5 \
      --P 10 30 100 300 1000 --out results/rhm/sweep.json
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

from src.utils.discrete_score import ScheduledDiscreteScoreMachine, cosine_mask_schedule, linear_mask_schedule
from src.utils.random_hierarchy import RandomHierarchyModel2D
from src.utils.rhm_grammar import evaluate_samples
from src.utils.rhm_sampling import (AdaptiveWindowDiscreteScoreMachine,
                                    PatchCountDiscreteScoreMachine)


def run_cell(rhm, P, k, n_gen, nsteps, device, seed, schedule, alpha_smooth, gen_batch,
             block_align, pad_mode, max_patches, k_adapt=None, h_min=None, h_mode='match'):
    """One (P, k) measurement: build the machine on P grids, generate, score.

    With ``k_adapt`` the window is not fixed: the machine holds one table per
    listed ``k`` and picks, per site and per step, the largest whose posterior
    entropy stays at or above ``h_min`` nats.
    """
    train_grids, _ = rhm.sample(P, seed=seed)
    t0 = time.time()
    if k_adapt:
        machine = AdaptiveWindowDiscreteScoreMachine(
            train_grids, vocab_size=rhm.v, kernel_sizes=k_adapt, h_min=h_min, h_mode=h_mode,
            alpha_smooth=alpha_smooth, device=device, block_align=block_align,
            pad_mode=pad_mode,
        )
    else:
        machine = PatchCountDiscreteScoreMachine(
            train_grids, vocab_size=rhm.v, kernel_size=k, alpha_smooth=alpha_smooth,
            device=device, block_align=block_align, pad_mode=pad_mode,
        )
    if machine.n_distinct > max_patches:
        return {'P': int(P), 'k': 'adapt' if k_adapt else int(k), 'skipped': True,
                'n_distinct_patches': machine.n_distinct,
                'n_patches_total': machine.n_patches_total}
    sampler = ScheduledDiscreteScoreMachine(
        machine, vocab_size=rhm.v, grid_size=rhm.grid_size,
        default_time_steps=nsteps, mask_schedule=schedule, mask_token=rhm.v, reveal_one=True,
    )
    gen = torch.Generator(device=device).manual_seed(seed + 9973)
    out = []
    for i in range(0, n_gen, gen_batch):
        b = min(gen_batch, n_gen - i)
        out.append(sampler.sample(n_samples=b, device=device, generator=gen).cpu().numpy())
    grids = np.concatenate(out, axis=0)

    res = evaluate_samples(rhm, train_grids, grids)
    res.update(P=int(P), k='adapt' if k_adapt else int(k), n_distinct_patches=machine.n_distinct,
               n_patches_total=machine.n_patches_total, seconds=round(time.time() - t0, 1),
               train_distinct=int(len(set(map(bytes, train_grids.astype(np.int8))))))
    if k_adapt:
        # k(t) as the entropy floor actually chose it, binned by masked fraction,
        # so the schedule can be read off rather than assumed
        tr = machine.trace
        bins = {}
        for frac, kbar in tr:
            bins.setdefault(round(frac, 1), []).append(kbar)
        res.update(k='adapt', k_set=list(k_adapt), h_min=float(h_min), h_mode=h_mode,
                   k_mean=float(np.mean([kb for _, kb in tr])) if tr else float('nan'),
                   k_schedule=sorted((f, round(float(np.mean(b)), 3)) for f, b in bins.items()))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--v', type=int, default=4)
    p.add_argument('--s', type=int, default=2)
    p.add_argument('--L', type=int, default=3)
    p.add_argument('--m', type=int, default=3)
    p.add_argument('--k', type=int, nargs='+', default=[3])
    p.add_argument('--P', type=int, nargs='+', required=True)
    p.add_argument('--n_gen', type=int, default=256)
    p.add_argument('--gen_batch', type=int, default=256)
    p.add_argument('--nsteps', type=int, default=None, help='default: 2 * number of sites')
    p.add_argument('--rule_seeds', type=int, default=3)
    p.add_argument('--alpha_smooth', type=float, default=1e-3)
    p.add_argument('--schedule', choices=['linear', 'cosine'], default='linear')
    p.add_argument('--block_align', type=int, nargs='+', default=[1],
                   help='1 = translation-invariant (the repo LS machine); s = phase-aware')
    p.add_argument('--pad_mode', choices=['zeros', 'outside'], default='outside',
                   help="'zeros' reproduces LocalDiscreteScoreMachine; 'outside' tells the "
                        'machine where the grid edge is, which it otherwise cannot see')
    p.add_argument('--k_adapt', type=int, nargs='+', default=None,
                   help='window sizes the adaptive machine may choose between, per site and '
                        'per step; when given, --k is ignored and --h_min is swept instead')
    p.add_argument('--h_min', type=float, nargs='+', default=[0.0],
                   help='posterior-entropy floor in nats for --k_adapt. The largest window '
                        'staying above it is used; 0 means always the largest that matched')
    p.add_argument('--h_mode', choices=['token', 'match'], nargs='+', default=['match'],
                   help="which entropy the floor applies to: 'token' = the predicted-token "
                        "posterior, 'match' = the posterior over which training patch matched")
    p.add_argument('--max_patches', type=int, default=400_000,
                   help='skip cells whose patch table exceeds this, to bound GPU time')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    schedule = linear_mask_schedule if args.schedule == 'linear' else cosine_mask_schedule
    nsteps = args.nsteps if args.nsteps is not None else 2 * (args.s ** args.L) ** 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rs in range(args.rule_seeds):
        rhm = RandomHierarchyModel2D(v=args.v, s=args.s, L=args.L, m=args.m, seed=1000 + rs)
        # with --k_adapt the swept axis is the entropy floor, not the window
        axis = ([(None, h, hmd) for hmd in args.h_mode for h in args.h_min]
                if args.k_adapt else [(k, None, None) for k in args.k])
        for ba in args.block_align:
            for k, hm, hmd in axis:
                for P in args.P:
                    r = run_cell(rhm, P, k, args.n_gen, nsteps, device, 7_000_000 + 1000 * rs + P,
                                 schedule, args.alpha_smooth, args.gen_batch, ba,
                                 args.pad_mode, args.max_patches,
                                 k_adapt=args.k_adapt, h_min=hm, h_mode=hmd)
                    r.update(v=args.v, s=args.s, L=args.L, m=args.m, rule_seed=rs, nsteps=nsteps,
                             alpha_smooth=args.alpha_smooth, schedule=args.schedule,
                             n_gen=args.n_gen, block_align=ba, pad_mode=args.pad_mode)
                    rows.append(r)
                    lab = (f"adapt{args.k_adapt} {hmd} h={hm}" if args.k_adapt else f"k={k}")
                    head = f"v={args.v} L={args.L} m={args.m} {lab} ba={ba} P={P:>7} rs={rs}:"
                    if r.get('skipped'):
                        print(f"{head} SKIPPED, {r['n_distinct_patches']} patches "
                              f"> --max_patches {args.max_patches}", flush=True)
                    else:
                        print(f"{head} gram={r['grammatical']:.3f} novel={r['novel']:.3f} "
                              f"succ={r['success']:.3f} "
                              f"lvl={[round(x, 3) for x in r['level_parse_fraction']]} "
                              f"patches={r['n_distinct_patches']} ({r['seconds']}s)", flush=True)
                    # written after every cell, so a job that runs out of time still leaves data
                    json.dump(rows, open(out, 'w'), indent=1)

    json.dump(rows, open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
