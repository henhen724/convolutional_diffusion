"""Does the straddling consistency check depend on the synonym rules?

hshunt, on why unaligned windows matter:

    "When there are pixels that see part of two different grammar rules they
    may be able to check for consistency between blocks in a way the block
    aligned model cannot.  It is a small signal, but I think *depending on the
    synonym rules*, you may be able to do better."

Section D of STATUS answered the first half: the check is real, it is one bit,
and it reaches the site's own enclosing parent rather than the root.  The
clause in italics has never been read, and it does not need a new job --
``rhm_straddle.py`` already enumerated every grid for 75 cells spanning
``m = 2..6``, so the answer is exact arithmetic on files already on disk.

Three locally-valid sets, all exact, all counted on the same enumeration of
``v ** 16`` grids (see rhm_straddle.py):

    lv_blocks   every aligned s**j* block is a legal rendering -- the
                independent-blocks coupon picture, = |R_j*| ** n_blocks
    lv_aligned  every k x k window is legal *for its own phase*  (block_align=s)
    lv_pooled   every k x k window is in the pooled window set   (block_align=1,
                the real LS machine)

and ``|R_L|`` is the truth.  ``lv_blocks >= lv_pooled >= |R_L|``, so the natural
scale-free readout is how much of the coupon picture's slack the straddling
windows actually remove:

    closure = log(lv_blocks / lv_pooled) / log(lv_blocks / |R_L|)

0 means straddling buys nothing and the machine is exactly the coupon picture;
1 means the machine's locally-valid set is the grammar itself.  This is the
quantity hshunt's sentence is about, and it is a pure ratio of exact counts --
no sampler, no peak fit, no estimator to blame.

REGISTERED BEFORE LOOKING (2026-11-05)
  1. closure FALLS with m.  More synonyms make more boundary pairs realizable,
     so a straddling window can rule out fewer configurations.  Bar: the slope
     c in  logit(closure) ~ a + c log m  is negative with a bootstrap CI over
     rule seeds excluding 0, in BOTH (v=2, k=3) and (v=3, k=3).
  2. The same sign for the phase-aware machine (closure computed from
     lv_aligned), i.e. this is a property of straddling, not of pooling.
  3. k=5 on a 4x4 grid is a degenerate control: the window covers the whole
     grid, so lv_pooled must equal |R_L| exactly and closure must be 1.000 in
     every cell.  If it is not, the pipeline is wrong and 1-2 mean nothing.

Usage:
    python3 scripts/rhm_straddle_synonyms.py --out results/.../straddle_m.json
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(20261105)


def load_cells(pattern):
    cells = []
    for f in sorted(glob.glob(str(ROOT / pattern))):
        for r in json.load(open(f)):
            if 'lv_pooled' not in r:          # the straddle_levels_* files
                continue
            r['src'] = Path(f).name
            cells.append(r)
    return cells


def closure(lv, blocks, R_L):
    """Fraction of the coupon picture's log-slack that the machine removes."""
    slack = np.log(blocks) - np.log(R_L)
    if slack <= 0:
        return np.nan
    return (np.log(blocks) - np.log(lv)) / slack


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def slope_ci(ms, ys, groups, n_boot=4000):
    """OLS slope of y on log m, bootstrapped by resampling whole rule seeds.

    Resampling seeds rather than cells keeps the m-ladder intact in every
    replicate -- the five seeds are the independent unit here, the five m
    values are the design.
    """
    ms, ys = np.asarray(ms, float), np.asarray(ys, float)
    groups = np.asarray(groups)
    ok = np.isfinite(ys)
    ms, ys, groups = ms[ok], ys[ok], groups[ok]
    if len(ys) < 3 or len(np.unique(ms)) < 2:
        return np.nan, (np.nan, np.nan)

    def fit(idx):
        x = np.log(ms[idx])
        if len(np.unique(x)) < 2:
            return np.nan
        return np.polyfit(x, ys[idx], 1)[0]

    point = fit(np.arange(len(ys)))
    seeds = np.unique(groups)
    boot = []
    for _ in range(n_boot):
        pick = RNG.choice(seeds, size=len(seeds), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == s) for s in pick])
        b = fit(idx)
        if np.isfinite(b):
            boot.append(b)
    if len(boot) < 100:
        return point, (np.nan, np.nan)
    return point, (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=None)
    a = p.parse_args()

    cells = load_cells('results/rhm_sampling/*/straddle_v*.json')
    for c in cells:
        c['closure_pooled'] = closure(c['lv_pooled'], c['lv_blocks'], c['R_L'])
        c['closure_aligned'] = closure(c['lv_aligned'], c['lv_blocks'], c['R_L'])
        c['ls_gain'] = c['lv_blocks'] / c['lv_pooled']

    print('cells: %d   (exact enumeration of v**16 grids, L=2 s=2)' % len(cells))
    print()

    # ---- control 3 first: if k >= side the machine sees the whole grid ----
    deg = [c for c in cells if c['k'] >= c['s'] ** c['L']]
    if deg:
        bad = [c for c in deg if c['lv_pooled'] != c['R_L']]
        print('--- control: k >= side, the window covers the whole grid ---')
        print('%d such cells; lv_pooled == |R_L| in %d of them, closure = %.3f..%.3f'
              % (len(deg), len(deg) - len(bad),
                 min(c['closure_pooled'] for c in deg),
                 max(c['closure_pooled'] for c in deg)))
        print('%s' % ('PASS -- the pipeline recovers the exact machine.' if not bad
                      else 'FAIL -- %d cells disagree; 1 and 2 below mean nothing.' % len(bad)))
        print()

    live = [c for c in cells if c['k'] < c['s'] ** c['L']]

    # ---- the m ladder, per (v, k) family ----
    fam = defaultdict(list)
    for c in live:
        fam[(c['v'], c['k'])].append(c)

    print('--- closure vs synonym count, per (v, k) ---')
    print('closure = log(lv_blocks/lv_pooled) / log(lv_blocks/|R_L|), exact')
    print('%-10s %6s %5s %9s %9s %9s %9s'
          % ('family', 'm', 'cells', 'closure', 'ls_gain', 'ceil_LS', '|R_L|'))
    out = {'cells': live, 'fits': {}}
    for key in sorted(fam):
        sel = fam[key]
        for m in sorted({c['m'] for c in sel}):
            g = [c for c in sel if c['m'] == m]
            print('%-10s %6d %5d %9.3f %9.3f %9.4f %9d'
                  % ('v=%d k=%d' % key, m, len(g),
                     float(np.median([c['closure_pooled'] for c in g])),
                     float(np.median([c['ls_gain'] for c in g])),
                     float(np.median([c['ceil_pooled'] for c in g])),
                     int(np.median([c['R_L'] for c in g]))))
        print()

    print('--- registered 1 and 2: slope of logit(closure) on log m ---')
    print('%-10s %-9s %8s %-22s %s'
          % ('family', 'machine', 'slope', '[95% CI, seed boot]', 'verdict'))
    for key in sorted(fam):
        sel = fam[key]
        for lab, fld in (('pooled', 'closure_pooled'), ('aligned', 'closure_aligned')):
            b, (lo, hi) = slope_ci([c['m'] for c in sel],
                                   logit([c[fld] for c in sel]),
                                   [c['rule_seed'] for c in sel])
            v = ('falls with m' if np.isfinite(hi) and hi < 0 else
                 'rises with m' if np.isfinite(lo) and lo > 0 else 'covers 0')
            print('%-10s %-9s %8.3f  [%6.3f, %6.3f]      %s'
                  % ('v=%d k=%d' % key, lab, b, lo, hi, v))
            out['fits']['v%d_k%d_%s' % (key[0], key[1], lab)] = {
                'slope': b, 'ci': [lo, hi], 'n': len(sel)}
    print()

    print('--- reading ---')
    live_fams = [k for k in sorted(fam) if k[1] < 4]
    falls = [k for k in live_fams
             if all(np.isfinite(out['fits']['v%d_k%d_%s' % (k[0], k[1], lab)]['ci'][1])
                    and out['fits']['v%d_k%d_%s' % (k[0], k[1], lab)]['ci'][1] < 0
                    for lab in ('pooled', 'aligned'))]
    if len(falls) == len(live_fams) and live_fams:
        print('Registered 1 and 2 both hold in %d of %d families: the straddling'
              % (len(falls), len(live_fams)))
        print('check is strongest when synonyms are few and washes out as the')
        print('grammar admits more renderings per symbol.  hshunt\'s "depending on')
        print('the synonym rules" is confirmed, with the sign he suggested.')
    elif not falls:
        print('Registered 1 and 2 fail everywhere: closure does not fall with m.')
        print('The straddling check does not depend on the synonym count the way')
        print('I registered, and the sentence needs a different reading.')
    else:
        print('Mixed: %d of %d families fall with m on both machines.'
              % (len(falls), len(live_fams)))
        print('Not enough to claim the dependence; report as suggestive only.')

    if a.out:
        o = Path(a.out)
        if not o.is_absolute():
            o = ROOT / o
        o.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(o, 'w'), indent=1)
        print('\nwrote %s' % o)


if __name__ == '__main__':
    main()
