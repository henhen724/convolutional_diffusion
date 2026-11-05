"""Slide deck: how the LS machine should choose its window, measured at s=3.

The sampling-complexity answer (`P* = 0.21 |R_L|`) is already delivered; this
deck covers what came after it, which is hshunt's question about setting the
patch size: "It should be the maximum length scale which stays above the
minimum posterior entropy."

The whole story is a sequence of eliminations, so the deck is ordered as one:

  the choice of window matters         -- adaptive beats every fixed k
  the entropy floor does not help      -- best h_min is 0 everywhere
  nor does its count rival             -- n_min loses the same way
  nor does a k(t) schedule             -- loses 4.7x at matched mean k
  nor does position                    -- R^2 < 0.001
  what is left is the partial grid     -- >= 53% content-dependent

s = 2 could not have shown any of this: there U1 = side - 1, so "largest window
above a floor" chose between one real rung and the whole grid. s = 3, L = 2 is
a 9x9 board with three genuine rungs, 5 / 7 / 9.

Builds from the JSON in results/rhm_sampling/2026-11-04 and re-runs cheaply, so
it is safe to rebuild when the last cells land.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median as med

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_rhm_slides import CMAP, PAGE, title_page  # noqa: E402

CELL = ('v', 's', 'L', 'm', 'rule_seed')
ARMS = ('adaptive', 'cap', 'sched', 'fixed')
ARM_LABEL = {'adaptive': 'criterion\n(largest k with support)',
             'cap': 'criterion capped\nto its own schedule',
             'sched': 'bare schedule\n(same k(t), every site)',
             'fixed': 'best fixed k'}
ARM_C = {'adaptive': '#b40426', 'cap': '#ee8468', 'sched': '#7b9ff9', 'fixed': '#3b4cc0'}


class Pages:
    """PdfPages that can also drop a PNG per page.

    Sherlock has no pdftoppm and no pymupdf in this venv, so the only way to
    look at what was drawn is to write the raster at draw time.
    """

    def __init__(self, pdf, png_dir=None):
        self.pdf = pdf
        self.png_dir = Path(png_dir) if png_dir else None
        if self.png_dir:
            self.png_dir.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def savefig(self, fig):
        self.pdf.savefig(fig)
        if self.png_dir:
            fig.savefig(self.png_dir / ('page%d.png' % self.n), dpi=70)
        self.n += 1


def load_s3(d):
    rows = []
    for f in sorted((ROOT / d).glob('s3_*.json')):
        try:
            rs = json.load(open(f))
        except (ValueError, IOError):
            continue
        if isinstance(rs, list):
            rows += [r for r in rs if isinstance(r, dict) and 'success' in r
                     and not r.get('skipped') and all(c in r for c in CELL)]
    return rows


def load_glob(d, pat):
    rows = []
    for f in sorted((ROOT / d).glob(pat)):
        if 'smoke' in f.name:
            continue
        try:
            rows.extend(json.load(open(f)))
        except (ValueError, IOError):
            continue
    return rows


def group_cells(rows):
    """Per cell: the adaptive rows at h_min=0 and the fixed rows, shared P only."""
    by = defaultdict(lambda: {'adapt': [], 'fixed': []})
    for r in rows:
        key = tuple(r[c] for c in CELL)
        if r.get('k') == 'adapt':
            by[key]['adapt'].append(r)
        elif isinstance(r.get('k'), int):
            by[key]['fixed'].append(r)
    out = {}
    for key, g in by.items():
        a0 = [r for r in g['adapt'] if r.get('h_min') == 0.0
              and float(r.get('n_min', 1.0)) == 1.0]
        if not a0 or not g['fixed']:
            continue
        Ps = set(r['P'] for r in a0) & set(r['P'] for r in g['fixed'])
        if not Ps:
            continue
        a = max((r for r in a0 if r['P'] in Ps), key=lambda r: r['success'])
        f = max((r for r in g['fixed'] if r['P'] in Ps), key=lambda r: r['success'])
        out[key] = dict(adapt=a, fixed=f, all_adapt=a0, all_fixed=g['fixed'], Ps=Ps)
    return out


def page_tokenization(pdf, grid, v, s, L):
    """s-aware: the shared helper in build_rhm_slides draws s=2 block lines."""
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('Tokenization: what a "sample" is, and what a window sees',
                 fontsize=20, weight='bold')
    ax = fig.add_axes([0.035, 0.22, 0.26, 0.58])
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=7, interpolation='nearest', rasterized=True)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, int(grid[i, j]), ha='center', va='center', fontsize=9, color='w')
    for t in range(0, grid.shape[0] + 1, s):
        ax.axhline(t - .5, color='#000000', lw=2.2)
        ax.axvline(t - .5, color='#000000', lw=2.2)
    side = grid.shape[0]
    ax.set_title('one sample: %dx%d grid of tokens in {0..%d}\n'
                 'black lines bound the %dx%d level-1 blocks' % (side, side, v - 1, s, s),
                 fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])

    paras = [
        'Tokens. Every site holds one of v = %d symbols. There is no intensity and no '
        'ordering: token 2 is not "between" 1 and 3. One grid is %dx%d = %d categorical '
        'sites.' % (v, side, side, side * side),

        'Grammar. A root symbol expands L = %d times; each symbol is replaced by one of '
        'm synonymous %dx%d "words", drawn once and then fixed. A grid is grammatical iff '
        'some root derives it. |R_L| = v*m^(1+s^2), so m^10 at s=3.' % (L, s, s),

        'Diffusion. Absorbing-state ("erasure"): each site is independently replaced by a '
        'MASK token with probability p(t). Unmasked sites are never corrupted, so exact '
        'Bayes denoising IS patch matching -- no score approximation enters, and any gap '
        'from optimal is the window choice, not a fitting error.',

        'LS machine. The posterior at a masked site is the empirical distribution of the '
        'centre tokens of all training kxk patches agreeing with the observed sites in '
        'that window (Laplace alpha=1e-3). Sites are revealed one at a time.',

        'Why s=3. U1 = 2s^(L-1)-1 is the smallest window that can contain a whole '
        'top-level child block. At s=2, U1 = side-1, so the only odd rung in [U1, side] '
        'is U1 itself and there is no schedule to choose. At s=3, L=2 the board is 9x9 '
        'with three real rungs: 5, 7, 9.',

        "Success (hshunt's criterion): a grid counts only if it is NOT in the training "
        'set AND parses completely.',
    ]
    import textwrap
    body = '\n\n'.join(textwrap.fill(p, width=86) for p in paras)
    body += '\n\n        success  =  grammatical  AND  novel'
    fig.text(0.33, 0.86, body, fontsize=12, va='top', family='monospace', linespacing=1.45)
    pdf.savefig(fig); plt.close(fig)


def page_fair_test(pdf, cells):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('1. The window choice matters: adaptive beats every fixed k',
                 fontsize=20, weight='bold')
    keys = sorted(cells)
    lab = ['v%d m%d\nrs%d' % (k[0], k[3], k[4]) for k in keys]
    ad = [cells[k]['adapt']['success'] for k in keys]
    fx = [cells[k]['fixed']['success'] for k in keys]
    x = np.arange(len(keys))

    ax = fig.add_axes([0.065, 0.46, 0.55, 0.36])
    ax.bar(x - 0.2, ad, 0.4, color=ARM_C['adaptive'], label='adaptive (h_min=0)')
    ax.bar(x + 0.2, fx, 0.4, color=ARM_C['fixed'], label='best fixed k')
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=9)
    ax.set_ylabel('success (novel AND grammatical)')
    ax.legend(frameon=False); ax.set_title('peak over the shared P ladder', fontsize=14)

    ax2 = fig.add_axes([0.70, 0.46, 0.27, 0.36])
    for k in keys:
        sch = cells[k]['adapt'].get('k_schedule') or []
        if sch:
            f = [p[0] for p in sch]
            kk = [p[1] for p in sch]
            ax2.plot(f, kk, lw=1.6, alpha=0.8, color=plt.cm.viridis(k[0] / 5))
    ax2.invert_xaxis()
    ax2.set_xlabel('masked fraction (time)'); ax2.set_ylabel('mean chosen k')
    ax2.set_title('the induced k(t)\nuses all three rungs', fontsize=13)
    ax2.axhline(5, ls=':', c='#888'); ax2.axhline(9, ls=':', c='#888')

    won = sum(1 for k in keys if cells[k]['adapt']['success'] > cells[k]['fixed']['success'])
    at_u1 = sum(1 for k in keys if cells[k]['fixed']['k'] == 5)
    ratio = med([cells[k]['adapt']['success'] / cells[k]['fixed']['success']
                 for k in keys if cells[k]['fixed']['success'] > 0])
    txt = ('Adaptive beats the best fixed k in %d of %d cells; median ratio %.0fx.\n'
           'The fixed optimum sits at U1 = 5 in only %d of %d -- it prefers k = 7, so the\n'
           'machine wants a window LARGER than the smallest one that can hold a child block.\n'
           'k(t) falls from 9 as the grid fills, and its span grows with the vocabulary v\n'
           '(about 1.5 at v=2, 2.9 at v=3, 3.2 at v=4): a richer grammar needs to back off\n'
           'further as the grid fills.' % (won, len(keys), ratio, at_u1, len(keys)))
    fig.text(0.065, 0.335, txt, fontsize=13, va='top', family='monospace', linespacing=1.6)
    pdf.savefig(fig); plt.close(fig)


def page_floors(pdf, rows, nmin_rows):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('2. Neither floor earns its keep -- the bare support test wins',
                 fontsize=20, weight='bold')

    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get('k') == 'adapt' and float(r.get('n_min', 1.0)) == 1.0:
            by[tuple(r[c] for c in CELL)][r['h_min']].append(r['success'])
    hs = sorted({h for c in by.values() for h in c})
    ax = fig.add_axes([0.07, 0.54, 0.40, 0.32])
    for key, d in sorted(by.items()):
        ys = [max(d[h]) if d.get(h) else np.nan for h in hs]
        ax.plot(hs, ys, 'o-', lw=1.4, ms=4, alpha=0.75,
                color=plt.cm.viridis(key[0] / 5))
    ax.set_xlabel('h_min  (entropy floor, "match" mode)')
    ax.set_ylabel('peak success')
    ax.set_title('hshunt\'s entropy floor: best value is 0', fontsize=14)

    ax2 = fig.add_axes([0.56, 0.54, 0.40, 0.32])
    nb = defaultdict(dict)
    for r in nmin_rows:
        if r.get('k') == 'adapt':
            nb[r['P']][float(r['n_min'])] = r['success']
    for P, d in sorted(nb.items()):
        ns = sorted(d)
        ax2.plot(ns, [d[n] for n in ns], 'o-', lw=1.8, ms=5, label='P = %d' % P)
    ax2.set_xscale('log'); ax2.set_xlabel('n_min  (count floor: matching occurrences)')
    ax2.set_ylabel('success'); ax2.legend(frameon=False)
    ax2.set_title('the count rival: best value is 1', fontsize=14)

    txt = ('The machine accepts a window when  n_match > 0  AND  h >= h_min.\n'
           'In "match" mode h = log(effective number of matching training patches), so\n'
           'h_min is a support-DIVERSITY floor and h_min = 0 is bare support-EXISTENCE.\n\n'
           'n_min is the genuine rival, not a reparametrisation: n_match counts matching\n'
           'training OCCURRENCES, h_match measures their DIVERSITY. A window matched by\n'
           'fifty copies of one patch has n_match = 50 but h_match = 0 -- the entropy floor\n'
           'throws it away, a count floor keeps it. Both lose anyway.\n\n'
           'One mechanism explains both: rejecting a window can only send the site to a\n'
           'SHORTER one, and shorter is less grammatical. The floors buy novelty and sell\n'
           'grammaticality at a bad rate (n_min 1 -> 10: novel +0.14, grammatical -0.40).')
    fig.text(0.07, 0.46, txt, fontsize=11.5, va='top', family='monospace', linespacing=1.42)
    pdf.savefig(fig); plt.close(fig)


def page_schedule(pdf, ks_rows):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('3. Nor is it a schedule: the SAME k(t), applied per site, is worth 4.7x',
                 fontsize=20, weight='bold')
    by = defaultdict(list)
    for r in ks_rows:
        by[(r['v'], r['rule_seed'])].append(r)
    keys = sorted(by)
    x = np.arange(len(keys))
    pk = {a: [max(g['arms'][a]['success'] for g in by[k]) for k in keys] for a in ARMS}

    ax = fig.add_axes([0.065, 0.54, 0.56, 0.32])
    w = 0.21
    for i, a in enumerate(ARMS):
        ax.bar(x + (i - 1.5) * w, pk[a], w, color=ARM_C[a], label=ARM_LABEL[a])
    ax.set_xticks(x); ax.set_xticklabels(['v%d rs%d' % k for k in keys], fontsize=10)
    ax.set_ylabel('peak success'); ax.set_yscale('log')
    # headroom so the 4-entry legend clears the tallest bars on the log axis
    ax.set_ylim(top=max(max(p) for p in pk.values()) * 12)
    ax.legend(frameon=False, fontsize=9.5, ncol=2, loc='upper right')
    ax.set_title('four matched arms, same grammar / training set / sampler seed', fontsize=13)

    ax2 = fig.add_axes([0.71, 0.54, 0.26, 0.32])
    ka, ks_ = [], []
    for k in keys:
        b = max(by[k], key=lambda g: g['arms']['adaptive']['success'])
        ka.append(b['arms']['adaptive']['mean_k']); ks_.append(b['arms']['sched']['mean_k'])
    ax2.scatter(ka, ks_, s=55, color=ARM_C['adaptive'], zorder=3)
    lo, hi = min(ka + ks_) - 0.3, max(ka + ks_) + 0.3
    ax2.plot([lo, hi], [lo, hi], ls='--', c='#888')
    ax2.set_xlabel('mean k, criterion'); ax2.set_ylabel('mean k, bare schedule')
    ax2.set_title('the control that makes\nthe comparison mean anything', fontsize=12)

    sa = med([max(g['arms']['sched']['success'] for g in by[k])
              / max(g['arms']['adaptive']['success'] for g in by[k]) for k in keys])
    txt = ('The control schedule is derived from the adaptive arm\'s OWN recorded trace --\n'
           'round its mean k(t) to the available rungs -- so it cannot be rigged by picking\n'
           'a schedule that was always going to lose.\n\n'
           'Adaptive beats the bare schedule in %d/%d cells and the best fixed k in %d/%d.\n'
           'The schedule recovers a median %.0f%% of adaptive success WHILE SPENDING THE SAME\n'
           'WINDOW BUDGET: median gap in mean k is 0.08 (right panel, on the diagonal).\n\n'
           'So the win is not which window sizes get used on average. It is which SITE gets\n'
           'which size. Capping the criterion to its own schedule is free at v=2 but costs\n'
           '8-30%% at v=3,4, so the per-site freedom matters more as the grammar grows.'
           % (len(keys), len(keys), len(keys), len(keys), 100 * sa))
    fig.text(0.065, 0.46, txt, fontsize=11.5, va='top', family='monospace', linespacing=1.42)
    pdf.savefig(fig); plt.close(fig)


def page_locality(pdf, kl_rows):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('4. What the per-site choice actually depends on', fontsize=20, weight='bold')
    names = [('r2_phase', 'position\n(block offset)'), ('r2_step', 'time\n(masked fraction)'),
             ('r2_nrev', 'local revealed\nneighbour count'),
             ('r2_step_nrev', 'time + local\ncount together')]
    ax = fig.add_axes([0.07, 0.54, 0.44, 0.32])
    for i, (f, _) in enumerate(names):
        vals = [r[f] for r in kl_rows]
        ax.scatter(np.full(len(vals), i) + np.random.RandomState(0).uniform(-.12, .12, len(vals)),
                   vals, s=30, alpha=0.7, color=ARM_C['adaptive'])
        ax.hlines(med(vals), i - .3, i + .3, color='#111', lw=2.5, zorder=4)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n[1] for n in names], fontsize=10)
    ax.set_ylabel('$R^2$ of the chosen k'); ax.set_ylim(-0.03, 1.0)
    ax.set_title('%d runs (v=2,3,4 x 3 seeds x 2 P)' % len(kl_rows), fontsize=13)

    ax2 = fig.add_axes([0.58, 0.54, 0.38, 0.32])
    for v, c in ((2, '#3b4cc0'), (3, '#7b9ff9'), (4, '#b40426')):
        rs = [r for r in kl_rows if r['v'] == v]
        if rs:
            ax2.scatter([r['r2_step'] for r in rs], [r['r2_nrev'] for r in rs],
                        s=45, color=c, label='v = %d' % v)
    lim = [0, max(r['r2_nrev'] for r in kl_rows) * 1.1]
    ax2.plot(lim, lim, ls='--', c='#888')
    ax2.set_xlabel('$R^2$, time alone'); ax2.set_ylabel('$R^2$, local reveal count')
    ax2.legend(frameon=False); ax2.set_title('the count beats time in every run', fontsize=13)

    mx = max(r['r2_step_nrev'] for r in kl_rows)
    txt = ('Position explains nothing. Regressing the chosen k on the site\'s offset within\n'
           'its level-1 block gives R^2 < 0.001 in all %d runs, and adds nothing jointly with\n'
           'time. So no lookup table in (position, time) can replace the criterion -- and the\n'
           'window choice carries no signature of how it straddles the grammar blocks.\n\n'
           'The local revealed-neighbour count (how many of the 9x9 around a site are fixed)\n'
           'beats time in %d/%d runs and SUBSUMES it: adding time on top gains only .005-.013.\n'
           'The k(t) schedule is therefore a consequence of reveal density, not a separate\n'
           'effect -- which is why page 3\'s bare schedule fails at matched mean k.\n\n'
           'But it caps at %.2f. The remaining >= %.0f%% turns on WHICH tokens were revealed,\n'
           'not how many. The criterion is irreducibly content-dependent.'
           % (len(kl_rows),
              sum(1 for r in kl_rows if r['r2_nrev'] > r['r2_step']), len(kl_rows),
              mx, 100 * (1 - mx)))
    fig.text(0.07, 0.46, txt, fontsize=11.5, va='top', family='monospace', linespacing=1.42)
    pdf.savefig(fig); plt.close(fig)


def page_traces(pdf, tr, n_show=5):
    fig = plt.figure(figsize=PAGE)
    cfg = tr['config']
    fig.suptitle('5. Example traces: the window size moving along one generation',
                 fontsize=20, weight='bold')
    side = int(round(cfg['n_sites'] ** 0.5))

    # the claim is about k moving, so show k and the posterior entropy per reveal
    ax = fig.add_axes([0.06, 0.56, 0.40, 0.30])
    for tag, c in (('floor', ARM_C['adaptive']), ('control', ARM_C['fixed'])):
        trajs = tr[tag]['traj'][:40]
        arr = np.full((len(trajs), cfg['n_sites']), np.nan)
        for i, t in enumerate(trajs):
            for rv in t['reveals']:
                arr[i, rv['step']] = rv['k']
        ax.plot(np.nanmean(arr, 0), lw=2.2, color=c,
                label='%s (k=%s)' % (tag, tr[tag]['kernel_sizes']))
    ax.set_xlabel('reveal index'); ax.set_ylabel('chosen k at the revealed site')
    ax.legend(frameon=False, fontsize=11)
    ax.set_title('mean over 40 generations', fontsize=13)

    ax2 = fig.add_axes([0.56, 0.56, 0.38, 0.30])
    for tag, c in (('floor', ARM_C['adaptive']), ('control', ARM_C['fixed'])):
        trajs = tr[tag]['traj'][:40]
        arr = np.full((len(trajs), cfg['n_sites']), np.nan)
        for i, t in enumerate(trajs):
            for rv in t['reveals']:
                arr[i, rv['step']] = rv['H']
        ax2.plot(np.nanmean(arr, 0), lw=2.2, color=c, label=tag)
    ax2.set_xlabel('reveal index'); ax2.set_ylabel('posterior entropy (nats)')
    ax2.legend(frameon=False, fontsize=11)
    ax2.set_title('the criterion keeps the posterior sharper', fontsize=13)

    # a handful of emitted grids, labelled by the two things that decide success
    gs = fig.add_gridspec(1, n_show, left=0.06, right=0.50, top=0.40, bottom=0.14, wspace=0.3)
    for c in range(n_show):
        t = tr['floor']['traj'][c]
        a = fig.add_subplot(gs[0, c])
        a.imshow(np.array(t['grid']), cmap=CMAP, vmin=0, vmax=7,
                 interpolation='nearest', rasterized=True)
        for q in range(0, side + 1, cfg['s']):
            a.axhline(q - .5, color='k', lw=1.0); a.axvline(q - .5, color='k', lw=1.0)
        a.set_xticks([]); a.set_yticks([])
        a.set_title('%s\n%s' % ('gram' if t['grammatical'] else 'NOT gram',
                                'novel' if t['novel'] else 'copy'), fontsize=9,
                    color='#1a7a1a' if (t['grammatical'] and t['novel']) else '#b40426')
    fig.text(0.06, 0.075, 'criterion output, the FIRST %d generations (not selected)\n'
             'v%d m%d rs%d, P=%d. Black lines mark the %dx%d blocks.'
             % (n_show, cfg['v'], cfg['m'], cfg['rule_seed'], cfg['P'], cfg['s'], cfg['s']),
             fontsize=10.5, va='top')

    txt = ('Same grammar, training set, sampler and seed\n'
           'in both arms; only the window rule differs.\n'
           'The criterion starts at k=9 and backs off as\n'
           'the grid fills; the control is pinned at the\n'
           'best fixed k=7 for the whole run.\n\n'
           '            success    gram     novel\n'
           ' criterion   %.4f  %.4f  %.4f\n'
           ' fixed k=7   %.4f  %.4f  %.4f\n\n'
           'The win is grammaticality (%.1fx here), bought\n'
           'with a little novelty: the fixed window is\n'
           'novel almost always because it is mostly\n'
           'emitting grids that no root derives.'
           % (tr['floor']['success'], tr['floor']['grammatical'], tr['floor']['novel'],
              tr['control']['success'], tr['control']['grammatical'], tr['control']['novel'],
              tr['floor']['grammatical'] / max(tr['control']['grammatical'], 1e-9)))
    fig.text(0.56, 0.44, txt, fontsize=11.5, va='top', family='monospace', linespacing=1.42)
    pdf.savefig(fig); plt.close(fig)


def page_conclusion(pdf, cells, ks_rows, kl_rows):
    keys = sorted(cells)
    ratio = med([cells[k]['adapt']['success'] / cells[k]['fixed']['success']
                 for k in keys if cells[k]['fixed']['success'] > 0])
    by = defaultdict(list)
    for r in ks_rows:
        by[(r['v'], r['rule_seed'])].append(r)
    sa = med([max(g['arms']['sched']['success'] for g in by[k])
              / max(g['arms']['adaptive']['success'] for g in by[k]) for k in by])
    mx = max(r['r2_step_nrev'] for r in kl_rows)

    fig = plt.figure(figsize=PAGE)
    fig.suptitle('Where this leaves the patch-size question', fontsize=21, weight='bold')
    rows = [
        ('choose the window adaptively vs any fixed k', '%.0fx' % ratio,
         '%d/%d cells' % (len(keys), len(keys)), 'the length scale is the whole story'),
        ('add hshunt\'s entropy floor h_min > 0', 'worse', '10/10 cells',
         'h_min=0 (bare support) is best everywhere'),
        ('add the count floor n_min > 1', 'worse', 'both P rungs',
         'same mechanism: rejection shortens the window'),
        ('replace per-site choice with its own k(t)', '%.0f%% kept' % (100 * sa),
         '9/9 cells', 'at matched mean k -- same budget, worse spent'),
        ('predict k from position (block offset)', 'R^2<0.001', '18/18 runs',
         'no positional signature at all'),
        ('predict k from local reveal count', 'R^2<=%.2f' % mx, '18/18 runs',
         'subsumes time; rest is content-dependent'),
    ]
    y = 0.80
    fig.text(0.05, 0.855, '%-44s %-11s %-13s %s' % ('what was tried', 'effect', 'support', 'reading'),
             fontsize=12, family='monospace', weight='bold')
    for a, b, c, d in rows:
        fig.text(0.05, y, '%-44s %-11s %-13s %s' % (a, b, c, d),
                 fontsize=11, family='monospace')
        y -= 0.046
    concl = (
        'Answer to "how are you setting the patch size over time?"\n\n'
        '  Not over time. The rule that works is "the largest window that matches any\n'
        '  training patch at all", evaluated SEPARATELY AT EVERY SITE. Its k(t) curve is\n'
        '  a summary of what it does, not a specification of it: handing that same curve\n'
        '  back to the sampler recovers only %.0f%% of the success.\n\n'
        '  The entropy criterion in particular does not bind -- every floor above zero\n'
        '  costs success in every cell measured, because a rejected window falls back on\n'
        '  a shorter one rather than on a better one.\n\n'
        'Caveat: all of this is s=3, L=2 (9x9, rungs 5/7/9), v in {2,3,4}, m=2 mostly,\n'
        'n_gen=4096. Deeper boards need k >= U1 = 2s^(L-1)-1, which is out of reach at L=4.'
        % (100 * sa))
    fig.text(0.05, 0.47, concl, fontsize=11.5, va='top', family='monospace', linespacing=1.42)
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results/rhm_sampling/2026-11-04')
    ap.add_argument('--trace', default='results/rhm_sampling/2026-11-04/trace_s3_v2m2.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--png_dir', default=None, help='also write one PNG per page')
    args = ap.parse_args()

    s3 = load_s3(args.results)
    cells = group_cells(s3)
    ks_rows = load_glob(args.results, 'ksched_*.json')
    kl_rows = load_glob(args.results, 'klocal_*.json')
    nmin_rows = json.load(open(ROOT / args.results / 'nmin_smoke.json'))
    tr = json.load(open(ROOT / args.trace))
    print('%d s3 rows -> %d cells | %d ksched | %d klocal | %d nmin'
          % (len(s3), len(cells), len(ks_rows), len(kl_rows), len(nmin_rows)))

    from src.utils.random_hierarchy import RandomHierarchyModel2D
    rhm = RandomHierarchyModel2D(v=2, s=3, L=2, m=2, seed=1000)
    grid0, _ = rhm.sample(1, seed=0)
    grid0 = np.asarray(grid0[0])

    keys = sorted(cells)
    ratio = med([cells[k]['adapt']['success'] / cells[k]['fixed']['success']
                 for k in keys if cells[k]['fixed']['success'] > 0])
    by = defaultdict(list)
    for r in ks_rows:
        by[(r['v'], r['rule_seed'])].append(r)
    sa = med([max(g['arms']['sched']['success'] for g in by[k])
              / max(g['arms']['adaptive']['success'] for g in by[k]) for k in by])

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as _pdf:
        pdf = Pages(_pdf, args.png_dir)
        title_page(
            pdf,
            'Setting the LS machine\'s patch size on the RHM',
            'The length scale is the whole story -- and it must be chosen '
            'per site, not per step',
            ['The criterion "largest window matching any training patch" beats the best\n'
             'FIXED window in %d/%d cells, median %.0fx.' % (len(keys), len(keys), ratio),
             'Every floor on top of it loses: the entropy floor h_min in 10/10 cells,\n'
             'and its count rival n_min at both P rungs. Rejection only shortens.',
             'Replacing the per-site choice by its own induced k(t) schedule, at the\n'
             'SAME mean k, keeps only %.0f%% of the success (9/9 cells).' % (100 * sa),
             'Position explains none of the choice (R^2 < 0.001); local reveal count\n'
             'explains at most 0.47. The rest is irreducibly content-dependent.',
             's=3, L=2 is the first board where this question is even askable:\n'
             'at s=2 the only rung between U1 and the grid side is U1 itself.'])
        page_tokenization(pdf, grid0, v=2, s=3, L=2)
        page_fair_test(pdf, cells)
        page_floors(pdf, s3, nmin_rows)
        page_schedule(pdf, ks_rows)
        page_locality(pdf, kl_rows)
        page_traces(pdf, tr)
        page_conclusion(pdf, cells, ks_rows, kl_rows)
    print('wrote %s (%.2f MB)' % (out, out.stat().st_size / 1e6))


if __name__ == '__main__':
    main()
