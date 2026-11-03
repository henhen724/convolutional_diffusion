#!/usr/bin/env python
"""Slide PDF for the RHM sampling-complexity result.

Eight landscape pages: the question and the headline, the tokenization, worked
example traces (generated grids with their bottom-up parse), the derived
ceiling, the measured success phase diagram, the exponent fits, the data cost,
and the conclusion.

Usage:
  python scripts/build_rhm_slides.py --results results/rhm_sampling/2026-11-02 \
      --out results/rhm_sampling/2026-11-02/rhm_sampling_complexity_slides.pdf
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAGE = (13.33, 7.5)
plt.rcParams.update({'font.size': 15, 'axes.titlesize': 18, 'axes.labelsize': 16,
                     'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13,
                     'figure.dpi': 150, 'savefig.dpi': 150})
CMAP = matplotlib.colors.ListedColormap(
    ['#3b4cc0', '#7b9ff9', '#c9d7f0', '#f2cbb7', '#ee8468', '#b40426', '#4a9b4a', '#d9c44a'])


def jstar(k, s):
    j = 0
    while 2 * s ** (j + 1) - 1 <= k:
        j += 1
    return j


def title_page(pdf, head, sub, bullets):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.06, 0.86, head, fontsize=27, weight='bold', va='top')
    fig.text(0.06, 0.70, sub, fontsize=18, va='top', color='#b40426', weight='bold')
    y = 0.57
    for b in bullets:
        # lines are pre-broken, so no wrap=True: it would re-flow them and cross the edge
        fig.text(0.07, y, b, fontsize=14, va='top', linespacing=1.4)
        y -= 0.08 + 0.040 * b.count('\n')
    pdf.savefig(fig); plt.close(fig)


def page_tokenization(pdf, rhm, grid):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('Tokenization: what a "sample" is, and what the window sees',
                 fontsize=20, weight='bold')
    ax = fig.add_axes([0.035, 0.28, 0.195, 0.48])
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=7, interpolation='nearest', rasterized=True)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, int(grid[i, j]), ha='center', va='center', fontsize=11, color='w')
    for b, c in [(4, '#000000'), (2, '#666666')]:
        for t in range(0, grid.shape[0] + 1, b):
            ax.axhline(t - .5, color=c, lw=2.5 if b == 4 else 1.2)
            ax.axvline(t - .5, color=c, lw=2.5 if b == 4 else 1.2)
    ax.set_title(f'one sample: {rhm.grid_size}x{rhm.grid_size} grid\n'
                 f'of tokens in {{0..{rhm.v - 1}}}', fontsize=15)
    ax.set_xticks([]); ax.set_yticks([])

    paras = [
        f'Tokens. Every site holds one of v = {rhm.v} symbols. There is no pixel intensity '
        f'and no ordering: token 3 is not "between" 2 and 4. A grid is {rhm.grid_size}x'
        f'{rhm.grid_size} = {rhm.grid_size ** 2} categorical sites.',

        f'Grammar. A root symbol expands {rhm.L} times. Each symbol is replaced by one of '
        f'm = {rhm.m} synonymous {rhm.s}x{rhm.s} "words", drawn once and then fixed. So the '
        'grey lines at left bound aligned 2x2 blocks (one level-3 symbol each) and the black '
        'lines bound aligned 4x4 blocks (one level-2 symbol each). A grid is grammatical iff '
        'some root derives it.',

        'Diffusion. Absorbing-state ("erasure"): each site is independently replaced by a '
        'MASK token with probability p(t). Unmasked sites are never corrupted, so exact '
        'Bayes denoising is patch matching -- no score approximation enters.',

        'LS machine. The posterior at a masked site is the empirical distribution of the '
        'centre tokens of all training k x k patches that agree with the observed sites '
        'inside that window, with Laplace smoothing alpha. Sites are revealed one at a time '
        '(exact ancestral sampling; revealing in parallel from marginals adds ~n^2/2T '
        'independence errors, which alone made 12% of grids ungrammatical).',

        "Success (hshunt's criterion). A generated grid counts only if it is NOT in the "
        'training set AND parses completely. Since every training grid is grammatical,',
    ]
    body = '\n\n'.join(textwrap.fill(p, width=84) for p in paras)
    body += '\n\n        success  =  grammatical  +  novel  -  1'
    fig.text(0.265, 0.855, body, fontsize=12.5, va='top',
             family='monospace', linespacing=1.5)
    pdf.savefig(fig); plt.close(fig)


def page_traces(pdf, rhm, traces, n_draws=6):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('Example traces: what the machine actually emits, parsed level by level',
                 fontsize=20, weight='bold')
    n = len(traces)
    gs = fig.add_gridspec(2, n, left=0.04, right=0.98, top=0.80, bottom=0.10,
                          hspace=0.45, wspace=0.25)
    for c, (label, grid, lv, note) in enumerate(traces):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(grid, cmap=CMAP, vmin=0, vmax=7, interpolation='nearest', rasterized=True)
        for t in range(0, grid.shape[0] + 1, 2):
            ax.axhline(t - .5, color='#666666', lw=1.0)
            ax.axvline(t - .5, color='#666666', lw=1.0)
        for t in range(0, grid.shape[0] + 1, 4):
            ax.axhline(t - .5, color='k', lw=2.2)
            ax.axvline(t - .5, color='k', lw=2.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=14)
        ax = fig.add_subplot(gs[1, c])
        ax.axis('off')
        # the grid above is one draw; the fractions are over all drawn with it
        txt = f'parse, bottom up\n(mean of {n_draws} draws):\n'
        names = ['leaves', '2x2 blocks', '4x4 blocks', 'whole grid']
        for lvl, nm in zip(lv, names):
            mark = 'OK ' if lvl >= 0.999 else 'XX '
            txt += f'  {mark}{nm:<11} {lvl:.2f}\n'
        txt += '\n' + note
        ax.text(0, 1, txt, fontsize=12.5, va='top', family='monospace', linespacing=1.5)
    pdf.savefig(fig); plt.close(fig)


def page_ceiling(pdf, D):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('Law 1 (derivation): the window sets a hard ceiling, independent of P',
                 fontsize=20, weight='bold')
    gs = fig.add_gridspec(1, 2, left=0.045, right=0.97, top=0.82, bottom=0.07,
                          wspace=0.10, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, 0]); ax.axis('off')
    ax.text(0, 1.02, (
        'A k x k window centred on a site reaches only\n'
        'k//2, so it contains the aligned block of side b\n'
        'through that site only when k >= 2b - 1. Write\n\n'
        '   j* = floor(log_s((k+1)/2))\n\n'
        'for the deepest level it can police. With\n'
        'unlimited data it then samples the locally-valid\n'
        'set R_j*^((d/s^j*)^2), of which the grammar is a\n'
        'fraction\n\n'
        '   C(k) = |R_L| / |R_j*|^(s^(2(L-j*)))    (exact)\n\n'
        'with R_j = distinct leaf renderings of an aligned\n'
        's^j block. With no word collisions\n'
        '|R_j| = v m^((s^2j-1)/(s^2-1)), and this collapses\n'
        'algebraically to\n\n'
        '   C(k) = f0^U,  f0 = m / v^(s^2-1),\n'
        '                 U = (s^(2(L-j*))-1)/(s^2-1)\n\n'
        'U counts the rule instances above the window\'s\n'
        'reach. Over the 150 enumerable L=2 cases the\n'
        'closed form is exact in exactly the 70 drawn\n'
        'without a word collision, and wrong in exactly\n'
        'the 80 with one -- no exceptions. There the\n'
        'counting form is the correct one. Collisions\n'
        'renormalize f0; they do not touch U.'),
        fontsize=11.5, va='top', family='monospace', linespacing=1.34)

    ax = fig.add_subplot(gs[0, 1])
    ks = [3, 7, 15]
    mc = defaultdict(list)
    for r in json.load(open(D / 'ceiling_L3.json')):
        if r['v'] == 4 and r['m'] == 3 and r['ceiling_mc'] is not None:
            mc[r['k']].append(r['ceiling_mc'])
    meas = {3: 0.0, 7: 0.021, 15: 0.867}
    pred = [(3 / 64.) ** ((4 ** (3 - jstar(k, 2)) - 1) // 3) for k in ks]
    x = np.arange(len(ks))
    ax.bar(x - 0.25, pred, 0.25, label='closed form  f0^U', color='#3b4cc0')
    ax.bar(x, [np.mean(mc[k]) if mc[k] else 1.0 for k in ks], 0.25,
           label='exact / Monte Carlo', color='#7b9ff9')
    ax.bar(x + 0.25, [meas[k] for k in ks], 0.25,
           label='measured max grammaticality', color='#b40426')
    for xi, k in zip(x, ks):
        if meas[k] == 0.0:
            ax.annotate('0 seen in\n8192 samples', (xi + 0.25, 2e-8), ha='center',
                        va='bottom', fontsize=11, color='#b40426', weight='bold')
    ax.set_yscale('log'); ax.set_ylim(1e-8, 3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}\nj*={jstar(k, 2)}, U={(4 ** (3 - jstar(k, 2)) - 1) // 3}'
                        for k in ks])
    ax.set_ylabel('grammaticality ceiling')
    ax.set_title('v=4, m=3, s=2, L=3 (8x8 grids)\nfive decades, set by the window alone',
                 fontsize=15)
    ax.legend(loc='upper left'); ax.grid(alpha=.3, axis='y')
    pdf.savefig(fig); plt.close(fig)


def page_phase(pdf, D):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('Law 2 (measurement): success peaks at a finite P, then more data destroys it',
                 fontsize=20, weight='bold')
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.98, top=0.87, bottom=0.29,
                          wspace=0.17, hspace=0.28, height_ratios=[1, 1.15])
    handles = {}
    for c, (f, lab, RL) in enumerate([('G_L2_pstar.json', 'v=4, m=3, L=2', 972),
                                      ('G_L2_m5.json', 'v=4, m=5, L=2', 12500)]):
        byk = defaultdict(lambda: defaultdict(list))
        for r in json.load(open(D / f)):
            if not r.get('skipped'):
                byk[r['k']][r['P']].append((r['grammatical'], r['novel'], r['success']))
        top = fig.add_subplot(gs[0, c])
        bot = fig.add_subplot(gs[1, c], sharex=top)
        for k, col in zip(sorted(byk), ['#b40426', '#3b4cc0', '#4a9b4a']):
            Ps = sorted(byk[k])
            mu = lambda i: [np.mean([x[i] for x in byk[k][p]]) for p in Ps]
            for i, (nm, ls) in enumerate([('grammatical', ':'), ('novel', '--')]):
                ln, = top.plot(Ps, mu(i), ls, color=col, lw=1.9, alpha=.85)
                handles.setdefault(f'k={k}  {nm}', ln)
            sc = mu(2)
            ln, = bot.plot(Ps, sc, '-o', color=col, lw=3.2, ms=5)
            handles.setdefault(f'k={k}  success', ln)
            if max(sc) > 0.01:
                bp = Ps[int(np.argmax(sc))]
                bot.annotate(f'P* = {bp}\n{max(sc):.3f}', xy=(bp, max(sc)),
                             xytext=(0, 13), textcoords='offset points', ha='center',
                             fontsize=12, weight='bold', color=col)
        for ax, yl in ((top, 'grammatical / novel'), (bot, 'success')):
            ax.axvline(RL, color='k', lw=1.4, ls='-.')
            ax.set_xscale('log'); ax.grid(alpha=.3)
            if c == 0:
                ax.set_ylabel(yl)
        top.set_ylim(-0.04, 1.12)
        top.annotate(f'|R_L| = {RL}', xy=(RL, 1.09), xytext=(-6, 0), ha='right', va='top',
                     textcoords='offset points', fontsize=11.5)
        top.set_title(lab, fontsize=16)
        top.tick_params(labelbottom=False)
        bot.set_ylim(-0.012, max(0.14, bot.get_ylim()[1]))
        bot.set_xlabel('training grids P')
    order = sorted(handles, key=lambda s: (s.split()[-1] != 'success', s))
    fig.legend([handles[o] for o in order], order, loc='lower center',
               bbox_to_anchor=(0.5, 0.085), ncol=3, fontsize=11.5, frameon=True)
    fig.text(0.5, 0.028,
             'Dash-dot = size of the whole grammar. k=7 = 2d-1 is the full window:\n'
             'grammaticality 0.999, novelty 0.0009; success 1.6e-4 at P=10, exactly 0 at the other 16 P.',
             fontsize=12.5, ha='center')
    pdf.savefig(fig); plt.close(fig)


def page_exponents(pdf, D):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('The exponents: success attains the derived ceiling', fontsize=20, weight='bold')
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.97, top=0.78, bottom=0.26, wspace=0.33)
    cl = defaultdict(list)
    for r in json.load(open(D / 'ceiling_L2.json')):
        cl[(r['v'], r['m'], r['k'])].append(r['ceiling_exact'])
    rows = []
    for f in sorted(D.glob('E_*.json')):
        rs = json.load(open(f))
        v, m = rs[0]['v'], rs[0]['m']
        sc = defaultdict(list)
        for r in rs:
            if not r.get('skipped'):
                sc[r['P']].append(r['success'])
        rows.append((v, m, max(np.mean(sc[p]) for p in sc), np.mean(cl[(v, m, 3)])))
    a = np.array(rows)
    for ax_i, (col, fixcol, fixval, lbl, pred) in enumerate(
            [(0, 1, 3, 'alphabet size v', -3), (1, 0, 4, 'rules per symbol m', +1)]):
        ax = fig.add_subplot(gs[0, ax_i])
        sel = a[a[:, fixcol] == fixval]
        sel = sel[np.argsort(sel[:, col])]
        ax.loglog(sel[:, col], sel[:, 2], 'o-', color='#b40426', ms=11, lw=2.5,
                  label='measured max success')
        ax.loglog(sel[:, col], sel[:, 3], 's--', color='#3b4cc0', ms=9, lw=2,
                  label='exact ceiling C(k)')
        sm = np.polyfit(np.log(sel[:, col]), np.log(sel[:, 2]), 1)[0]
        se = np.polyfit(np.log(sel[:, col]), np.log(sel[:, 3]), 1)[0]
        ax.set_xlabel(lbl); ax.grid(alpha=.3, which='both')
        ax.set_xticks(sel[:, col].astype(int), minor=False)
        ax.set_xticks([], minor=True)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(f'L=2, k=3 (U=1)\nmeasured {sm:+.2f}, exact {se:+.2f}, '
                     f'predicted {pred:+d}', fontsize=14)
        if ax_i == 0:
            ax.set_ylabel('max over P of success')
        ax.legend(fontsize=11)

    ax = fig.add_subplot(gs[0, 2])
    Us, ss, ps = [], [], []
    for L in (1, 2, 3):
        rs = json.load(open(D / f'F_L{L}.json'))
        sc = defaultdict(list)
        for r in rs:
            if not r.get('skipped'):
                sc[r['P']].append(r['success'])
        U = (4 ** (L - 1) - 1) // 3
        Us.append(U); ss.append(max(np.mean(sc[p]) for p in sc)); ps.append(0.375 ** U)
    # L=1 is the degenerate end (k=3 = 2d-1 there, so the window is the whole grid and
    # novelty is 0 by construction); show it as an isolated marker, not part of the trend.
    ax.semilogy(Us[1:], ss[1:], 'o-', color='#b40426', ms=12, lw=2.5,
                label='measured max success')
    ax.semilogy(Us[1:], ps[1:], 's--', color='#3b4cc0', ms=9, lw=2, label='f0^U, f0 = 0.375')
    ax.semilogy([Us[0]], [3e-3], 'v', color='#777777', ms=13, label='L=1: success = 0\n(full window)')
    for U, s_, L in zip(Us[1:], ss[1:], (2, 3)):
        ax.annotate(f'L={L}, n={4 ** L}', (U, s_), textcoords='offset points',
                    xytext=(-6, 16), fontsize=12.5, ha='center')
    ax.set_xlim(-0.6, 6.0); ax.set_ylim(1e-3, 3)
    ax.set_xticks([0, 1, 5])
    ax.set_xlabel('U = unenforced rules'); ax.grid(alpha=.3, which='both')
    ax.set_title('depth law: v=2, m=3, k=3\nf0 fixed, only U varies', fontsize=14)
    ax.legend(fontsize=10.5, loc='lower left')
    fig.text(0.5, 0.055,
             'The v and m panels measure the ceiling exponents directly: success tracks the '
             'exactly-computed C(k) across two decades.\n'
             'L=1 is excluded from the depth trend because there k=3 = 2d-1 is already the full '
             'window, so grammaticality is 1 and novelty 0.\nBoth ends of the k ladder give '
             'success = 0, for opposite reasons.', fontsize=12.5, ha='center')
    pdf.savefig(fig); plt.close(fig)


def page_cost(pdf, D):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('Law 3: the data cost of a larger window, N_k = distinct k x k windows',
                 fontsize=20, weight='bold')
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.97, top=0.80, bottom=0.13, wspace=0.26)
    rows = []
    for f in ['counts_m.json', 'counts_v.json', 'counts_k7.json']:
        rows += json.load(open(D / f))
    ax = fig.add_subplot(gs[0, 0])
    for k, col in zip([3, 5, 7], ['#b40426', '#3b4cc0', '#4a9b4a']):
        cur = [r for r in rows if r['v'] == 4 and r['m'] == 3 and r['L'] == 3 and r['k'] == k]
        if not cur:
            continue
        c = dict(cur[0]['curve'])
        ax.loglog(sorted(c), [c[p] for p in sorted(c)], 'o-', color=col, ms=8, lw=2.5,
                  label=f'k={k} (j*={jstar(k, 2)})')
    ax.set_xlabel('training grids P'); ax.set_ylabel('distinct windows seen, N_k')
    ax.set_title('v=4, m=3, L=3 (8x8)\nk=3 flattens; k=5,7 never do', fontsize=15)
    ax.grid(alpha=.3, which='both'); ax.legend()

    ax = fig.add_subplot(gs[0, 1]); ax.axis('off')
    agg = defaultdict(list)
    for r in rows:
        c = dict(r['curve']); Ps = sorted(c)
        # growth over the last decade of P, not the last point: the ladder steps by 3,
        # so a per-point ratio understates it. Same measure as the results notebook.
        i10 = max([0] + [i for i, P in enumerate(Ps) if P <= Ps[-1] / 10])
        agg[(r['v'], r['m'], r['k'], r['L'])].append((c[Ps[-1]], c[Ps[-1]] / c[Ps[i10]]))
    txt = ('L = 3, at P = 3e5 (k=3,5) / 3e4 (k=7)\n\n'
           f'{"v":>3}{"m":>3}{"k":>3} {"N_k":>10} {"x/decade":>9}  verdict\n')
    growth = defaultdict(list)
    for key in sorted(agg):
        v, m, k, L = key
        if L != 3:
            continue
        N, g = np.mean(agg[key], axis=0)
        growth[k].append(g)
        if k == 3 or (v, m) == (4, 3):
            txt += (f'{v:>3}{m:>3}{k:>3} {N:>10.0f} {g:>9.2f}  '
                    f'{"flat" if g < 1.03 else "still climbing"}\n')
    # k=3 is flat at small v and still creeping at v=6,8: a bigger polynomial,
    # not a different regime, so say so rather than calling the whole row saturated.
    g3 = {(v, m): float(np.mean(agg[(v, m, k, L)], axis=0)[1])
          for (v, m, k, L) in agg if k == 3 and L == 3}
    slow = ', '.join(f'v={v} {g:.2f}x' for (v, m), g in sorted(g3.items()) if g >= 1.03)
    txt += (f'\nk=3 is within 3% per decade in '
            f'{sum(g < 1.03 for g in g3.values())} of the {len(g3)} grammars\n'
            f'({slow} still climbing) -- a bigger\n'
            'polynomial, not another regime.\n'
            f'Across all 8, k=5 grows {min(growth[5]):.2f}-{max(growth[5]):.2f} per decade\n'
            f'and k=7 {min(growth[7]):.2f}-{max(growth[7]):.2f}: neither ever saturates.\n\n'
            'Fitted at k=3:  N_3 ~ v^4.4 m^2.8 at L=3,\n'
            '                     v^2.1 m^3.7 at L=2 --\n'
            'polynomial either way, which is why k=3 is the\n'
            'only window learnable at all at L=3.\n\n'
            'A window of side k = 2b-1 is fixed by the 3x3\n'
            'aligned b-blocks it touches, so N_k grows like\n'
            'm^Theta(k^2/(s^2-1)): EXPONENTIAL IN WINDOW AREA.\n\n'
            'And P* (argmax of success) tracks the grammar\n'
            'itself: 500 at |R_L|=972, 3000 at |R_L|=12500.')
    ax.text(0, 1.02, txt, fontsize=11.5, va='top', family='monospace', linespacing=1.38)
    pdf.savefig(fig); plt.close(fig)


def page_conclusion(pdf):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('The scaling law', fontsize=22, weight='bold')
    fig.text(0.05, 0.88, (
        'Combining the two legs. Success >= eps needs U = O(1), i.e. s^(2(L-j*)) = O(1),\n'
        'i.e. the window must scale with the whole grid:   k = Theta(d) = Theta(sqrt(n)).\n'
        'But then the patch table has |R_j*| = v m^(Theta(n/(s^2-1))) entries, and P must fill\n'
        'it while staying below the memorization point |R_L| -- and at k = 2d-1 those two\n'
        'coincide exactly, so the interval where success > 0 closes.'),
        fontsize=13.5, va='top', family='monospace', linespacing=1.45)
    fig.text(0.05, 0.665,
             'P*  =  v . m^(Theta(n / (s^2 - 1)))  =  exp(Theta(n)),      n = s^(2L) sites',
             fontsize=17, weight='bold', color='#b40426', family='monospace')
    fig.text(0.05, 0.60, (
        'At fixed window k:  success ~ f0^((n/a - 1)/(s^2-1)),  a = s^(2j*) ~ ((k+1)/2)^2\n'
        '                    so  ln(1/success) = Theta(n / k^2).\n\n'
        'The LS machine extracts NO benefit from the hierarchy: its sampling complexity\n'
        'is exponential in the number of sites, the same as memorizing the whole\n'
        'distribution. Under the joint criterion (novel AND grammatical) there is no\n'
        'finite P at all -- grammaticality reaches 1 only at the window size where\n'
        'novelty is identically 0.\n\n'
        "Previous best on this eval: none. This is the repo's first measurement of RHM\n"
        'sampling complexity for the LS machine; the grammar oracle (rhm_grammar.py) and\n'
        'the patch-count machine were written for it.\n\n'
        'Why this matters for MLS: the obstruction is compositional, not statistical.\n'
        'One window cannot police two scales at once. A hierarchy of windows -- a 3x3\n'
        'machine whose output feeds a second 3x3 machine on inferred level-1 symbols --\n'
        'would police scale s^2 at the cost of two tables of size |R_1|: POLYNOMIAL\n'
        'instead of exponential. That is the natural next experiment.'),
        fontsize=12.5, va='top', family='monospace', linespacing=1.45)
    pdf.savefig(fig); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    D = Path(args.results)

    import torch
    from src.utils.discrete_score import ScheduledDiscreteScoreMachine, linear_mask_schedule
    from src.utils.random_hierarchy import RandomHierarchyModel2D
    from src.utils.rhm_grammar import evaluate_samples, feasible_symbol_masks
    from src.utils.rhm_sampling import PatchCountDiscreteScoreMachine

    rhm = RandomHierarchyModel2D(v=4, s=2, L=3, m=3, seed=1000)
    true_grids, _ = rhm.sample(600, seed=4242)

    traces = [('a true RHM sample', true_grids[0], [1., 1., 1., 1.],
               'grammatical by construction.')]
    for k, note in [(3, 'window reaches 1 site:\nseals 2x2 blocks,\nblind to 4x4 and up.'),
                    (7, 'window reaches 3:\nseals 4x4 too, but N_7\nis far too big to fill.'),
                    (15, 'window = whole grid:\nperfect grammar,\nbut it is a copy.')]:
        mach = PatchCountDiscreteScoreMachine(
            true_grids, vocab_size=rhm.v, kernel_size=k, alpha_smooth=1e-3,
            device='cpu', block_align=2 if k < 15 else 1, pad_mode='outside')
        sampler = ScheduledDiscreteScoreMachine(
            mach, vocab_size=rhm.v, grid_size=rhm.grid_size, default_time_steps=128,
            mask_schedule=linear_mask_schedule, mask_token=rhm.v, reveal_one=True)
        g = torch.Generator(device='cpu').manual_seed(11 + k)
        out = sampler.sample(n_samples=6, device='cpu', generator=g).cpu().numpy()
        ev = evaluate_samples(rhm, true_grids, out)
        lv = list(reversed(ev['level_parse_fraction']))
        traces.append((f'LS machine, k={k}\n(P=600)', out[0], lv,
                       note + f'\nnovel {ev["novel"]:.2f}'))

    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        title_page(pdf, 'Sampling complexity of the Random Hierarchy Model\nunder the LS '
                        '(local score) machine',
                   'There is no finite P. Sampling complexity is exp(Theta(n)) in the number '
                   'of sites\n-- the LS machine gains nothing from the hierarchy.',
                   ['Criterion (hshunt): a generated grid counts only if it is NOT in the '
                    'training set AND fully grammatical.',
                    'Because every training grid is grammatical, success = grammatical + '
                    'novel - 1, so the question is only\n'
                    'whether grammaticality ever beats the memorization rate. Over '
                    'k in {3,5,7,15}, v in {2,3,4,6,8},\n'
                    'm in {2..6}, L in {1,2,3} and P from 10 to 300,000, it never does by '
                    'more than a margin that\nvanishes exponentially in the number of sites.',
                    'Two legs: a window of side k can police the grammar only up to block '
                    'side ~k/2, which caps\ngrammaticality at a value independent of P '
                    '(derivation, exact); and widening the window costs\n'
                    'data exponentially in its area (measured).',
                    f'Repo convolutional_diffusion @ {commit}; rule seeds 1000-1002; '
                    f'alpha = 1e-3;\none site revealed at a time (exact ancestral sampling); '
                    f'n_gen 512-8192.'])
        page_tokenization(pdf, rhm, true_grids[0])
        page_traces(pdf, rhm, traces)
        page_ceiling(pdf, D)
        page_phase(pdf, D)
        page_exponents(pdf, D)
        page_cost(pdf, D)
        page_conclusion(pdf)
    print('wrote', out, f'({out.stat().st_size / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
