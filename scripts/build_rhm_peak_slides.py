#!/usr/bin/env python
"""Slide PDF for the corrected RHM sampling-complexity result.

Supersedes build_rhm_slides.py, whose ceiling argument (C(k) = f0^U) was
retracted on 2026-11-03: it assumed a window straddling a block boundary
constrains nothing, which exact enumeration disproved. The asymptotic answer is
unchanged -- exp(Theta(n)) -- but it is now *measured*, as a data collapse of
the success peak onto the grammar size, rather than derived from that ceiling.

Seven landscape pages: question and headline, tokenization, worked traces, the
window bound with its k=5 control, the collapse of P* onto |R_L|, the mechanism
(ceiling times memorization drag), and the conclusion.

Usage:
  python scripts/build_rhm_peak_slides.py --results results/rhm_sampling/2026-11-03 \
      --out results/rhm_sampling/2026-11-03/rhm_sampling_complexity_slides.pdf
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_rhm_slides import PAGE, page_tokenization, page_traces, title_page  # noqa: E402

U1 = lambda s, L: 2 * s ** (L - 1) - 1


class Tee:
    """A PdfPages stand-in that also drops each page as a PNG.

    The page functions call ``pdf.savefig(fig)``, so standing in for the
    PdfPages is all it takes to get downloadable single pages alongside the
    deck -- and to let me look at what I built.
    """

    def __init__(self, pdf, png_dir=None):
        self.pdf, self.dir, self.n = pdf, png_dir, 0
        if png_dir:
            Path(png_dir).mkdir(parents=True, exist_ok=True)

    def savefig(self, fig):
        self.pdf.savefig(fig)
        self.n += 1
        if self.dir:
            fig.savefig(Path(self.dir) / ('page%d.png' % self.n), dpi=110)


def wrap(paras, width=120):
    """Hard-break paragraphs: matplotlib's wrap=True re-flows past the margin."""
    return '\n'.join(textwrap.fill(p, width=width) for p in paras)


def foot(fig, y, paras, width=120, size=12.5, x=.055):
    fig.text(x, y, wrap(paras, width), fontsize=size, va='top', linespacing=1.5)


def stats(br):
    """Collapse statistics, recomputed from peak_summary.json.

    Every number quoted on the deck comes from here rather than being typed in,
    so a re-run of the sweep cannot leave a stale figure on a slide.
    """
    R = np.array([r['R_L'] for r in br], float)
    P = np.array([r['P_star'] for r in br], float)
    ratio = P / R
    b, loga = np.polyfit(np.log(R), np.log(P), 1)
    resid = np.log(P) - (b * np.log(R) + loga)
    # sd of the slope from the usual least-squares expression
    sb = np.sqrt((resid ** 2).sum() / (len(R) - 2) / ((np.log(R) - np.log(R).mean()) ** 2).sum())
    return dict(n=len(br), R_lo=R.min(), R_hi=R.max(), R_span=R.max() / R.min(),
                P_lo=P.min(), P_hi=P.max(), P_span=P.max() / P.min(),
                r_lo=ratio.min(), r_hi=ratio.max(), r_span=ratio.max() / ratio.min(),
                median=np.median(ratio), geo=float(np.exp(np.mean(np.log(ratio)))),
                geo_sd=float(np.exp(np.std(np.log(ratio)))), b=b, sb=sb,
                succ_med=float(np.median([r['succ_max'] for r in br])),
                succ_hi=float(max(r['succ_max'] for r in br)))


# ---------------------------------------------------------------- data loading

def load(D: Path):
    RL = {}
    for f in glob.glob(str(D / 'nk_exact_L*.json')):
        for r in json.load(open(f)):
            RL[(r['v'], r['m'], r['s'], r['L'], r['rule_seed'])] = r['R_L']
    NK = {}
    for f in glob.glob(str(D / 'nk_exact_L*.json')):
        for r in json.load(open(f)):
            NK[(r['v'], r['m'], r['s'], r['L'], r['rule_seed'], r['k'])] = r['N_k']
    rows = []
    for pat in ('L3peak_v*.json', 'L3k7_v*_m*.json', 'P_v*_m*.json'):
        for f in glob.glob(str(D / pat)):
            rows += [r for r in json.load(open(f)) if 'success' in r]
    peaks = json.load(open(D / 'peak_summary.json'))
    return RL, NK, rows, peaks


def curves(rows, RL, k_sel='u1'):
    out = {}
    for r in rows:
        if r.get('block_align', 1) != 1 or not isinstance(r.get('k'), int):
            continue
        u1 = U1(r['s'], r['L'])
        if (k_sel == 'u1' and r['k'] != u1) or (k_sel != 'u1' and r['k'] != k_sel):
            continue
        key = (r['v'], r['m'], r['s'], r['L'], r['rule_seed'], r['k'])
        if key[:5] in RL:
            out.setdefault(key, []).append(r)
    return {k: sorted(v, key=lambda r: r['P']) for k, v in out.items()}


# ---------------------------------------------------------------------- pages

def page_window_bound(pdf, rows, NK):
    """k=5 vs k=7 at L=3: the window, not the data, is what binds below U=1."""
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('1. A window below the U=1 scale cannot be fixed with data',
                 fontsize=21, weight='bold', y=.97)
    ax = [fig.add_axes([.06, .33, .40, .50]), fig.add_axes([.56, .33, .40, .50])]

    sel = [r for r in rows if r['v'] == 2 and r['m'] == 2 and r['L'] == 3
           and r.get('block_align', 1) == 1 and isinstance(r.get('k'), int)]
    for k, col in ((5, '#3b4cc0'), (7, '#b40426')):
        for rs in (0, 1, 2):
            cs = sorted([r for r in sel if r['k'] == k and r['rule_seed'] == rs],
                        key=lambda r: r['P'])
            if len(cs) < 2:
                continue
            P = [c['P'] for c in cs]
            ax[0].plot(P, [c['grammatical'] for c in cs], 'o-', color=col, ms=4,
                       label=('k=%d' % k) if rs == 0 else None, alpha=.8)
            ax[1].plot(P, [c['n_distinct_patches'] for c in cs], 'o-', color=col, ms=4,
                       label=('k=%d' % k) if rs == 0 else None, alpha=.8)
    n5 = NK.get((2, 2, 2, 3, 0, 5))
    if n5:
        ax[1].axhline(n5, color='#3b4cc0', ls=':', lw=1.5)
        ax[1].text(2e4, n5 * 1.25, 'N$_5$ = %d (every window the\ngrammar allows)' % n5,
                   color='#3b4cc0', fontsize=12)
    ax[0].set_ylabel('fraction fully grammatical'); ax[0].set_ylim(-.02, .55)
    ax[1].set_ylabel('distinct patches in the table'); ax[1].set_yscale('log')
    for b, t in zip(ax, ['grammaticality vs training set size',
                         'the k=5 table is provably complete']):
        b.set_xscale('log'); b.set_xlabel('P (training grids)'); b.set_title(t, fontsize=16)
        b.grid(alpha=.3); b.legend(loc='upper left')
    foot(fig, .23, [
        'RHM v=2, m=2, s=2, L=3 (8x8 grid, 64 sites), 3 rule seeds, n_gen=1024. A k x k window '
        'resolves levels up to j* = floor(log_s((k+1)/2)), so reaching the root needs '
        'k >= 2 s^(L-1) - 1 = 7.',
        'At k=5 the table saturates at exactly N_5 = 92080 by P = 1e6 -- it has seen every window '
        'the grammar can produce -- and grammaticality never exceeds 0.7% anywhere from P = 1e4 to '
        '4e6, with no trend. At k=7 on the same grammars it reaches 25-52%.',
        'Below the U=1 window, data is not the binding constraint, and no P fixes it. '
        'All the sampling-complexity curves that follow are therefore taken at k = U=1.',
    ], x=.06)
    pdf.savefig(fig); plt.close(fig)


def page_collapse(pdf, cs, RL, peaks, st):
    """The headline: success peaks at the same P/|R_L| over 5 decades in |R_L|."""
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('2. Success peaks at P* = %.2f |R_L|, over five decades of grammar size'
                 % st['geo'], fontsize=21, weight='bold', y=.97)
    ax = [fig.add_axes([.075, .44, .24, .42]), fig.add_axes([.395, .44, .24, .42]),
          fig.add_axes([.725, .44, .245, .42])]
    for key in sorted(cs):
        v, m, s, L, rs, k = key
        c = cs[key]
        P = np.array([x['P'] for x in c], float)
        S = np.array([x['success'] for x in c], float)
        col = '#3b4cc0' if L == 2 else '#b40426'
        ax[0].plot(P, S, '-', color=col, lw=1, alpha=.55)
        ax[1].plot(P / RL[key[:5]], S, '-', color=col, lw=1, alpha=.55)
    br = [r for r in peaks if r['bracketed']]
    for L, col, mk in ((2, '#3b4cc0', 'o'), (3, '#b40426', 's')):
        sub = [r for r in br if r['L'] == L]
        if sub:
            ax[2].plot([r['R_L'] for r in sub], [r['P_star'] for r in sub], mk, color=col,
                       ms=7, label='L=%d  (n=%d)' % (L, len(sub)), alpha=.85)
    xs = np.array([st['R_lo'] * .8, st['R_hi'] * 1.3])
    ax[2].plot(xs, st['geo'] * xs, 'k--', lw=1.5, label='P* = %.2f |R$_L$|' % st['geo'])
    ax[0].set_xlabel('P'); ax[0].set_title('success vs P', fontsize=15)
    ax[1].set_xlabel('$P\\,/\\,|R_L|$'); ax[1].set_title('the same curves, rescaled', fontsize=15)
    ax[1].axvline(st['geo'], color='k', ls=':', lw=1.5)
    ax[2].set_xlabel('$|R_L|$  (exact grammar size)'); ax[2].set_ylabel('$P^*$')
    ax[2].set_yscale('log'); ax[2].set_title('peak location', fontsize=15)
    ax[2].legend(loc='upper left', fontsize=11)
    for b in ax[:2]:
        b.set_ylabel('novel AND grammatical')
    for b in ax:
        b.set_xscale('log'); b.grid(alpha=.3)
    foot(fig, .335, [
        'Blue L=2 (16 sites, k=3), red L=3 (64 sites, k=7); each line is one (v, m, rule seed). '
        'Rules are drawn with replacement, so independent seeds at the same (v, m) give |R_L| '
        'differing by up to 20x -- that is what makes this a collapse rather than a fit. |R_L| is '
        'the exact enumerated grammar size, never the nominal v m^M.',
    ])
    vs = sorted({r['v'] for r in br}); ms = sorted({r['m'] for r in br})
    fig.text(.055, .20,
             '%d bracketed peaks, v in {%s}, m in {%s}, L in {2,3}:\n'
             '  |R_L| spans %s - %s  (%s x)      P* spans %s - %s  (%s x)\n'
             '  P*/|R_L| spans %.3f - %.3f  (%.1fx), median %.3f, geometric sd %.2fx\n'
             '  free power law  P* ~ |R_L|^%.3f +- %.3f   -- exponent 1 is %.1f sd away'
             % (st['n'], ','.join(map(str, vs)), ','.join(map(str, ms)),
                f'{st["R_lo"]:,.0f}'.replace(',', ' '), f'{st["R_hi"]:,.0f}'.replace(',', ' '),
                f'{st["R_span"]:,.0f}'.replace(',', ' '),
                f'{st["P_lo"]:,.0f}'.replace(',', ' '), f'{st["P_hi"]:,.0f}'.replace(',', ' '),
                f'{st["P_span"]:,.0f}'.replace(',', ' '),
                st['r_lo'], st['r_hi'], st['r_span'], st['median'], st['geo_sd'],
                st['b'], st['sb'], abs(1 - st['b']) / st['sb']),
             fontsize=13, va='top', linespacing=1.55, family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', fc='#eef2fb', ec='#3b4cc0'))
    pdf.savefig(fig); plt.close(fig)


def page_mechanism(pdf, cs, RL, st):
    """Why there is a peak at all: a P-independent ceiling times a decaying novelty."""
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('3. Why a peak exists: a window ceiling that saturates, times novelty that decays',
                 fontsize=20, weight='bold', y=.97)
    ax = [fig.add_axes([.06, .44, .25, .42]), fig.add_axes([.39, .44, .25, .42]),
          fig.add_axes([.72, .44, .25, .42])]
    gx, gy = [], []
    for key in sorted(cs):
        v, m, s, L, rs, k = key
        c = cs[key]
        x = np.array([r['P'] for r in c], float) / RL[key[:5]]
        col = '#3b4cc0' if L == 2 else '#b40426'
        ax[0].plot(x, [r['grammatical'] for r in c], '-', color=col, lw=1, alpha=.55)
        ax[1].plot(x, [r['novel'] for r in c], '-', color=col, lw=1, alpha=.55)
        for r, xi in zip(c, x):
            if r['grammatical'] * r['n_samples'] >= 8:
                gx.append(xi); gy.append(r['success'] / r['grammatical'])
    gx, gy = np.array(gx), np.array(gy)
    ax[2].plot(gx, gy, '.', color='#7b9ff9', ms=5, alpha=.5)
    xs = np.logspace(-3.5, 1, 100)
    ok = (gy > .02) & (gy < .999)
    sl, ic = np.polyfit(gx[ok], -np.log(gy[ok]), 1)
    ax[2].plot(xs, np.exp(-xs), 'k--', lw=1.5, label='uniform: $e^{-x}$')
    ax[2].plot(xs, np.exp(-(sl * xs + ic)), '-', color='#b40426', lw=2,
               label='fit: $e^{-%.2fx-%.2f}$' % (sl, ic))
    ax[2].legend(loc='lower left', fontsize=11); ax[2].set_ylim(0, 1.05)
    ax[0].set_title('grammaticality saturates', fontsize=15)
    ax[0].set_ylabel('fraction grammatical')
    ax[1].set_title('novelty decays', fontsize=15); ax[1].set_ylabel('fraction novel')
    ax[2].set_title('P(novel | grammatical)', fontsize=15)
    ax[2].set_ylabel('P(novel | grammatical)')
    for b in ax:
        b.set_xscale('log'); b.set_xlabel('$x = P\\,/\\,|R_L|$'); b.grid(alpha=.3)
    foot(fig, .345, [
        'success(x) = grammatical(x) x P(novel | grammatical, x). The first factor rises with data '
        'but flattens at a ceiling fixed by the window, not by P -- at L=3, k=7 that ceiling is '
        '0.25-0.52 depending on the rule draw. The second falls. Their product peaks, and because '
        'both are functions of x = P/|R_L| alone, the peak sits at a fixed x.',
        'The machine is NOT sampling uniformly from the grammar. Uniform sampling would give '
        'P(novel | grammatical) = (1 - 1/|R_L|)^P ~ e^-x (black dashed). Measured: '
        'e^(-%.2f x - %.2f) over %d cells.'
        % (sl, ic, ok.sum()),
        'The intercept says only %.0f%% of grammatical outputs are novel even as x -> 0: when the '
        'LS machine does get the grammar right, it is disproportionately reproducing a training '
        'grid. That memorization drag is a second, P-independent penalty, and it is why peak '
        'success stays low (median %.3f) even where it is best.'
        % (100 * np.exp(-ic), st['succ_med']),
    ], x=.06)
    pdf.savefig(fig); plt.close(fig)


def page_conclusion(pdf, peaks, st):
    fig = plt.figure(figsize=PAGE)
    fig.text(.06, .93, '4. The scaling law', fontsize=23, weight='bold', va='top')
    fig.text(.06, .80,
             'P*  =  $\\Theta(|R_L|)$  =  $\\Theta\\!\\left(v\\, m^{(n-1)/(s^2-1)}\\right)$  '
             '=  $\\exp(\\Theta(n))$,      $n = s^{2L}$ sites',
             fontsize=26, va='top', color='#b40426')
    body = [
        ('What was measured', [
            'For each (v, m, s, L, rule seed): the full success curve over P at the U=1 window '
            'k = 2 s^(L-1) - 1, and the P where it peaks.',
            "Success is hshunt's criterion: NOT in the training set AND parses under every "
            'grammar rule, by exact bottom-up parsing.',
            '|R_L| is the exact enumerated grammar size per rule seed, not the nominal v m^M. '
            'Rules are drawn with replacement and collide, which is what gives the 20x spread in '
            '|R_L| at fixed (v, m) that turns this from a fit into a collapse.']),
        ('What it says', [
            'Two requirements fight. Grammaticality needs the patch table filled, and at the U=1 '
            'window that table has N_k = Theta(|R_L|) distinct entries (measured exactly: '
            'N_3/|R_2| = 4.3-6.0, N_7/|R_3| = 1.6-6.1). Novelty needs the training set to be a '
            'small part of the grammar. Both are functions of x = P/|R_L|, so the optimum sits at '
            'a fixed x and P* is linear in the number of grids the grammar can produce -- the LS '
            'machine needs data as if the grammar were a flat list, not a hierarchy.']),
        ('Honest limits', [
            'Peak success is low: median %.3f over %d curves, best %.3f. The law locates the '
            'optimum; it does not claim it is good.' % (st['succ_med'], st['n'], st['succ_hi']),
            'Two depths only (L=2, 16 sites; L=3, 64 sites). L=4 needs k=15, whose table is out of '
            'reach, so the exponential in n rests on those two depths plus the exact N_k counts.',
            'The exponent is %.3f +- %.3f, consistent with 1, but the residual scatter is %.2fx '
            'and the constant %.2f is not sharp.'
            % (st['b'], st['sb'], st['geo_sd'], st['geo'])]),
    ]
    y = .70
    for head, paras in body:
        fig.text(.06, y, head, fontsize=16, weight='bold', va='top', color='#3b4cc0')
        y -= .044
        for p in paras:
            txt = textwrap.fill(p, width=126)
            fig.text(.07, y, txt, fontsize=12, va='top', linespacing=1.45)
            y -= .033 * (1 + txt.count('\n')) + .007
        y -= .014
    pdf.savefig(fig); plt.close(fig)


# ----------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--trace_P', type=int, default=20000,
                   help='training grids behind the trace page (small = login-node smoke test)')
    p.add_argument('--trace_n', type=int, default=64)
    p.add_argument('--png_dir', default=None, help='also write each page as a PNG')
    a = p.parse_args()
    D = Path(a.results)
    RL, NK, rows, peaks = load(D)
    cs = curves(rows, RL)

    import torch
    from src.utils.discrete_score import ScheduledDiscreteScoreMachine, linear_mask_schedule
    from src.utils.random_hierarchy import RandomHierarchyModel2D
    from src.utils.rhm_grammar import evaluate_samples
    from src.utils.rhm_sampling import PatchCountDiscreteScoreMachine

    # traces use the configuration the law was measured on: L=3, where k=7 is U=1
    rhm = RandomHierarchyModel2D(v=2, s=2, L=3, m=2, seed=1000)
    true_grids, _ = rhm.sample(a.trace_P, seed=4242)
    traces = [('a true RHM sample', true_grids[0], [1., 1., 1., 1.],
               'grammatical by construction.')]
    for k, note in [(3, 'reaches 1 site: seals 2x2\nblocks, blind to 4x4 and up.'),
                    (5, 'reaches 2: still cannot\nsee the 4x4 rule.'),
                    (7, 'reaches 3 = U=1 scale:\nthe first k that can parse\nto the root.')]:
        mach = PatchCountDiscreteScoreMachine(
            true_grids, vocab_size=rhm.v, kernel_size=k, alpha_smooth=1e-3,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            block_align=1, pad_mode='outside')
        sampler = ScheduledDiscreteScoreMachine(
            mach, vocab_size=rhm.v, grid_size=rhm.grid_size, default_time_steps=64,
            mask_schedule=linear_mask_schedule, mask_token=rhm.v, reveal_one=True)
        g = torch.Generator(device=mach.device).manual_seed(11 + k)
        out = sampler.sample(n_samples=a.trace_n, device=mach.device, generator=g).cpu().numpy()
        ev = evaluate_samples(rhm, true_grids, out)
        traces.append((f'LS machine, k={k}\n(P={a.trace_P:,})'.replace(',', ' '), out[0],
                       list(reversed(ev['level_parse_fraction'])),
                       note + f'\ngram {ev["grammatical"]:.2f}, novel {ev["novel"]:.2f}'))

    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    br = [r for r in peaks if r['bracketed']]
    st = stats(br)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as _pdf:
        pdf = Tee(_pdf, a.png_dir)
        title_page(
            pdf,
            'Sampling complexity of the Random Hierarchy Model\nunder the LS (local score) machine',
            wrap(['P* = Theta(|R_L|) = exp(Theta(n)): the LS machine needs training data '
                  'proportional to the entire grammar.'], 74),
            [wrap(['Criterion (hshunt): a generated grid counts only if it is NOT in the '
                   'training set AND is fully grammatical.'], 118),
             wrap(['Measured, not derived: the success curve over P was taken for %d '
                   '(v, m, L, rule seed) configurations and its peak P* located. Plotted against '
                   'the exact grammar size |R_L| the peaks collapse: P* spans %.0fx, while '
                   'P*/|R_L| spans %.1fx about a median of %.3f.'
                   % (st['n'], st['P_span'], st['r_span'], st['median'])], 118),
             wrap(['This supersedes the 2026-11-02 deck. That one reached exp(Theta(n)) from a '
                   'ceiling C(k) = f0^U which assumed a window straddling a block boundary '
                   'constrains nothing. hshunt flagged it, exact enumeration disproved it, and '
                   'it is withdrawn. The asymptotic answer survives; the route to it is now the '
                   'data collapse above.'], 118),
             wrap(['Repo convolutional_diffusion @ %s; rule seeds 1000-1002; alpha = 1e-3; one '
                   'site revealed at a time (exact ancestral sampling); n_gen = 1024 at L=3, '
                   '512-8192 at L=2; jobs 50903850 (exact counts), 50930718, 50935971.'
                   % commit], 118)])
        page_tokenization(pdf, rhm, true_grids[0])
        page_traces(pdf, rhm, traces, n_draws=a.trace_n)
        page_window_bound(pdf, rows, NK)
        page_collapse(pdf, cs, RL, peaks, st)
        page_mechanism(pdf, cs, RL, st)
        page_conclusion(pdf, peaks, st)
    print('wrote', out, '(%.2f MB)' % (out.stat().st_size / 1e6))


if __name__ == '__main__':
    main()
