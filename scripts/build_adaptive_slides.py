#!/usr/bin/env python
"""Slide PDF: hshunt's patch-size criterion, scored on novel-and-grammatical.

The criterion is "take the largest length scale whose posterior entropy stays
above a floor".  ``AdaptiveWindowDiscreteScoreMachine`` implements it, the
2026-11-03 sweep (50903583) measured it, and every reading of that sweep until
now quoted grammaticality alone -- the one number the criterion is not about,
since a machine that copies its training set scores 1.0 on it.  This deck scores
it on the project's actual criterion, novel *and* grammatical, and then runs the
one-variable follow-up (51098252) that the first result pointed at.

Seven landscape pages: headline, tokenization, the re-read, the induced k(t),
the three-arm comparison, reveal-by-reveal traces, and what it means for the
patch-size question.

Usage:
  python scripts/build_adaptive_slides.py --results results/rhm_sampling/2026-11-04 \
      --out results/rhm_sampling/2026-11-04/rhm_adaptive_window_slides.pdf
"""

from __future__ import annotations

import argparse
import json
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

from scripts.build_rhm_slides import CMAP, PAGE, page_tokenization, title_page  # noqa: E402
from scripts.build_rhm_peak_slides import Tee, foot  # noqa: E402

C_F3, C_A135, C_A35 = '#6a6a6a', '#b40426', '#1a7f37'


def load(results):
    d = Path(results)
    cmp15 = json.load(open(d / 'adaptive_u1_compare.json'))
    cmp21 = json.load(open(d / 'adaptive_vs_fixed.json'))
    tr = json.load(open(d / 'adaptive_traces.json'))
    old, new = [], []
    for f in sorted(Path(ROOT / 'results/rhm_sampling/2026-11-03').glob('adapt_L2_*.json')):
        old += [r for r in json.load(open(f)) if r.get('h_mode') == 'match']
    for f in sorted(d.glob('adaptu1_L2_*.json')):
        new += json.load(open(f))
    return cmp15, cmp21, tr, old, new


def sched_at(rows, cell, h_min):
    """The k(t) schedule for one cell and floor, at the P where it does best."""
    rs = [r for r in rows if (r['v'], r['m'], r['rule_seed']) == cell and r['h_min'] == h_min]
    if not rs:
        return None, None
    r = max(rs, key=lambda q: q['success'])
    s = sorted(r['k_schedule'], reverse=True)
    return np.array([f for f, _ in s]), np.array([k for _, k in s])


def page_title(pdf, cmp15, cmp21):
    n21 = len(cmp21)
    lose21 = sum(1 for r in cmp21 if r['adapt_succ'] <= r['fixed_succ'])
    n15 = len(cmp15)
    beat_old = sum(1 for r in cmp15 if r['a35_succ'] > r['a135_succ'])
    bind = sum(1 for r in cmp15 if r['a35_hmin'] > 0)
    lose_f3 = sum(1 for r in cmp15 if r['a35_succ'] <= r['f3_succ'])
    title_page(
        pdf,
        'Setting the window by posterior entropy',
        'The criterion picks the right direction and the wrong operating point:\n'
        'it beats every adaptive variant tried and still loses to a fixed k = U1.',
        [
            'The criterion. Take the largest window whose posterior entropy is still above a\n'
            'floor h_min. Entropy falls with both k and time, so k shrinks along a generation:\n'
            'coarse scales while the grid is ambiguous, fine scales last.',

            'Scored for the first time on novel AND grammatical, not grammaticality alone.\n'
            'A machine that copies its training set scores 1.0 on grammaticality.',

            f'Re-read of the 2026-11-03 sweep: adaptive loses to the best fixed k in\n'
            f'{lose21}/{n21} cells (median success ratio 0.672) -- but beats the fixed k = 5 it\n'
            f'actually imitates in 11/12. The choice set was wrong, not the adaptivity.',

            f'One variable changed (51098252): drop k = 1, leaving k >= U1 = 3. Now a binding\n'
            f'floor wins in {bind}/{n15} cells where before it lost in 21/21, and the new machine\n'
            f'beats the old in {beat_old}/{n15}. It still loses to fixed k = 3 in {lose_f3}/{n15}.',
        ])


def page_reread(pdf, cmp21, old):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('The re-read: grammaticality said yes, the real criterion said no',
                 fontsize=20, weight='bold')

    ax = fig.add_axes([0.07, 0.42, 0.40, 0.42])
    f = np.array([r['fixed_succ'] for r in cmp21])
    a = np.array([r['adapt_succ'] for r in cmp21])
    ax.scatter(f, a, s=70, color=C_A135, zorder=3, edgecolor='k', linewidth=.5)
    lim = [0, max(f.max(), a.max()) * 1.08]
    ax.plot(lim, lim, '--', color='#444444', lw=1.5, label='equal')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('best fixed k, peak success', fontsize=13)
    ax.set_ylabel('adaptive {1,3,5}, peak success', fontsize=13)
    ax.set_title('every point below the line:\n20 of 21 cells', fontsize=14)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=.3)

    # the floor's ledger, which is where the refutation becomes a mechanism:
    # novelty is bought exactly as the criterion intends and grammaticality pays
    # more than it is worth
    ax2 = fig.add_axes([0.57, 0.42, 0.38, 0.42])
    cell = (2, 3, 0)
    hs = sorted(set(r['h_min'] for r in old))
    g, n, s, km = [], [], [], []
    for h in hs:
        rs = [r for r in old if (r['v'], r['m'], r['rule_seed']) == cell and r['h_min'] == h]
        r = max(rs, key=lambda q: q['success'])
        g.append(r['grammatical']); n.append(r['novel'])
        s.append(r['success']); km.append(r['k_mean'])
    ax2.plot(hs, g, 'o-', color='#1a7f37', lw=2.5, ms=8, label='grammatical (sold)')
    ax2.plot(hs, n, 's-', color='#2c7fb8', lw=2.5, ms=8, label='novel (bought)')
    ax2.plot(hs, s, '^-', color=C_A135, lw=3, ms=9, label='both = success')
    ax2.set_xlabel('entropy floor h_min (nats)', fontsize=13)
    ax2.set_ylabel('fraction', fontsize=13)
    ax2.set_title('k set {1,3,5}, v=2 m=3: the floor works\nand the trade is still bad',
                  fontsize=14)
    ax2.legend(fontsize=10.5, loc='center right')
    ax2.grid(alpha=.3)

    foot(fig, 0.30, [
        'Left. Peak success over the shared training-set ladder, adaptive against the best fixed k on '
        'the same grammar and seeds. The adaptive machine is below the diagonal in 20 of 21 cells, '
        'median success ratio 0.672. It is not a close call and it is not noise: the same ordering '
        'holds in every grammar family measured (v = 2, 3, 4; m = 2, 3, 5; L = 2 and 3).',

        'Right. Why. Raising the floor does exactly what it was designed to do -- novelty climbs '
        f'{n[0]:.3f} -> {n[-1]:.3f} as the mean window falls {km[0]:.2f} -> {km[-1]:.2f} -- and '
        f'grammaticality falls {g[0]:.3f} -> {g[-1]:.3f} faster, so success drops {s[0]:.3f} -> '
        f'{s[-1]:.3f}. The floor stops the machine memorizing and bankrupts it doing so. Its best '
        'setting was h_min = 0, i.e. no floor at all, in 21 of 21 cells.',
    ])
    pdf.savefig(fig); plt.close(fig)


def page_kt(pdf, old, new):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('The induced k(t): the direction is right at every floor',
                 fontsize=20, weight='bold')
    cell = (2, 3, 0)
    for i, (rows, name, ks) in enumerate(((old, 'k set {1,3,5}', '{1,3,5}'),
                                          (new, 'k set {3,5}  (k >= U1)', '{3,5}'))):
        ax = fig.add_axes([0.08 + 0.48 * i, 0.44, 0.37, 0.40])
        for h, col in zip([0.0, 0.1, 0.5, 1.0],
                          ['#bbbbbb', '#8fbcd4', '#2c7fb8', '#b40426']):
            f, k = sched_at(rows, cell, h)
            if f is None:
                continue
            ax.plot(1 - f, k, 'o-', color=col, lw=2.5, ms=6, label='h_min = %.1f' % h)
        ax.axhline(3, color='#1a7f37', ls='--', lw=2)
        ax.text(0.02, 3.08, 'U1 = 3', color='#1a7f37', fontsize=11.5, weight='bold')
        ax.set_xlabel('fraction of sites revealed  (trajectory ->)', fontsize=13)
        ax.set_ylabel('mean chosen k', fontsize=13)
        ax.set_ylim(0.8, 5.3)
        ax.set_title('%s   (cell v%d m%d rs%d)' % (name, cell[0], cell[1], cell[2]), fontsize=13)
        ax.grid(alpha=.3)
        if i == 0:
            ax.legend(fontsize=10.5, loc='lower left')

    f0, k0 = sched_at(old, cell, 1.0)
    f1, k1 = sched_at(new, cell, 1.0)
    foot(fig, 0.31, [
        'k shrinks monotonically along every trajectory at every floor, in both machines -- the '
        'direction hshunt predicted, and it is produced by the entropy criterion rather than imposed. '
        'Both panels are one cell, v2 m3 rs0, so the curves can be read; the win-counts on the other '
        'pages are over all 15-21 cells. With no floor the schedule is nearly flat here (5.00 at the '
        'first reveal, 4.13 at the last): the '
        'largest window almost always still matches something, so "largest that matched" barely '
        'varies. The floor is what gives the schedule range.',

        'Left, the failure. At h_min = 1.0 the unrestricted machine ends the trajectory at mean k = '
        f'{k0[-1]:.2f}, well below U1 = 2 s^(L-1) - 1 = 3, the smallest window that can contain a '
        'complete top-level child block. Right, the fix: with k = 1 removed the same floor lands at '
        f'{k1[-1]:.2f} and the collapse does not happen. Note the honest limit -- above U1 at L = 2 '
        'there are only two rungs, 3 and 5, so this is a thin test of a schedule, not a rich one.',
    ])
    pdf.savefig(fig); plt.close(fig)


def page_three(pdf, cmp15):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('Three machines, same grammars, same ladder, same seeds',
                 fontsize=20, weight='bold')
    ax = fig.add_axes([0.065, 0.44, 0.58, 0.40])
    x = np.arange(len(cmp15))
    w = 0.27
    lbl = ['v%dm%d.%d' % (r['v'], r['m'], r['rule_seed']) for r in cmp15]
    for off, key, col, nm in ((-w, 'f3_succ', C_F3, 'fixed k = 3 (= U1)'),
                              (0.0, 'a135_succ', C_A135, 'adaptive {1,3,5}'),
                              (w, 'a35_succ', C_A35, 'adaptive {3,5}')):
        ax.bar(x + off, [r[key] for r in cmp15], w, color=col, label=nm)
    ax.set_xticks(x); ax.set_xticklabels(lbl, rotation=60, fontsize=8.5, ha='right')
    ax.set_ylabel('peak success', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=.3, axis='y')

    ax2 = fig.add_axes([0.775, 0.44, 0.185, 0.40])
    r35 = np.array([r['a35_succ'] / r['f3_succ'] for r in cmp15 if r['f3_succ'] > 0])
    r13 = np.array([r['a35_succ'] / r['a135_succ'] for r in cmp15 if r['a135_succ'] > 0])
    ax2.boxplot([r13, r35], labels=['vs\n{1,3,5}', 'vs\nk=3'], widths=.6)
    ax2.axhline(1, color='#444444', ls='--', lw=1.5)
    ax2.set_yscale('log')
    ax2.set_ylabel('success ratio, {3,5} / other', fontsize=12)
    ax2.set_title('median %.2f  /  %.2f' % (np.median(r13), np.median(r35)), fontsize=12)
    ax2.grid(alpha=.3, axis='y')

    nb = sum(1 for r in cmp15 if r['a35_hmin'] > 0)
    nw = sum(1 for r in cmp15 if r['a35_succ'] > r['a135_succ'])
    foot(fig, 0.31, [
        f'Removing the sub-U1 windows helps, unambiguously: the new machine beats the old in '
        f'{nw} of {len(cmp15)} cells (median ratio {np.median(r13):.2f}), and -- the real reversal -- '
        f'its best floor is a binding one (h_min > 0) in {nb} of {len(cmp15)} cells, where in the '
        'unrestricted machine h_min = 0 won in all 21. Once the floor can no longer select a window '
        'that cannot see the root, the entropy criterion starts paying.',

        f'It still is not enough. Fixed k = 3 wins in {sum(1 for r in cmp15 if r["a35_succ"] <= r["f3_succ"])} '
        f'of {len(cmp15)} cells, median ratio {np.median(r35):.2f}; the three wins are +0.0032, +0.0004 '
        'and +0.0142, which is not a result. The prediction registered before the run was that {3,5} '
        'with a binding floor would beat both the control and fixed k = 3. Half of that held. The '
        'null stated alongside it -- that at L = 2, U1 already is the schedule -- is the better fit.',
    ])
    pdf.savefig(fig); plt.close(fig)


def page_traces(pdf, tr, n_show=10):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('Two generations, reveal by reveal: what the floor changes',
                 fontsize=20, weight='bold')
    cfg = tr['config']
    F, C = tr['floor']['traj'], tr['control']['traj']
    # the same sample index on both sides -- same sampler, same seed, same draw
    # order, only h_min different -- so the two columns are one controlled
    # comparison rather than two unrelated generations.  Without this the pair
    # that comes out is usually the *same* trajectory twice: the runs share
    # their early reveals and only diverge once the posterior sharpens.
    want = [b for b in range(len(F))
            if F[b]['novel'] and F[b]['grammatical'] and not C[b]['novel']]
    if not want:
        want = [b for b in range(len(F))
                if (F[b]['novel'], F[b]['grammatical']) != (C[b]['novel'], C[b]['grammatical'])]
    b0 = want[0] if want else 0
    picks = {'floor': F[b0], 'control': C[b0]}

    for i, tag in enumerate(('floor', 'control')):
        q = picks[tag]
        ax = fig.add_axes([0.06 + 0.47 * i, 0.46, 0.17, 0.34])
        g = np.array(q['grid'])
        ax.imshow(g, cmap=CMAP, vmin=0, vmax=7, interpolation='nearest', rasterized=True)
        for a in range(g.shape[0]):
            for b in range(g.shape[1]):
                ax.text(b, a, int(g[a, b]), ha='center', va='center', fontsize=12, color='w')
        for t_ in range(0, g.shape[0] + 1, 2):
            ax.axhline(t_ - .5, color='#666666', lw=1.2)
            ax.axvline(t_ - .5, color='#666666', lw=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        # three short lines rather than one wide one: centred on the 0.17-wide
        # grid axes, a single "novel NO (copy), grammatical yes" runs out over
        # the monospace trace block to its right
        ax.set_title('%s  (h_min = %.1f)\nnovel: %s\ngrammatical: %s'
                     % (tag, tr[tag]['h_min'],
                        'yes' if q['novel'] else 'NO (copy)',
                        'yes' if q['grammatical'] else 'no'), fontsize=11.5)

        # the last reveals, not the first: with a shared seed both runs open
        # identically at k = 5 on an empty grid and only separate once the
        # posterior sharpens, so the first ten rows would be the same twice
        rows = ['step  site   k      H(nats)  token']
        for rv in q['reveals'][-n_show:]:
            rows.append('%4d  %4d  %3.0f  %9.3f  %5d'
                        % (rv['step'], rv['site'], rv['k'], rv['H'], rv['token']))
        kk = [rv['k'] for rv in q['reveals']]
        hh = [rv['H'] for rv in q['reveals']]
        rows.insert(1, '  (first %d reveals omitted, k = 5 in both)'
                    % max(0, len(q['reveals']) - n_show))
        rows.append('mean k %.2f   total H %.2f nats' % (np.mean(kk), np.sum(hh)))
        fig.text(0.255 + 0.47 * i, 0.80, '\n'.join(rows), fontsize=9.5,
                 family='monospace', va='top', linespacing=1.45)

    fk = [rv['k'] for rv in picks['floor']['reveals']]
    ck = [rv['k'] for rv in picks['control']['reveals']]
    foot(fig, 0.33, [
        f'Both columns are generation #{b0} of the same batch: same grammar and training set '
        f'(v = {cfg["v"]}, s = {cfg["s"]}, L = {cfg["L"]}, m = {cfg["m"]}, P = {cfg["P"]}, rule seed '
        f'{cfg["rule_seed"]}), same sampler, same random seed. Only h_min differs, so every '
        'difference below is caused by the floor. Sites are listed in the order the '
        f'sampler revealed them, one per step, over all {cfg["n_sites"]} sites. k is the window the '
        'entropy criterion chose at that site at that step, H the entropy of the posterior it was '
        'then sampled from.',

        f'The floor run holds a mean window of {np.mean(fk):.2f} against {np.mean(ck):.2f} for the '
        f'control, and the gap opens late: both start at k = 5 while the grid is empty, and only the '
        f'floored run steps down as the window fills and the posterior sharpens. Over these 64 '
        f'generations the floor gives novelty {tr["floor"]["novel"]:.3f} against '
        f'{tr["control"]["novel"]:.3f}, grammaticality '
        f'{tr["floor"]["grammatical"]:.3f} against {tr["control"]["grammatical"]:.3f}, and success '
        f'{tr["floor"]["success"]:.3f} against {tr["control"]["success"]:.3f} -- the same direction '
        'as the 4096-generation sweep, which is the check that these two columns are typical.',
    ])
    pdf.savefig(fig); plt.close(fig)


def page_conclusion(pdf, cmp15):
    fig = plt.figure(figsize=PAGE)
    fig.suptitle('What this says about setting the patch size', fontsize=21, weight='bold')
    txt = [
        ('1. The direction is confirmed and it is not an assumption.',
         'The entropy criterion makes k shrink along the generation at every floor and in both '
         'machines, because the observed sites inside a window only accumulate and the posterior '
         'only sharpens. Nobody imposed that schedule; it falls out of the floor.'),
        ('2. The floor does stop memorization, exactly as intended.',
         'On the unrestricted machine, h_min 0 -> 1.0 lifts novelty 0.460 -> 0.997, median over the '
         '21 cells (0.487 -> 0.975 on the single cell plotted earlier). A working anti-copying '
         'device -- just not, by itself, a working sampler.'),
        ('3. The failure was the choice set, and fixing it reversed the floor verdict.',
         'Letting the floor pick k = 1 spends grammaticality on windows too small to contain a '
         'top-level child block. With k >= U1 the best floor becomes a binding one in 15/15 cells, '
         'against 0/21 before, and the machine beats its predecessor in 15/15.'),
        ('4. It still loses to simply fixing k = U1 (12/15 cells, median ratio 0.775).',
         'So at L = 2 the honest answer is the null: U1 already is the schedule. The criterion '
         'earns its keep against other adaptive machines, not against the fixed window.'),
        ('5. The test is thin where it matters, and that is the next experiment.',
         'Above U1 at L = 2 there are only two rungs, 3 and 5. A schedule has nowhere to go. L = 3 '
         'has U1 = 7 and real room, but its k = 7 tables overflow the int64 window code -- fixing '
         'that is the prerequisite for a fair test, and it is now the top of the queue.'),
    ]
    y = 0.845
    for head, body in txt:
        fig.text(0.055, y, head, fontsize=14.5, weight='bold', va='top', color='#b40426')
        foot(fig, y - 0.043, [body], width=124, size=11.5)
        y -= 0.148
    # wrapped by hand: matplotlib's wrap=True breaks at the figure edge, which
    # ran this four-line caveat up into the last bullet's body
    fig.text(0.055, 0.035, textwrap.fill(
        'Caveat on "cannot be grammatical". A window below U1 cannot see the root in one look, '
        'but overlapping windows still chain constraints: in 1D at n = 16 a k = 3 window scores '
        '0.06 grammaticality, and the success optimum there sits at k = 11, below U1 = 15. '
        '"k >= U1" is the right floor for this 2D L = 2 machine, not a general law.', 138),
        fontsize=10.5, va='bottom', style='italic', color='#444444', linespacing=1.4)
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/rhm_sampling/2026-11-04')
    ap.add_argument('--out', required=True)
    ap.add_argument('--png_dir', default=None)
    args = ap.parse_args()

    cmp15, cmp21, tr, old, new = load(ROOT / args.results)
    from src.utils.random_hierarchy import RandomHierarchyModel2D
    cfg = tr['config']
    rhm = RandomHierarchyModel2D(v=cfg['v'], s=cfg['s'], L=cfg['L'], m=cfg['m'],
                                 seed=1000 + cfg['rule_seed'])
    grid0 = np.array(cfg['train_grids'][0])

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as raw:
        pdf = Tee(raw, args.png_dir)
        page_title(pdf, cmp15, cmp21)
        page_tokenization(pdf, rhm, grid0)
        page_reread(pdf, cmp21, old)
        page_kt(pdf, old, new)
        page_three(pdf, cmp15)
        page_traces(pdf, tr)
        page_conclusion(pdf, cmp15)
    print('wrote %s (%.0f kB)' % (out, out.stat().st_size / 1024))


if __name__ == '__main__':
    main()
