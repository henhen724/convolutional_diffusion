"""Deck: the entropy budget answers "why k=11" and fails to close the 1D/2D gap.

Reuses the peak-deck helpers so the page furniture matches the three decks
already sent, and rebuilds its own trace page: the point of this run is what
happens *inside* a generation, so the traces carry the per-reveal entropy
rather than only the posterior on one token.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_1d_slides import page_tokenization
from scripts.build_rhm_peak_slides import PAGE, Tee, foot
from scripts.build_rhm_slides import title_page
from scripts.rhm_1d_depth import MASK, LS1D, RHM1D, is_grammatical
from scripts.rhm_entropy_window import generate_H


def med(x):
    x = [v for v in x if np.isfinite(v)]
    return float(np.median(x)) if x else float('nan')


def by_k(peaks, n, field):
    sel = [p for p in peaks if p['dim'] == 1 and p['n'] == n and p['bracketed']]
    ks = sorted({p['k'] for p in sel})
    return ks, [med([p[field] for p in sel if p['k'] == k]) for k in ks]


def page_why_k11(pdf, peaks):
    """The answer to the question the run was launched for."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=PAGE)
    fig.subplots_adjust(left=.07, right=.97, top=.86, bottom=.47, wspace=.28)
    fig.text(.055, .95, '3. Why success peaks at k=11: the window is paid for in entropy',
             fontsize=20, weight='bold', va='top')

    ks, H = by_k(peaks, 16, 'H_at_peak')
    _, S = by_k(peaks, 16, 'succ_max')
    ax.plot(ks, H, 'o-', color='#2c7fb8', lw=2.5, ms=7)
    ax.axvline(15, ls='--', color='k', lw=1.5)
    ax.text(14.7, min(H) + .15, 'U=1 (k=15)', rotation=90, fontsize=10, ha='right')
    ax.axvline(11, ls=':', color='#b40426', lw=2)
    ax.text(11.3, min(H) + .15, 'best k = 11', rotation=90, fontsize=10, color='#b40426')
    ax.set_xticks(ks)
    ax.set_xlabel('window k   (n = 16 sites)', fontsize=12)
    ax.set_ylabel('H_tot at the peak  [nats per generation]', fontsize=12)
    ax.set_title('a wider window removes entropy (one reversal, at k=9)', fontsize=12.5)
    ax.grid(alpha=.3)

    # the ledger itself, rather than success-against-H: plotted against H the
    # two sides double back on each other at k=7/9, where H reverses, and the
    # resulting fold reads as noise instead of as the trade it is
    _, G = by_k(peaks, 16, 'gram_at')
    _, N = by_k(peaks, 16, 'novel_at')
    ax2.plot(ks, G, 'o-', color='#1a7f37', lw=2.5, ms=7, label='grammatical (bought)')
    ax2.plot(ks, N, 's-', color='#2c7fb8', lw=2.5, ms=7, label='novel (paid)')
    ax2.plot(ks, S, '^-', color='#b40426', lw=3, ms=8, label='both = success')
    ax2.axvline(11, ls=':', color='#b40426', lw=2)
    ax2.set_xticks(ks)
    ax2.set_xlabel('window k   (n = 16 sites)', fontsize=12)
    ax2.set_ylabel('fraction at the peak', fontsize=12)
    ax2.set_title('the trade: grammar rises, novelty falls, product peaks', fontsize=12.5)
    ax2.legend(fontsize=10, loc='center left')
    ax2.grid(alpha=.3)

    foot(fig, .38, [
        'THE LEDGER. Over k = 3..17 at n=16 the trajectory entropy falls 11.82 -> 8.21 nats, '
        'a third of what is there, with one reversal (k=7 -> 9) and no feature whatever at '
        'U1=15. Grammaticality over the same range rises 0.06 -> 0.78 while novelty at the '
        'peak falls 1.00 -> 0.52. The window buys grammar and pays in entropy, and entropy is '
        'the currency novelty is denominated in -- so the optimum is wherever the marginal '
        'grammar gain stops covering the marginal novelty loss.',
        'At n=16 that is k=11, at H_tot = 10.11 nats of a possible 16 ln 3 = 17.6. At n=8 it '
        'is k=5. Neither is U=1, and nothing in the entropy curve knows where U=1 is -- which '
        'is the point: the block structure sets where a window can first be grammatical at '
        'all, and the entropy sets where it should actually sit.',
    ], width=132, size=11.5)
    pdf.savefig(fig); plt.close(fig)


def page_mechanism(pdf, rows1, rows2):
    """Copies come from low-entropy trajectories -- the claim's one survivor."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=PAGE)
    fig.subplots_adjust(left=.07, right=.97, top=.86, bottom=.47, wspace=.28)
    fig.text(.055, .95, '4. The mechanism holds at the level of single samples',
             fontsize=20, weight='bold', va='top')

    for rows, dim, c in ((rows1, 1, '#2c7fb8'), (rows2, 2, '#b40426')):
        sel = [r for r in rows if 0.02 < r['novel'] < 0.98
               and np.isfinite(r['H_novel']) and np.isfinite(r['H_copy'])]
        d = [r['H_novel'] - r['H_copy'] for r in sel]
        ax.hist(d, bins=40, alpha=.6, color=c, label='%dD  (%d runs, median %+.2f)'
                % (dim, len(sel), float(np.median(d))))
    ax.axvline(0, color='k', lw=1.5)
    ax.set_xlabel('H(novel generations) - H(copies)   [nats]', fontsize=12)
    ax.set_ylabel('runs', fontsize=12)
    ax.set_title('within every run, copies are the low-entropy trajectories', fontsize=12.5)
    ax.legend(fontsize=10)
    ax.grid(alpha=.3)

    sel = [r for r in rows1 if 0.02 < r['novel'] < 0.98]
    ax2.scatter([r['P'] * np.exp(-r['H_tot']) for r in sel],
                [-np.log(r['novel']) for r in sel], s=9, alpha=.35, color='#2c7fb8',
                rasterized=True, label='1D')
    sel2 = [r for r in rows2 if 0.02 < r['novel'] < 0.98]
    ax2.scatter([r['P'] * np.exp(-r['H_tot']) for r in sel2],
                [-np.log(r['novel']) for r in sel2], s=14, alpha=.6, color='#b40426',
                rasterized=True, label='2D')
    lo, hi = 1e-9, 10
    ax2.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='the prediction (identity)')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('predicted:  P / exp(H_tot)', fontsize=12)
    ax2.set_ylabel('observed:  -ln(novel fraction)', fontsize=12)
    ax2.set_title('but the quantitative law misses by 2x in 1D, 600x in 2D', fontsize=12.5)
    # lower right: the upper-left corner holds the 2D points that fall furthest
    # below the identity, which are the ones the panel exists to show
    ax2.legend(fontsize=9.5, loc='lower right')
    ax2.grid(alpha=.3)

    foot(fig, .38, [
        'WHAT SURVIVES. Split each run\'s generations into the novel ones and the copies and '
        'compare their mean trajectory entropy. The copies come out lower in 84% of 1D runs, '
        'by a median 1.75 nats, and in 100% of 2D runs, by 7.42. The sampler reproduces a '
        'training string exactly when the windows pinned it, and pinning is visible as missing '
        'entropy. That is a mechanism, not a correlation: it is measured inside the run that '
        'produced the sample.',
        'WHAT DOES NOT. Turning it into a count fails. If the machine spread its mass over '
        'exp(H_tot) strings, novelty would be exp(-P/exp(H_tot)) -- the dashed line. It is off '
        'by 2.06x in 1D with a 15x spread and by 597x in 2D with a 171x spread. exp(H_tot) is '
        'not the size of an effective support; the distribution is far too uneven for its '
        'entropy to count anything.',
    ], width=132, size=11.5)
    pdf.savefig(fig); plt.close(fig)


def page_gap(pdf, peaks):
    """x* before and after, on the same bracketed-peak eval."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=PAGE)
    fig.subplots_adjust(left=.08, right=.97, top=.86, bottom=.47, wspace=.30)
    fig.text(.055, .95, '5. x* before and after: the budget makes the gap 258x worse',
             fontsize=20, weight='bold', va='top')

    def grp(dim, n, u1_only):
        s = [p for p in peaks if p['dim'] == dim and p['n'] == n and p['bracketed']]
        if u1_only:
            s = [p for p in s if p['k'] == p['U1']]
        return s
    a, b = grp(1, 16, True), grp(2, 16, False)
    meas = [med([p['ratio'] for p in a]), med([p['ratio'] for p in b])]
    pred = [med([p['x_pred'] for p in a]), med([p['x_pred'] for p in b])]

    x = np.arange(2)
    ax.bar(x - .18, meas, .34, label='x* measured  = P*/|R_L|', color='#2c7fb8')
    ax.bar(x + .18, pred, .34, label='x* predicted = exp(H_tot)/|R_L|', color='#b40426')
    for i, (u, v) in enumerate(zip(meas, pred)):
        ax.text(i - .18, u * 1.15, '%.3f' % u, ha='center', fontsize=11, weight='bold')
        ax.text(i + .18, v * 1.15, '%.3g' % v, ha='center', fontsize=11, weight='bold')
    ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels(['1D, n=16\n(k = U1 = 15)', '2D, n=16\n(k = U1 = 3)'])
    ax.set_ylabel('x*', fontsize=12)
    ax.set_title('measured 1.92x apart; predicted 496x apart', fontsize=12.5)
    ax.legend(fontsize=9.5, loc='upper left')
    ax.grid(alpha=.3, axis='y')

    sel = [p for p in peaks if p['bracketed'] and np.isfinite(p['H_at_peak'])]
    for dim, c, mk in ((1, '#2c7fb8', 'o'), (2, '#b40426', '^')):
        s = [p for p in sel if p['dim'] == dim]
        ax2.scatter([p['H_at_peak'] for p in s], [p['P_star'] for p in s],
                    s=26, marker=mk, alpha=.7, color=c, label='%dD (%d peaks)' % (dim, len(s)))
    H = np.array([p['H_at_peak'] for p in sel])
    lP = np.log([p['P_star'] for p in sel])
    c0, c1 = np.polyfit(H, lP, 1)
    xs = np.linspace(H.min(), H.max(), 20)
    ax2.plot(xs, np.exp(c0 * xs + c1), 'k--', lw=1.5,
             label='pooled fit, slope %.2f' % c0)
    ax2.set_yscale('log')
    ax2.set_xlabel('H_tot at the peak  [nats]', fontsize=12)
    ax2.set_ylabel('P*', fontsize=12)
    ax2.set_title('H explains almost none of ln P*: slope 0.10, not 1', fontsize=12.5)
    ax2.legend(fontsize=9.5)
    ax2.grid(alpha=.3)

    foot(fig, .38, [
        'BEFORE, the same bracketed-peak eval both decks use, but for the first time with one '
        'instrument on both sides rather than two runs: 1D n=16 at U1=15 gives x* = 0.083 '
        '(6 cells), 2D n=16 gives 0.159 (all 25 bracketed L=2 cells). Gap 1.92x. The 2D '
        'figure reproduces the published 0.193 to 1.2x as a population median, but cell by '
        'cell the peak estimate scatters 4.8x -- 2048 generations and a 3-point parabola on '
        'a doubling ladder -- so it is a population number and no single cell should be '
        'quoted from it.',
        'AFTER, substituting the entropy prediction x* = exp(H_tot)/|R_L|: 1D 0.68, 2D 337.6. '
        'Gap 496x -- 258 times worse than the thing it was meant to explain. And as a mere '
        'collapse variable H fails too: ln P* = 0.100 H + 4.18 pooled, residual 5.5x, and '
        'adding a dimension term still leaves a 2.1x 2D offset. The entropy budget joins '
        'coverage and table size on the refuted pile.',
    ], width=132, size=11.5)
    pdf.savefig(fig); plt.close(fig)


# P=8192 on this cell, because the page needs both outcomes side by side and
# that rung is where they are closest to even: novel 0.71, copies 0.29.
def page_traces(pdf, v=3, s=2, L=4, m=2, rs=0, P=8192, k=11, n_show=10):
    """One novel and one copy generation, with the entropy of every reveal."""
    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '6. A novel generation and a copy, reveal by reveal',
             fontsize=20, weight='bold', va='top')
    model = RHM1D(v, s, L, m, seed=1000 + rs)
    d = model.d
    train = model.sample(P, seed=1234 + 7 * rs)
    tr = set(map(bytes, np.ascontiguousarray(train.astype(np.int8))))
    dev = torch.device('cpu')
    machine = LS1D(train, v, k, dev, alpha_smooth=1e-3)
    fig.text(.055, .885, 'v=%d, s=%d, L=%d (n=%d sites), m=%d;  P=%d training strings;  '
             'window k=%d;  alphabet {0,1,2}, "." = still masked'
             % (v, s, L, d, m, P, k), fontsize=12.5, va='top', color='#333')

    # generate a batch, then show one that is novel and one that is a copy
    g = torch.Generator(device=dev).manual_seed(1234 + 7 * rs + 9973)
    gs, H = generate_H(machine, 256, d, dev, g)
    arr = np.ascontiguousarray(gs.numpy().astype(np.int8))
    novel = np.array([bytes(r) not in tr for r in arr])
    gram = is_grammatical(model, gs.numpy())
    # the novel column should show a *success*, i.e. novel and grammatical both,
    # since that is the event the whole project measures; fall back to any novel
    # generation if this rung happens not to produce one
    ok = np.where(novel & gram)[0]
    if not len(ok):
        ok = np.where(novel)[0]
    pick = {'novel': int(ok[0]), 'copy': int(np.where(~novel)[0][0])}

    # replay the two chosen trajectories one at a time so the per-step entropy
    # can be printed; the sampler is deterministic given its own generator, so
    # each replay reproduces the sample it is labelled with
    for col, (lab, i) in enumerate(pick.items()):
        gg = torch.Generator(device=dev).manual_seed(1234 + 7 * rs + 9973)
        xt = torch.full((256, d), MASK, dtype=torch.int64)
        rows_ = torch.arange(256)
        log = []
        for _ in range(d):
            msk = xt == MASK
            if not bool(msk.any()):
                break
            noise = torch.rand(256, d, generator=gg)
            pk = torch.where(msk, noise, torch.full_like(noise, -1.0)).argmax(1)
            p = machine.posterior(xt, pk).clamp_min(1e-12)
            tok = torch.multinomial(p, 1, generator=gg)[:, 0]
            keep = msk[rows_, pk]
            h = -(p * p.log()).sum(1)
            log.append((''.join('.' if t < 0 else str(int(t)) for t in xt[i]),
                        int(pk[i]), int(tok[i]), float(h[i])))
            xt[rows_[keep], pk[keep]] = tok[keep]

        x0 = .055 + .47 * col
        tot = float(H[i])
        fig.text(x0, .82, '%s   (H_tot = %.2f nats)' % (lab.upper(), tot),
                 fontsize=14, weight='bold', va='top',
                 color='#1a7f37' if lab == 'novel' else '#b40426')
        y = .775
        for (st, si, tk, h) in log[:n_show]:
            fig.text(x0, y, '%s  site %2d -> %d   H = %.2f' % (st, si, tk, h),
                     fontsize=9.5, va='top', family='monospace')
            y -= .026
        fig.text(x0, y, '...  final %s' % ''.join(str(int(t)) for t in xt[i]),
                 fontsize=9.5, va='top', family='monospace')
        y -= .032
        fig.text(x0, y, '     grammatical: %s      novel: %s'
                 % ('yes' if gram[i] else 'NO', 'yes' if novel[i] else 'NO'),
                 fontsize=10.5, va='top', weight='bold',
                 color='#1a7f37' if (gram[i] and novel[i]) else '#b40426')

    foot(fig, .40, [
        'Each row is one reveal: the string so far, the site the sampler chose uniformly among '
        'those still masked, the token it drew, and H, the entropy in nats of the posterior it '
        'drew from. H_tot is the sum over all %d reveals -- the quantity every plot in this '
        'deck is built on. The first %d reveals are shown.' % (d, n_show),
        'The contrast is the mechanism in miniature, and it is worth being exact about its '
        'size: these two trajectories differ by 1.2 nats, against a median gap of 1.75 over '
        'the 781 1D runs. What makes the difference is reveals the windows have already '
        'pinned -- H = 0.00 exactly, where the k=%d windows admit one continuation and no '
        'other. The copy hits two in its first ten; the novel generation hits none, and every '
        'reveal with two or more live tokens is a place it can leave the training set without '
        'leaving the grammar.' % k,
    ], width=132, size=11.5)
    pdf.savefig(fig); plt.close(fig)


def page_conclusion(pdf):
    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '7. Where this leaves the 1D/2D gap', fontsize=22,
             weight='bold', va='top')
    foot(fig, .87, [
        'ANSWERED. Why success peaks at k=11 rather than at the root-reaching U1=15. Widening '
        'the window from k=3 to k=17 removes 3.61 of 11.82 nats from the generation while '
        'grammaticality rises 0.06 -> 0.78 and novelty falls 1.00 -> 0.52. Copies are the '
        'low-entropy trajectories -- 1.75 nats below the novel ones in 1D, 7.42 in 2D, in 84% '
        'and 100% of runs. So the window is bought with exactly the quantity novelty needs, '
        'and the optimum is where the exchange stops paying. Nothing in the entropy curve '
        'marks U=1; the block structure says where a window can first be grammatical, the '
        'entropy says where it should sit.',
        '',
        'STILL OPEN, AND NOW WITH A THIRD REFUTATION. x* = P*/|R_L| measured head-to-head on '
        'one instrument: 1D 0.083, 2D 0.159, 1.92x apart. Substituting exp(H_tot)/|R_L| gives '
        '0.68 and 337.6 -- 496x apart, 258x worse. Pooled, ln P* = 0.100 H + 4.18 with a 5.5x '
        'residual, so H carries almost none of ln P*. Coverage, table size and now entropy '
        'have all failed to collapse the dimensions.',
        '',
        'WHAT THE FAILURE ITSELF SAYS. exp(H_tot)/|R_L| is useless as a constant but reads '
        'cleanly as a regime: it is 0.68 in 1D and 338 in 2D. The 2D machine spreads its mass '
        'over vastly more strings than the grammar contains, so its peak is grammar-limited '
        '(peak success 0.022); the 1D machine is spread comparably to the grammar, so its peak '
        'is novelty-limited (peak success 0.49). Two different binding constraints is a reason '
        'no single ratio should collapse them, and it is measured rather than assumed. I am '
        'not calling it the explanation -- it is a reinterpretation of a number that failed '
        'its own test, and it predicts nothing yet.',
        '',
        'NEXT. The AdaptiveWindow machine already in the repo picks k per site and per step off '
        'an entropy floor, which is what page 3 says the optimum actually is -- but it has only '
        'ever been scored on grammaticality, never on novel-and-grammatical. Scoring it is the '
        'measurement these three decks have been converging on.',
    ], width=130, size=11.5)
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--peaks', default='results/rhm_entropy/2026-11-04/entropy_peaks.json')
    ap.add_argument('--e1d', default='results/rhm_entropy/2026-11-04/entropy_1d.json')
    ap.add_argument('--e2d', default='results/rhm_entropy/2026-11-04/entropy_2d_L2.json')
    ap.add_argument('--out', default='results/rhm_entropy/2026-11-04/rhm_entropy_slides.pdf')
    ap.add_argument('--png_dir', default='results/rhm_entropy/2026-11-04/pages_ent')
    args = ap.parse_args()

    peaks = json.load(open(ROOT / args.peaks))
    rows1 = json.load(open(ROOT / args.e1d))
    rows2 = json.load(open(ROOT / args.e2d))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as raw:
        pdf = Tee(raw, ROOT / args.png_dir)
        title_page(
            pdf,
            'Why the LS machine\'s best window is k=11,\nand why entropy still does not'
            ' close the 1D/2D gap',
            'The window is bought with entropy, and copies are the low-entropy\n'
            'trajectories -- but x* goes from 1.92x apart to 496x apart when the\n'
            'entropy prediction is substituted for it. Third refutation.',
            # 3/2/3/2 lines: title_page steps y by 0.08 + 0.040 per line and does
            # not wrap, so 13 bullet lines under a 3-line subtitle runs off the page
            ['THE PREDICTION, REGISTERED BEFORE THE RUN. A generation escapes a training\n'
             'set of P strings only if it injects more than ln P nats, so the effective\n'
             'support is exp(H_tot) and x* = exp(H_tot)/|R_L| with no fitted constant.',
             'THE RUN. Jobs 51073161 and 51076386. 988 1D rows over 50 bracketed curves,\n'
             'plus all 25 bracketed 2D L=2 cells on the same instrument -- first head-to-head.',
             'CONFIRMED. The mechanism, sample by sample: copies carry 1.75 fewer nats in\n'
             '1D and 7.42 fewer in 2D, in 84% and 100% of runs. Widening k costs 11.82 ->\n'
             '8.21 nats over k = 3..17, which is why the optimum is k=11, not U1=15.',
             'REFUTED. Every quantitative form. novel = exp(-P/exp(H_tot)) misses by 2x in\n'
             '1D, 600x in 2D; pooled ln P* = 0.100 H + 4.18, slope 0.10 where it needs 1.'])
        # the shared page's default closing line calls U=1 "the whole story of
        # this deck", which was true of the first two decks and is the opposite
        # of this one's finding
        page_tokenization(pdf, closing='and this deck is about why generating at that '
                          'window is the wrong choice.')
        page_why_k11(pdf, peaks)
        page_mechanism(pdf, rows1, rows2)
        page_gap(pdf, peaks)
        page_traces(pdf)
        page_conclusion(pdf)
    print('wrote %s (%.0f kB, %d pages)' % (out, out.stat().st_size / 1e3, pdf.n))


if __name__ == '__main__':
    main()
