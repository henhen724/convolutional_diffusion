#!/usr/bin/env python
"""Slide PDF: the coverage hypothesis, tested and refuted; the window optimum.

The 1D depth probe (2026-11-03) ended with one open "why": x* = P*/|R_L| is
0.050 in 1D against 0.193 in 2D at matched n=16, and the standing explanation
was window coverage.  This deck reports the direct test -- sweep k at fixed n,
dimension and grammar -- which refutes it with the sign backwards, and the
positive result that came out of the same sweep: the root-reaching window U=1,
used throughout this program, is past the optimum for generating samples.

Seven landscape pages: headline, tokenization, the refutation, the alternative
that also fails, the window optimum with its mechanism, worked traces at the
two window sizes, and what changes.

Usage:
  python scripts/build_coverage_slides.py \
      --peaks results/rhm_1d/2026-11-04/coverage_peaks.json \
      --out results/rhm_1d/2026-11-04/rhm_coverage_slides.pdf
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
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_rhm_peak_slides import PAGE, Tee, foot                  # noqa: E402
from scripts.build_rhm_slides import title_page                           # noqa: E402
from scripts.build_1d_slides import page_tokenization                     # noqa: E402

# measured 2D values these are tested against (STATUS A/C, 30 bracketed peaks)
TWO_D = {16: dict(x=0.193, cov=9 / 16, k=3), 64: dict(x=0.207, cov=49 / 64, k=7)}
MAX_CHANCE = 0.05


def med(x):
    return float(np.median(x)) if len(x) else float('nan')


def by_k(peaks, n, field='ratio', bracketed=True):
    """Median of ``field`` per k at depth n, over density-passing cells."""
    sel = [p for p in peaks if p['n'] == n and p['chance'] <= MAX_CHANCE
           and (p['bracketed'] or not bracketed)]
    ks = sorted({p['k'] for p in sel})
    return ks, [med([p[field] for p in sel if p['k'] == k]) for k in ks]


def page_refutation(pdf, peaks):
    """x* against window size at fixed n -- the registered prediction, refuted.

    The prediction is drawn on the plot rather than described beside it: the
    coverage hypothesis is a *horizontal* line through 2D's 0.193, since it
    says cells at equal coverage have equal x*.  The measured curve crosses it
    once, steeply, which is the refutation in one picture.
    """
    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '3. The test: sweep k at fixed n. Coverage is refuted, '
             'sign backwards', fontsize=22, weight='bold', va='top')

    ax = fig.add_axes([.07, .40, .40, .45])
    for n, c in ((8, '#2c7fb8'), (16, '#b40426')):
        ks, xs = by_k(peaks, n)
        cov = [k / n for k in ks]
        ax.plot(cov, xs, 'o-', color=c, lw=2, ms=7, label='1D, n=%d' % n)
    for n, t in TWO_D.items():
        ax.plot([t['cov']], [t['x']], '*', ms=20, color='k',
                label='2D, n=%d (measured)' % n)
    ax.axhline(TWO_D[16]['x'], ls='--', lw=1.5, color='#888')
    # below the line, not above: above it collides with the axis title
    ax.text(.17, TWO_D[16]['x'] * .60, 'coverage hypothesis:\n1D should meet 2D here',
            fontsize=10.5, color='#555')
    ax.set_yscale('log')
    ax.set_xlabel('window coverage  k / n', fontsize=12)
    ax.set_ylabel('x* = P* / |R_L|', fontsize=12)
    ax.set_title('x* rises with coverage; the hypothesis needs it to fall',
                 fontsize=12.5, pad=10)
    ax.legend(fontsize=9.5, loc='lower right')
    ax.grid(alpha=.3)

    ax2 = fig.add_axes([.57, .40, .38, .45])
    labels = ['matched n\n(n=16)', 'matched coverage\n(9/16)', 'matched k\n(k=3)']
    ks16, xs16 = by_k(peaks, 16)
    g_n = TWO_D[16]['x'] / xs16[ks16.index(15)]
    g_c = TWO_D[16]['x'] / xs16[ks16.index(9)]
    g_k = TWO_D[16]['x'] / xs16[ks16.index(3)]
    gaps = [g_n, g_c, g_k]
    ax2.bar(labels, gaps, color=['#2c7fb8', '#b40426', '#999'])
    for i, g in enumerate(gaps):
        ax2.text(i, g * 1.05, '%.1fx' % g, ha='center', fontsize=13, weight='bold')
    ax2.set_yscale('log')
    ax2.set_ylabel('1D-to-2D gap in x*  (1 = explained)', fontsize=12)
    ax2.axhline(1, color='k', lw=1)
    ax2.set_title('every way of matching leaves a gap -- coverage widens it',
                  fontsize=12.5)
    ax2.grid(alpha=.3, axis='y')

    foot(fig, .30, [
        'REGISTERED BEFORE THE RUN. Coverage drives x* -> x* rises as k falls, reaching '
        '~0.19 at k=9, whose coverage 9/16 is exactly 2D L=2\'s. Seeing the root drives it '
        '-> x* flat for k >= U1, success collapsing below.',
        'MEASURED. x* *falls* as k falls, monotonically: 0.004 at k=3 to 0.100 at k=17, a '
        '25x spread, Spearman(coverage, x*) = +1.00 at n=16 and +0.80 at n=8. The '
        'hypothesis required the opposite sign. At matched coverage 1D gives 0.038 against '
        '2D\'s 0.193 -- 5.1x apart, where matched n alone leaves 2.3x on these same cells '
        '(the depth probe, a wider cell mix, put it at 3.2x). Matching on coverage more '
        'than doubles the discrepancy it was introduced to explain.',
    ], width=132, size=11.5)
    pdf.savefig(fig); plt.close(fig)


def page_table_size(pdf, peaks, cv):
    """The other candidate -- table saturation -- fails in both directions.

    Worth its own page because it is the more principled of the two: the
    machine sees windows, not samples, so "P* is set by how many distinct
    windows the grammar has" is the hypothesis a coupon-collector argument
    actually suggests.  It is still wrong.
    """
    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '4. The principled alternative fails too: it is not table size',
             fontsize=22, weight='bold', va='top')

    ax = fig.add_axes([.07, .42, .40, .43])
    for dim, c, mk in ((1, '#2c7fb8', 'o'), (2, '#b40426', 's')):
        sel = [q for q in cv if q['dim'] == dim]
        ax.scatter([q['N_k'] / q['n'] for q in sel], [q['P_star'] for q in sel],
                   s=28, color=c, marker=mk, alpha=.75, label='%dD (%d cells)'
                   % (dim, len(sel)))
    lo = min(q['N_k'] / q['n'] for q in cv)
    hi = max(q['N_k'] / q['n'] for q in cv)
    xx = np.array([lo, hi])
    ax.plot(xx, xx, 'k--', lw=1.5, label='P* = N_k / n  (full table)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('N_k / n   (windows the grammar admits, per sample)', fontsize=12)
    ax.set_ylabel('P*', fontsize=12)
    ax.set_title('the two dimensions sit on parallel lines, not one', fontsize=12.5)
    ax.legend(fontsize=9.5, loc='upper left')
    ax.grid(alpha=.3)

    ax2 = fig.add_axes([.57, .42, .38, .43])
    ks, xs = by_k(peaks, 16)
    ys = [q for q in zip(ks, xs)]
    ks2, yv = by_k(peaks, 16, field='ratio')
    # y* per k recomputed from the saturated table recorded in the sweep
    ax2.plot(ks, np.array(xs) / xs[0], 'o-', color='#2c7fb8', lw=2, ms=7,
             label='x* = P*/|R_L|   (25x)')
    ystar = cv_ystar_by_k(peaks)
    if ystar:
        kk, yy = ystar
        ax2.plot(kk, np.array(yy) / yy[-1], 's-', color='#b40426', lw=2, ms=7,
                 label='y* = P* n/N_k   (61x)')
    ax2.set_yscale('log')
    ax2.set_xlabel('window k   (n = 16)', fontsize=12)
    ax2.set_ylabel('spread, normalised to its own minimum', fontsize=12)
    ax2.set_title('y* is the *less* stable of the two across k', fontsize=12.5)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=.3)

    foot(fig, .32, [
        'The machine never sees a sample; it sees n windows per sample, drawn from the N_k '
        'the grammar admits. So the natural law is P* ~ N_k/n, and N_k knows about window '
        'geometry -- the one thing that genuinely differs between 1D and 2D at equal n. '
        'N_k is exact here, enumerated over the whole grammar by odometer, not sampled.',
        'It fails both ways. Across dimensions y* = P* n/N_k is 4.4x apart where x* is '
        '3.2x (both on the depth-probe cells of this panel, not page 3\'s k-sweep cells, '
        'which put the x* gap at 2.3x); within 1D y* spreads 61x across k, x* 25x. Note y* '
        'crosses 1 between the dimensions: 2D peaks at y* = 1.5-4.5, *after* the training '
        'set has supplied one window per distinct window, while 1D peaks at 0.35 with the '
        'table a third full. Neither quantity is the collapse variable, and the 1D/2D '
        'offset survives as a genuine dimensional effect rather than a bookkeeping error.',
    ], width=132, size=11.5)
    pdf.savefig(fig); plt.close(fig)


def cv_ystar_by_k(peaks):
    """y* per k at n=16, from each cell's own saturated table size."""
    import json as _j
    p = ROOT / 'results/rhm_1d/2026-11-04/coverage.json'
    if not p.exists():
        return None
    cells = {}
    for r in _j.load(open(p)):
        cells.setdefault((r['v'], r['m'], r['s'], r['L'], r['rule_seed'], r['k']),
                         []).append((r['P'], r['n_distinct_patches']))
    out = {}
    for pk in peaks:
        if pk['n'] != 16 or not pk['bracketed'] or pk['chance'] > MAX_CHANCE:
            continue
        key = (pk['v'], pk['m'], pk['s'], pk['L'], pk['rule_seed'], pk['k'])
        c = sorted(cells.get(key, []))
        if len(c) < 2 or c[-1][1] <= 0 or (c[-1][1] - c[-2][1]) / c[-1][1] > .02:
            continue
        out.setdefault(pk['k'], []).append(pk['P_star'] * pk['n'] / c[-1][1])
    if not out:
        return None
    ks = sorted(out)
    return ks, [med(out[k]) for k in ks]


def page_optimum(pdf, peaks):
    """The positive result: U=1 is past the optimum, and why.

    Grammaticality and novelty are plotted at the located peak rather than at
    their own best P, because success is their product *at a single P* and the
    trade is only meaningful read that way.
    """
    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '5. The window optimum sits BELOW the root-reaching window',
             fontsize=22, weight='bold', va='top')

    for i, n in enumerate((16, 8)):
        ax = fig.add_axes([.07 + .50 * i, .42, .38, .43])
        ks, g = by_k(peaks, n, 'gram_at')
        _, nv = by_k(peaks, n, 'novel_at')
        _, sc = by_k(peaks, n, 'succ_max')
        ax.plot(ks, g, 'o-', color='#2c7fb8', lw=2, ms=6, label='grammatical')
        ax.plot(ks, nv, 's-', color='#f4a261', lw=2, ms=6, label='novel')
        ax.plot(ks, sc, 'D-', color='#b40426', lw=2.5, ms=7, label='success (both)')
        u1 = [p for p in peaks if p['n'] == n][0]['U1']
        ax.axvline(u1, ls='--', color='k', lw=1.5)
        ax.text(u1 - .25, .04, 'U=1 (k=%d)' % u1, rotation=90, fontsize=10,
                ha='right', va='bottom')
        best = ks[int(np.argmax(sc))]
        ax.plot([best], [max(sc)], '*', ms=22, color='#b40426', zorder=5)
        # point the label into open space: leftward normally, but rightward when
        # the optimum sits near the left edge (n=8, best k=5), where a leftward
        # offset runs into the y-axis label and the legend
        left = best - ks[0] <= .45 * (ks[-1] - ks[0])
        ax.annotate('best k = %d' % best, xy=(best, max(sc)),
                    xytext=(best + (2.0 if left else -3.2), max(sc) + .17),
                    ha='left' if left else 'center',
                    fontsize=11, weight='bold', color='#b40426',
                    arrowprops=dict(arrowstyle='->', color='#b40426', lw=1.4))
        ax.set_xticks(ks)
        ax.set_ylim(0, 1.18)
        ax.set_xlabel('window k', fontsize=12)
        ax.set_ylabel('fraction, at the located peak P*', fontsize=12)
        ax.set_title('n = %d sites  (U=1 is k=%d)' % (n, u1), fontsize=12.5)
        ax.legend(fontsize=9.5, loc='upper left')
        ax.grid(alpha=.3)

    ks, sc = by_k(peaks, 16, 'succ_max')
    b16, u16 = max(sc), sc[ks.index(15)]
    foot(fig, .32, [
        'MECHANISM. Widening the window buys grammaticality and spends novelty. At n=16, '
        'across k = 3..17, grammaticality at the peak rises 0.06 -> 0.78 while novelty '
        'falls 1.00 -> 0.57. Success is their product, so it turns over -- at k=11, two '
        'rungs below the root-reaching U1=15, beating it by %.0f%% (%.3f vs %.3f). At n=8 '
        'the best is k=5 against U1=7, +8%%.' % (100 * (b16 / u16 - 1), b16, u16),
        'WHY IT MATTERS. U=1 is the window this whole program has used, chosen because it '
        'is the smallest that can see the root and therefore the smallest that can be '
        'fully grammatical. That reasoning optimises the wrong thing: a window that can '
        'always reach the root also reproduces its training strings, and hshunt\'s '
        'criterion scores a reproduced string as a failure. The useful window is the '
        'largest one that stays short of the root.',
    ], width=132, size=11.5)
    pdf.savefig(fig); plt.close(fig)


def page_traces(pdf):
    """Two real generations from one grammar, at k=11 and k=15.

    Recomputed live: the sweep stores rates, and the claim on page 5 is about
    what an individual generation does differently at the two window sizes --
    which only a trace shows.
    """
    import torch
    from scripts.rhm_1d_depth import RHM1D, LS1D, exact_RL, is_grammatical, MASK

    dev = torch.device('cpu')
    model = RHM1D(v=2, s=2, L=4, m=2, seed=1000)
    d, R = model.d, exact_RL(model)
    P = 64
    train = model.sample(P, seed=1234)
    tr = set(map(bytes, np.ascontiguousarray(train.astype(np.int8))))

    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '6. The same grammar at two windows, revealed one site at a time',
             fontsize=22, weight='bold', va='top')
    fig.text(.055, .885, 'v=2, s=2, L=4 (n=16 sites), m=2;  |R_L| = %d;  P = %d '
             'training strings.  Alphabet {0,1}, "." = still masked.'
             % (R, P), fontsize=12, va='top', color='#444')

    for col, k in enumerate((11, 15)):
        mach = LS1D(train, model.v, k, dev, alpha_smooth=1e-3)
        g = torch.Generator(device=dev).manual_seed(11)
        xt = torch.full((2, d), MASK, dtype=torch.int64)
        rows_ = torch.arange(2)
        log = [[] for _ in range(2)]
        for _ in range(d):
            masked = xt == MASK
            noise = torch.rand(2, d, generator=g)
            pick = torch.where(masked, noise, torch.full_like(noise, -1.)).argmax(1)
            p = mach.posterior(xt, pick).clamp_min(1e-12)
            tok = torch.multinomial(p, 1, generator=g)[:, 0]
            for b in range(2):
                log[b].append((''.join('.' if t < 0 else str(int(t)) for t in xt[b]),
                               int(pick[b]), p[b].numpy(), int(tok[b])))
            keep = masked[rows_, pick]
            xt[rows_[keep], pick[keep]] = tok[keep]
        gram = is_grammatical(model, xt.numpy())
        x0 = .055 + .47 * col
        fig.text(x0, .83, 'window k = %d   %s' % (k, '(U=1, sees the root)' if k >= d - 1
                                                  else '(below U=1)'),
                 fontsize=14, weight='bold', va='top',
                 color='#b40426' if k < d - 1 else '#2c7fb8')
        y = .79
        for b in range(2):
            fig.text(x0, y, 'generation %d' % (b + 1), fontsize=11, weight='bold',
                     va='top', color='#555')
            y -= .035
            for (st, si, pr, tk) in log[b][:6]:
                fig.text(x0, y, '%s   site %2d -> %d   p(1)=%.2f'
                         % (st, si, tk, pr[1]), fontsize=9.5, va='top',
                         family='monospace')
                y -= .026
            fin = ''.join(str(int(t)) for t in xt[b])
            novel = bytes(np.ascontiguousarray(xt[b].numpy().astype(np.int8))) not in tr
            fig.text(x0, y, '...  final %s' % fin, fontsize=9.5, va='top',
                     family='monospace')
            y -= .030
            fig.text(x0, y, '     grammatical: %s      novel: %s'
                     % ('yes' if gram[b] else 'NO', 'yes' if novel else 'NO'),
                     fontsize=10.5, va='top', weight='bold',
                     color='#1a7f37' if (gram[b] and novel) else '#b40426')
            y -= .045

    foot(fig, .25, [
        'Each row is the state before one reveal: the string so far, the site the sampler '
        'chose uniformly among those still masked, the token drawn, and the posterior the '
        'window put on "1". The first six of sixteen reveals are shown, then the finished '
        'string. Both columns use the same grammar, the same training set and the same '
        'random seed; only the window differs.',
        'The k=15 column is the failure mode page 5 quantifies: with a window that spans '
        'all but one site, matching training windows pins almost the whole string, so the '
        'generation is grammatical and *not new*. The k=11 column leaves the sampler room '
        'to recombine.',
    ], width=132, size=11.5)
    pdf.savefig(fig); plt.close(fig)


def page_conclusion(pdf, peaks):
    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '7. What this changes, and what it does not',
             fontsize=22, weight='bold', va='top')
    ks, sc = by_k(peaks, 16, 'succ_max')
    _, xs = by_k(peaks, 16)
    foot(fig, .86, [
        'UNCHANGED, AND STILL THE DELIVERABLE. P* = exp(Theta(n)) with n the number of '
        'sites, measured as the peak of "novel AND fully grammatical" against training-set '
        'size. In 2D at the U=1 window, P* = 0.21 |R_L| (30 bracketed peaks, |R_L| spanning '
        '49 152x, free exponent 0.965 +- 0.041). Nothing here touches that.',
        '',
        'RETIRED. The coverage hypothesis, which was the last standing explanation for why '
        'the 0.21 is 2D-specific. It is refuted with the sign backwards: x* rises with '
        'coverage (Spearman +1.00), and matching on coverage puts 1D and 2D 5.1x apart '
        'where matching on n leaves 2.3x. The table-saturation alternative fails too, 4.4x '
        'across dimensions and a 61x spread across k. I have no mechanism for the 1D/2D '
        'offset and am no longer proposing one as if I did.',
        '',
        'NEW, AND THE REASON THIS WAS WORTH RUNNING. x* is a steep function of the window: '
        '25x across k = 3..17 at fixed n. So "0.21" was never a property of the RHM alone '
        '-- it is a property of the RHM *at the U=1 window*, and both dimensions had only '
        'ever been measured at that one convention. And U=1 is the wrong convention for '
        'generation: success peaks at k=11 against U1=15 (+18%) at n=16, and at k=5 against '
        'U1=7 (+8%) at n=8, because widening the window buys grammaticality with novelty.',
        '',
        'HONEST LIMITS. Two depths, n=8 and n=16, and 1D only; the n=8 row leans on few '
        'cells after the density cut, since |R_L| >= 200 at v=2 means a grammar covering '
        '78-100% of all 2^8 strings, where noise scores 1.00 on grammaticality. The +18% is '
        'a median over 11 rule seeds, not a single run, but it has not been checked in 2D, '
        'where the sub-U=1 windows that would test it produce <0.7% grammaticality and no '
        'locatable peak.',
        '',
        'NEXT. The window optimum is the actionable finding and it points straight at the '
        'open question hshunt has already raised -- setting the patch size from the '
        'posterior-entropy floor rather than from the block structure. The AdaptiveWindow '
        'machine in the repo already chooses k per site and per step; it has never been '
        'scored against hshunt\'s novel-and-grammatical criterion, only against '
        'grammaticality. That is the next measurement.',
    ], width=130, size=11.5)
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--peaks', default='results/rhm_1d/2026-11-04/coverage_peaks.json')
    ap.add_argument('--collapse', default='results/rhm_1d/2026-11-04/collapse_variable.json')
    ap.add_argument('--out', default='results/rhm_1d/2026-11-04/rhm_coverage_slides.pdf')
    ap.add_argument('--png_dir', default='results/rhm_1d/2026-11-04/pages_cov')
    args = ap.parse_args()

    peaks = json.load(open(ROOT / args.peaks))
    cv = json.load(open(ROOT / args.collapse))
    ks, sc = by_k(peaks, 16, 'succ_max')
    best, u1i = max(sc), sc[ks.index(15)]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as raw:
        pdf = Tee(raw, ROOT / args.png_dir)
        title_page(
            pdf,
            'The RHM coverage hypothesis, tested and refuted\n'
            '-- and a better window than U=1',
            # pre-broken, and kept short: title_page does not wrap, so a long
            # subtitle line runs off the right edge at 18pt bold
            'Coverage does not explain the 1D/2D gap -- matching on it leaves 5.1x,\n'
            'worse than the 2.3x matching on n. But success peaks at k=11, below\n'
            'the root-reaching U1=15, and beats it by %.0f%%.' % (100 * (best / u1i - 1)),
            ['THE OPEN QUESTION. P* = exp(Theta(n)) holds in both dimensions, the constant\n'
             'does not: x* = P*/|R_L| is 0.050 in 1D against 0.193 in 2D at matched n=16.\n'
             'Coverage was the standing explanation -- U=1 spans 94% of 1D, 56% of 2D.',
             'THE TEST. Sweep k at fixed n, dimension and grammar: the lever neither depth\n'
             'probe could pull, since there k and n moved together. 1D L=4 runs k = 3..17,\n'
             'and k=9 has coverage 9/16 -- exactly 2D L=2\'s. 2 432 runs, job 51058208.',
             'REFUTED, SIGN BACKWARDS. x* rises with coverage (Spearman +1.00), and at\n'
             'matched coverage the dimensions are 5.1x apart -- worse than matched n.\n'
             'The table-size alternative fails too: 4.4x across dimensions, 61x across k.',
             'WHAT REPLACED IT. x* moves 25x across k, so "0.21" is a property of the U=1\n'
             'convention, not of the RHM -- and U=1 is the wrong convention for generating.'])
        page_tokenization(pdf)
        page_refutation(pdf, peaks)
        page_table_size(pdf, peaks, cv)
        page_optimum(pdf, peaks)
        page_traces(pdf)
        page_conclusion(pdf, peaks)
    print('wrote %s (%.0f kB, %d pages)' % (out, out.stat().st_size / 1e3, pdf.n))


if __name__ == '__main__':
    main()
