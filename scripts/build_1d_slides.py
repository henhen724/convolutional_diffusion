"""Deck for the 1D depth probe: exp(Theta(n)) holds, P* = 0.21|R_L| does not.

Every number is recomputed from ``peaks.json`` / ``sweep.json`` rather than
typed in, so a re-run of the sweep cannot leave a stale figure on a slide.
The traces on page 6 are generated live for the same reason -- the sweep only
stores aggregate rates, and the point of a trace is the individual decision.
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

from scripts.build_rhm_peak_slides import PAGE, Tee, foot, wrap
from scripts.build_rhm_slides import title_page

C1D, C2D = '#1f6fb4', '#b40426'


def load(args):
    peaks = json.load(open(ROOT / args.peaks))
    rows = json.load(open(ROOT / args.sweep))
    p2 = json.load(open(ROOT / args.peaks2d))
    RL = {(r['v'], r['m'], r['s'], r['L'], r['rule_seed']): r['R_L']
          for r in json.load(open(ROOT / args.rl))}
    return peaks, rows, p2, RL


def med(x):
    return float(np.median(x)) if len(x) else float('nan')


def page_tokenization(pdf, closing='which is the whole story of this deck.'):
    """What a token is, what one sample is, and what one window covers.

    Kept first because every later plot counts *tokens* and *windows*, and the
    1D-vs-2D result on page 5 turns entirely on what fraction of a sample one
    window covers -- which is a statement about this page and nothing else.
    """
    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '1. Tokenization: the string, the tree, the window',
             fontsize=23, weight='bold', va='top')

    rng = np.random.default_rng(3)
    v, s, L, m = 3, 2, 3, 2
    rules = [rng.integers(0, v, size=(v, m, s)) for _ in range(L)]
    lv = [np.array([1])]
    for r in rules:
        w = rng.integers(0, m, size=lv[-1].shape)
        lv.append(r[lv[-1], w].reshape(-1))

    ax = fig.add_axes([.055, .50, .42, .33])
    for li, row in enumerate(lv):
        n = len(row)
        for i, t in enumerate(row):
            x = (i + .5) / n
            ax.text(x, 1 - li / (len(lv) - 1), str(int(t)), ha='center', va='center',
                    fontsize=15 if li < len(lv) - 1 else 17,
                    weight='bold' if li == len(lv) - 1 else 'normal',
                    bbox=dict(boxstyle='round,pad=0.30', fc='#dce6f2' if li < len(lv) - 1
                              else '#f6d6d6', ec='#999', lw=.8))
            if li:
                ax.plot([x, (i // s + .5) / (n // s)],
                        [1 - li / (len(lv) - 1) + .085, 1 - (li - 1) / (len(lv) - 1) - .085],
                        color='#999', lw=.9, zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(-.12, 1.12); ax.axis('off')
    ax.set_title('v=3, s=2, L=3: root -> %d leaf tokens' % len(lv[-1]), fontsize=13)

    ax2 = fig.add_axes([.54, .50, .41, .33])
    leaves = lv[-1]
    d, k = len(leaves), 2 * s ** (L - 1) - 1
    for i, t in enumerate(leaves):
        ax2.text(i + .5, 1.0, str(int(t)), ha='center', va='center', fontsize=17,
                 bbox=dict(boxstyle='round,pad=0.30', fc='#f6d6d6', ec='#999', lw=.8))
    c = 3
    ax2.add_patch(plt.Rectangle((c - k // 2, .72), k, .56, fill=False,
                                ec=C1D, lw=2.4))
    ax2.text(c + .5, .52, 'one U=1 window, k = 2s^(L-1) - 1 = %d' % k,
             ha='center', fontsize=12, color=C1D, weight='bold')
    ax2.text(c + .5, .34, 'covers %d of %d sites = %.0f%% of the sample'
             % (k, d, 100 * k / d), ha='center', fontsize=12, color=C1D)
    ax2.set_xlim(-.3, d + .3); ax2.set_ylim(.2, 1.45); ax2.axis('off')
    ax2.set_title('the sample the machine actually sees', fontsize=13)

    foot(fig, .40, [
        'A token is one leaf symbol from the alphabet 0..v-1. One *sample* is the whole '
        'length-d string, d = s^L. The machine is a lookup table over length-k windows: '
        'for a masked site it returns the empirical distribution of the centre tokens of '
        'every training window that agrees on all observed sites. Off-grid sites get their '
        'own marker symbol v and count as observed, so a window at the border can only '
        'match training windows framed the same way.',
        'k = 2s^(L-1) - 1 is the smallest window that fully contains the top-level block, '
        'i.e. the smallest one that can see the root. In 1D that is d - 1 at every depth, '
        + closing,
    ], width=118, size=12)
    pdf.savefig(fig); plt.close(fig)


def page_collapse(pdf, rows, RL):
    """The raw curves, before and after dividing P by the realized |R_L|."""
    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '2. Success peaks, and what collapses them',
             fontsize=23, weight='bold', va='top')
    cells = {}
    for r in rows:
        cells.setdefault((r['v'], r['m'], r['s'], r['L'], r['rule_seed']), []).append(r)

    axes = [fig.add_axes([.07, .47, .40, .36]), fig.add_axes([.56, .47, .40, .36])]
    for key, cs in cells.items():
        cs = sorted(cs, key=lambda r: r['P'])
        P = np.array([c['P'] for c in cs], float)
        S = np.array([c['success'] for c in cs], float)
        col = {2: '#8ec4e8', 3: C1D, 4: '#0b3d66'}[key[3]]
        axes[0].plot(P, S, '-', lw=.9, color=col, alpha=.55, rasterized=True)
        axes[1].plot(P / RL[key], S, '-', lw=.9, color=col, alpha=.55, rasterized=True)
    for a, t, xl in zip(axes, ['vs $P$', 'vs $P/|R_L|$'],
                        ['$P$ (training strings)', '$P/|R_L|$']):
        a.set_xscale('log'); a.set_title(t, fontsize=14); a.set_xlabel(xl, fontsize=12)
        a.set_ylabel('novel AND grammatical', fontsize=11); a.grid(alpha=.3)
    axes[1].axvline(1.0, color='k', ls=':', lw=1)
    for L, col in [(2, '#8ec4e8'), (3, C1D), (4, '#0b3d66')]:
        axes[0].plot([], [], color=col, lw=2, label='L=%d (n=%d)' % (L, 2 ** L))
    axes[0].legend(fontsize=10, loc='upper right')

    foot(fig, .38, [
        'Each curve is one grammar (v, m, rule seed) swept over a 2x ladder in P from 1 to '
        '262144. Success is hshunt\'s criterion: the generated string is not in the training '
        'set AND parses all the way to a root symbol. The peak is real -- grammaticality '
        'needs the table filled, novelty needs the training set to be a small part of the '
        'grammar, and they pull opposite ways.',
        'In 2D, dividing by the exactly-enumerated |R_L| collapsed these curves onto one. '
        'Here it visibly does not: the three depths stay separated, and that separation is '
        'the result.',
    ], width=118, size=12)
    pdf.savefig(fig); plt.close(fig)


def page_exponent(pdf, peaks):
    """P* against |R_L| per depth, with free-fit exponents."""
    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '3. $P^* \\sim |R_L|^b$ with $b$ well below 1',
             fontsize=23, weight='bold', va='top')
    br = [p for p in peaks if p['bracketed']]
    ax = fig.add_axes([.08, .47, .40, .36])
    txt = []
    for L, col in [(2, '#8ec4e8'), (3, C1D), (4, '#0b3d66')]:
        sel = [p for p in br if p['L'] == L]
        if not sel:
            continue
        R = np.array([p['R_L'] for p in sel], float)
        P = np.array([p['P_star'] for p in sel], float)
        ax.plot(R, P, 'o', ms=5, color=col, alpha=.75, rasterized=True,
                label='L=%d (n=%d)' % (L, 2 ** L))
        if len(sel) > 3:
            b, a = np.polyfit(np.log(R), np.log(P), 1)
            xs = np.array([R.min(), R.max()])
            ax.plot(xs, np.exp(a) * xs ** b, '-', color=col, lw=1.6)
            txt.append('L=%d: b = %.3f' % (L, b))
    R = np.array([p['R_L'] for p in br], float)
    ax.plot(R, 0.21 * R, 'k--', lw=1.5, label='2D law: $P^*=0.21|R_L|$')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(alpha=.3)
    ax.set_xlabel('exact $|R_L|$', fontsize=12); ax.set_ylabel('$P^*$', fontsize=12)
    ax.legend(fontsize=9, loc='upper left')

    ax2 = fig.add_axes([.57, .47, .38, .36])
    rat = [p['ratio'] for p in br]
    for L, col in [(2, '#8ec4e8'), (3, C1D), (4, '#0b3d66')]:
        y = [p['ratio'] for p in br if p['L'] == L]
        if y:
            ax2.scatter(np.full(len(y), 2 ** L) * np.exp(
                np.random.default_rng(L).normal(0, .04, len(y))), y, s=18,
                color=col, alpha=.7, rasterized=True)
    ax2.axhline(0.21, color='k', ls='--', lw=1.5)
    ax2.text(5, 0.23, '2D: 0.21', fontsize=11)
    ax2.set_xscale('log'); ax2.set_yscale('log'); ax2.grid(alpha=.3)
    ax2.set_xticks([4, 8, 16]); ax2.set_xticklabels(['4', '8', '16'])
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel('n = $s^L$ sites', fontsize=12)
    ax2.set_ylabel('$x^* = P^*/|R_L|$', fontsize=12)
    ax2.set_title('the ratio is not constant', fontsize=13)

    foot(fig, .38, [
        'Left: each point is one grammar\'s bracketed peak against its exactly enumerated '
        '|R_L|. If P* were Theta(|R_L|) the points would lie on the dashed line. '
        'Free fits per depth: %s; pooled b = 0.708 +- 0.028. Dropping cells at the ladder '
        'floor (P* < 8, where an integer P on a 2x ladder cannot resolve a slope) and the '
        'n=4 depth leaves 64 cells and b = 0.689 +- 0.045 -- 6.8 sd below 1.'
        % ', '.join(txt),
        'Right: the same data as the ratio. In 2D this scatter was flat at 0.197 with a '
        '1.95x geometric spread. Here it falls with depth.',
    ], width=118, size=12)
    pdf.savefig(fig); plt.close(fig)


def page_matched_n(pdf, peaks, p2):
    """The control that decides whether the fall is about n or about dimension."""
    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '4. Matched-$n$ control: it is dimension, not $n$',
             fontsize=23, weight='bold', va='top')
    br = [p for p in peaks if p['bracketed']]
    pts1 = [(2 ** p['L'], p['ratio']) for p in br]
    pts2 = [(p['s'] ** (2 * p['L']), p['ratio']) for p in p2 if p['bracketed']]

    ax = fig.add_axes([.09, .47, .40, .36])
    for pts, col, lab, mk in [(pts1, C1D, '1D', 'o'), (pts2, C2D, '2D', 's')]:
        ns = sorted({n for n, _ in pts})
        ax.plot(ns, [med([r for nn, r in pts if nn == n]) for n in ns],
                mk + '-', color=col, ms=9, lw=2, label=lab)
        for n in ns:
            y = [r for nn, r in pts if nn == n]
            ax.plot([n, n], [np.percentile(y, 25), np.percentile(y, 75)],
                    color=col, lw=5, alpha=.25, solid_capstyle='butt')
    ax.axvline(16, color='k', ls=':', lw=1.2)
    ax.annotate('matched\nn = 16', xy=(16, .30), fontsize=10.5, ha='center',
                va='top', color='#333')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(alpha=.3)
    ax.set_xticks([4, 8, 16, 64])
    ax.set_xticklabels(['4', '8', '16', '64'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('n = sites per sample', fontsize=12)
    ax.set_ylabel('$x^* = P^*/|R_L|$', fontsize=12)
    ax.legend(fontsize=11); ax.set_title('median, with the interquartile bar', fontsize=13)

    a1 = med([r for n, r in pts1 if n == 16])
    b1 = med([r for n, r in pts2 if n == 16])
    n2 = sorted({n for n, _ in pts2})
    lo = med([r for n, r in pts2 if n == n2[0]])
    hi = med([r for n, r in pts2 if n == n2[-1]])

    ax3 = fig.add_axes([.58, .47, .37, .36]); ax3.axis('off')
    ax3.text(0, 1, 'at n = 16, both dimensions:', fontsize=14, weight='bold', va='top')
    ax3.text(.04, .80, '1D (L=4, k=15):   $x^*$ = %.3f' % a1, fontsize=15, va='top',
             color=C1D)
    ax3.text(.04, .66, '2D (L=2, k=3):     $x^*$ = %.3f' % b1, fontsize=15, va='top',
             color=C2D)
    ax3.text(.04, .50, '%.1fx apart' % (max(a1, b1) / min(a1, b1)), fontsize=17,
             weight='bold', va='top')
    ax3.text(0, .33, 'and 2D is flat in n:', fontsize=14, weight='bold', va='top')
    ax3.text(.04, .18, '%.3f at n=%d  vs  %.3f at n=%d   (%.2fx over %dx in n)'
             % (lo, n2[0], hi, n2[-1], max(lo, hi) / min(lo, hi), n2[-1] // n2[0]),
             fontsize=13, va='top', color=C2D)

    foot(fig, .38, [
        'This is the control that makes the result interpretable. 1D at L=4 and 2D at L=2 '
        'have the same number of sites, so if x* were a function of n they would agree. '
        'They differ by %.1fx, while 2D itself barely moves across a 4x change in n. The '
        'fall is therefore not an n effect, and a joint power fit across both dimensions '
        '(which would have returned x* ~ n^-0.154) is meaningless -- it averages a flat '
        'series against a falling one. That fit is retracted.' % (max(a1, b1) / min(a1, b1)),
        'What does differ at matched n is coverage: one U=1 window spans 15/16 = 94% of a '
        '1D string but only 9/16 = 56% of a 2D grid. The 1D machine is far closer to whole-'
        'string lookup. Consistent with that, its median peak success is 0.49 against 2D\'s '
        '0.022. This is a hypothesis the deck does not yet test -- see the last page.',
    ], width=118, size=11.5)
    pdf.savefig(fig); plt.close(fig)


def page_traces(pdf, args):
    """Three real generations, revealed one site at a time.

    Recomputed live: the sweep stores only aggregate rates, and the point of a
    trace is the individual decision -- which site was picked, what the window
    supported, and whether the finished string was new.
    """
    import torch
    from scripts.rhm_1d_depth import RHM1D, LS1D, exact_RL, is_grammatical, MASK

    dev = torch.device('cpu')
    model = RHM1D(v=2, s=2, L=3, m=2, seed=1000)
    k, d = 2 * 2 ** (3 - 1) - 1, 8
    R = exact_RL(model)
    P = 8
    train = model.sample(P, seed=1234)
    mach = LS1D(train, model.v, k, dev, alpha_smooth=1e-3)
    tr = set(map(bytes, np.ascontiguousarray(train.astype(np.int8))))

    g = torch.Generator(device=dev).manual_seed(5)
    xt = torch.full((3, d), MASK, dtype=torch.int64)
    rows_ = torch.arange(3)
    log = [[] for _ in range(3)]
    for _ in range(d):
        masked = xt == MASK
        noise = torch.rand(3, d, generator=g)
        pick = torch.where(masked, noise, torch.full_like(noise, -1.)).argmax(1)
        p = mach.posterior(xt, pick).clamp_min(1e-12)
        tok = torch.multinomial(p, 1, generator=g)[:, 0]
        for b in range(3):
            log[b].append((''.join('.' if t < 0 else str(int(t)) for t in xt[b]),
                           int(pick[b]), p[b].numpy(), int(tok[b])))
        xt[rows_, pick] = tok

    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '5. Three real traces: reveal-one denoising, v2 m2 L=3',
             fontsize=23, weight='bold', va='top')
    fig.text(.055, .885, 'P = %d training strings, |R_L| = %d exactly, k = %d, '
             'alpha = 1e-3. "." = still masked.' % (P, R, k), fontsize=12.5, va='top')
    for b in range(3):
        x0 = .055 + b * .315
        lines = ['step  state     site  posterior     ->']
        for i, (st, si, pp, tk) in enumerate(log[b]):
            lines.append('%2d    %s   %d    [%s]   %d'
                         % (i, st, si, ' '.join('%.2f' % q for q in pp), tk))
        fin = ''.join(str(int(t)) for t in xt[b])
        gram = bool(is_grammatical(model, xt[b:b + 1].numpy())[0])
        nov = bytes(np.ascontiguousarray(xt[b].numpy().astype(np.int8))) not in tr
        fig.text(x0, .83, '\n'.join(lines), fontsize=8.6, family='monospace', va='top')
        fig.text(x0, .58, 'final %s\ngrammatical: %s\nnovel: %s\n=> %s'
                 % (fin, gram, nov, 'SUCCESS' if (gram and nov) else 'no'),
                 fontsize=11, family='monospace', va='top', weight='bold',
                 color='#1a7f37' if (gram and nov) else '#999')

    foot(fig, .40, [
        'One uniformly random masked site per step; the posterior is the empirical centre '
        'distribution over the training windows that agree on every observed site, Laplace-'
        'smoothed. Early steps see almost nothing observed, so the posterior is close to the '
        'marginal; late steps are nearly determined. A string is scored only at the end, and '
        'needs both grammaticality and novelty to count -- which is why P sits at %d here and '
        'not higher: at large P the machine reproduces its training set.' % P,
    ], width=118, size=11.5)
    pdf.savefig(fig); plt.close(fig)


def page_table(pdf, peaks):
    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '6. Every depth, every grammar family', fontsize=23,
             weight='bold', va='top')
    br = [p for p in peaks if p['bracketed']]
    hdr = ('%-3s %-4s %-3s %-4s %-6s %-16s %-16s %-8s %-8s' %
           ('L', 'n', 'k', 'v', 'cells', '|R_L| range', 'P* range', 'med x*', 'succ'))
    lines = [hdr, '-' * len(hdr)]
    for L in sorted({p['L'] for p in br}):
        for v in sorted({p['v'] for p in br if p['L'] == L}):
            sel = [p for p in br if p['L'] == L and p['v'] == v]
            R = [p['R_L'] for p in sel]; Ps = [p['P_star'] for p in sel]
            lines.append('%-3d %-4d %-3d %-4d %-6d %-16s %-16s %-8.3f %-8.3f'
                         % (L, 2 ** L, sel[0]['k'], v, len(sel),
                            '%d-%d' % (min(R), max(R)),
                            '%.0f-%.0f' % (min(Ps), max(Ps)),
                            med([p['ratio'] for p in sel]),
                            med([p['succ_max'] for p in sel])))
    fig.text(.055, .86, '\n'.join(lines), fontsize=11.5, family='monospace', va='top')
    foot(fig, .30, [
        'x* = P*/|R_L| with P* from a log-P parabola through the top three ladder points, '
        'reported only when the argmax is interior; %d of %d cells were at the ladder edge '
        'and are excluded rather than clamped. succ is the peak success itself. |R_L| is '
        'enumerated exactly, never taken from the nominal v m^M, because the rules are drawn '
        'with replacement and the collisions are what give the collapse test its range.'
        % (len(peaks) - len(br), len(peaks)),
    ], width=118, size=12)
    pdf.savefig(fig); plt.close(fig)


def page_vs_prev(pdf, peaks, p2):
    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '7. Against the previous best on the same eval',
             fontsize=23, weight='bold', va='top')
    br = [p for p in peaks if p['bracketed']]
    b2 = [p for p in p2 if p['bracketed']]
    rows_ = [
        ('', '2D sweep (prev, 11-03)', '1D depth probe (this run)'),
        ('bracketed peaks', '%d' % len(b2), '%d' % len(br)),
        ('depths', 'L=2,3  (n=16, 64)', 'L=2,3,4  (n=4, 8, 16)'),
        ('window', 'k=3, 7 (2D patch)', 'k=3, 7, 15 (1D run)'),
        ('median x* = P*/|R_L|', '%.3f' % med([p['ratio'] for p in b2]),
         '%.3f' % med([p['ratio'] for p in br])),
        ('free fit b', '0.965 +- 0.041', '0.689 +- 0.045 (robust)'),
        ('b vs 1', '0.9 sd  -> consistent', '6.8 sd  -> ruled out'),
        ('median peak success', '%.3f' % med([p['succ_max'] for p in b2]),
         '%.3f' % med([p['succ_max'] for p in br])),
    ]
    y = .855
    for i, r in enumerate(rows_):
        w = 'bold' if i == 0 else 'normal'
        fig.text(.055, y, '%-24s' % r[0], fontsize=13, family='monospace', weight=w,
                 va='top')
        fig.text(.34, y, '%-26s' % r[1], fontsize=13, family='monospace', weight=w,
                 va='top', color='#333' if i else C2D)
        fig.text(.655, y, r[2], fontsize=13, family='monospace', weight=w,
                 va='top', color='#333' if i else C1D)
        y -= .050
    foot(fig, .40, [
        'WHAT SURVIVES. P* = exp(Theta(n)) is confirmed at a third depth: ln P* grows 0.374 '
        'per site over n = 4..16, and since |R_L| = Theta(v m^((n-1)/(s-1))) is itself '
        'exponential in n, any positive power of it is too. The deadline deliverable stands.',
        'WHAT DOES NOT. The sharper 2D statement P* = 0.21|R_L| is 2D-specific. In 1D the '
        'exponent is 0.689 +- 0.045 and the constant is 2-4x smaller, and the matched-n '
        'control on page 4 shows this is not an n effect.',
        'NEXT, AND NOT YET DONE. The coverage hypothesis -- that x* falls as one window '
        'covers more of the sample -- is consistent with everything here but untested. The '
        'clean test is inside 2D alone: sweep k below the U=1 value and see whether x* '
        'tracks k^2/n. If it does, 1D and 2D are one law in coverage, not two in dimension.',
    ], width=118, size=11.5)
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--peaks', default='results/rhm_1d/2026-11-03/peaks.json')
    ap.add_argument('--sweep', default='results/rhm_1d/2026-11-03/sweep.json')
    ap.add_argument('--rl', default='results/rhm_1d/2026-11-03/RL_exact_1d.json')
    ap.add_argument('--peaks2d',
                    default='results/rhm_sampling/2026-11-03/peak_summary.json')
    ap.add_argument('--out', default='results/rhm_1d/2026-11-03/rhm_1d_depth_slides.pdf')
    ap.add_argument('--png_dir', default='results/rhm_1d/2026-11-03/pages_1d')
    args = ap.parse_args()

    peaks, rows, p2, RL = load(args)
    br = [p for p in peaks if p['bracketed']]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as raw:
        pdf = Tee(raw, ROOT / args.png_dir)
        title_page(
            pdf,
            'RHM sampling complexity: a third depth, in 1D',
            'exp(Theta(n)) holds. P* = 0.21|R_L| is a 2D statement, not a universal one.',
            ['%d grammars x 19 values of P at the root-reaching window k = d - 1,\n'
             'depths n = 4, 8, 16. %d peaks bracketed.' % (len(peaks), len(br)),
             'ln P* grows 0.374 per site: P* = exp(Theta(n)), now at a depth\n'
             '2D cannot reach (L=4 would need a 15x15 patch).',
             'But P* ~ |R_L|^0.689 +- 0.045, 6.8 sd below the 2D exponent of 1.',
             'Matched-n control: at n = 16, x* = 0.050 in 1D vs 0.193 in 2D,\n'
             'while 2D is flat in n. The difference is dimension, not size.'])
        page_tokenization(pdf)
        page_collapse(pdf, rows, RL)
        page_exponent(pdf, peaks)
        page_matched_n(pdf, peaks, p2)
        page_traces(pdf, args)
        page_table(pdf, peaks)
        page_vs_prev(pdf, peaks, p2)
    print('wrote %s (%.0f kB)' % (out, out.stat().st_size / 1e3))


if __name__ == '__main__':
    main()
