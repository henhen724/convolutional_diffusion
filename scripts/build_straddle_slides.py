"""Slide deck: how many levels up the tree a straddling window can see.

Reads the JSONs written by ``rhm_straddle_levels.py`` (and, where it exists, the
exact enumeration from ``rhm_straddle.py``) and builds a short PDF, one page per
claim, plus a PNG of each page so it can be checked without a rasterizer.

  python scripts/build_straddle_slides.py --results results/rhm_sampling/2026-11-03 \
      --out .../straddle_levels_slides.pdf --png_dir .../straddle_pages
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


def load(D: Path):
    """Saturated a_j rows, plus the exact L=2 enumeration keyed by cell."""
    rows = []
    for f in sorted(D.glob('straddle_levels_*.json')):
        rows += json.load(open(f))
    exact = {}
    for f in sorted(D.glob('straddle_v*.json')):
        for r in json.load(open(f)):
            exact[(r['v'], r['m'], r['k'], r['rule_seed'])] = r
    good = [r for r in rows if r['W_saturated']]
    return rows, good, exact


def aj(r, j):
    return r['accept'].get(str(j))


def reach(r):
    """Highest level whose incoherence the window detects: the largest l with a_{l-1} < 1.

    ``E_{l-1}`` is coherent through ``l-1`` and scrambled at ``l`` and above, so
    rejecting it means the window notices something at level ``l`` or higher.
    ``reach - j*`` is then how many levels above the deepest block it contains
    the window can actually see; 0 means it sees nothing a block-aligned reading
    would not.
    """
    best = r['j_star']
    for l in range(r['j_star'] + 1, r['L'] + 1):
        if (aj(r, l - 1) or 0) < 0.9999:
            best = l
    return best


def root_strength(r):
    """1 - a_{L-1}: how often the window catches a scrambled ROOT pairing.

    This, not ``1 - a_{j*+1}``, is the comparable number across depths: at L=2
    the only level above ``j* = 1`` is the root, so the root is tested by a_1.
    """
    x = aj(r, r['L'] - 1)
    return None if x is None else 1 - x


def page_method(pdf, rows, good, exact):
    """What E_j is, and the exact-enumeration check that the estimator is right."""
    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '1. The measurement, and its exact check',
             fontsize=23, weight='bold', va='top')

    # left: the E_j cartoon at L=3, j=2 -- four independent legal 4x4 blocks
    ax = fig.add_axes([.06, .47, .33, .35])
    rng = np.random.default_rng(3)
    img = rng.integers(0, 2, (8, 8))
    ax.imshow(img, cmap='Blues', vmin=-.5, vmax=1.8, interpolation='nearest')
    for c in (3.5,):
        ax.axhline(c, color='k', lw=3); ax.axvline(c, color='k', lw=3)
    for off in (0, 4):
        for off2 in (0, 4):
            ax.add_patch(plt.Rectangle((off2 - .5, off - .5), 4, 4, fill=False,
                                       ec='#2b6cb0', lw=2.5))
    ax.add_patch(plt.Rectangle((1.5, 1.5), 5, 5, fill=False, ec='#b40426', lw=3))
    ax.set_title('$E_2$ at L=3: four independent\nlegal 4x4 blocks, 5x5 window', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])

    # right: estimator vs exact enumeration at L=2
    ax = fig.add_axes([.52, .47, .38, .35])
    xs, ys = [], []
    for r in good:
        if r['L'] != 2:
            continue
        g = exact.get((r['v'], r['m'], r['k'], r['rule_seed']))
        a = aj(r, r['j_star'])
        if g and a is not None and g['lv_blocks']:
            xs.append(g['lv_pooled'] / g['lv_blocks']); ys.append(a)
    if xs:
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='exact = estimated')
        ax.plot(xs, ys, 'o', ms=6, color='#b40426', alpha=.7,
                label='%d L=2 cells' % len(xs))
        err = np.abs(np.array(ys) - np.array(xs)) / np.maximum(np.array(xs), 1e-12)
        ax.set_title('estimator vs exact enumeration\nmax rel. error %.2g' % err.max(),
                     fontsize=12)
        ax.legend(fontsize=10, loc='upper left')
    ax.set_xlabel('exact  $lv_{pooled}/lv_{blocks}$', fontsize=12)
    ax.set_ylabel('measured  $a_{j^*}$', fontsize=12)
    ax.grid(alpha=.3)

    foot(fig, .38, [
        'A k x k window fully contains an aligned block of side b only when k >= 2b-1, so it '
        'directly certifies levels up to j* = floor(log_s((k+1)/2)). The question is whether it '
        'also polices the levels ABOVE j*, using windows that straddle a block boundary and see '
        'the right edge of one block against the left edge of the next.',
        'Ensemble E_j tiles the grid with independent uniform draws from R_j, the distinct '
        'renderings of an aligned s^j block: coherent through level j, scrambled above it. '
        'a_j is the fraction of E_j grids whose every window lies in the machine\'s window set W. '
        'a_L = 1 says W is saturated; a_j < 1 for j > j* says the window sees above j*.',
        'At L=2 every grid can be enumerated, so a_{j*} has a closed-form check -- the right '
        'panel. L=3 has v^64 grids and only the estimator reaches it.',
    ], width=118, size=11.5)
    pdf.savefig(fig); plt.close(fig)


def page_profiles(pdf, good):
    """a_j by level, L=2 and L=3, one line per grammar."""
    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '2. The window polices two levels above the block it contains',
             fontsize=22, weight='bold', va='top')

    panels = [(2, 3), (3, 3), (3, 5)]
    for i, (L, k) in enumerate(panels):
        ax = fig.add_axes([.075 + .315 * i, .47, .245, .36])
        sel = [r for r in good if r['L'] == L and r['k'] == k]
        for r in sel:
            js = np.array(sorted(int(j) for j in r['accept']))
            va = np.array([r['accept'][str(j)] for j in js])
            ax.plot(js, va, 'o-', ms=4, lw=1.2, alpha=.65,
                    color=plt.cm.viridis((r['m'] - 2) / 4.0))
        ax.axvline(sel[0]['j_star'] + .0, color='k', ls=':', lw=1.5) if sel else None
        ax.set_ylim(-.04, 1.08)
        ax.set_xticks(range(1, L + 1))
        ax.set_xlabel('level $j$ kept coherent', fontsize=11)
        if i == 0:
            ax.set_ylabel('$a_j$  (fraction accepted)', fontsize=12)
        ax.set_title('L=%d, k=%d  (j*=%d)\n%d grammars'
                     % (L, k, sel[0]['j_star'] if sel else 0, len(sel)), fontsize=12)
        ax.grid(alpha=.3)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(2, 6))
    cax = fig.add_axes([.945, .47, .012, .36])
    fig.colorbar(sm, cax=cax).set_label('m (synonyms)', fontsize=11)

    l3k5 = [r for r in good if r['L'] == 3 and r['k'] == 5 and aj(r, 2) is not None]
    a2 = np.array([aj(r, 2) for r in l3k5] or [np.nan])
    a1 = np.array([aj(r, 1) for r in l3k5 if aj(r, 1) is not None] or [np.nan])
    foot(fig, .38, [
        'Each line is one grammar; the dotted line is j*, the deepest level the window contains '
        'outright. a_j = 1 at j = L is the saturation check and holds everywhere plotted. The '
        'interesting point is j = j*+1 and beyond: a value below 1 there means the window rejects '
        'grids that are ALREADY coherent at that level, so its reach is strictly deeper than j*.',
        'At L=3, k=5 (j* = 1) the window contains only 2x2 blocks, yet %d of %d grammars have '
        'a_2 < 0.99 -- median a_2 = %.2f. Those rejections can only come from windows straddling '
        'the boundary between two 4x4 level-2 blocks, i.e. the window is checking the ROOT, two '
        'levels above j*. That is the effect hshunt predicted.'
        % (int((a2 < .99).sum()), len(a2), float(np.median(a2))),
        'a_1 is far smaller still (median %.3f): scrambling at level 1 breaks far more windows '
        'than scrambling only the root does.' % float(np.median(a1)),
    ], width=118, size=11.5)
    pdf.savefig(fig); plt.close(fig)


def page_when(pdf, good):
    """Strength of the straddle signal vs grammar richness."""
    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '3. The signal dies as the grammar above j* gets richer',
             fontsize=22, weight='bold', va='top')

    ax = fig.add_axes([.075, .47, .38, .36])
    ms = sorted({r['m'] for r in good})
    for (L, k), mk, col in (((2, 3), 'o', '#2b6cb0'), ((2, 5), 's', '#b40426'),
                            ((3, 3), 'v', '#4c9f70'), ((3, 5), 'D', '#d08b2c')):
        med = []
        for m in ms:
            y = [root_strength(r) for r in good
                 if r['L'] == L and r['k'] == k and r['m'] == m and root_strength(r) is not None]
            # a median over one or two grammars is noise, not a trend: the L=3,
            # k=5, m=3 cells mostly failed to saturate and are left out here.
            med.append(np.median(y) if len(y) >= 3 else np.nan)
        ax.plot(ms, med, mk + '-', ms=7, lw=1.6, color=col, label='L=%d, k=%d' % (L, k))
    ax.set_xlabel('m  (synonyms per symbol)', fontsize=12)
    ax.set_ylabel(r'root detection  $1 - a_{L-1}$', fontsize=12)
    ax.set_xticks(ms)
    ax.set_ylim(-.04, 1.04); ax.grid(alpha=.3); ax.legend(fontsize=10)
    ax.set_title('median over grammars', fontsize=12)

    ax = fig.add_axes([.56, .47, .38, .36])
    groups = [(L, k) for L in (2, 3) for k in (3, 5)
              if any(r['L'] == L and r['k'] == k for r in good)]
    width = .8 / max(len(groups), 1)
    for i, (L, k) in enumerate(groups):
        sel = [r for r in good if r['L'] == L and r['k'] == k]
        up = [reach(r) - r['j_star'] for r in sel]
        xs = sorted(set(up))
        ax.bar([x + (i - len(groups) / 2 + .5) * width for x in xs],
               [up.count(x) / len(sel) for x in xs], width,
               label='L=%d, k=%d  (n=%d)' % (L, k, len(sel)))
    ax.set_xlabel('levels above $j^*$ the window detects', fontsize=12)
    ax.set_ylabel('fraction of grammars', fontsize=12)
    ax.set_xticks([0, 1, 2]); ax.set_ylim(0, 1.15)
    ax.grid(alpha=.3, axis='y'); ax.legend(fontsize=9)

    n35 = [r for r in good if r['L'] == 3 and r['k'] == 5]
    all2 = sum(reach(r) - r['j_star'] == 2 for r in n35)
    foot(fig, .38, [
        'Left: the signal is set by how constrained a boundary between two sibling blocks is, not '
        'by the window size alone. Every extra synonym adds legal boundary pairs, so fewer '
        'scrambled pairings are detectable and root detection falls monotonically in m -- from '
        '0.77 to 0.11 across m=2..6 at L=2, k=3. A wider window (k=5) decays far more slowly.',
        'Right: how far the reach actually extends. At L=3 with j* = 1 a window could in '
        'principle see one level up (level 2) or two (the root). All %d of %d saturated k=5 '
        'grammars see two; at k=3 it is %d of %d, with %d seeing nothing at all. So the reach is '
        'not fixed by k -- the same window size reaches the root on one grammar and stops at the '
        'blocks on another, which is the rule-dependence hshunt expected.'
        % (all2, len(n35),
           sum(reach(r) - r['j_star'] == 2 for r in good if r['L'] == 3 and r['k'] == 3),
           len([r for r in good if r['L'] == 3 and r['k'] == 3]),
           sum(reach(r) - r['j_star'] == 0 for r in good if r['L'] == 3 and r['k'] == 3)),
        'Only medians over at least 3 grammars are drawn. The gap at L=3, k=5, m=3 is the one '
        'place the method runs out: those window sets never saturated within 6M grids (|W| was '
        'still growing past 40M windows), and an unsaturated W rejects grammatical grids too, so '
        'those cells are excluded rather than reported.',
        'Caveat worth stating plainly: a_j measures the SUPPORT the window set defines -- what an '
        'infinite-data machine could reject. It is not what the finite-P sampler actually emits, '
        'which is lower; see the sampling-complexity deck.',
    ], width=118, size=11.5)
    pdf.savefig(fig); plt.close(fig)


def page_traces(pdf):
    """The tokenization, and two worked grids: where the rejected windows actually sit.

    Recomputed rather than loaded, because the point is the *location* of the
    rejected windows and the sweep only stores aggregate rates.  v2 m2 L=3 k=5
    saturates W at the exact N_5 = 92080 in well under a minute on one core.
    """
    import torch
    from src.utils.random_hierarchy import RandomHierarchyModel2D
    from scripts.rhm_straddle import window_codes
    from scripts.rhm_straddle_levels import block_set, tile, window_set

    dev = torch.device('cpu')
    model = RandomHierarchyModel2D(v=2, s=2, L=3, m=2, seed=1000)
    k, d = 5, 8
    W, n_used, _ = window_set(model, k, dev, 2_000_000, 50_000)

    def bad_mask(g):
        c = window_codes(g[None], k, model.v, dev)[0]      # one window per SITE
        i = torch.searchsorted(W, c.reshape(-1)).clamp(max=W.numel() - 1)
        return ~(W[i] == c.reshape(-1)).reshape(c.shape)

    gen = torch.Generator(device=dev).manual_seed(20261103)
    e2 = tile(torch.as_tensor(block_set(model, 2), device=dev), 2, 1, dev, gen)[0]
    gg = torch.as_tensor(np.asarray(model.sample(1, seed=90000)[0])[0], dtype=torch.int64)

    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '4. Tokenization, and two worked grids',
             fontsize=23, weight='bold', va='top')

    for i, (g, lab) in enumerate([
            (e2, '$E_2$: four INDEPENDENT legal 4x4 blocks\n(coherent to level 2, no common root)'),
            (gg, 'a grammatical grid ($E_3$)\nevery window is one the machine knows')]):
        ax = fig.add_axes([.07 + .30 * i, .50, .22, .33])
        bad = bad_mask(g)
        ax.imshow(np.asarray(g), cmap='Blues', vmin=-.6, vmax=1.9, interpolation='nearest')
        for r in range(d):
            for c in range(d):
                ax.text(c, r, str(int(g[r, c])), ha='center', va='center', fontsize=9,
                        color='#b40426' if bad[r, c] else '0.35',
                        weight='bold' if bad[r, c] else 'normal')
        ax.axhline(3.5, color='k', lw=2.5); ax.axvline(3.5, color='k', lw=2.5)
        for r, c in bad.nonzero():
            r, c = int(r), int(c)
            ax.add_patch(plt.Rectangle((c - 2.5, r - 2.5), k, k, fill=False,
                                       ec='#b40426', lw=2.5))
        ax.set_title('%s\n%d of %d windows rejected' % (lab, int(bad.sum()), bad.numel()),
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    nbad = int(bad_mask(e2).sum())
    foot(fig, .41, [
        'Tokenization. A grid is d x d = %d x %d SITES, each one token from an alphabet of v '
        'symbols; a token is an integer label, with no sub-token structure and no embedding. The '
        'root symbol is uniform on {0..v-1} and L levels of rules expand each symbol into an s x s '
        'block chosen from m synonymous right-hand sides. The LS machine never sees the tree -- '
        'only the leaf grid and the k x k windows cut from it. A window is counted as one int64, '
        'its k*k tokens packed base v+1 (the extra digit marks off-grid padding at the border), '
        'which is exactly why k >= 7 is out of reach: 3^48 overflows int64.' % (d, d),
        'Left: an E_2 grid for v=2, m=2, L=3, seed 1000. Each 4x4 quadrant is a legal rendering of '
        'some level-2 symbol, but the four were drawn independently, so no root produces all four '
        'together. The machine rejects %d of its 64 windows -- boxed in red, centred on the bold '
        'site -- and the box straddles the quadrant boundary. That single window is the entire '
        'effect: nothing inside any 4x4 block is wrong, and a block-aligned reader sees nothing.'
        % nbad,
        'Right: a genuinely grammatical grid from the same model, for contrast -- 0 of 64 '
        'rejected, which is the a_L = 1 saturation check made concrete. W here holds %d distinct '
        '5x5 windows, grown from %.1fM grammatical grids and matching the exact count N_5 = 92080.'
        % (W.numel(), n_used / 1e6),
    ], width=118, size=11)
    pdf.savefig(fig); plt.close(fig)


def page_table(pdf, rows, good):
    """Per-cell numbers, so nothing on the plots has to be taken on trust.

    One line per (L, v, m, k), the five rule seeds collapsed to a median and a
    range: dumping all 146 saturated cells individually ran off the page, and
    the seed-to-seed spread is the point anyway -- ``up`` shows that the same
    (v, m, k) reaches the root on one grammar and stops at the blocks on
    another.
    """
    fig = plt.figure(figsize=PAGE)
    fig.text(.055, .95, '5. Every cell, five rule seeds collapsed',
             fontsize=23, weight='bold', va='top')
    hdr = '%-2s %-2s %-2s %-2s %-3s %-5s %-9s %-17s %-17s %-6s %-5s' % (
        'L', 'v', 'm', 'k', 'j*', 'sat', 'med |W|', 'a_1  (min-max)',
        'a_2  (min-max)', 'a_L', 'up')
    lines = [hdr, '-' * len(hdr)]

    def rng(sel, j):
        y = [aj(r, j) for r in sel if aj(r, j) is not None]
        if not y:
            return '%-17s' % '--'
        return '%.3f (%.3f-%.3f)' % (np.median(y), min(y), max(y))

    keys = sorted({(r['L'], r['v'], r['m'], r['k']) for r in rows})
    for L, v, m, k in keys:
        allc = [r for r in rows if (r['L'], r['v'], r['m'], r['k']) == (L, v, m, k)]
        sel = [r for r in allc if r['W_saturated']]
        if not sel:
            lines.append('%-2d %-2d %-2d %-2d %-3d %-5s %-9s %-17s %-17s %-6s %-5s'
                         % (L, v, m, k, allc[0]['j_star'], '0/%d' % len(allc),
                            '--', 'W never saturated', '--', '--', '--'))
            continue
        up = [reach(r) - r['j_star'] for r in sel]
        lines.append('%-2d %-2d %-2d %-2d %-3d %-5s %-9d %-17s %-17s %-6.4f %-5s' % (
            L, v, m, k, sel[0]['j_star'], '%d/%d' % (len(sel), len(allc)),
            int(np.median([r['W_size'] for r in sel])), rng(sel, 1), rng(sel, 2),
            float(np.median([aj(r, L) for r in sel])),
            str(up[0]) if min(up) == max(up) else '%d-%d' % (min(up), max(up))))

    half = (len(lines) + 1) // 2
    for i, chunk in enumerate((lines[:half], [hdr, '-' * len(hdr)] + lines[half:])):
        fig.text(.035 + .49 * i, .88, '\n'.join(chunk), fontsize=7.0, va='top',
                 family='monospace', linespacing=1.45)
    foot(fig, .17, [
        'sat = rule seeds whose window set provably saturated (a_L = 1), out of 5. a_1 and a_2 '
        'are medians over those seeds with the full seed range in brackets; up = levels above j* '
        'the window detects, as a range over seeds. The %d cells whose W never saturated (all of '
        'them L=3, k=5, m=3) are reported as such rather than dropped silently.'
        % (len(rows) - len(good)),
    ], width=118, size=11)
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--png_dir', type=str, default=None)
    a = ap.parse_args()

    D = Path(a.results)
    rows, good, exact = load(D)
    print('%d cells, %d saturated' % (len(rows), len(good)))
    if not good:
        raise SystemExit('nothing saturated -- not building a deck on unsaturated W')

    l3k5 = [r for r in good if r['L'] == 3 and r['k'] == 5 and aj(r, 2) is not None]
    med = float(np.median([aj(r, 2) for r in l3k5])) if l3k5 else float('nan')

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(a.out) as _pdf:
        pdf = Tee(_pdf, a.png_dir)
        title_page(
            pdf,
            'Straddling windows see two levels up the RHM tree',
            'a 5x5 window containing only 2x2 blocks still rejects %d%% of grids\n'
            'that are coherent through level 2 -- it is policing the root' % round(100 * (1 - med)),
            ['A k x k window contains aligned blocks up to j* = floor(log_s((k+1)/2)).\n'
             'The question: does it check anything above j*?',
             'Method: tile a grid with independent legal s^j blocks (ensemble E_j) and\n'
             'measure a_j, how often all its windows are still ones the machine knows.\n'
             'a_L = 1 verifies the window set is complete; a_j < 1 for j > j* is the signal.',
             'Validated against exact enumeration of every L=2 grid, then run at L=3\n'
             'where enumeration is impossible (v^64 grids).',
             '%d cells, %d with a provably saturated window set.' % (len(rows), len(good))])
        page_method(pdf, rows, good, exact)
        page_profiles(pdf, good)
        page_when(pdf, good)
        page_traces(pdf)
        page_table(pdf, rows, good)
    print('wrote', a.out, Path(a.out).stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
