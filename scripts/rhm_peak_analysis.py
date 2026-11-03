#!/usr/bin/env python
"""Locate the success peak P* and test whether it scales as Theta(|R_L|).

hshunt's criterion is that the LS machine samples the RHM successfully when its
output is *both* novel and fully grammatical. Those two pull in opposite
directions in P:

  * grammaticality needs the patch table filled, and the table has N_k distinct
    windows with N_k = Theta(|R_L|) at the U=1 window k = 2 s^(L-1) - 1
    (measured: N_3/|R_2| = 4.3-6.0, N_7/|R_3| = 1.6-6.1), so it needs P large;
  * novelty needs the training set to be a small part of the grammar, so it
    needs P << |R_L|.

So success has a peak, and the claim to test is that the peak sits at
P* = Theta(|R_L|). The test that makes it a law rather than a fit: rules are
drawn with replacement, so independent rule seeds at the *same* (v, m, L) give
|R_L| differing by up to 20x (v3m2 at L=3: 6291456, 301440, 6291456). If the
curves collapse when plotted against P/|R_L| but not against P, the scaling
variable is |R_L| and nothing else.

|R_L| is taken from the exact enumerations in nk_exact_L{2,3}.json, never from
the nominal v*m^M, precisely because the collisions are the point.

Usage:
  python scripts/rhm_peak_analysis.py --out results/rhm_sampling/2026-11-03
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

U1 = lambda s, L: 2 * s ** (L - 1) - 1          # window that just reaches the root


def load_exact_RL(root: Path) -> dict:
    """(v, m, s, L, rule_seed) -> exact |R_L|, from the enumeration runs."""
    out = {}
    for f in glob.glob(str(root / 'nk_exact_L*.json')):
        for r in json.load(open(f)):
            out[(r['v'], r['m'], r['s'], r['L'], r['rule_seed'])] = r['R_L']
    return out


def load_cells(root: Path, patterns) -> list:
    rows = []
    for pat in patterns:
        for f in glob.glob(str(root / pat)):
            rows += [r for r in json.load(open(f)) if 'success' in r]
    return rows


def curves(rows, RL, u1_only=True):
    """Group cells into (v, m, s, L, rule_seed, k) curves ordered by P."""
    out = {}
    for r in rows:
        if r.get('block_align', 1) != 1 or not isinstance(r.get('k'), int):
            continue
        if u1_only and r['k'] != U1(r['s'], r['L']):
            continue
        key = (r['v'], r['m'], r['s'], r['L'], r['rule_seed'], r['k'])
        if key[:5] not in RL:
            continue
        out.setdefault(key, []).append(r)
    for key in out:
        out[key] = sorted(out[key], key=lambda r: r['P'])
    return out


def peak_of(cs, RL_key, RL):
    """Peak success and where it sits, with a log-P parabola through the top 3.

    The raw argmax is quantised to the ladder, which is coarse (3x steps), so
    the parabola refines it. It is only reported when the argmax is interior;
    at an endpoint the peak is not bracketed and P* is a bound, not a value.
    """
    P = np.array([c['P'] for c in cs], float)
    S = np.array([c['success'] for c in cs], float)
    i = int(np.argmax(S))
    interior = 0 < i < len(S) - 1 and S[i] > 0
    Pstar = P[i]
    if interior:
        x, y = np.log(P[i - 1:i + 2]), S[i - 1:i + 2]
        a, b, _ = np.polyfit(x, y, 2)
        if a < 0:
            xv = -b / (2 * a)
            if x[0] <= xv <= x[2]:
                Pstar = float(np.exp(xv))
    return dict(P_argmax=float(P[i]), P_star=float(Pstar), succ_max=float(S[i]),
                bracketed=bool(interior), R_L=RL[RL_key],
                ratio=float(Pstar / RL[RL_key]),
                gram_at=float(cs[i]['grammatical']), novel_at=float(cs[i]['novel']))


def conditional_novelty(rows, RL, min_gram_count=8):
    """P(novel | grammatical) against x = P/|R_L|, and what that says.

    This is the quantity the peak is really about. ``success`` mixes two
    effects -- how often the machine is grammatical at all, which rises with P,
    and how often a grammatical output is new, which falls -- and dividing them
    isolates the second. A machine that sampled uniformly from the grammar
    would give exactly ``(1 - 1/|R_L|)^P ~ exp(-x)``, so the departure measures
    how far the machine is from that.
    """
    pts = []
    for r in rows:
        if r.get('block_align', 1) != 1 or not isinstance(r.get('k'), int):
            continue
        if r['k'] != U1(r['s'], r['L']) or r['grammatical'] <= 0:
            continue
        key = (r['v'], r['m'], r['s'], r['L'], r['rule_seed'])
        if key not in RL or r['grammatical'] * r['n_samples'] < min_gram_count:
            continue
        pts.append((r['P'] / RL[key], r['success'] / r['grammatical']))
    if len(pts) < 10:
        return
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    ok = (y > 0.02) & (y < 0.999)
    slope, icpt = np.polyfit(x[ok], -np.log(y[ok]), 1)
    print('\nP(novel | grammatical): fit -ln y = %.3f x + %.3f over n=%d cells'
          % (slope, icpt, ok.sum()))
    print('  uniform sampling from R_L would give slope 1, intercept 0.')
    print('  intercept %.2f => only %.0f%% of grammatical outputs are novel even at'
          ' x -> 0: the machine is biased towards reproducing training grids,'
          % (icpt, 100 * np.exp(-icpt)))
    print('  which is a second, P-independent drag on success on top of the'
          ' exp(-x) coverage term.')
    return dict(slope=float(slope), intercept=float(icpt), n=int(ok.sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/rhm_sampling/2026-11-03')
    p.add_argument('--png', default=None, help='write the collapse figure here')
    a = p.parse_args()
    root = Path(a.out)

    RL = load_exact_RL(root)
    rows = load_cells(root, ['L3peak_v*_k7.json', 'L3k7_v*_m*.json', 'P_v*_m*.json'])
    cs = curves(rows, RL)
    if not cs:
        raise SystemExit('no U=1 curves with an exact |R_L| yet')

    print('%-20s %9s %10s %9s %8s %7s %7s %6s' % (
        'v,m,L,rs  k', '|R_L|', 'P*', 'P*/|R_L|', 'succ_max', 'gram', 'novel', 'brack'))
    summary = []
    for key in sorted(cs):
        v, m, s, L, rs, k = key
        pk = peak_of(cs[key], key[:5], RL)
        summary.append(dict(v=v, m=m, s=s, L=L, rule_seed=rs, k=k, n_P=len(cs[key]), **pk))
        print('v%dm%d L%d rs%d  k%-3d %9d %10.3g %9.3f %8.3f %7.3f %7.3f %6s' % (
            v, m, L, rs, k, pk['R_L'], pk['P_star'], pk['ratio'], pk['succ_max'],
            pk['gram_at'], pk['novel_at'], pk['bracketed']))

    conditional_novelty(rows, RL)

    br = [r for r in summary if r['bracketed']]
    for lab, sub in (('L=2', [r for r in br if r['L'] == 2]),
                     ('L=3', [r for r in br if r['L'] == 3]),
                     ('all', br)):
        if not sub:
            continue
        ra = np.array([r['ratio'] for r in sub])
        print('\n%s bracketed peaks: n=%d  P*/|R_L| median %.3f  geo-sd %.2fx  range %.3f-%.3f'
              % (lab, len(sub), np.median(ra), np.exp(np.std(np.log(ra))), ra.min(), ra.max()))

    # the collapse test: scatter in P vs in P/|R_L|, same points
    if br:
        allc = [(r['L'], r['P_star'], r['ratio']) for r in br]
        for L in sorted({c[0] for c in allc}):
            ps = np.array([c[1] for c in allc if c[0] == L])
            rs_ = np.array([c[2] for c in allc if c[0] == L])
            if len(ps) > 1:
                print('L=%d: spread of P* is %.1fx, spread of P*/|R_L| is %.1fx'
                      % (L, ps.max() / ps.min(), rs_.max() / rs_.min()))

    json.dump(summary, open(root / 'peak_summary.json', 'w'), indent=1)
    print('\nwrote', root / 'peak_summary.json')

    if a.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for key in sorted(cs):
            v, m, s, L, rs, k = key
            P = np.array([c['P'] for c in cs[key]], float)
            S = np.array([c['success'] for c in cs[key]], float)
            lab = 'v%dm%d L%d rs%d' % (v, m, L, rs)
            col = 'C%d' % (L - 2)
            ax[0].plot(P, S, 'o-', ms=3, lw=1, color=col, alpha=.6, label=lab)
            ax[1].plot(P / RL[key[:5]], S, 'o-', ms=3, lw=1, color=col, alpha=.6)
        for b, t in zip(ax, ['success vs $P$', 'success vs $P/|R_L|$ (collapse)']):
            b.set_xscale('log'); b.set_ylabel('novel AND grammatical'); b.set_title(t)
            b.grid(alpha=.3)
        ax[0].set_xlabel('$P$'); ax[1].set_xlabel('$P/|R_L|$')
        ax[1].axvline(1.0, color='k', ls=':', lw=1)
        fig.tight_layout(); fig.savefig(a.png, dpi=150)
        print('wrote', a.png)


if __name__ == '__main__':
    main()
