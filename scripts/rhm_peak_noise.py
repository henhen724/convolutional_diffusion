"""How much of the scatter in x* = P*/|R_L| is the estimator, not the grammar?

STATUS files "2D peaks scatter 4.8x per cell though medians reproduce to 1.2x"
as an open problem and suggests raising n_gen.  Before spending a run on that,
this asks a cheaper question: how much of that scatter is *measurement noise in
the peak estimator* rather than real cell-to-cell variation in the grammar?

The reason to suspect the estimator: ``success`` is a binomial fraction over
``n_gen`` draws, the success-vs-P curve is flat near its top, and
``rhm_peak_analysis.peak_of`` takes the argmax and then fits a parabola through
the top THREE points.  When the top of the curve is a statistical tie -- and at
n_gen=4096, success~0.03 has a standard error of ~0.003, so ties are the normal
case -- the argmax jumps between tied points and drags the whole 3-point fit
window with it.  That is a high-variance estimator by construction.

The decomposition needs no new run and no assumption that the true curve is
smooth.  For each measured curve, treat the observed success fractions as the
truth, resample success counts at each P from Binomial(n_gen, p_hat), and
re-estimate P* on each replicate.  The spread of P* across replicates is pure
estimator noise on a FIXED grammar.  Comparing it to the observed spread across
rule seeds within one (v, m) splits that observed spread into

    log-var(observed across seeds)  =  log-var(estimator)  +  log-var(grammar)

so the grammar's own contribution is what is left after the estimator's is
subtracted.  Everything is done in log space because these ratios are
log-normal-ish (``rhm_collapse_variable.geo`` makes the same choice).

Two estimators are compared on identical replicates:

  A  current: argmax, then a parabola in log P through the top 3 points
     (``rhm_peak_analysis.peak_of``, reimplemented here so both run on the
     same bootstrap draws).

  B  proposed: a parabola in log P fitted to every point within a factor
     ``--frac`` of the maximum, weighted by inverse binomial variance.  It uses
     more of the curve, so a single tied point cannot move it far.

B being smoother will mechanically show less spread, which on its own proves
nothing -- a constant estimator has zero variance and is useless.  So B is only
preferred if it ALSO tracks the same peak: the script reports the median
|log| displacement between A and B on the real (un-resampled) curves, and the
bootstrap bias of each estimator against the curve it was drawn from.  An
estimator that is quiet but systematically displaced is worse, not better.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

# Both estimators fit a parabola to 3-5 points spanning a narrow range of
# log P, so numpy's conditioning warning fires on nearly every bootstrap draw
# and would bury the output.  The ill-conditioning is the thing being measured
# (it is exactly why A is noisy), not a fault to fix, and both estimators
# already reject a fit whose vertex falls outside its own window.
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def U1(s, L):
    return 2 * s ** (L - 1) - 1


def load_exact_RL(root):
    out = {}
    for f in glob.glob(str(root / 'nk_exact_L*.json')):
        for r in json.load(open(f)):
            out[(r['v'], r['m'], r['s'], r['L'], r['rule_seed'])] = r['R_L']
    return out


def load_curves(root, patterns):
    """(v, m, s, L, rule_seed, k) -> rows ordered by P, U=1 windows only."""
    rows = []
    for pat in patterns:
        for f in glob.glob(str(root / pat)):
            rows += [r for r in json.load(open(f)) if 'success' in r]
    out = defaultdict(list)
    for r in rows:
        if r.get('block_align', 1) != 1 or not isinstance(r.get('k'), int):
            continue
        if r['k'] != U1(r['s'], r['L']):
            continue
        out[(r['v'], r['m'], r['s'], r['L'], r['rule_seed'], r['k'])].append(r)
    return {k: sorted(v, key=lambda r: r['P']) for k, v in out.items()}


def est_A(P, S, _n):
    """Current estimator: argmax + parabola through the top 3 points."""
    i = int(np.argmax(S))
    if not (0 < i < len(S) - 1 and S[i] > 0):
        return np.nan
    x, y = np.log(P[i - 1:i + 2]), S[i - 1:i + 2]
    a, b, _ = np.polyfit(x, y, 2)
    if a < 0:
        xv = -b / (2 * a)
        if x[0] <= xv <= x[2]:
            return float(np.exp(xv))
    return float(P[i])


def est_B(P, S, n_gen, frac=0.5, fallback=False):
    """Inverse-variance weighted parabola over every point near the top.

    The window is defined by the DATA's own maximum, so on a flat top it spans
    many points instead of three, and no single tied point can carry the fit.

    ``fallback`` makes a rejected fit return the argmax, which is what A does.
    Without it B returns nan, and because the rejections track curve shape
    rather than noise, a median over the surviving cells is taken on a biased
    subsample -- on the 2026-11-05 L=2 sweep B kept only 2 of 8 cells at v=3.
    It is off by default so the bootstrap in this script keeps measuring B
    where B is actually defined.
    """
    i = int(np.argmax(S))
    if not (0 < i < len(S) - 1 and S[i] > 0):
        return np.nan
    miss = float(P[i]) if fallback else np.nan
    sel = S >= frac * S[i]
    if sel.sum() < 4:
        lo, hi = max(0, i - 2), min(len(S), i + 3)
        sel = np.zeros_like(S, bool)
        sel[lo:hi] = True
    x, y = np.log(P[sel]), S[sel]
    var = np.clip(y * (1 - y) / max(n_gen, 1), 1e-12, None)
    try:
        a, b, _ = np.polyfit(x, y, 2, w=1.0 / np.sqrt(var))
    except Exception:
        return miss
    if a >= 0:
        return miss
    xv = -b / (2 * a)
    if not (x.min() <= xv <= x.max()):
        return miss
    return float(np.exp(xv))


def geo_sd(vals):
    v = np.asarray([x for x in vals if np.isfinite(x) and x > 0], float)
    if len(v) < 2:
        return np.nan
    return float(np.exp(np.std(np.log(v), ddof=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/rhm_sampling/2026-11-03')
    ap.add_argument('--patterns', nargs='+',
                    default=['L3peak_v*_k7.json', 'L3k7_v*_m*.json',
                             'P_v*_m*.json'])
    ap.add_argument('--boot', type=int, default=400)
    ap.add_argument('--frac', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    root = ROOT / a.root
    RL = load_exact_RL(root)
    cs = load_curves(root, a.patterns)
    if not cs:
        raise SystemExit('no U=1 curves found')
    rng = np.random.default_rng(a.seed)

    per_cell, rows_out = {}, []
    for key, rows in sorted(cs.items()):
        v, m, s, L, rs, k = key
        if key[:5] not in RL:
            continue
        P = np.array([r['P'] for r in rows], float)
        S = np.array([r['success'] for r in rows], float)
        ng = int(rows[0].get('n_gen', 4096))
        pa, pb = est_A(P, S, ng), est_B(P, S, ng, a.frac)
        if not np.isfinite(pa):
            continue                       # unbracketed: P* is a bound, skip
        ba, bb = [], []
        for _ in range(a.boot):
            Sb = rng.binomial(ng, np.clip(S, 0, 1)) / ng
            ba.append(est_A(P, Sb, ng))
            bb.append(est_B(P, Sb, ng, a.frac))
        rec = dict(v=v, m=m, s=s, L=L, rule_seed=rs, k=k, n_gen=ng,
                   R_L=RL[key[:5]], P_star_A=pa, P_star_B=pb,
                   x_A=pa / RL[key[:5]],
                   x_B=(pb / RL[key[:5]]) if np.isfinite(pb) else float('nan'),
                   boot_sd_A=geo_sd(ba), boot_sd_B=geo_sd(bb),
                   boot_bias_A=float(np.exp(np.nanmedian(np.log(np.array(ba) / pa)))),
                   boot_bias_B=(float(np.exp(np.nanmedian(np.log(np.array(bb) / pb))))
                                if np.isfinite(pb) else float('nan')),
                   succ_max=float(S.max()), n_P=len(P))
        rows_out.append(rec)
        per_cell[key] = rec

    dim = lambda s_, L_, rec: 2      # every curve here is the 2D model
    print('curves scored: %d' % len(rows_out))
    print()
    print('--- 1. estimator noise on a FIXED grammar (bootstrap, %d draws) ---'
          % a.boot)
    print('%-22s %10s %10s %10s %10s' % ('cell', 'A geo-sd', 'B geo-sd',
                                         'A bias', 'B bias'))
    for rec in rows_out:
        print('v%dm%d L%d rs%d %9s %10s %10s %10s %10s'
              % (rec['v'], rec['m'], rec['L'], rec['rule_seed'], '',
                 '%.2fx' % rec['boot_sd_A'],
                 '%.2fx' % rec['boot_sd_B'] if np.isfinite(rec['boot_sd_B']) else '-',
                 '%.3f' % rec['boot_bias_A'],
                 '%.3f' % rec['boot_bias_B'] if np.isfinite(rec['boot_bias_B']) else '-'))
    medA = np.nanmedian([r['boot_sd_A'] for r in rows_out])
    medB = np.nanmedian([r['boot_sd_B'] for r in rows_out])
    print('median estimator noise:  A %.2fx   B %.2fx' % (medA, medB))

    disp = [abs(np.log(r['x_B'] / r['x_A'])) for r in rows_out
            if np.isfinite(r['x_B'])]
    print('median |log| displacement B vs A on the real curves: %.3f (= %.2fx)'
          % (np.median(disp), float(np.exp(np.median(disp)))))

    print()
    print('--- 2. observed spread across rule seeds, same (v,m,L) ---')
    g = defaultdict(list)
    for r in rows_out:
        g[(r['v'], r['m'], r['L'])].append(r)
    print('%-16s %6s %10s %10s %12s %12s'
          % ('(v,m,L)', 'seeds', 'obs sd A', 'obs sd B', 'est sd A', 'est sd B'))
    tot = {'obsA': [], 'obsB': [], 'estA': [], 'estB': []}
    for key in sorted(g):
        sel = g[key]
        if len(sel) < 2:
            continue
        oA, oB = geo_sd([r['x_A'] for r in sel]), geo_sd([r['x_B'] for r in sel])
        eA = float(np.nanmedian([r['boot_sd_A'] for r in sel]))
        eB = float(np.nanmedian([r['boot_sd_B'] for r in sel]))
        for nm, val in (('obsA', oA), ('obsB', oB), ('estA', eA), ('estB', eB)):
            tot[nm].append(val)
        print('v%dm%d L%-9d %6d %10s %10s %12s %12s'
              % (key[0], key[1], key[2], len(sel),
                 '%.2fx' % oA if np.isfinite(oA) else '-',
                 '%.2fx' % oB if np.isfinite(oB) else '-',
                 '%.2fx' % eA, '%.2fx' % eB))

    def split(obs, est):
        """Grammar-only spread: log-variances subtract, floored at 1.0x."""
        lo, le = np.log(np.nanmedian(obs)), np.log(np.nanmedian(est))
        return float(np.exp(np.sqrt(max(lo ** 2 - le ** 2, 0.0))))

    print()
    print('--- 3. the split (medians over cells, log-variances subtract) ---')
    for tag, o, e in (('A current', tot['obsA'], tot['estA']),
                      ('B proposed', tot['obsB'], tot['estB'])):
        print('  %-11s observed %.2fx = estimator %.2fx  (x)  grammar %.2fx'
              % (tag, np.nanmedian(o), np.nanmedian(e), split(o, e)))

    print()
    print('--- 4. what the headline becomes under each estimator ---')

    def fit_b(xs, ys):
        """P* ~ |R_L|^b by least squares in log-log; b and its standard error."""
        X, Y = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
        n = len(X)
        if n < 3:
            return float('nan'), float('nan')
        A_ = np.vstack([X, np.ones(n)]).T
        coef, res, *_ = np.linalg.lstsq(A_, Y, rcond=None)
        resid = Y - A_ @ coef
        s2 = float(resid @ resid) / (n - 2)
        se = float(np.sqrt(s2 / np.sum((X - X.mean()) ** 2)))
        return float(coef[0]), se

    print('%-12s %6s %10s %10s %16s' % ('estimator', 'cells', 'med x*',
                                        'geo-sd', 'b in P*~|R_L|^b'))
    for tag, xk, pk in (('A current', 'x_A', 'P_star_A'),
                        ('B proposed', 'x_B', 'P_star_B')):
        sel = [r for r in rows_out if np.isfinite(r[xk]) and r[xk] > 0]
        xs = np.array([r[xk] for r in sel])
        b, se = fit_b([r['R_L'] for r in sel], [r[pk] for r in sel])
        # CI on the median from resampling cells, so it reflects how few
        # independent grammars there actually are, not the bootstrap draws.
        boots = [np.median(rng.choice(xs, len(xs), replace=True))
                 for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print('%-12s %6d %10.4f %10s %16s' % (
            tag, len(sel), float(np.median(xs)), '%.2fx' % geo_sd(xs),
            '%.3f +- %.3f' % (b, se)))
        print('%-12s %6s   95%% CI on the median  [%.4f, %.4f]' % ('', '', lo, hi))

    if a.out:
        p = ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows_out, open(p, 'w'), indent=1)
        print('\nwrote %s (%d curves)' % (p, len(rows_out)))


if __name__ == '__main__':
    main()
