"""Score the per-site-vs-schedule arms across cells.

``rhm_k_schedule.py`` runs four matched arms per cell -- the criterion, the
criterion capped to its own induced schedule, the bare schedule, and the best
fixed k.  One cell said the per-site choice is worth 4.7x at matched mean k;
this asks whether that holds across v, rule seed and P, and settles two
questions that were registered before the data existed:

1. ``rhm_k_schedule.sbatch`` predicted the sched/adaptive ratio *falls* as v
   grows, on the reasoning that a larger vocabulary makes support sparser and
   more unevenly spread across sites.
2. ``rhm_k_locality.py`` then found the opposite tendency in the regression --
   the schedule explains *more* of the chosen k at high v (R^2 .09-.21 at v2,
   .30-.41 at v4) -- which argues the ratio should *rise* with v.

They cannot both be right, and the numbers here decide it.  The mean-k columns
are the control that makes the comparison mean anything: if the bare schedule
drifts to a different mean k than the adaptive arm, it is no longer the same
window budget spent differently.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median as med

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ARMS = ('adaptive', 'cap', 'sched', 'fixed')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results/rhm_sampling/2026-11-04')
    args = ap.parse_args()

    rows = []
    for p in sorted((ROOT / args.results).glob('ksched_*.json')):
        if 'smoke' in p.name:
            continue
        rows.extend(json.load(open(p)))
    if not rows:
        print('no ksched results under %s' % args.results)
        return
    print('%d cell-rows from %s\n' % (len(rows), args.results))

    print('=== peak over the P ladder, per cell ===')
    print('  v rs |    adaptive       cap      sched     fixed |  sched/ad  ad/fixed'
          ' | meank ad/sched')
    by_v = defaultdict(list)
    peaks = []
    for (v, rs), grp in sorted(_group(rows).items()):
        pk = {a: max(g['arms'][a]['success'] for g in grp) for a in ARMS}
        # mean k read at the P where the adaptive arm peaked, so the window
        # budget being compared is the one that produced the headline number
        best = max(grp, key=lambda g: g['arms']['adaptive']['success'])
        ka, ks = best['arms']['adaptive']['mean_k'], best['arms']['sched']['mean_k']
        sa = pk['sched'] / pk['adaptive'] if pk['adaptive'] else float('nan')
        af = pk['adaptive'] / pk['fixed'] if pk['fixed'] else float('inf')
        by_v[v].append((sa, af))
        peaks.append((v, rs, pk, sa, af, ka, ks))
        print('  %d %2d | %9.4f %9.4f %9.4f %9.4f | %9.3f %9s | %5.2f %5.2f'
              % (v, rs, pk['adaptive'], pk['cap'], pk['sched'], pk['fixed'],
                 sa, ('%.1f' % af) if af != float('inf') else 'inf', ka, ks))

    print('\n=== does the per-site choice survive across cells? ===')
    n = len(peaks)
    print('  adaptive > bare schedule : %d / %d cells' % (
        sum(1 for p in peaks if p[2]['adaptive'] > p[2]['sched']), n))
    print('  adaptive > best fixed k  : %d / %d cells' % (
        sum(1 for p in peaks if p[2]['adaptive'] > p[2]['fixed']), n))
    print('  cap within 20%% of adaptive: %d / %d cells' % (
        sum(1 for p in peaks if p[2]['adaptive']
            and abs(p[2]['cap'] - p[2]['adaptive']) / p[2]['adaptive'] < 0.2), n))
    print('  median sched/adaptive    : %.3f  (1.0 would mean the schedule is enough)'
          % med([p[3] for p in peaks]))
    print('  median |mean k| gap      : %.2f  (the arms spend the same window budget)'
          % med([abs(p[5] - p[6]) for p in peaks]))

    print('\n=== the registered disagreement: how does sched/adaptive move with v? ===')
    print('  v | cells | median sched/adaptive | median adaptive/fixed')
    for v in sorted(by_v):
        print('  %d | %5d | %21.3f | %21s'
              % (v, len(by_v[v]), med([x[0] for x in by_v[v]]),
                 '%.1f' % med([x[1] for x in by_v[v] if x[1] != float('inf')])
                 if any(x[1] != float('inf') for x in by_v[v]) else 'inf'))
    vs = sorted(by_v)
    if len(vs) > 1:
        lo, hi = med([x[0] for x in by_v[vs[0]]]), med([x[0] for x in by_v[vs[-1]]])
        print('  -> ratio %s from v=%d to v=%d (%.3f -> %.3f): %s'
              % ('rises' if hi > lo else 'falls', vs[0], vs[-1], lo, hi,
                 'the sbatch prediction is refuted, the locality caveat holds'
                 if hi > lo else 'the sbatch prediction holds'))


def _group(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r['v'], r['rule_seed'])].append(r)
    return g


if __name__ == '__main__':
    main()
