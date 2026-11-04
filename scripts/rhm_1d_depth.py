"""Sampling complexity of the LS machine on the *1D* Random Hierarchy Model.

Why 1D.  The measured law ``P* = Theta(|R_L|) = exp(Theta(n))`` currently rests
on two depths, L=2 and L=3, because the window that just reaches the root has
side ``k = 2 s^(L-1) - 1 = d - 1`` and at L=4 in 2D that is a 15x15 = 225-site
patch: the table never fills and the int64 window code overflows.  In 1D the
same window is 15 *sites*, so L=4 (n = 16 sites) is routine and the law gets a
third depth -- the cheapest honest test of the asymptotics there is.

Everything else is the 2D procedure with one axis removed, deliberately: the
same absorbing-state ("erasure") diffusion, the same *pooled* translation-
invariant window table with the off-grid padding marker, the same reveal-one
ancestral sampler, the same Laplace smoothing, and hshunt's success criterion
(a generated string counts only if it is **not in the training set** and
**fully grammatical**).  So a difference between 1D and 2D is a statement about
dimension, not about implementation.

In 1D the grammar has ``|R_L| <= v m^M`` with ``M = (n-1)/(s-1)``, ``n = s^L``
sites.  Rules are drawn with replacement, so words collide and the realized
``|R_L|`` is smaller and varies between rule seeds -- which is what makes the
collapse against the *exact* enumerated ``|R_L|`` a test rather than a fit.

Usage:
  python scripts/rhm_1d_depth.py --L 2 3 4 --v 2 3 4 --m 2 3 4 \
      --rule_seeds 3 --out results/rhm_1d/2026-11-03/sweep.json
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MASK = -1          # internal marker; the public alphabet is 0..v-1 and v = off-grid


class RHM1D:
    """1D Random Hierarchy Model, matching RandomHierarchyModel2D's conventions.

    Root symbol uniform on ``0..v-1``; ``rules[0]`` applied first; each level's
    ``(v, m, s)`` table drawn i.i.d. **with replacement**, so distinct symbols
    and distinct synonyms can render the same word.
    """

    def __init__(self, v, s, L, m, seed):
        self.v, self.s, self.L, self.m, self.seed = v, s, L, m, seed
        rng = np.random.default_rng(seed)
        self.rules = [rng.integers(0, v, size=(v, m, s)) for _ in range(L)]
        self.d = s ** L

    def sample(self, n, seed):
        """(n, d) int64 array of leaf tokens."""
        rng = np.random.default_rng(seed)
        x = rng.integers(0, self.v, size=(n, 1))
        for rule in self.rules:
            w = rng.integers(0, self.m, size=x.shape)
            x = rule[x, w].reshape(n, -1)              # (n, len, s) -> (n, len*s)
        return x.astype(np.int64)


def feasible_masks(model, strings):
    """Bottom-up feasible-symbol bitmasks, one array per level (see rhm_grammar)."""
    v, s, L = model.v, model.s, model.L
    if v > 63:
        raise ValueError('v=%d exceeds the 63 symbols a uint64 bitmask holds' % v)
    cur = np.uint64(1) << np.asarray(strings, dtype=np.uint64)
    out = [None] * (L + 1)
    out[L] = cur
    for l in range(L - 1, -1, -1):
        child = out[l + 1]
        n, length = child.shape
        blocks = child.reshape(n, length // s, s)
        rule, acc = model.rules[l], np.zeros((n, length // s), dtype=np.uint64)
        for a in range(v):
            for wi in range(model.m):
                fits = np.ones((n, length // s), dtype=bool)
                for di in range(s):
                    fits &= (blocks[:, :, di] & (np.uint64(1) << np.uint64(rule[a, wi, di]))) != 0
                    if not fits.any():
                        break
                acc |= np.where(fits, np.uint64(1) << np.uint64(a), np.uint64(0))
        out[l] = acc
    return out


def is_grammatical(model, strings):
    """(N,) bool: does each string parse all the way to a root symbol?"""
    return feasible_masks(model, strings)[0][:, 0] != 0


def exact_RL(model, max_words=1_000_000):
    """Exact |R_L|, the number of distinct leaf strings the grammar can produce.

    Bottom-up over distinct renderings, the 1D twin of ``rhm_ceiling_exact``'s.
    Returns ``None`` once any intermediate set passes ``max_words``, in which
    case the cell is skipped rather than reported against a nominal ``v m^M``
    that the collisions make wrong.

    The cap is checked inside *both* loops and ``max_words`` is deliberately
    modest.  Job 51009204 was OOM-killed here, not in the measurement: the
    v4 m>=3 L=4 cells were always going to be skipped, but enumerating them far
    enough to prove it held ``v`` symbol sets of millions of Python ``bytes``
    at once.  A set only ever needs to be big enough to decide the comparison
    against ``R_max``, and callers use ``R`` only when it is far below this cap.
    """
    cur = {a: {bytes([a])} for a in range(model.v)}
    for l in range(model.L - 1, -1, -1):
        nxt = {}
        for a in range(model.v):
            acc = set()
            for wi in range(model.m):
                parts = [cur[int(model.rules[l][a, wi, di])] for di in range(model.s)]
                words = parts[0]
                for p in parts[1:]:
                    words = {x + y for x in words for y in p}
                    if len(words) > max_words:
                        return None
                acc |= words
                if len(acc) > max_words:
                    return None
            nxt[a] = acc
        cur = nxt
    return len(set().union(*cur.values()))


def window_table(strings, k, v, device, chunk=200_000):
    """Distinct length-k windows and their multiplicities, pooled over position.

    One window per predicted site, the string framed in the off-grid marker
    ``v`` so border sites get a window too and it records where the edge is --
    the 1D form of ``distinct_patch_table(..., pad_mode='outside')``.
    """
    s = torch.as_tensor(strings, dtype=torch.int64, device=device)
    n, d = s.shape
    pad = k // 2
    framed = torch.full((n, d + k - 1), v, dtype=torch.int64, device=device)
    framed[:, pad:pad + d] = s
    w = framed.unfold(1, k, 1).reshape(-1, k)                      # (n*d, k)
    # pack base (v+1) so unique() runs on one int64 column, not k of them
    powers = (v + 1) ** torch.arange(k, device=device, dtype=torch.int64)
    if (k - 1) * np.log(v + 1) >= 63 * np.log(2):
        raise SystemExit('v=%d k=%d: base-%d code of %d digits overflows int64' % (v, k, v + 1, k))
    codes = (w * powers).sum(-1)
    uniq, counts = torch.unique(codes, return_counts=True)
    digits = torch.stack([(uniq // (v + 1) ** i) % (v + 1) for i in range(k)], dim=1)
    return digits, counts.double()


class LS1D:
    """Pooled local-score machine: posterior = empirical centre distribution.

    Identical in content to ``PatchCountDiscreteScoreMachine`` with
    ``block_align=1, pad_mode='outside'``: a window matches a training window
    iff every *observed* site agrees, where off-grid sites count as observed
    (their own symbol ``v``) so a border window can only match a training window
    framed the same way.  Masked sites are one-hot zero and constrain nothing.
    """

    def __init__(self, strings, v, k, device, alpha_smooth=1e-3, chunk=32768):
        self.v, self.k, self.device = v, k, device
        self.alpha, self.chunk = alpha_smooth, chunk
        patches, counts = window_table(strings, k, v, device)
        C = v + 1
        # float32 directly: a float64 intermediate here is ~2.5 GB at the largest
        # L=4 cell, and with at most k=15 ones per row the dot products are exact
        # in float32 anyway
        onehot = torch.nn.functional.one_hot(patches, num_classes=C).float()    # (NP, k, C)
        self.flat = onehot.reshape(patches.shape[0], -1)                        # (NP, k*C)
        self.weight = counts.to(torch.float32)
        self.center = torch.nn.functional.one_hot(patches[:, k // 2],
                                                  num_classes=C)[:, :v].float()
        self.nout = (patches == v).sum(1).float()
        self.n_distinct = int(patches.shape[0])
        self.n_total = float(counts.sum())

    def posterior(self, xt, sites):
        """(B, v) posterior over the true token at ``sites[b]`` of ``xt[b]``."""
        v, k, d = self.v, self.k, self.k // 2
        B, n = xt.shape
        C = v + 1
        off = torch.arange(-d, d + 1, device=self.device)
        idx = sites[:, None] + off[None, :]                        # (B, k)
        inside = (idx >= 0) & (idx < n)
        safe = idx.clamp(0, n - 1)
        tok = torch.gather(xt, 1, safe)                            # (B, k)
        q = torch.zeros(B, k, C, device=self.device)
        obs = inside & (tok != MASK)
        q.scatter_(2, torch.where(obs, tok, torch.zeros_like(tok)).unsqueeze(-1),
                   obs.float().unsqueeze(-1))
        q[:, :, v] = (~inside).float()                             # off-grid is observed
        need = obs.sum(1).float() + (~inside).sum(1).float()
        need_out = (~inside).sum(1).float()

        qf = q.reshape(B, -1)
        cnt = torch.zeros(B, v, device=self.device)
        nm = torch.zeros(B, device=self.device)
        for i in range(0, self.flat.shape[0], self.chunk):
            sl = slice(i, i + self.chunk)
            agree = qf @ self.flat[sl].T                           # (B, chunk)
            match = ((agree - need[:, None]).abs() < 0.5) & \
                    (self.nout[None, sl] == need_out[:, None])
            mw = match.float() * self.weight[None, sl]
            cnt += mw @ self.center[sl]
            nm += mw.sum(1)
        return (cnt + self.alpha) / (nm[:, None] + self.alpha * v)


def generate(machine, n_samples, n_sites, device, generator):
    """Reveal-one ancestral sampling from the all-masked string."""
    xt = torch.full((n_samples, n_sites), MASK, dtype=torch.int64, device=device)
    rows = torch.arange(n_samples, device=device)
    for _ in range(n_sites):
        masked = xt == MASK
        if not bool(masked.any()):
            break
        noise = torch.rand(n_samples, n_sites, device=device, generator=generator)
        pick = torch.where(masked, noise, torch.full_like(noise, -1.0)).argmax(1)
        p = machine.posterior(xt, pick).clamp_min(1e-12)
        tok = torch.multinomial(p, 1, generator=generator)[:, 0]
        keep = masked[rows, pick]
        xt[rows[keep], pick[keep]] = tok[keep]
    return xt


def run_cell(model, P, k, n_gen, device, seed, alpha_smooth, gen_batch):
    """One (P, k) measurement: build the table on P strings, generate, score."""
    t0 = time.time()
    train = model.sample(P, seed=seed)
    machine = LS1D(train, model.v, k, device, alpha_smooth=alpha_smooth)
    gen = torch.Generator(device=device).manual_seed(seed + 9973)
    out = []
    for i in range(0, n_gen, gen_batch):
        b = min(gen_batch, n_gen - i)
        out.append(generate(machine, b, model.d, device, gen).cpu().numpy())
    gs = np.concatenate(out, 0)

    gram = is_grammatical(model, gs)
    tr = set(map(bytes, np.ascontiguousarray(train.astype(np.int8))))
    novel = np.array([bytes(r) not in tr
                      for r in np.ascontiguousarray(gs.astype(np.int8))], dtype=bool)
    # recorded every cell, not just when something breaks: job 51009204 was
    # OOM-killed and the only way to tell which phase did it was the log's last
    # line.  ru_maxrss is a high-water mark for the whole process, so it only
    # ever rises -- read the *jumps* between cells, not the absolute value.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # KB -> GB
    gpu = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    return dict(P=int(P), k=int(k), v=model.v, s=model.s, L=model.L, m=model.m,
                rule_seed=model.seed - 1000, block_align=1,
                n_samples=int(len(gs)), grammatical=float(gram.mean()),
                novel=float(novel.mean()), success=float((gram & novel).mean()),
                n_distinct_patches=machine.n_distinct,
                n_patches_total=machine.n_total, rss_gb=round(rss, 3),
                gpu_peak_gb=round(gpu, 3),
                train_distinct=len(tr), seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v', type=int, nargs='+', default=[2, 3, 4])
    ap.add_argument('--s', type=int, default=2)
    ap.add_argument('--L', type=int, nargs='+', default=[2, 3, 4])
    ap.add_argument('--m', type=int, nargs='+', default=[2, 3, 4])
    ap.add_argument('--rule_seeds', type=int, default=3)
    ap.add_argument('--n_gen', type=int, default=2048)
    ap.add_argument('--gen_batch', type=int, default=1024)
    ap.add_argument('--alpha_smooth', type=float, default=1e-3)
    ap.add_argument('--R_max', type=float, default=1.5e5,
                    help='skip grammars too large for the P ladder to bracket the peak')
    ap.add_argument('--P_min', type=int, default=1)
    ap.add_argument('--P_max', type=int, default=262144)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--rl_out', type=str, default=None,
                    help='where to write the exact |R_L| table (default: beside --out)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # one absolute ladder for every cell, so where the peak lands is a
    # measurement and not a consequence of having centred the grid on 0.2|R_L|
    ladder = [p for p in (args.P_min * 2 ** i for i in range(40)) if p <= args.P_max]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rl_out = Path(args.rl_out) if args.rl_out else out.with_name('RL_exact_1d.json')

    rows, rl_rows = [], []
    for L in args.L:
        for v in args.v:
            for m in args.m:
                for rs in range(args.rule_seeds):
                    model = RHM1D(v, args.s, L, m, seed=1000 + rs)
                    k = 2 * args.s ** (L - 1) - 1          # U=1: just reaches the root
                    nominal = v * m ** ((model.d - 1) // (args.s - 1))
                    # the smallest realized/nominal ratio seen anywhere in this
                    # sweep is 0.011, so a grammar 1000x over R_max cannot come
                    # back under it; don't pay to enumerate what collisions
                    # could never rescue
                    R = None if nominal > 1000 * args.R_max else exact_RL(model)
                    if R is None or R > args.R_max:
                        print('skip  v%d m%d L%d rs%d  |R_L|=%s (nominal %d) > R_max'
                              % (v, m, L, rs, R, nominal), flush=True)
                        continue
                    rl_rows.append(dict(v=v, s=args.s, L=L, m=m, rule_seed=rs, k=k,
                                        R_L=int(R), R_L_nominal=int(nominal)))
                    json.dump(rl_rows, open(rl_out, 'w'), indent=1)
                    print('cell  v%d m%d L%d rs%d  n=%d k=%d  |R_L|=%d (nominal %d, %.2fx)'
                          % (v, m, L, rs, model.d, k, R, nominal, R / nominal), flush=True)
                    for P in ladder:
                        r = run_cell(model, P, k, args.n_gen, device, 1234 + 7 * rs,
                                     args.alpha_smooth, args.gen_batch)
                        r['R_L'] = int(R)
                        rows.append(r)
                        print('   P=%-7d gram %.4f  novel %.4f  succ %.4f  |table|=%-7d'
                              '  rss %.2fG  gpu %.2fG  [%.0fs]'
                              % (P, r['grammatical'], r['novel'], r['success'],
                                 r['n_distinct_patches'], r['rss_gb'], r['gpu_peak_gb'],
                                 r['seconds']), flush=True)
                        json.dump(rows, open(out, 'w'), indent=1)
    print('wrote %s (%d rows) and %s (%d grammars)'
          % (out, len(rows), rl_out, len(rl_rows)))


if __name__ == '__main__':
    main()
