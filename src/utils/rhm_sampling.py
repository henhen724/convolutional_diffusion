"""
A patch-count form of the local discrete score machine, for sampling-complexity
sweeps on the Random Hierarchy Model.

``LocalDiscreteScoreMachine`` re-reads the whole training set on every call and
convolves the noisy batch against every training patch individually, so its cost
grows linearly in the number of training samples ``P``. Measuring a sampling
complexity means pushing ``P`` up by orders of magnitude until the generated
samples become grammatical, which that cost makes impractical.

Nothing is lost by collapsing duplicates first. The posterior

    p(center = c | window) = (sum over matching training patches of [center = c] + alpha)
                             / (number of matching training patches + alpha * V)

depends on the training patches only through *how many times* each distinct
``k x k`` patch occurs, so counting distinct patches once and carrying their
multiplicities as weights gives the identical posterior at a cost set by the
number of *distinct* patches. That number saturates: once ``P`` is large enough
that every patch the grammar allows has been seen, more data only sharpens the
weights. This is what makes the sweep affordable, and it is also exactly the
quantity the scaling law turns out to be about.

:class:`PatchCountDiscreteScoreMachine` is a drop-in backbone for
``ScheduledDiscreteScoreMachine``; ``test_rhm_sampling.py`` checks it reproduces
``LocalDiscreteScoreMachine``'s posterior to floating-point tolerance.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def distinct_patch_table(grids, k: int, vocab_size: int, phase=None, block_align: int = 1,
                         pad_mode: str = 'zeros'):
    """Distinct ``k x k`` patches of a token-grid stack, with multiplicities.

    Parameters
    ----------
    grids : (P, H, W) int array/tensor of training grids.
    k : window side length.
    vocab_size : number of real token classes. ``pad_mode='outside'`` uses
        ``vocab_size`` itself as the off-grid marker, so patches carry
        ``vocab_size + 1`` symbols.
    phase : optional ``(pi, pj)``. If given, only windows that predict a site
        ``(i, j)`` with ``i % block_align == pi`` and ``j % block_align == pj``
        are collected. The predicted site is the window's centre element.
    block_align : modulus defining the phase classes; 1 pools every offset,
        which is what the translation-invariant machine does.
    pad_mode : ``'zeros'`` collects only windows lying wholly inside the grid,
        matching ``LocalDiscreteScoreMachine``. ``'outside'`` first frames each
        grid in an explicit off-grid marker, so border windows are collected too
        and carry where the edge is.

    Returns
    -------
    patches : (NP, k, k) int64 tensor of distinct patches.
    counts : (NP,) float64 tensor of how often each occurs across the grids and
        the sliding positions retained.
    """
    g = torch.as_tensor(np.asarray(grids), dtype=torch.long)
    d = k // 2
    if pad_mode == 'outside':
        g = F.pad(g[:, None], (d, k - 1 - d, d, k - 1 - d), value=vocab_size)[:, 0]
        site_offset = 0  # window at padded top-left r predicts grid site r
    else:
        site_offset = d  # window at top-left r predicts grid site r + d
    nr, nc = g.shape[1] - k + 1, g.shape[2] - k + 1
    keep_r = keep_c = None
    if phase is not None and block_align > 1:
        keep_r = (torch.arange(nr) + site_offset) % block_align == phase[0]
        keep_c = (torch.arange(nc) + site_offset) % block_align == phase[1]

    # One row per sliding-window position. Unfolding the whole stack at once
    # would materialise P * nr * nc rows of k*k int64 -- 2.6e8 rows, about
    # 100 GB, at P = 4e6 with k = 7 -- because reshaping a strided unfold view
    # copies. So the chunking happens *before* the unfold: each slice of grids
    # is unfolded and folded into the running table, whose height is bounded by
    # N_k rather than by P.
    def chunks():
        step = max(1, _MAX_FOLD_ROWS // max(1, nr * nc))
        for i in range(0, g.shape[0], step):
            w = g[i:i + step].unfold(1, k, 1).unfold(2, k, 1)
            if keep_r is not None:
                w = w[:, keep_r][:, :, keep_c]
            yield w.reshape(-1, k * k)

    uniq, counts = _unique_with_counts(chunks(), vocab_size + 1)
    if uniq.shape[0] == 0:
        return torch.zeros(0, k, k, dtype=torch.long), torch.zeros(0, dtype=torch.float64)
    return uniq.reshape(-1, k, k), counts


def _pack_rows(rows, base):
    """Pack rows of digits in ``[0, base)`` into as few int64 words as fit.

    Returns the ``(N, nwords)`` key tensor and the per-word digit widths, which
    :func:`_unpack_rows` needs to invert it.
    """
    per = max(1, int(62 / math.log2(base)))         # digits that fit one word
    words, spans = [], []
    for i in range(0, rows.shape[1], per):
        chunk = rows[:, i:i + per]
        w = torch.zeros(rows.shape[0], dtype=torch.long)
        for j in range(chunk.shape[1]):
            w = w * base + chunk[:, j]
        words.append(w)
        spans.append(chunk.shape[1])
    return torch.stack(words, 1), spans


def _unpack_rows(keys, spans, base):
    """Inverse of :func:`_pack_rows`."""
    out = []
    for c, width in enumerate(spans):
        w = keys[:, c].clone()
        digits = torch.zeros(keys.shape[0], width, dtype=torch.long)
        for j in range(width - 1, -1, -1):
            digits[:, j] = w % base
            w = w // base
        out.append(digits)
    return torch.cat(out, 1)


_MAX_FOLD_ROWS = 20_000_000


def _unique_with_counts(chunk_iter, base):
    """Distinct rows with multiplicities, over an iterable of row chunks.

    Rows are packed into a few int64 words -- 49 base-3 digits fit in two -- so
    the lexicographic sort inside ``torch.unique`` compares 2 columns instead of
    49, and each chunk is folded into the running table as it arrives rather
    than concatenated first.
    """
    spans = None
    uniq = counts = None
    for rows in chunk_iter:
        if rows.shape[0] == 0:
            continue
        keys, spans = _pack_rows(rows, base)
        if uniq is not None:
            keys = torch.cat([uniq, keys], 0)
        u, inv = torch.unique(keys, dim=0, return_inverse=True)
        c = torch.zeros(u.shape[0], dtype=torch.float64)
        w = torch.ones(keys.shape[0], dtype=torch.float64)
        if uniq is not None:
            w[:uniq.shape[0]] = counts          # carry the running multiplicities
        c.scatter_add_(0, inv, w)
        uniq, counts = u, c
    if uniq is None:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.float64)
    return _unpack_rows(uniq, spans, base), counts


class PatchCountDiscreteScoreMachine(nn.Module):
    """Exact local Bayes denoiser driven by a distinct-patch count table.

    Parameters mirror ``LocalDiscreteScoreMachine`` where they overlap. The
    training set is consumed once, at construction, into the patch table.

    ``block_align`` controls whether the machine is translation invariant.
    With the default ``1`` it is, exactly like ``LocalDiscreteScoreMachine``:
    every sliding offset feeds one shared table, so a window carries no
    information about *where* it sits. With ``block_align = s`` the table is
    split by the window's top-left corner mod ``s``, giving the machine the
    phase of the RHM's block grid -- the minimum extra information needed to
    tell an aligned ``s x s`` block from a straddling one. The two differ only
    in that bookkeeping; the window, the posterior and the sampler are the same.
    """

    def __init__(
        self,
        grids,
        vocab_size: int,
        kernel_size: int = 3,
        mask_token: Optional[int] = None,
        alpha_smooth: float = 1.0,
        device=None,
        chunk: int = 8192,
        block_align: int = 1,
        pad_mode: str = 'zeros',
        use_gemm: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.kernel_size = kernel_size
        self.mask_token = vocab_size if mask_token is None else mask_token
        self.alpha_smooth = alpha_smooth
        self.chunk = chunk
        self.block_align = block_align
        self.pad_mode = pad_mode
        self.use_gemm = use_gemm
        # 'outside' adds one symbol, the off-grid marker, to the matching alphabet
        self.n_channels = vocab_size + 1 if pad_mode == 'outside' else vocab_size
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        self.device = device
        # Agreement counts are integers bounded by k*k (one-hot inputs), so fp16
        # represents them exactly up to 2048 and the tensor cores accumulate in
        # fp32 anyway. Only the matching GEMM uses it; the weights, the counts
        # and the posterior all stay fp32.
        self.match_dtype = torch.float16 if device.type == 'cuda' else torch.float32

        d = kernel_size // 2
        a = block_align
        self.phases = [(0, 0)] if a <= 1 else [(pi, pj) for pi in range(a) for pj in range(a)]
        self.n_distinct = 0
        self.n_patches_total = 0.0
        for idx, ph in enumerate(self.phases):
            patches, counts = distinct_patch_table(
                grids, kernel_size, vocab_size, phase=None if a <= 1 else ph, block_align=a,
                pad_mode=pad_mode,
            )
            onehot = F.one_hot(patches, num_classes=self.n_channels).permute(0, 3, 1, 2).float()
            self.register_buffer(f'patch_onehot_{idx}', onehot.to(device))           # (NP, C, k, k)
            # Flattened copy for the GEMM matching path. Same (C, k, k) element
            # order as F.unfold produces, so the two index the window alike.
            self.register_buffer(f'patch_flat_{idx}',
                                 onehot.reshape(onehot.shape[0], -1).to(device=device,
                                                                        dtype=self.match_dtype))
            self.register_buffer(f'patch_weight_{idx}', counts.float().to(device))   # (NP,)
            # the centre is always a real in-grid site, never the off-grid marker
            self.register_buffer(f'patch_center_{idx}', onehot[:, :vocab_size, d, d].to(device))
            # how many of the patch's cells are off-grid: a window may only match a
            # patch framed exactly like itself, whether or not its sites are masked
            nout = (onehot[:, vocab_size].sum(dim=(1, 2)) if pad_mode == 'outside'
                    else torch.zeros(onehot.shape[0]))
            self.register_buffer(f'patch_nout_{idx}', nout.to(device))
            self.n_distinct += int(patches.shape[0])
            self.n_patches_total += float(counts.sum())

    def _table(self, idx):
        return (getattr(self, f'patch_onehot_{idx}'), getattr(self, f'patch_weight_{idx}'),
                getattr(self, f'patch_center_{idx}'), getattr(self, f'patch_nout_{idx}'))

    def _match_gemm(self, idx, xt_pad, need, need_out, k, b, h, w, sites=None):
        """Same match-and-accumulate as the conv2d loop, as one GEMM per chunk.

        ``conv2d`` with ``NP`` output channels over a ``(h+k-1)`` input is the
        wrong kernel shape for this -- the arithmetic is a plain inner product
        between each window and each patch, and casting it as a matmul over
        unfolded windows runs it on the tensor cores instead. The result is
        identical: agreement counts are small integers, so fp16 inputs with
        fp32 accumulation reproduce them exactly.
        """
        onehot, weight, center, nout = self._table(idx)
        flat = getattr(self, f'patch_flat_{idx}')
        np_, V = flat.shape[0], self.vocab_size
        # (b, C*k*k, h*w) -> (b*h*w, C*k*k), one row per output site
        cols = F.unfold(xt_pad.to(self.match_dtype), (k, k))
        cols = cols.transpose(1, 2).reshape(b * h * w, -1)
        need_f = need.reshape(-1, 1)
        out_f = (None if need_out is None
                 else need_out.reshape(1, h * w).expand(b, h * w).reshape(-1, 1))
        if sites is not None:
            # Only one site per batch element is ever read by the reveal-one
            # sampler, and the match is per site, so restricting here drops the
            # GEMM from b*h*w rows to b -- a factor of n on the whole trajectory.
            sel = torch.arange(b, device=cols.device) * (h * w) + sites.to(cols.device)
            cols, need_f = cols[sel], need_f[sel]
            out_f = None if out_f is None else out_f[sel]
        rows_out = cols.shape[0]
        c_flat = torch.zeros(rows_out, V, device=cols.device)
        n_flat = torch.zeros(rows_out, device=cols.device)
        wl_flat = torch.zeros(rows_out, device=cols.device)
        for i in range(0, np_, self.chunk):
            sl = slice(i, min(i + self.chunk, np_))
            agree = (cols @ flat[sl].T).float()                      # (B', chunk)
            match = (agree - need_f).abs() < 0.5
            if out_f is not None:
                match &= nout[None, sl] == out_f
            mw = match.float() * weight[None, sl]
            c_flat += mw @ center[sl]
            n_flat += mw.sum(1)
            wl_flat += mw @ weight[sl].clamp_min(1e-12).log()
        if sites is not None:
            # back to full grid shape; untouched sites keep n_match = 0, which
            # the caller turns into the uniform smoothing prior and never reads
            c_full = torch.zeros(b * h * w, V, device=cols.device)
            n_full = torch.zeros(b * h * w, device=cols.device)
            wl_full = torch.zeros(b * h * w, device=cols.device)
            c_full[sel], n_full[sel], wl_full[sel] = c_flat, n_flat, wl_flat
            c_flat, n_flat, wl_flat = c_full, n_full, wl_full
        return (c_flat.reshape(b, h, w, V).permute(0, 3, 1, 2).contiguous(),
                n_flat.reshape(b, h, w), wl_flat.reshape(b, h, w))

    supports_sites = True

    def forward(self, t, xt: torch.Tensor, label=None, device=None, k: Optional[int] = None,
                return_match: bool = False, sites: Optional[torch.Tensor] = None):
        """(B, V, H, W) posterior over the true token at every site.

        ``return_match`` additionally returns the (B, H, W) count of training
        patches that matched the window at each site. A site with no match is
        running on the smoothing prior alone, which
        :class:`AdaptiveWindowDiscreteScoreMachine` has to tell apart from a
        genuinely flat posterior.
        """
        device = self.patch_onehot_0.device
        k = self.kernel_size if k is None else k
        d = k // 2
        V = self.vocab_size

        xt = xt.to(device)
        b, h, w = xt.shape
        obs = (xt != self.mask_token).float()
        safe = torch.where(xt == self.mask_token, torch.zeros_like(xt), xt)
        xt_onehot = F.one_hot(safe, num_classes=V).permute(0, 3, 1, 2).float()

        # Asymmetric when k is even, so the output stays (h, w) and output site
        # (i, j) is always the patch element (d, d) of the window whose top-left
        # corner is (i - d, j - d). For odd k this is the usual symmetric padding.
        pad = (d, k - 1 - d, d, k - 1 - d)
        C = self.n_channels
        if self.pad_mode == 'outside':
            # Off-grid sites are *known*, not missing: give them their own channel
            # and count them as observed, so a window only matches training patches
            # that sit at the same distance from the edge.
            xt_pad = torch.zeros(b, C, h + k - 1, w + k - 1, device=device)
            xt_pad[:, V] = 1.0
            xt_pad[:, :V, d:d + h, d:d + w] = xt_onehot * obs[:, None]
            xt_pad[:, V, d:d + h, d:d + w] = 0.0
            obs_pad = torch.ones(b, 1, h + k - 1, w + k - 1, device=device)
            obs_pad[:, :, d:d + h, d:d + w] = obs[:, None]
        else:
            obs_pad = F.pad(obs[:, None], pad, value=0.0)
            xt_pad = F.pad(xt_onehot * obs[:, None], pad, value=0.0)
        ones_k = torch.ones(1, 1, k, k, device=device)
        need = F.conv2d(obs_pad, ones_k)[:, 0]  # (b, h, w)
        if self.pad_mode == 'outside':
            frame = torch.zeros(1, 1, h + k - 1, w + k - 1, device=device)
            frame[:, :, :d, :] = 1.0
            frame[:, :, d + h:, :] = 1.0
            frame[:, :, :, :d] = 1.0
            frame[:, :, :, d + w:] = 1.0
            need_out = F.conv2d(frame, ones_k)[0, 0]  # (h, w), purely geometric
        else:
            need_out = None

        counts = torch.zeros(b, V, h, w, device=device)
        n_match = torch.zeros(b, h, w, device=device)
        # sum of w log w over matching patches, for the entropy of the posterior
        # over *which* training patch matched (see AdaptiveWindow, h_mode='match')
        wlogw = torch.zeros(b, h, w, device=device)
        rows = torch.arange(h, device=device)
        cols = torch.arange(w, device=device)
        for idx, ph in enumerate(self.phases):
            onehot, weight, center, nout = self._table(idx)
            if onehot.shape[0] == 0:
                continue
            if self.use_gemm:
                c_phase, n_phase, wl_phase = self._match_gemm(
                    idx, xt_pad, need, need_out, k, b, h, w, sites=sites)
                if self.block_align <= 1:
                    counts, n_match = counts + c_phase, n_match + n_phase
                    wlogw = wlogw + wl_phase
                else:
                    sel = ((rows % self.block_align == ph[0])[:, None]
                           & (cols % self.block_align == ph[1])[None, :])
                    counts += c_phase * sel
                    n_match += n_phase * sel
                    wlogw += wl_phase * sel
                continue
            c_phase = torch.zeros(b, V, h, w, device=device)
            n_phase = torch.zeros(b, h, w, device=device)
            wl_phase = torch.zeros(b, h, w, device=device)
            for i in range(0, onehot.shape[0], self.chunk):
                pw = onehot[i:i + self.chunk]
                agree = F.conv2d(xt_pad, pw)                                  # (b, np, h, w)
                match = (torch.round(agree) == torch.round(need[:, None]))
                if need_out is not None:
                    match &= nout[i:i + self.chunk][None, :, None, None] == need_out[None, None]
                match = match.float()
                match = match * weight[i:i + self.chunk][None, :, None, None]
                c_phase += torch.einsum('bphw,pv->bvhw', match, center[i:i + self.chunk])
                n_phase += match.sum(dim=1)
                # match already carries the weight, so match * log(weight) is w log w
                wl_phase += (match * weight[i:i + self.chunk].clamp_min(1e-12)
                             .log()[None, :, None, None]).sum(dim=1)
            if self.block_align <= 1:
                counts, n_match = counts + c_phase, n_match + n_phase
                wlogw = wlogw + wl_phase
            else:
                # the table was keyed by the site each window predicts, so the
                # output site's own coordinates select it
                sel = ((rows % self.block_align == ph[0])[:, None]
                       & (cols % self.block_align == ph[1])[None, :])
                counts += c_phase * sel
                n_match += n_phase * sel
                wlogw += wl_phase * sel

        posterior = (counts + self.alpha_smooth) / (n_match[:, None] + self.alpha_smooth * V)
        out = torch.where(obs[:, None].bool(), xt_onehot, posterior)
        if not return_match:
            return out
        # H = log W - (sum_j w_j log w_j) / W, the entropy of the normalized
        # weights of the training patches this window matched
        safe_n = n_match.clamp_min(1e-12)
        h_match = torch.where(n_match > 0, safe_n.log() - wlogw / safe_n,
                              torch.zeros_like(n_match))
        return out, n_match, h_match


class AdaptiveWindowDiscreteScoreMachine(nn.Module):
    """LS machine whose window size is chosen per site, per step, by entropy.

    A fixed ``k`` is the wrong object. Along a reverse-erasure trajectory the
    number of observed sites inside a window only grows, so the posterior that
    window induces only sharpens: a window that is safely informative while most
    of the grid is masked ends up, near the end of the trajectory, matching a
    single training patch. Its posterior entropy is then zero and the machine is
    copying -- grammatical, because training grids are, and never novel.

    hshunt's criterion is to take the largest length scale whose posterior
    entropy is still above a floor. Because entropy falls with both ``k`` and
    time, that makes ``k`` *shrink* along the trajectory: coarse scales are
    resolved while there is still enough ambiguity to support them, fine scales
    last. The floor is what keeps the machine off the memorizing branch.

    At each site the chosen window is the largest ``k`` that both matched at
    least one training patch and left entropy at least ``h_min`` nats; if none
    qualifies the smallest ``k`` is used, which is the most conservative
    available posterior. ``no match`` has to be excluded explicitly: with
    smoothing, a window matching nothing returns the uniform posterior, whose
    entropy is *maximal*, so an entropy test alone would prefer exactly the
    windows that know nothing.

    Which entropy is the right one is not obvious, so ``h_mode`` selects it and
    the sweep decides:

    ``'token'``
        entropy of the predicted-token posterior. This is the literal reading,
        but it penalises a window that is confident *and right*: many training
        patches matching and all agreeing is generalization, not memorization,
        and a floor of 0.2 nats on this quantity drops grammaticality from 0.71
        to 0.03 at v=2, m=3, L=2 by forcing the machine down to k=1.
    ``'match'``
        entropy of the posterior over *which* training patch matched,
        ``log W - (sum_j w_j log w_j) / W``. This is the memorization signal
        proper: it goes to zero exactly when one training patch explains the
        window, whatever the token posterior looks like.

    ``n_min`` is the rival criterion, and it is not a reparametrization of
    ``h_min``. ``n_match`` is the total number of matching training *occurrences*
    and ``h_match`` their *diversity*, so a window matched by fifty copies of one
    patch has ``n_match = 50`` and ``h_match = 0``. Both entropy floors throw
    that window away, and the ``'token'`` note above says why that is the wrong
    instinct: many matches that all agree is generalization, not memorization.
    A floor on the count keeps it and instead rejects the window whose single
    accidental match pins the continuation to one training grid. The s=3 sweep
    found the best ``h_min`` to be 0 in 8 of 8 cells -- that is ``n_min = 1``,
    the weakest member of the family -- which is what makes ``n_min > 1`` the
    experiment that has not been run. Default 1 reproduces ``n_match > 0``
    exactly, so every earlier result is unchanged.

    ``trace`` records ``(masked fraction, mean chosen k)`` per call, which is
    the k(t) schedule the criterion induces rather than one imposed by hand.

    ``k_cap_schedule`` and ``require_support`` exist to ask what that induced
    schedule is worth on its own. The criterion does two separable things: it
    shrinks the window as the grid fills, and it picks a *different* window at
    different sites of the same grid. Give ``k_cap_schedule`` a list of
    ``(masked_fraction, k)`` breakpoints and no kernel above the cap is
    considered; set ``require_support = False`` on top of that and the support
    test never runs, so every masked site uses the capped ``k`` whether or not a
    training patch matches it. The three settings are the three arms: no cap is
    the criterion, cap alone is the criterion restricted to the schedule it
    induced, and cap without support is the bare schedule. Feeding back the
    measured ``trace`` as the cap makes the control exactly matched rather than
    a schedule chosen by hand to lose.
    """

    def __init__(
        self,
        grids,
        vocab_size: int,
        kernel_sizes,
        h_min: float,
        h_mode: str = 'match',
        n_min: float = 1.0,
        k_cap_schedule=None,
        require_support: bool = True,
        mask_token: Optional[int] = None,
        alpha_smooth: float = 1.0,
        device=None,
        chunk: int = 8192,
        block_align: int = 1,
        pad_mode: str = 'outside',
    ):
        super().__init__()
        self.kernel_sizes = sorted(int(k) for k in kernel_sizes)
        self.vocab_size = vocab_size
        self.h_min = float(h_min)
        if h_mode not in ('token', 'match'):
            raise ValueError(f"h_mode must be 'token' or 'match', got {h_mode!r}")
        self.h_mode = h_mode
        self.n_min = float(n_min)
        # sorted by threshold descending, so the cap is the k of the first
        # entry whose threshold the current masked fraction has fallen to
        self.k_cap_schedule = (None if k_cap_schedule is None else
                               sorted(((float(f), int(k)) for f, k in k_cap_schedule),
                                      key=lambda fk: -fk[0]))
        self.require_support = bool(require_support)
        self.mask_token = vocab_size if mask_token is None else mask_token
        self.machines = nn.ModuleList([
            PatchCountDiscreteScoreMachine(
                grids, vocab_size=vocab_size, kernel_size=k, mask_token=mask_token,
                alpha_smooth=alpha_smooth, device=device, chunk=chunk,
                block_align=block_align, pad_mode=pad_mode)
            for k in self.kernel_sizes
        ])
        self.n_distinct = sum(mm.n_distinct for mm in self.machines)
        self.n_patches_total = sum(mm.n_patches_total for mm in self.machines)
        self.trace = []

    def cap_at(self, masked_fraction: float) -> int:
        """Largest kernel the schedule allows at this masked fraction."""
        if self.k_cap_schedule is None:
            return self.kernel_sizes[-1]
        for f, k in self.k_cap_schedule:
            if masked_fraction >= f:
                return k
        return self.k_cap_schedule[-1][1]

    def forward(self, t, xt: torch.Tensor, label=None, device=None):
        masked = (xt.to(self.machines[0].device) == self.mask_token)
        cap = self.cap_at(float(masked.float().mean()))
        chosen = None
        kmap = None
        for k, mach in zip(self.kernel_sizes, self.machines):
            # kernel_sizes is ascending, so the first is always evaluated and
            # the cap only ever removes larger candidates
            if chosen is not None and k > cap:
                break
            post, n_match, h_match = mach(t, xt, return_match=True)
            if chosen is None:
                chosen = post
                kmap = torch.full_like(n_match, float(k))
                continue
            if self.h_mode == 'token':
                h = -(post.clamp_min(1e-12) * post.clamp_min(1e-12).log()).sum(1)
            else:
                h = h_match
            # n_min <= 1 goes through the strict-positive test rather than
            # >= 1.0, so the default is bit-for-bit the old behaviour whatever
            # the float weights do
            enough = (n_match > 0) if self.n_min <= 1 else (n_match >= self.n_min)
            # with the support test off the capped kernel is taken at every
            # masked site, matched or not: that is the bare schedule, and a
            # site it cannot match falls back on smoothing rather than on a
            # shorter window
            ok = (masked & enough & (h >= self.h_min) if self.require_support
                  else masked)
            chosen = torch.where(ok[:, None], post, chosen)
            kmap = torch.where(ok, torch.full_like(kmap, float(k)), kmap)
        # the per-site choice, kept so a caller can read it without repeating
        # the accept test and risking drift from it
        self._last_kmap = kmap
        if masked.any():
            self.trace.append((float(masked.float().mean()),
                               float(kmap[masked].mean())))
        return chosen


def locally_valid_fraction(machine, grids, k: int, vocab_size: int, pad_mode: str = 'outside'):
    """Fraction of ``grids`` all of whose ``k x k`` windows are in the patch table.

    Separates the two ways the machine can emit an ungrammatical grid. If a grid
    is *not* locally valid the sampler left the table: smoothing let it place a
    token no training window supports, so the failure is the sampler's. If it is
    locally valid but ungrammatical, the table itself admits it -- that is the
    ceiling, the quantity ``rhm_straddle.py`` enumerates exactly at ``L=2``.

    Only meaningful for the pooled (``block_align = 1``) machine, whose table is
    a single set of windows.
    """
    g = torch.as_tensor(np.asarray(grids), dtype=torch.long)
    d = k // 2
    if pad_mode == 'outside':
        g = F.pad(g[:, None], (d, k - 1 - d, d, k - 1 - d), value=vocab_size)[:, 0]
    npos = (g.shape[1] - k + 1) * (g.shape[2] - k + 1)
    w = g.unfold(1, k, 1).unfold(2, k, 1).reshape(-1, k * k)
    # the table back as token ids: the one-hot was built from them
    tab = machine.patch_onehot_0.argmax(dim=1).reshape(-1, k * k).cpu()
    if tab.shape[0] == 0:
        return 0.0

    # same packing as the table build, so the two agree on key order and the
    # word split adapts to the base -- a fixed two-way split overflows int64
    # once v+1 is large enough (9**25 at v=8, k=7)
    base = vocab_size + 1
    tk, _ = _pack_rows(tab, base)
    qk, _ = _pack_rows(w, base)

    # membership by sorted lookup rather than a Python set: at N_k ~ 2e7 a set
    # of key tuples costs several GB of objects, while the void view is one
    # int64 per word and searchsorted stays in C.
    dt = [('w%d' % c, '<i8') for c in range(tk.shape[1])]
    tv = np.sort(np.ascontiguousarray(tk.numpy()).view(dt).ravel())
    qv = np.ascontiguousarray(qk.numpy()).view(dt).ravel()
    idx = np.searchsorted(tv, qv)
    hit = tv[np.clip(idx, 0, tv.size - 1)] == qv
    return float(hit.reshape(-1, npos).all(axis=1).mean())
