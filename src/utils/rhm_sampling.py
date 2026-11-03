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
    # (P, nr, nc, k, k) -> one entry per sliding-window position.
    windows = g.unfold(1, k, 1).unfold(2, k, 1)
    if phase is not None and block_align > 1:
        rows = (torch.arange(windows.shape[1]) + site_offset) % block_align == phase[0]
        cols = (torch.arange(windows.shape[2]) + site_offset) % block_align == phase[1]
        windows = windows[:, rows][:, :, cols]
    windows = windows.reshape(-1, k * k)
    if windows.shape[0] == 0:
        return torch.zeros(0, k, k, dtype=torch.long), torch.zeros(0, dtype=torch.float64)
    uniq, inverse = torch.unique(windows, dim=0, return_inverse=True)
    counts = torch.zeros(uniq.shape[0], dtype=torch.float64)
    counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float64))
    return uniq.reshape(-1, k, k), counts


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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.kernel_size = kernel_size
        self.mask_token = vocab_size if mask_token is None else mask_token
        self.alpha_smooth = alpha_smooth
        self.chunk = chunk
        self.block_align = block_align
        self.pad_mode = pad_mode
        # 'outside' adds one symbol, the off-grid marker, to the matching alphabet
        self.n_channels = vocab_size + 1 if pad_mode == 'outside' else vocab_size
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

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

    def forward(self, t, xt: torch.Tensor, label=None, device=None, k: Optional[int] = None):
        """(B, V, H, W) posterior over the true token at every site."""
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
        rows = torch.arange(h, device=device)
        cols = torch.arange(w, device=device)
        for idx, ph in enumerate(self.phases):
            onehot, weight, center, nout = self._table(idx)
            if onehot.shape[0] == 0:
                continue
            c_phase = torch.zeros(b, V, h, w, device=device)
            n_phase = torch.zeros(b, h, w, device=device)
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
            if self.block_align <= 1:
                counts, n_match = counts + c_phase, n_match + n_phase
            else:
                # the table was keyed by the site each window predicts, so the
                # output site's own coordinates select it
                sel = ((rows % self.block_align == ph[0])[:, None]
                       & (cols % self.block_align == ph[1])[None, :])
                counts += c_phase * sel
                n_match += n_phase * sel

        posterior = (counts + self.alpha_smooth) / (n_match[:, None] + self.alpha_smooth * V)
        return torch.where(obs[:, None].bool(), xt_onehot, posterior)
