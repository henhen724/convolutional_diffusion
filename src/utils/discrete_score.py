"""
Local Bayes-optimal denoising for discrete diffusion (e.g. the Random
Hierarchy Model, see ``random_hierarchy.py``).

This is the discrete analogue of the ``Local*ScoreModule`` family in
``idealscore.py``. Those modules estimate the score of a *continuous*
diffusion process by kernel-weighting nearby training patches with a
Gaussian likelihood. Here the corruption process is the standard
**absorbing-state ("erasure") discrete diffusion**: independently at every
site, the clean token is either left untouched or replaced by a special
``MASK`` sentinel, with probability ``p_mask(t)`` set by a masking schedule.

Because an unmasked site is *never* corrupted, the likelihood of a training
"word" (a local k x k window of tokens) given a noisy window is either 0 (it
disagrees with the noisy window somewhere it is observed) or a constant
(it agrees everywhere observed) -- there is no Gaussian kernel to evaluate.
The Bayes-optimal posterior over a masked site's true token is therefore
exactly the empirical distribution, over every k x k word seen anywhere in
the training set, of the center token among words that are consistent with
the observed (unmasked) tokens of the window. ``LocalDiscreteScoreMachine``
computes this posterior directly (with Dirichlet/Laplace smoothing to
handle windows with zero training matches), and ``ScheduledDiscreteScoreMachine``
uses it as the reverse-transition model of an ancestral sampler.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def linear_mask_schedule(t):
    """p_mask(t) = t: masking probability grows linearly with t in [0, 1]."""
    return t


def cosine_mask_schedule(t):
    """p_mask(t) = 1 - cos(pi t / 2): slow start, fast finish (MaskGIT-style)."""
    if torch.is_tensor(t):
        return 1 - torch.cos(t * math.pi / 2)
    return 1 - math.cos(t * math.pi / 2)


def mask_tokens(x0: torch.Tensor, p_mask, mask_token: int, generator: Optional[torch.Generator] = None):
    """Apply absorbing-state ("erasure") forward corruption.

    Parameters
    ----------
    x0 : LongTensor (B, H, W) clean token grids, values in [0, vocab_size).
    p_mask : float or Tensor broadcastable to (B,), independent per-site masking
        probability at the current noise level.
    mask_token : sentinel id (e.g. vocab_size) marking a masked/absorbed site.
    generator : optional torch.Generator for reproducible masking.

    Returns
    -------
    xt : LongTensor (B, H, W), equal to x0 at unmasked sites and mask_token elsewhere.
    mask : BoolTensor (B, H, W), True where a site was masked.
    """
    b = x0.shape[0]
    if not torch.is_tensor(p_mask):
        p_mask = torch.full((b,), float(p_mask), device=x0.device)
    p_mask = p_mask.to(x0.device).reshape(b, 1, 1)
    u = torch.rand(x0.shape, device=x0.device, generator=generator)
    mask = u < p_mask
    xt = torch.where(mask, torch.full_like(x0, mask_token), x0)
    return xt, mask


class LocalDiscreteScoreMachine(nn.Module):
    """Exact local Bayes-optimal denoiser for absorbing-state discrete diffusion.

    Parameters
    ----------
    dataset : yields (LongTensor (H, W) token grid, LongTensor scalar label) pairs,
        e.g. :class:`~src.utils.random_hierarchy.RandomHierarchy2DDataset` with
        ``onehot=False``.
    vocab_size : number of real token classes (mask_token is *not* one of these).
    kernel_size : side length k of the local "word" window used for matching.
    batch_size, max_samples, shuffle : control the training-set DataLoader,
        as in the continuous ``Local*ScoreModule`` classes.
    mask_schedule : only used by ``ScheduledDiscreteScoreMachine``; kept here
        for interface parity (matching itself needs no explicit noise level,
        it is read off directly from which sites in ``xt`` are masked).
    mask_token : sentinel id for a masked site; defaults to ``vocab_size``.
    alpha_smooth : symmetric Dirichlet/Laplace smoothing count added per class,
        so windows with zero training matches fall back to a uniform posterior
        instead of being undefined.
    """

    def __init__(
        self,
        dataset,
        vocab_size: int,
        kernel_size: int = 3,
        batch_size: int = 64,
        mask_schedule=linear_mask_schedule,
        max_samples: Optional[int] = None,
        mask_token: Optional[int] = None,
        alpha_smooth: float = 1.0,
        shuffle: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dataset = dataset
        self.trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.vocab_size = vocab_size
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.mask_schedule = mask_schedule
        self.max_samples = max_samples
        self.mask_token = vocab_size if mask_token is None else mask_token
        self.alpha_smooth = alpha_smooth

    def forward(self, t, xt: torch.Tensor, label=None, device=None, k: Optional[int] = None):
        """Compute the exact Bayes posterior over the true token at every site.

        ``t`` is accepted for interface parity with the continuous score
        machines but is not used: the posterior depends only on which sites
        of ``xt`` are currently masked, not on the numeric noise level.

        Parameters
        ----------
        xt : LongTensor (B, H, W), values in [0, vocab_size) at observed sites
            and self.mask_token at masked sites.

        Returns
        -------
        posterior : FloatTensor (B, vocab_size, H, W), a valid probability
            distribution over the vocabulary at every site (a one-hot delta
            at already-observed sites, the Bayes posterior at masked sites).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if k is None:
            k = self.kernel_size

        V = self.vocab_size
        xt = xt.to(device)
        b, h, w = xt.shape
        d = k // 2

        obs = (xt != self.mask_token).float()  # (b, h, w)
        safe_xt = torch.where(xt == self.mask_token, torch.zeros_like(xt), xt)
        xt_onehot = F.one_hot(safe_xt, num_classes=V).permute(0, 3, 1, 2).float()  # (b, V, h, w)
        xt_onehot_masked = xt_onehot * obs[:, None, :, :]

        obs_pad = F.pad(obs[:, None, :, :], (d, d, d, d), value=0.0)  # (b, 1, h+2d, w+2d)
        xt_pad = F.pad(xt_onehot_masked, (d, d, d, d), value=0.0)  # (b, V, h+2d, w+2d)

        ones_kernel = torch.ones(1, 1, k, k, device=device)
        need = F.conv2d(obs_pad, ones_kernel)[:, 0, :, :]  # (b, h, w): # of observed sites in each window

        counts = torch.zeros(b, V, h, w, device=device)
        n_matches = torch.zeros(b, h, w, device=device)

        i = 0
        for images, labels in self.trainloader:
            if label is not None:
                images = images[labels == label]
            if images.shape[0] == 0:
                continue

            images = images.to(device)
            bsize = images.shape[0]
            i += bsize
            if self.max_samples is not None and i > self.max_samples:
                break

            image_onehot = F.one_hot(images, num_classes=V).permute(0, 3, 1, 2).float()  # (bsize, V, H, W)
            patches = F.unfold(image_onehot, k, stride=1, padding=0)  # (bsize, V*k*k, Lpos)
            lpos = patches.shape[-1]
            patches = patches.permute(2, 0, 1).reshape(lpos * bsize, V, k, k)  # (NP, V, k, k)
            pcenters = patches[:, :, d, d]  # (NP, V), one-hot center token of each training word

            pdotx = F.conv2d(xt_pad, patches)  # (b, NP, h, w): agreement count on observed sites
            match = torch.round(pdotx) == torch.round(need[:, None, :, :])  # exact match everywhere observed

            match_f = match.float()
            counts += torch.einsum("bphw,pv->bvhw", match_f, pcenters)
            n_matches += match_f.sum(dim=1)

        posterior = (counts + self.alpha_smooth) / (n_matches[:, None, :, :] + self.alpha_smooth * V)
        # Already-observed sites are known exactly; collapse their posterior to a delta.
        posterior = torch.where(obs[:, None, :, :].bool(), xt_onehot, posterior)
        return posterior


class ScheduledDiscreteScoreMachine(nn.Module):
    """Ancestral sampler for absorbing-state discrete diffusion.

    Mirrors ``ScheduledScoreMachine`` in idealscore.py, but for the
    erasure/masking corruption process: at each step, currently-masked sites
    are revealed (sampled from the backbone's posterior) with exactly the
    probability needed to match the target masking schedule, using the
    standard common-random-number coupling for absorbing diffusion:
    ``P(masked at t_prev | masked at t) = p_mask(t_prev) / p_mask(t)``.
    """

    def __init__(
        self,
        backbone,
        vocab_size: int,
        grid_size: int,
        default_time_steps: int = 50,
        mask_schedule=linear_mask_schedule,
        mask_token: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.vocab_size = vocab_size
        self.grid_size = grid_size
        self.default_time_steps = default_time_steps
        self.mask_schedule = mask_schedule
        self.mask_token = vocab_size if mask_token is None else mask_token

    def forward(
        self,
        xt: torch.Tensor,
        nsteps: Optional[int] = None,
        label=None,
        device=None,
        generator=None,
        t_start: float = 1.0,
    ):
        """Run the reverse (denoising) process starting from noise level ``t_start``.

        ``xt`` need not be fully masked: passing an already-corrupted-to-``t_start``
        grid (e.g. via ``mask_tokens(x0, mask_schedule(t_start), ...)``) implements
        the "corrupt to t_start, then denoise back to 0" experiment used to probe
        which scales of structure survive forward-then-backward diffusion.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xt = xt.clone().to(device)
        b = xt.shape[0]
        if nsteps is None:
            nsteps = self.default_time_steps

        i_start = max(1, round(t_start * nsteps))
        for i in range(i_start, 0, -1):
            still_masked = xt == self.mask_token
            if not bool(still_masked.any()):
                continue

            t = i / nsteps
            t_prev = (i - 1) / nsteps
            p_t = float(self.mask_schedule(t))
            p_prev = float(self.mask_schedule(t_prev))

            reveal_prob = 1.0 if p_t <= 0 else min(1.0, max(0.0, 1.0 - p_prev / p_t))
            if i == 1:
                reveal_prob = 1.0  # force any remaining masked sites to reveal on the last step

            posterior = self.backbone(
                torch.full((b,), t, device=device), xt, label=label, device=device
            )  # (b, V, H, W)

            reveal = still_masked & (torch.rand(xt.shape, device=device, generator=generator) < reveal_prob)
            if bool(reveal.any()):
                probs = posterior.permute(0, 2, 3, 1).reshape(-1, self.vocab_size).clamp_min(1e-12)
                sampled = torch.multinomial(probs, 1, generator=generator).reshape(b, xt.shape[1], xt.shape[2])
                xt = torch.where(reveal, sampled, xt)

        return xt

    def sample(self, n_samples: int = 1, nsteps: Optional[int] = None, label=None, device=None, generator=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xt = torch.full(
            (n_samples, self.grid_size, self.grid_size), self.mask_token, dtype=torch.long, device=device
        )
        return self(xt, nsteps=nsteps, label=label, device=device, generator=generator)
