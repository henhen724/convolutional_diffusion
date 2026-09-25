"""
Dynamical correlation function and susceptibility for forward-backward
diffusion experiments, following Sclocchi, Favero, Levi & Wyart,
"Probing the Latent Hierarchical Structure of Data via Diffusion Models"
(ICLR 2025, arXiv:2410.13770), which generalizes the class phase transition
of Sclocchi, Favero & Wyart (PNAS 2025, arXiv:2402.16991) to real images
and text.

Recipe (identical for discrete and continuous data): corrupt a starting
datum x0 to inversion time t, denoise it back to t=0 many times (many
independent stochastic "trajectories"), and look at whether/how much each
site i changed relative to x0:

    sigma_i(t) = 1[x_hat_0,i(t) != x0,i]          (discrete: masking)
    sigma_i(t) = || x0,i - x_hat_0,i(t) ||         (continuous: images)

The dynamical correlation function is the covariance of sigma across sites,
averaged over trajectories and then over starting data:

    C_ij(t) = <sigma_i(t) sigma_j(t)> - <sigma_i(t)><sigma_j(t)>

and the dynamical susceptibility is the normalized total correlation

    chi(t) = sum_ij C_ij(t) / sum_i C_ii(t)

which is exactly the statistical-physics susceptibility associated with the
correlation function: chi = 1 when every site changes independently, and
chi = N (all sites) when the whole system changes as one correlated block.
It peaks at the phase transition, where a system-spanning cluster of sites
(e.g. every leaf under the root symbol) flips together.

This module intentionally does not care whether sigma came from a discrete
absorbing-diffusion trajectory (ScheduledDiscreteScoreMachine) or a
continuous Gaussian-diffusion trajectory (ScheduledScoreMachine) -- only
`dynamical_susceptibility(sigma_by_x0)` is needed, fed with whichever sigma
definition is appropriate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def dynamical_susceptibility(sigma_by_x0: np.ndarray) -> Tuple[float, np.ndarray]:
    """Dynamical susceptibility chi(t) from per-site "did it change" values.

    Parameters
    ----------
    sigma_by_x0 : array (n_x0, R, N) -- for each of n_x0 starting data
        points, R independent stochastic reverse-diffusion trajectories,
        each giving a length-N per-site sigma_i value (0/1 mismatch
        indicator for discrete data, or a nonnegative L2 distance for
        continuous data).

    Returns
    -------
    chi : float, sum_ij C_ij / sum_i C_ii, with C averaged (post-covariance)
        over the n_x0 starting data points as in the paper ("<.> over
        trajectories, overline over starting data").
    C_mean : (N, N) array, the starting-data-averaged covariance matrix.
    """
    sigma_by_x0 = np.asarray(sigma_by_x0, dtype=np.float64)
    if sigma_by_x0.ndim != 3:
        raise ValueError("sigma_by_x0 must have shape (n_x0, R, N)")
    n_x0, r, n = sigma_by_x0.shape
    if r < 2:
        raise ValueError("need at least 2 trajectories (R >= 2) to estimate a covariance")

    c_sum = np.zeros((n, n), dtype=np.float64)
    for i in range(n_x0):
        c_sum += np.cov(sigma_by_x0[i], rowvar=False)
    c_mean = c_sum / n_x0

    trace = np.trace(c_mean)
    if trace <= 1e-12:
        return 0.0, c_mean
    chi = float(c_mean.sum() / trace)
    return chi, c_mean


def pool_sigma_field(sigma_field: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Average-pool a per-pixel sigma field down to a coarser site grid.

    Real images have far more raw pixels than independent stochastic
    trajectories one can afford to sample, which makes a pixel-resolution
    covariance matrix badly under-determined. Pooling into a coarser grid
    of "sites" (still fine enough to resolve spatial structure) keeps the
    number of sites well below the trajectory count, following the spirit
    of the paper's own patch/token-level (not raw-pixel-level) analysis.

    Parameters
    ----------
    sigma_field : (..., H, W) tensor.
    pool_size : side length of the (non-overlapping) pooling window.

    Returns
    -------
    (..., H // pool_size, W // pool_size) tensor.
    """
    shape = sigma_field.shape
    x = sigma_field.reshape(-1, 1, shape[-2], shape[-1])
    pooled = F.avg_pool2d(x, pool_size)
    return pooled.reshape(*shape[:-2], pooled.shape[-2], pooled.shape[-1])


def find_peak(t_values, chi_values) -> Tuple[float, float, int]:
    """Return (t at peak, peak chi value, index of peak) for a chi(t) curve."""
    idx = int(np.argmax(chi_values))
    return float(t_values[idx]), float(chi_values[idx]), idx
