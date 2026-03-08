"""
Channel theory of scale calibration: pixel-space ELS and LS.

Utilities for computing posterior stats (entropy, variance, Binder cumulant) vs kernel size k,
loading scale calibration files, and inferring the rate-distortion weight β from the
cost minimizer. Cost to minimize: β⟨d(x,x')⟩ − S (higher entropy S is better).
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def t_to_snr(t: Union[float, np.ndarray, torch.Tensor], schedule: Callable) -> np.ndarray:
    """
    Signal-to-noise ratio at time t: α_t / β_t = (1 - β_t) / β_t.
    schedule(t) returns β_t; α_t = 1 - β_t.
    """
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    tt = torch.tensor(t)
    bt = schedule(tt)
    if isinstance(bt, torch.Tensor):
        bt = bt.numpy()
    bt = np.atleast_1d(bt)
    at = 1.0 - bt
    snr = at / (bt + 1e-12)
    return snr[0] if snr.size == 1 else snr


def compute_posterior_stats_at_t(
    module: Any,
    t: torch.Tensor,
    x_batch: torch.Tensor,
    device: Optional[torch.device] = None,
    label: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, float]]:
    """
    Run forward_with_posterior_stats on the module and return scalar stats.

    Args:
        module: Score module with forward_with_posterior_stats (LS or ELS).
        t: Timestep tensor, shape (1,) or (b,).
        x_batch: Noisy batch (b, c, h, w).
        device: Device for computation; defaults to x_batch.device.
        label: Optional class label for conditional modules.

    Returns:
        Dict with keys: avg_entropy, center_variance, center_binder, total_variance, total_binder.
        avg_entropy is in nats (from forward_with_posterior_stats entropy_map).
        Returns None if the module does not support forward_with_posterior_stats.
    """
    if not hasattr(module, "forward_with_posterior_stats"):
        return None
    if device is None:
        device = x_batch.device
    with torch.no_grad():
        out = module.forward_with_posterior_stats(t, x_batch, label=label, device=device)
    # Backward-compatible parsing:
    # old: (score, E_x0, entropy_map, center_variance_map, center_binder_map)
    # new: (score, E_x0, entropy_map, center_variance_map, center_binder_map,
    #       patch_variance_map, patch_binder_map)
    if len(out) == 5:
        score, E_x0, entropy_map, center_variance_map, center_binder_map = out
        patch_variance_map = center_variance_map.mean(dim=1)
        patch_binder_map = center_binder_map.mean(dim=1)
    else:
        (
            score,
            E_x0,
            entropy_map,
            center_variance_map,
            center_binder_map,
            patch_variance_map,
            patch_binder_map,
        ) = out
    avg_entropy = entropy_map.mean().item()
    # "Center pixel" means the center of each local patch, at every (h, w),
    # not only the global image-center coordinate.
    center_var = center_variance_map.mean().item()
    center_binder = center_binder_map.mean().item()
    total_var = patch_variance_map.mean().item()
    total_binder = patch_binder_map.mean().item()
    return {
        "avg_entropy": avg_entropy,
        "center_variance": center_var,
        "center_binder": center_binder,
        "total_variance": total_var,
        "total_binder": total_binder,
    }


def infer_beta_at_k_star(
    entropy_by_k: np.ndarray,
    reg_by_k: np.ndarray,
    k_vals: List[int],
    k_star: int,
) -> float:
    """
    Infer β such that k_star is a minimizer of cost = β×⟨d⟩ − S (finite-diff at k_star).
    Higher entropy S is better; we minimize cost so minimize β×reg − entropy.

    At the minimizer, d(cost)/dk = 0 ⇒ β×d(reg)/dk − d(S)/dk = 0 ⇒ β = d(S)/dk / d(reg)/dk.

    Args:
        entropy_by_k: Entropy values for each k (same order as k_vals).
        reg_by_k: ⟨d⟩ (e.g. variance or Binder) for each k.
        k_vals: List of kernel sizes.
        k_star: Calibrated kernel size (assumed minimizer).

    Returns:
        Inferred β, or np.nan if k_star not in k_vals or d_reg ≈ 0.
        Uses one-sided finite difference when k_star is at boundary.
    """
    if k_star not in k_vals:
        return np.nan
    idx = list(k_vals).index(k_star)
    n = len(k_vals)
    if n < 2:
        return np.nan
    # Central difference when possible, else one-sided
    if idx > 0 and idx < n - 1:
        d_entropy = (entropy_by_k[idx + 1] - entropy_by_k[idx - 1]) / (
            k_vals[idx + 1] - k_vals[idx - 1]
        )
        d_reg = (reg_by_k[idx + 1] - reg_by_k[idx - 1]) / (
            k_vals[idx + 1] - k_vals[idx - 1]
        )
    elif idx < n - 1:
        d_entropy = (entropy_by_k[idx + 1] - entropy_by_k[idx]) / (
            k_vals[idx + 1] - k_vals[idx]
        )
        d_reg = (reg_by_k[idx + 1] - reg_by_k[idx]) / (
            k_vals[idx + 1] - k_vals[idx]
        )
    else:
        d_entropy = (entropy_by_k[idx] - entropy_by_k[idx - 1]) / (
            k_vals[idx] - k_vals[idx - 1]
        )
        d_reg = (reg_by_k[idx] - reg_by_k[idx - 1]) / (
            k_vals[idx] - k_vals[idx - 1]
        )
    if abs(d_reg) < 1e-12:
        return np.nan
    return float(d_entropy / d_reg)


def infer_beta_range_at_k_star(
    entropy_by_k: np.ndarray,
    reg_by_k: np.ndarray,
    k_vals: List[int],
    k_star: int,
) -> Tuple[float, float]:
    """
    Range of β for which k_star remains the minimizer of cost = β×reg − S.
    So cost(k) >= cost(k_star) for all k: β*(reg(k)-reg*) >= S(k)-S*.

    Returns:
        (beta_lo, beta_hi): for beta in [beta_lo, beta_hi], k_star is a minimizer.
        Uses np.nan for unbounded side (no competing k on that side).
    """
    if k_star not in k_vals:
        return np.nan, np.nan
    idx = list(k_vals).index(k_star)
    S_star = float(entropy_by_k[idx])
    R_star = float(reg_by_k[idx])
    n = len(k_vals)
    beta_lo, beta_hi = -np.inf, np.inf
    for i in range(n):
        if i == idx:
            continue
        S_k = entropy_by_k[i]
        R_k = reg_by_k[i]
        dR = R_k - R_star
        dS = S_k - S_star
        if abs(dR) < 1e-12:
            continue
        b = dS / dR
        if dR > 0:
            # β*(R_k - R*) >= S_k - S*  =>  β >= b
            if b > beta_lo:
                beta_lo = b
        else:
            # β <= b
            if b < beta_hi:
                beta_hi = b
    return (
        float(beta_lo) if np.isfinite(beta_lo) else np.nan,
        float(beta_hi) if np.isfinite(beta_hi) else np.nan,
    )


def load_scale_calibration(
    scale_file: Optional[str],
    tld: str = "./checkpoints/",
) -> Optional[torch.Tensor]:
    """
    Load a scale calibration median tensor (one k per timestep).

    File is expected to be from scales_calibration.py: *_median.pt with shape (nsteps,).

    Args:
        scale_file: Basename of the file (e.g. 'scales_MNIST_ResNet_LS_median.pt').
        tld: Top-level directory for checkpoints.

    Returns:
        1D tensor of median kernel size per step, or None if file missing/invalid.
    """
    if not scale_file:
        print("WARNING: No scale file provided")
        return None
    path = os.path.join(tld, scale_file)
    if not os.path.isfile(path) and not scale_file.endswith("_median.pt"):
        alt = os.path.join(tld, scale_file.replace(".pt", "_median.pt"))
        if os.path.isfile(alt):
            path = alt
    if not os.path.isfile(path):
        print("WARNING: Scale file path is invalid: ", str(path))
        return None
    median = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(median, torch.Tensor) and median.dim() == 1:
        return median
    elif type(median) == list and torch.Tensor(median).dim() == 1:
        return torch.Tensor(median)
    else:
        print("Scale array had an unexpceted type: ", type(median), "\n\n",median)
    return None


def collect_stats_vs_k(
    modules_by_k: List[Tuple[int, Any]],
    t: torch.Tensor,
    x_batch: torch.Tensor,
    device: torch.device,
    label: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute posterior stats for each (k, module); return arrays indexed by k.

    Args:
        modules_by_k: List of (kernel_size, module) for LS or ELS.
        t: Timestep tensor.
        x_batch: Noisy batch.
        device: Device.
        label: Optional label.

    Returns:
        Dict with keys k_vals, avg_entropy, center_variance, center_binder,
        total_variance, total_binder (each a 1D array over k).
    """
    k_vals = [k for k, _ in modules_by_k]
    entropy = []
    cvar = []
    cbinder = []
    tvar = []
    tbinder = []
    for _, mod in modules_by_k:
        s = compute_posterior_stats_at_t(mod, t, x_batch, device, label)
        if s is None:
            break
        entropy.append(s["avg_entropy"])
        cvar.append(s["center_variance"])
        cbinder.append(s["center_binder"])
        tvar.append(s["total_variance"])
        tbinder.append(s["total_binder"])
    return {
        "k_vals": k_vals,
        "avg_entropy": np.array(entropy),
        "center_variance": np.array(cvar),
        "center_binder": np.array(cbinder),
        "total_variance": np.array(tvar),
        "total_binder": np.array(tbinder),
    }


def infer_betas_for_rate_distortion(
    stats_ls: Dict[str, np.ndarray],
    stats_els: Dict[str, np.ndarray],
    median_ls: Optional[torch.Tensor],
    median_els: Optional[torch.Tensor],
    timestep_idx: int,
) -> Dict[Tuple[str, str], float]:
    """
    Infer β for each (score_type, reg_type) at the given timestep using calibrated k.

    Args:
        stats_ls: From collect_stats_vs_k for LS.
        stats_els: From collect_stats_vs_k for ELS.
        median_ls: Median k per step for LS (or None).
        median_els: Median k per step for ELS (or None).
        timestep_idx: Step index (0-based).

    Returns:
        Dict mapping (score_type, reg_type) to inferred β (e.g. ("LS", "total_variance")).
    """
    k_vals = stats_ls["k_vals"]
    betas: Dict[Tuple[str, str], float] = {}
    for score_name, stats, median in [
        ("LS", stats_ls, median_ls),
        ("ELS", stats_els, median_els),
    ]:
        if median is None or timestep_idx >= median.numel():
            continue
        k_star = int(median[timestep_idx].item())
        for reg_name in ("center_variance", "center_binder", "total_variance", "total_binder"):
            beta = infer_beta_at_k_star(
                stats["avg_entropy"],
                stats[reg_name],
                k_vals,
                k_star,
            )
            betas[(score_name, reg_name)] = beta
    return betas


def cost_at_beta(
    entropy: np.ndarray,
    reg: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Cost = β×⟨d⟩ − S (element-wise along k). Minimize this; higher entropy is better."""
    return beta * reg - entropy
