"""
Pixel-pixel correlation regions: how much of a noisy image is statistically
informative about a single clean pixel, as a function of training-set size
and corruption level.

Definition: the "informative region" around clean pixel x is

    R(D, t) = { pixel a : |Corr(x0[x], xt[a])| > z / sqrt(|D|) }

where |D| is the number of samples used to *estimate* the correlation, and
z/sqrt(|D|) is a Bonferroni-corrected significance threshold for a Pearson
correlation estimated from |D| i.i.d. draws (the standard error of a sample
correlation under the null of zero true correlation is ~1/sqrt(|D|); z is
raised above 1 by the correction below). The measurement intentionally uses
exactly |D| samples for *both* the correlation estimate and the threshold
(rather than a large noise-free reference map with only the threshold
varying), so the region's size and shape include realistic finite-sample
fluctuation, not just the smooth underlying image statistics.

A bare z=1 threshold, applied independently to every pixel in the image
simultaneously, gives a per-pixel false-positive rate of ~32% under the
null of zero true correlation -- a multiple-comparisons/"look-elsewhere"
effect that, combined with real (if weak) long-range image correlations,
pushes local false/true-positive density above the ~0.59 site-percolation
threshold for 4-connectivity on a finite grid. Past that point, positives
spuriously percolate into one cluster spanning most of the image, which no
amount of connected-component post-processing can undo. `region_mask`
therefore raises z via a Bonferroni correction on the number of pixels
tested, and `connected_region_containing_target` additionally isolates the
single spatially contiguous component touching the target -- together
keeping both the false-positive rate and the percolation risk low at every
|D| and image size.

Motivating scaling hypothesis (deviation from scale invariance): natural
images have an approximately power-law spatial power spectrum, giving a
real-space two-point correlation function

    <phi(x) phi(0)> ~ |x|^(-epsilon)

with epsilon small (close to the marginal/critical case). Setting this
signal equal to the |D|-dependent detectability threshold z/sqrt(|D|) and
solving for the boundary radius gives

    R(|D|) ~ |D|^(1/(2*epsilon))          =>          area ~ |D|^(1/epsilon)

i.e. a *plain power law* in |D| (not a logarithm) -- so epsilon can be
extracted two independent ways: (1) directly, by radially averaging the
CLEAN-image correlation map and fitting its power-law tail
(`radial_correlation_profile` + `fit_radial_power_law`), and (2) indirectly,
from the power-law exponent of how the informative region's area grows
with |D| (epsilon = 1/area_exponent). The two should agree if this picture
is right; `nested_correlation_curve`'s area-vs-|D| fit and the direct radial
fit are meant to be compared against each other, not treated as
independent hypotheses.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def sample_x0_xt(
    dataset: Dataset,
    n_samples: int,
    t: float,
    schedule,
    channel_mode: str = "mean",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw n_samples images and one independent forward-diffusion corruption each.

    Parameters
    ----------
    dataset : yields (image_tensor (C,H,W), label) pairs, e.g. from get_dataset.
    n_samples : number of images to draw (without replacement).
    t : diffusion inversion time in [0, 1]; beta_t = schedule(t).
    schedule : noise schedule, e.g. cosine_noise_schedule (returns beta_t).
    channel_mode : "mean" averages channels to grayscale; "first" takes channel 0.
    seed : RNG seed for sample selection and noise draw.

    Returns
    -------
    (x0, xt) : each (n_samples, H, W) float64 arrays.
    """
    if n_samples > len(dataset):
        raise ValueError(f"n_samples={n_samples} exceeds dataset size {len(dataset)}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=n_samples, replace=False)
    imgs = torch.stack([dataset[int(i)][0] for i in idx])  # (n, C, H, W)
    if channel_mode == "mean":
        imgs = imgs.mean(dim=1)
    elif channel_mode == "first":
        imgs = imgs[:, 0]
    else:
        raise ValueError(f"unknown channel_mode: {channel_mode}")

    beta_t = float(schedule(torch.tensor(float(t))))
    alpha_t = 1.0 - beta_t
    gen = torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1)))
    noise = torch.randn(imgs.shape, generator=gen)
    xt = (alpha_t**0.5) * imgs + (beta_t**0.5) * noise
    return imgs.numpy().astype(np.float64), xt.numpy().astype(np.float64)


def pixel_correlation_map(x0: np.ndarray, xt: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
    """Pearson correlation between the clean value at `target` and the noisy
    value at every pixel, estimated from the samples in x0/xt.

    Parameters
    ----------
    x0, xt : (N, H, W) arrays, paired samples (same N images, same order).
    target : (row, col) of the clean reference pixel x.

    Returns
    -------
    (H, W) array of Corr(x0[target], xt[i, j]) for every (i, j).
    """
    if x0.shape != xt.shape:
        raise ValueError("x0 and xt must have the same shape")
    n = x0.shape[0]
    x0_target = x0[:, target[0], target[1]]
    x0_target = x0_target - x0_target.mean()
    xt_centered = xt - xt.mean(axis=0, keepdims=True)

    cov = np.einsum("n,nij->ij", x0_target, xt_centered) / n
    std_target = x0_target.std()
    std_map = xt_centered.std(axis=0)
    denom = std_target * std_map
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-12)
    return np.clip(corr, -1.0, 1.0)


def radial_correlation_profile(
    corr_map: np.ndarray, target: Tuple[int, int], r_max: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Radially average |corr_map| by (rounded) integer distance from `target`.

    Used to directly measure the real-space decay of the natural-image
    correlation function <phi(x)phi(0)> ~ |x|^(-epsilon), independent of any
    diffusion corruption or dataset-size detectability threshold.

    `r_max` defaults to the largest radius for which a *complete* circle
    around `target` still fits inside the image -- i.e. the distance from
    `target` to the nearest edge, minus 1 -- so every averaged radius uses
    a full, unbiased ring of pixels rather than a partial one clipped by
    the image boundary.

    Returns
    -------
    (radii, mean_abs_corr) : both 1D arrays, radii = 1..r_max.
    """
    h, w = corr_map.shape
    if r_max is None:
        r_max = int(min(target[0], h - 1 - target[0], target[1], w - 1 - target[1]))
    yy, xx = np.mgrid[0:h, 0:w]
    r_int = np.round(np.sqrt((yy - target[0]) ** 2 + (xx - target[1]) ** 2)).astype(int)
    radii = np.arange(1, r_max + 1)
    values = np.array([np.abs(corr_map[r_int == rr]).mean() for rr in radii])
    return radii, values


def fit_radial_power_law(
    radii: np.ndarray, values: np.ndarray, r_min: int = 2, r_max: Optional[int] = None
) -> Tuple[float, float]:
    """Fit |C(r)| ~ r^(-epsilon) over [r_min, r_max] via log-log linear
    regression. Returns (epsilon, r_squared); epsilon is the decay
    exponent (positive for decaying correlations).
    """
    if r_max is None:
        r_max = int(radii.max())
    mask = (radii >= r_min) & (radii <= r_max) & (values > 0)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    log_r, log_v = np.log(radii[mask]), np.log(values[mask])
    slope, intercept = np.polyfit(log_r, log_v, 1)
    pred = slope * log_r + intercept
    ss_res = np.sum((log_v - pred) ** 2)
    ss_tot = np.sum((log_v - log_v.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return -float(slope), float(r2)


def region_mask(corr_map: np.ndarray, n_samples: int, fwer_alpha: float = 0.05) -> np.ndarray:
    """Boolean mask of sites whose |correlation| is significant at a
    Bonferroni-corrected threshold, controlling the family-wise false
    positive rate at `fwer_alpha` across all `corr_map.size` simultaneous
    per-pixel tests.

    A bare single-comparison ~1-sigma threshold (1/sqrt(n_samples)) gives
    ~32% of *genuinely uncorrelated* pixels crossing it by chance alone
    (P(|Z|>1) for a standard normal) -- and with ~10^3 pixels tested at
    once, that false-positive density (or the real density once combined
    with real, weak, long-range image correlations) can exceed the
    4-connectivity site-percolation threshold (~0.59) on a finite image, at
    which point false positives spuriously percolate into (and merge with)
    the true region around the target, contaminating any area/shape
    measurement no matter how the resulting mask is post-processed. The
    Bonferroni correction below raises the z-score with the number of
    pixels tested so the false-positive density stays low (~fwer_alpha)
    regardless of image size or n_samples, keeping it safely under the
    percolation threshold. `connected_region_containing_target` still
    isolates genuine spatial contiguity on top of this.
    """
    from scipy.stats import norm

    n_pixels = corr_map.size
    z = norm.ppf(1.0 - fwer_alpha / (2.0 * n_pixels))
    threshold = z / np.sqrt(n_samples)
    return np.abs(corr_map) > threshold


def connected_region_containing_target(mask: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
    """The 4-connected component of `mask` containing `target`.

    This is what "the region around pixel x" means geometrically: the
    contiguous blob of correlated sites touching x, as distinct from
    isolated, scattered pixels elsewhere that cross the threshold purely
    from multiple-comparisons sampling noise (see `region_mask`). Returns
    an all-False mask if `target` itself doesn't clear the threshold.
    """
    from scipy import ndimage

    if not mask[target]:
        return np.zeros_like(mask)
    labeled, _ = ndimage.label(mask)  # default structure: 4-connectivity in 2D
    return labeled == labeled[target]


def region_shape_stats(mask: np.ndarray) -> Dict[str, object]:
    """Area, effective radius, and best-fit-ellipse shape of a boolean mask,
    via the second moments of the mask's pixel coordinates.
    """
    area = int(mask.sum())
    if area == 0:
        return {
            "area": 0,
            "r_eff": 0.0,
            "semi_axes": (0.0, 0.0),
            "aspect_ratio": float("nan"),
            "eccentricity": float("nan"),
            "centroid": (float("nan"), float("nan")),
        }
    ys, xs = np.nonzero(mask)
    cy, cx = float(ys.mean()), float(xs.mean())
    dy, dx = ys - cy, xs - cx
    cov = np.array([[np.mean(dy * dy), np.mean(dy * dx)], [np.mean(dy * dx), np.mean(dx * dx)]])
    eigvals = np.clip(np.linalg.eigvalsh(cov), 1e-12, None)  # ascending: [minor, major]
    semi_axes = 2.0 * np.sqrt(eigvals)  # ~2 std along each principal axis
    r_eff = float(np.sqrt(area / np.pi))
    aspect_ratio = float(semi_axes[1] / semi_axes[0])
    eccentricity = float(np.sqrt(max(0.0, 1.0 - (semi_axes[0] / semi_axes[1]) ** 2)))
    return {
        "area": area,
        "r_eff": r_eff,
        "semi_axes": (float(semi_axes[0]), float(semi_axes[1])),
        "aspect_ratio": aspect_ratio,
        "eccentricity": eccentricity,
        "centroid": (cy, cx),
    }


def correlation_region(
    dataset: Dataset,
    n_samples: int,
    t: float,
    schedule,
    target: Tuple[int, int],
    channel_mode: str = "mean",
    seed: Optional[int] = None,
    fwer_alpha: float = 0.05,
) -> Dict[str, object]:
    """End-to-end: sample, correlate, threshold, and measure shape in one call.

    Shape statistics (`area`, `r_eff`, etc.) are computed on the connected
    component touching `target`, not the raw thresholded mask -- the raw
    mask can still carry scattered false positives near the Bonferroni-
    corrected threshold (see `region_mask`). Both masks are returned for
    diagnostic comparison.
    """
    x0, xt = sample_x0_xt(dataset, n_samples, t, schedule, channel_mode=channel_mode, seed=seed)
    corr_map = pixel_correlation_map(x0, xt, target)
    raw_mask = region_mask(corr_map, n_samples, fwer_alpha=fwer_alpha)
    mask = connected_region_containing_target(raw_mask, target)
    stats = region_shape_stats(mask)
    stats["corr_map"] = corr_map
    stats["mask"] = mask
    stats["raw_mask"] = raw_mask
    stats["n_samples"] = n_samples
    stats["t"] = t
    return stats


def nested_correlation_curve(
    dataset: Dataset,
    n_values,
    t: float,
    schedule,
    target: Tuple[int, int],
    channel_mode: str = "mean",
    seed: Optional[int] = None,
    fwer_alpha: float = 0.05,
) -> Dict[int, Dict[str, object]]:
    """Region stats for every n in n_values, using *nested* subsets of one
    draw at max(n_values) rather than independent draws per n.

    This is both cheaper (one sample-and-corrupt pass instead of one per n)
    and more faithful to "what happens as you add more data": since n=100 is
    literally a prefix of the n=1000 draw, adjacent points on the resulting
    curve differ only by the samples actually *added*, not by an entirely
    fresh, independently-noisy re-estimate -- important for a scaling-law
    fit, since independent re-draws would inject extra point-to-point noise
    on top of the true n-dependence we're trying to measure.
    """
    n_values = sorted(n_values)
    n_max = n_values[-1]
    x0_full, xt_full = sample_x0_xt(dataset, n_max, t, schedule, channel_mode=channel_mode, seed=seed)

    out: Dict[int, Dict[str, object]] = {}
    for n in n_values:
        x0, xt = x0_full[:n], xt_full[:n]
        corr_map = pixel_correlation_map(x0, xt, target)
        raw_mask = region_mask(corr_map, n, fwer_alpha=fwer_alpha)
        mask = connected_region_containing_target(raw_mask, target)
        stats = region_shape_stats(mask)
        stats["corr_map"] = corr_map
        stats["mask"] = mask
        stats["raw_mask"] = raw_mask
        stats["n_samples"] = n
        stats["t"] = t
        out[n] = stats
    return out
