"""
Exact-where-possible mutual information between clean and erasure-corrupted
discrete patches (e.g. from the Random Hierarchy Model, see
``random_hierarchy.py`` and ``discrete_score.py``).

Because the corruption process is the absorbing-state ("erasure") channel
used throughout ``discrete_score.py`` -- every site of a patch is
independently kept or replaced by ``MASK`` with probability ``p`` -- the
conditional entropy of the noisy patch given the clean patch does not need
to be estimated at all:

    H(Xt | X0 = x0) = (number of sites) * H_b(p)

for *every* x0, since the only randomness left once x0 is fixed is which
sites got erased, and that is independent of x0 by construction. So

    I(X0; Xt) = H(Xt) - H(Xt | X0) = H(Xt) - k**2 * H_b(p)

and only the noisy-patch marginal entropy H(Xt) needs to be estimated from
data, via a (Miller-Madow bias-corrected) plug-in estimate on the empirical
distribution of every k x k sliding-window patch found anywhere in a large
sample of clean grids -- i.e. using *all* patches, not just ones aligned to
a hierarchy level's block boundaries.

An equivalent, more intuitive identity used to sanity-check the code and to
build exact test cases:

    I(X0; Xt) = E_S[ H(X0_S) ]

where S is the (random, independent of X0) subset of sites revealed by the
erasure channel: the mutual information is exactly the expected entropy of
whichever sub-patch happens to survive.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def binary_entropy_bits(p) -> np.ndarray:
    """H_b(p) in bits, vectorized; H_b(0) = H_b(1) = 0."""
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros_like(p)
    interior = (p > 0) & (p < 1)
    pi = p[interior]
    out[interior] = -(pi * np.log2(pi) + (1 - pi) * np.log2(1 - pi))
    return out


def extract_all_patches(grids: torch.Tensor, k: int) -> torch.Tensor:
    """Every k x k sliding-window patch at every position of every grid.

    Parameters
    ----------
    grids : LongTensor (N, H, W) of discrete token grids.
    k : patch side length.

    Returns
    -------
    LongTensor (M, k, k), M = N * (H-k+1) * (W-k+1), one row per
    (image, position) pair -- translation-invariant, not restricted to any
    hierarchy level's block alignment.
    """
    x = grids.unsqueeze(1).to(torch.float32)  # (N, 1, H, W)
    patches = F.unfold(x, k, stride=1, padding=0)  # (N, k*k, L)
    n, kk, L = patches.shape
    patches = patches.permute(0, 2, 1).reshape(n * L, k, k)
    return patches.round().long()


def empirical_entropy_bits(values: Iterable, correction: bool = True) -> Tuple[float, int, int]:
    """Plug-in (optionally Miller-Madow corrected) discrete entropy, in bits.

    Returns (entropy_bits, n_samples, n_distinct_observed).
    """
    counts = Counter(values)
    n = sum(counts.values())
    if n == 0:
        return 0.0, 0, 0
    probs = np.fromiter(counts.values(), dtype=np.float64) / n
    h = float(-np.sum(probs * np.log2(probs)))
    if correction:
        h += (len(counts) - 1) / (2.0 * n * math.log(2))
    return h, n, len(counts)


def mask_flat_patches(flat_patches: np.ndarray, p: float, mask_symbol: int, rng: np.random.Generator) -> np.ndarray:
    """Apply the i.i.d.-per-site erasure channel to (M, n_sites) clean patches."""
    mask = rng.random(flat_patches.shape) < p
    return np.where(mask, mask_symbol, flat_patches)


def patch_erasure_mutual_information(
    flat_patches: np.ndarray,
    n_sites: int,
    p_values: Sequence[float],
    mask_symbol: int = -1,
    correction: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Dict[float, Dict[str, float]]:
    """Mutual information between a clean patch and its erasure-corrupted view.

    Uses the exact identity ``I = H(Xt) - n_sites * H_b(p)``: the conditional
    term is analytic (no estimation needed, since which sites are erased is
    independent of the clean patch's value by construction of the channel),
    so only ``H(Xt)`` -- the entropy of the noisy-patch marginal -- has to be
    estimated, from the empirical distribution of ``flat_patches`` after
    masking.

    Parameters
    ----------
    flat_patches : (M, n_sites) array of clean patch values (one row per
        patch instance, e.g. from :func:`extract_all_patches` reshaped).
    n_sites : number of sites per patch (e.g. k*k for a k x k patch).
    p_values : masking probabilities to evaluate.
    mask_symbol : sentinel value used for masked sites (must not collide
        with a real token value).
    correction : apply Miller-Madow bias correction to the entropy estimate.
    rng : numpy Generator for the masking draws.

    Returns
    -------
    dict p -> {"mi_bits", "h_xt_bits", "h_cond_bits", "n_samples", "n_distinct"}
    """
    if rng is None:
        rng = np.random.default_rng(0)
    results: Dict[float, Dict[str, float]] = {}
    for p in p_values:
        xt = mask_flat_patches(flat_patches, p, mask_symbol, rng)
        h_xt, n, n_distinct = empirical_entropy_bits(map(tuple, xt), correction=correction)
        h_cond = n_sites * float(binary_entropy_bits(np.array([p]))[0])
        mi = max(0.0, h_xt - h_cond)
        results[p] = {
            "mi_bits": mi,
            "h_xt_bits": h_xt,
            "h_cond_bits": h_cond,
            "n_samples": n,
            "n_distinct": n_distinct,
        }
    return results


def patch_size_mutual_information_curves(
    grids: torch.Tensor,
    k_values: Sequence[int],
    p_values: Sequence[float],
    mask_symbol: int = -1,
    correction: bool = True,
    seed: int = 0,
) -> Dict[int, Dict[float, Dict[str, float]]]:
    """``I(X0_patch; Xt_patch)`` for every (patch size, masking probability) pair.

    Patches are every k x k sliding window from every position of every grid
    in ``grids`` (translation-invariant; no alignment to hierarchy levels).
    """
    rng = np.random.default_rng(seed)
    out: Dict[int, Dict[float, Dict[str, float]]] = {}
    for k in k_values:
        patches = extract_all_patches(grids, k)
        flat = patches.reshape(patches.shape[0], k * k).numpy()
        out[k] = patch_erasure_mutual_information(
            flat, n_sites=k * k, p_values=p_values, mask_symbol=mask_symbol, correction=correction, rng=rng
        )
    return out


def extract_block_contained_patches(
    leaf_grid: np.ndarray, block_values: np.ndarray, block_size: int, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Every k x k patch fully contained inside a single hierarchy block,
    paired with the ground-truth latent symbol governing that block.

    Parameters
    ----------
    leaf_grid : (N, H, W) leaf-level (finest) grids.
    block_values : (N, nb, nb) latent symbol at some coarser level, where
        ``nb = H // block_size`` and block (bi, bj) spans leaf rows/cols
        ``[bi*block_size:(bi+1)*block_size, bj*block_size:(bj+1)*block_size]``
        (this is exactly ``levels[l]`` from :meth:`RandomHierarchyModel2D.sample`
        with ``return_hierarchy=True``, for ``block_size = s**(L-l)``).
    block_size : side length of one block.
    k : patch side length; must be <= block_size (a patch that doesn't fit
        inside one block has no single well-defined governing latent).

    Returns
    -------
    (flat_patches (M, k*k), labels (M,)) -- one row per (sample, block,
    in-block offset) triple.
    """
    if k > block_size:
        raise ValueError(f"k={k} must be <= block_size={block_size} for a block-contained patch")
    n, h, w = leaf_grid.shape
    nb = h // block_size
    blocks = leaf_grid.reshape(n, nb, block_size, nb, block_size).transpose(0, 1, 3, 2, 4)
    # blocks: (n, nb, nb, block_size, block_size)
    n_offsets = block_size - k + 1
    patch_chunks = []
    label_chunks = []
    flat_labels = block_values.reshape(-1)
    for di in range(n_offsets):
        for dj in range(n_offsets):
            sub = blocks[:, :, :, di : di + k, dj : dj + k]  # (n, nb, nb, k, k)
            patch_chunks.append(sub.reshape(-1, k * k))
            label_chunks.append(flat_labels)
    flat_patches = np.concatenate(patch_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    return flat_patches, labels


def label_patch_mutual_information(
    flat_patches: np.ndarray,
    labels: np.ndarray,
    p_values: Sequence[float],
    mask_symbol: int = -1,
    correction: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Dict[float, Dict[str, float]]:
    """``I(Z; Xt_patch)`` between a discrete label Z and its paired clean
    patch, after the patch (not Z) is corrupted by i.i.d. erasure at
    probability p.

    Unlike :func:`patch_erasure_mutual_information`, there is no shortcut
    analytic conditional-entropy term here (Z is not "the patch before
    corruption", so ``H(Xt | Z)`` is not simply a function of p alone) --
    all three terms of ``I(Z;Xt) = H(Z) + H(Xt) - H(Z,Xt)`` are estimated
    with the same Miller-Madow-corrected plug-in estimator for consistency.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    labels_int = labels.astype(np.int64).tolist()
    h_z, n, _ = empirical_entropy_bits(labels_int, correction=correction)
    results: Dict[float, Dict[str, float]] = {}
    for p in p_values:
        xt = mask_flat_patches(flat_patches, p, mask_symbol, rng)
        xt_tuples = [tuple(row) for row in xt]
        h_xt, _, _ = empirical_entropy_bits(xt_tuples, correction=correction)
        joint = ((z, xt_t) for z, xt_t in zip(labels_int, xt_tuples))
        h_joint, _, n_distinct = empirical_entropy_bits(joint, correction=correction)
        mi = max(0.0, h_z + h_xt - h_joint)
        results[p] = {
            "mi_bits": mi,
            "h_z_bits": h_z,
            "h_xt_bits": h_xt,
            "h_joint_bits": h_joint,
            "n_samples": n,
            "n_distinct_joint": n_distinct,
        }
    return results


def resolvable_latent_height_grid(
    levels: Sequence[np.ndarray],
    s: int,
    k_values: Sequence[int],
    p_values: Sequence[float],
    threshold_frac: float = 0.5,
    mask_symbol: int = -1,
    correction: bool = True,
    seed: int = 0,
) -> Dict[str, object]:
    """Height (distance from the leaves) of the coarsest hierarchy level
    whose latent is "resolvable" from a k x k patch under corruption p,
    for every (k, p) on the requested grid.

    A level l (0 = root, L = leaf) is resolvable at (k, p) if
    ``I(Z_l ; Xt_patch) >= threshold_frac * H(Z_l)``, using patches fully
    contained in one of level l's own blocks (patch size capped at the
    block size ``s**(L-l)``, since a patch bigger than the block has no
    single well-defined governing latent -- larger k does not add more
    resolving power for that level beyond what its own block already gives).

    Two consistency corrections are applied on top of the raw threshold
    test, both consequences of the data-processing inequality that a finite
    Monte Carlo estimate can spuriously violate:

    1. Monotonicity in p: erasing at probability p2 > p1 can always be
       simulated as erasing at p1 and then independently re-erasing the
       survivors, i.e. Xt(p2) is a further degradation of Xt(p1). So
       ``I(Z_l; Xt(p2)) <= I(Z_l; Xt(p1))`` *must* hold in the true
       population for any fixed (l, k); once resolvability fails at some p
       (ascending order), it is forced to also fail at every larger p, even
       if sampling noise makes a later raw estimate look like it clears the
       bar again.
    2. Contiguity across levels: a level only counts as resolvable if every
       finer level down to the leaf is *also* resolvable, so an isolated
       noisy blip at one level can't count on its own.

    Parameters
    ----------
    levels : ``levels[l]`` for ``l = 0..L`` (root to leaf), each of shape
        ``(N, s**l, s**l)``, e.g. from ``RandomHierarchyModel2D.sample(...,
        return_hierarchy=True)``.
    p_values : masking probabilities, must be given in ascending order (the
        monotonicity correction above assumes this).

    Returns
    -------
    dict with:
      "height_grid": (len(p_values), len(k_values)) int array; L - l_min,
          or -1 if nothing (not even the leaf) is resolvable.
      "mi_by_level": {level: {k: {p: mi_bits}}} (raw, uncorrected estimates).
      "h_z_by_level": {level: H(Z_l) in bits}.
    """
    if list(p_values) != sorted(p_values):
        raise ValueError("p_values must be given in ascending order")

    L = len(levels) - 1
    leaf = levels[L]
    rng = np.random.default_rng(seed)

    mi_by_level: Dict[int, Dict[int, Dict[float, float]]] = {}
    resolvable_by_level: Dict[int, Dict[int, Dict[float, bool]]] = {}
    h_z_by_level: Dict[int, float] = {}

    for l in range(L + 1):
        block_size = s ** (L - l)
        block_values = levels[l]
        mi_by_level[l] = {}
        resolvable_by_level[l] = {}
        cache: Dict[int, Dict[str, object]] = {}
        for k in k_values:
            k_eff = min(k, block_size)
            if k_eff not in cache:
                flat_patches, labels = extract_block_contained_patches(leaf, block_values, block_size, k_eff)
                res = label_patch_mutual_information(
                    flat_patches, labels, p_values, mask_symbol=mask_symbol, correction=correction, rng=rng
                )
                raw_mi = {p: res[p]["mi_bits"] for p in p_values}
                h_z = res[p_values[0]]["h_z_bits"]
                resolvable_flags: Dict[float, bool] = {}
                still_ok = True
                for p in p_values:  # ascending: enforce monotone non-increasing resolvability
                    still_ok = still_ok and (h_z <= 1e-9 or raw_mi[p] >= threshold_frac * h_z)
                    resolvable_flags[p] = still_ok
                cache[k_eff] = {"mi": raw_mi, "resolvable": resolvable_flags, "h_z": h_z}
                h_z_by_level[l] = h_z
            mi_by_level[l][k] = cache[k_eff]["mi"]
            resolvable_by_level[l][k] = cache[k_eff]["resolvable"]

    height_grid = np.full((len(p_values), len(k_values)), -1, dtype=np.int64)
    for pi, p in enumerate(p_values):
        for ki, k in enumerate(k_values):
            l_min = None
            for l in range(L, -1, -1):  # leaf up to root
                if resolvable_by_level[l][k][p]:
                    l_min = l
                else:
                    break
            if l_min is not None:
                height_grid[pi, ki] = L - l_min

    return {"height_grid": height_grid, "mi_by_level": mi_by_level, "h_z_by_level": h_z_by_level}
