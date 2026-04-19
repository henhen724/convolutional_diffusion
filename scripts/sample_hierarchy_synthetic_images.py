#!/usr/bin/env python3
"""
Sample synthetic image datasets: hierarchical mixture of Gaussians and the
random-hierarchy sparse pattern model (see sample_random_hierarchy_images).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np


Array = np.ndarray


def sample_hierarchical_mixture_gaussian_images(
    n_samples: int,
    image_size: int = 32,
    num_channels: int = 3,
    num_components: int = 16,
    mu_0: float = 1.0,
    sigma: float = 0.15,
    seed: Optional[int] = 0,
) -> Array:
    """
    Sample images from a **hierarchical mixture of Gaussians**.

    Generative process:

    1. **Component means**  M = (μ_1, …, μ_K): each μ_k ∈ R^{C×H×W} is drawn i.i.d. with
       vec(μ_k) ~ N(0, μ_0² I)  (zero-mean, isotropic prior with variance μ_0² per dimension).
    2. **Data**  x_1, …, x_D: for each sample independently, pick a component
       z ~ Uniform({1, …, K}) and emit  x | (z = k) ~ N(μ_k, σ² I).

    So there are K mixture components; all share spherical noise variance σ².

    Parameters
    ----------
    n_samples
        Number of images D to draw (dataset size).
    image_size
        H and W (square images).
    num_channels
        Number of channels C.
    num_components
        K, the number of Gaussian components (number of mean tensors M).
    mu_0
        μ_0: prior scale (std dev along each coordinate of each mean); covariance is μ_0² I
        over the flattened mean vector for each component.
    sigma
        σ: within-component noise std; covariance is σ² I per observation.
    seed
        RNG seed for means and mixture sampling. None → non-deterministic.

    Returns
    -------
    x : np.ndarray, shape (n_samples, C, H, W), dtype float32
    """
    if n_samples < 0:
        raise ValueError("n_samples must be nonnegative")
    if num_components < 1:
        raise ValueError("num_components must be >= 1")
    if mu_0 < 0:
        raise ValueError("mu_0 must be nonnegative")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")

    rng = np.random.default_rng(seed)
    # K mean images, each ~ N(0, mu_0^2 I) coordinate-wise
    means = rng.normal(
        loc=0.0,
        scale=mu_0,
        size=(num_components, num_channels, image_size, image_size),
    ).astype(np.float32)

    z = rng.integers(0, num_components, size=n_samples)
    eps = rng.standard_normal(
        size=(n_samples, num_channels, image_size, image_size),
        dtype=np.float32,
    )
    x = means[z] + np.float32(sigma) * eps
    return x


def sample_random_hierarchy_images(
    n_samples: int,
    image_size: int = 32,
    num_channels: int = 3,
    m: int = 4,
    L: int = 4,
    seed: Optional[int] = None,
) -> Array:
    """
    **Random hierarchy** sparse pattern images (RGB or grayscale duplicated).

    **Geometry**

    - Flatten the ``image_size``×``image_size`` image in **row-major** order.
    - Split into **B = m^L** contiguous blocks; **each block has exactly ``m`` pixels**.
    - So total pixels ``N = B·m = m^(L+1)``. Requires ``image_size**2 == m**(L+1)``.

    **Within each block**

    Exactly one pixel is **on** (value ``1.0`` in float32, i.e. white if scaled to 255)
    and the other ``m-1`` pixels are **off** (``0.0``, black). The on-pixel index inside
    the block is in ``{0, …, m-1}`` (offset within that block along the flatten order).

    **Latents**

    For each level ``l = 1, …, L``, sample **2^l** i.i.d. symbols, each uniform on
    ``{0, …, m-1}``.

    **Readout (“which pixel is white”)**

    Let ``z_L`` be the length-``2^L`` vector at level ``L``. The **B** block offsets are
    filled from ``z_L`` in order, **cycling** if ``2^L < B`` (so every component of ``z_L``
    is used at least once when ``B > 2^L``). If ``2^L > B``, only the first ``B`` entries
    are used. When **m = 2**, ``B = 2^L`` and ``z_L`` is exactly one symbol per block.

    Coarser levels ``l < L`` are sampled as part of the hierarchy (for downstream analysis /
    conditioning); only level ``L`` drives the white-pixel choice as above.

    Parameters
    ----------
    n_samples
        Number of independent images.
    image_size
        Side length ``H``; image is ``H×H``.
    num_channels
        The same binary pattern is copied to every channel (e.g. RGB all match).
    m
        Block size (pixels per block) and alphabet size for latents.
    L
        Number of hierarchy levels; number of blocks ``B = m**L``.
    seed
        RNG seed; ``None`` → non-deterministic.

    Returns
    -------
    x : np.ndarray, shape ``(n_samples, C, H, W)``, dtype float32, values in ``{0.0, 1.0}``.
    """
    if n_samples < 0:
        raise ValueError("n_samples must be nonnegative")
    if m < 2:
        raise ValueError("m must be >= 2 (need at least two pixel positions per block)")
    if L < 1:
        raise ValueError("L must be >= 1")
    if num_channels < 1:
        raise ValueError("num_channels must be >= 1")

    n_pix = image_size * image_size
    n_expect = m ** (L + 1)
    if n_pix != n_expect:
        raise ValueError(
            f"random hierarchy requires image_size^2 == m^(L+1); got "
            f"{n_pix} != {n_expect} for m={m}, L={L} (try e.g. m=4,L=4 for 32×32)."
        )

    rng = np.random.default_rng(seed)
    b_blocks = m**L
    out = np.zeros((n_samples, num_channels, image_size, image_size), dtype=np.float32)

    for s in range(n_samples):
        hierarchy = {ell: rng.integers(0, m, size=2**ell, dtype=np.int64) for ell in range(1, L + 1)}
        z_l = hierarchy[L]
        if z_l.shape[0] >= b_blocks:
            block_white = z_l[:b_blocks].astype(np.int64, copy=False)
        else:
            reps = int(np.ceil(b_blocks / z_l.shape[0]))
            block_white = np.tile(z_l, reps)[:b_blocks]

        flat = np.zeros(n_pix, dtype=np.float32)
        for blk in range(b_blocks):
            j = int(block_white[blk]) % m
            flat[blk * m + j] = 1.0
        plane = flat.reshape(image_size, image_size)
        for c in range(num_channels):
            out[s, c] = plane

    return out


def _save_npy(path: Path, arr: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _write_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample hierarchical-mixture or random-hierarchy synthetic images."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=("hierarchical_mixture", "random_hierarchy"),
        required=True,
    )
    parser.add_argument("--out_dir", type=str, default="./data/hierarchy_synthetic")
    parser.add_argument("--n_samples", type=int, default=10_000)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    # Hierarchical mixture–specific (see sample_hierarchical_mixture_gaussian_images docstring)
    parser.add_argument("--num_components", type=int, default=16)
    parser.add_argument("--mu_0", type=float, default=1.0, help="Prior std μ_0 for each mean (cov μ_0² I).")
    parser.add_argument("--sigma", type=float, default=0.15, help="Within-component noise std σ (cov σ² I).")
    # Random hierarchy (see sample_random_hierarchy_images): need image_size^2 == m^(L+1)
    parser.add_argument("--m", type=int, default=4, help="Random hierarchy: m pixels per block; latent alphabet 0..m-1.")
    parser.add_argument("--L", type=int, default=4, help="Random hierarchy: L levels, B=m^L blocks.")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    if args.model == "hierarchical_mixture":
        x = sample_hierarchical_mixture_gaussian_images(
            n_samples=args.n_samples,
            image_size=args.image_size,
            num_channels=args.num_channels,
            num_components=args.num_components,
            mu_0=args.mu_0,
            sigma=args.sigma,
            seed=args.seed,
        )
        data_path = out_dir / "hierarchical_mixture_train.npy"
        _save_npy(data_path, x)
        _write_metadata(
            out_dir / "hierarchical_mixture_metadata.json",
            {
                "model": "hierarchical_mixture_gaussian",
                "n_samples": int(args.n_samples),
                "image_size": int(args.image_size),
                "num_channels": int(args.num_channels),
                "num_components": int(args.num_components),
                "mu_0": float(args.mu_0),
                "sigma": float(args.sigma),
                "seed": int(args.seed),
                "data_file": data_path.name,
            },
        )
        print(f"Wrote {x.shape} float32 array to {data_path}")
    else:
        x = sample_random_hierarchy_images(
            n_samples=args.n_samples,
            image_size=args.image_size,
            num_channels=args.num_channels,
            m=args.m,
            L=args.L,
            seed=args.seed,
        )
        data_path = out_dir / "random_hierarchy_train.npy"
        _save_npy(data_path, x)
        _write_metadata(
            out_dir / "random_hierarchy_metadata.json",
            {
                "model": "random_hierarchy",
                "n_samples": int(args.n_samples),
                "image_size": int(args.image_size),
                "num_channels": int(args.num_channels),
                "m": int(args.m),
                "L": int(args.L),
                "B_blocks": int(args.m**args.L),
                "seed": int(args.seed) if args.seed is not None else None,
                "data_file": data_path.name,
            },
        )
        print(f"Wrote {x.shape} float32 array to {data_path}")


if __name__ == "__main__":
    main()
