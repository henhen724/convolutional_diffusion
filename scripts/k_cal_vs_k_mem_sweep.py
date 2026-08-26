#!/usr/bin/env python
"""
Sweep k_cal (cosine-calibrated LS scale vs. a trained UNet) and posterior entropy
(for k_mem,eps, the memorization transition) along real model-driven reverse-diffusion
trajectories, on the same noisy samples for both quantities.

Companion compute for notebooks/MatchingCalbratedScaleAndCollapseScale.ipynb -- this script
does the heavy GPU sweep and caches the result; the notebook loads the cache to search for
the best-matching eps and plot k_cal(sigma_t^2) vs. k_mem,eps(sigma_t^2).

Example usage:
  python scripts/k_cal_vs_k_mem_sweep.py --dataset cifar10 \
      --model_path checkpoints/backbone_CIFAR10_UNet_zeros_conditional.pt --conditional
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data import get_dataset
from src.utils.idealscore import LocalScoreModule
from src.utils.noise_schedules import cosine_noise_schedule

# Legacy backbone_*.pt checkpoints were pickled with class module "models"/"utils"
# (pre-refactor layout); register aliases so torch.load's unpickler can find them.
import src.models
sys.modules["models"] = sys.modules["src.models"]
import src.utils
sys.modules["utils"] = sys.modules["src.utils"]


def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return (a @ b) / (a.norm() * b.norm() + 1e-8)


def central_entropy_mean(entropy_map, k):
    """entropy_map: (b, h, w). Exclude a border of width k//2 on each side (patches that
    needed zero-padding in LocalScoreModule's F.unfold(..., padding=k//2)), so only pixels
    with a fully-real k x k patch count. Matches a coauthor's convention of excluding
    boundary patches from the memorization-transition entropy (only affects k_mem, not the
    k_cal cosine-similarity comparison)."""
    margin = k // 2
    if margin == 0:
        return entropy_map.mean()
    h, w = entropy_map.shape[-2:]
    if 2 * margin >= h or 2 * margin >= w:
        return entropy_map.mean()  # margin too large for image size; fall back
    return entropy_map[:, margin:h - margin, margin:w - margin].mean()


def run_calibration_and_entropy_sweep(
    model, dataset, image_size, in_channels,
    kernelsizes, nsteps, nsamps, max_samples,
    score_batch_size, schedule, device, seed=0,
    conditional=False, nlabels=10, exclude_boundary_patches=True,
):
    """Sweep k_cal (cosine-calibrated vs. model) and posterior entropy (for k_mem,eps),
    evaluated on the same noisy samples along real model-driven reverse trajectories.
    If conditional, a random label is drawn per trajectory and passed to both the model
    and the LS modules (so the LS posterior is restricted to that class' training patches,
    matching how the model itself is conditioned). The label must stay on CPU:
    LocalScoreModule compares it directly against its trainloader's (CPU) labels, while
    the model moves label to its own device internally (EmbeddingModule.forward).
    exclude_boundary_patches: see central_entropy_mean.
    """
    ls_modules = {
        k: LocalScoreModule(
            dataset, kernel_size=k, image_size=image_size,
            batch_size=min(score_batch_size, len(dataset)),
            schedule=schedule, max_samples=max_samples,
        ).to(device).eval()
        for k in kernelsizes
    }

    nk = len(kernelsizes)
    cos_sum = torch.zeros(nsteps, nk)
    entropy_sum = torch.zeros(nsteps, nk)
    sigma2 = torch.zeros(nsteps)

    torch.manual_seed(seed)
    t_start = time.time()
    with torch.no_grad():
        for s in range(nsamps):
            label = torch.randint(0, nlabels, (1,)) if conditional else None
            x = torch.randn(1, in_channels, image_size, image_size, device=device)
            for i in range(nsteps, 0, -1):
                step_idx = i - 1
                t = i * torch.ones(1, device=device) / nsteps
                beta_t = schedule(t)
                alpha_t = 1 - beta_t
                if s == 0:
                    sigma2[step_idx] = beta_t.item()

                eps_nn = model(t, x, label=label)

                for j, k in enumerate(kernelsizes):
                    score, _, entropy_map, *_ = ls_modules[k].forward_with_posterior_stats(
                        t, x, label=label, device=device, k=k
                    )
                    eps_ls = -score * beta_t.sqrt()
                    cos_sum[step_idx, j] += cosine_sim(eps_ls, eps_nn).item()
                    if exclude_boundary_patches:
                        entropy_sum[step_idx, j] += central_entropy_mean(entropy_map, k).item()
                    else:
                        entropy_sum[step_idx, j] += entropy_map.mean().item()

                beta_t_prev = schedule(t - 1 / nsteps) if i > 1 else torch.zeros_like(beta_t)
                alpha_t_prev = 1 - beta_t_prev
                x = x * torch.sqrt(alpha_t_prev / alpha_t)[:, None, None, None]
                x = x + (
                    torch.sqrt(beta_t_prev)[:, None, None, None]
                    - torch.sqrt(alpha_t_prev / alpha_t)[:, None, None, None] * torch.sqrt(beta_t)[:, None, None, None]
                ) * eps_nn
            print(f"  trajectory {s + 1}/{nsamps} done ({time.time() - t_start:.0f}s elapsed)", flush=True)

    return {
        "kernelsizes": list(kernelsizes),
        "sigma2": sigma2.numpy(),
        "cos_by_step_k": (cos_sum / nsamps).numpy(),
        "entropy_by_step_k": (entropy_sum / nsamps).numpy(),
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep k_cal and posterior entropy vs. a UNet")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--conditional", action="store_true", default=False)
    parser.add_argument("--kernelsizes", type=int, nargs="*",
                         default=[3, 5, 7, 9, 11, 13, 15, 17, 19, 23, 27, 31])
    parser.add_argument("--nsteps", type=int, default=20)
    parser.add_argument("--nsamps", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--score_batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                         help="Defaults to results/channel_theory/k_cal_vs_k_mem_<dataset>.pt")
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--include_boundary_patches", action="store_true", default=False,
                         help="Average entropy over the whole image instead of excluding "
                              "boundary (zero-padded) patches.")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    dataset, metadata = get_dataset(args.dataset, root="./data")
    in_channels = metadata["num_channels"]
    image_size = metadata["image_size"]
    nlabels = metadata["num_classes"]
    if max(args.kernelsizes) > image_size:
        raise SystemExit(f"kernelsizes must not exceed image_size={image_size}")

    model = torch.load(args.model_path, map_location=device, weights_only=False)
    model.eval().to(device)
    print(f"Loaded {args.model_path}: {sum(p.numel() for p in model.parameters())} params")

    result = run_calibration_and_entropy_sweep(
        model, dataset, image_size, in_channels,
        args.kernelsizes, args.nsteps, args.nsamps, args.max_samples,
        args.score_batch_size, cosine_noise_schedule, device, seed=args.seed,
        conditional=args.conditional, nlabels=nlabels,
        exclude_boundary_patches=not args.include_boundary_patches,
    )

    output = args.output or os.path.join("results", "channel_theory", f"k_cal_vs_k_mem_{args.dataset}.pt")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(result, output)
    print(f"Saved result to {output}")


if __name__ == "__main__":
    main()
