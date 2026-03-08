"""
Channel theory of scale calibration: pixel-space ELS and LS.

Loads scale calibration files (median k per timestep), and at a given timestep
computes and plots vs kernel size k:
- Average entropy of ELS and LS posteriors
- Center-pixel variance and Binder cumulant
- Total patch variance and total patch Binder cumulant

Theory: fitted scale minimizes cost = β⟨d(x,x')⟩ − S (higher entropy S is better).
This script plots the implied beta for different rate-distortion choices (variance vs Binder,
center vs total) as a function of timestep.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.channel_theory import (
    collect_stats_vs_k,
    cost_at_beta,
    infer_beta_at_k_star,
    load_scale_calibration,
    t_to_snr,
)
from src.utils.data import get_dataset
from src.utils.idealscore import (
    LocalEquivScoreModule,
    LocalScoreModule,
)
from src.utils.noise_schedules import cosine_noise_schedule


def run_channel_theory(
    dataset_name="mnist",
    kernelsizes=(3, 5, 7, 9, 11),
    scale_file_ls=None,
    scale_file_els=None,
    tld="./checkpoints/",
    nsteps=20,
    nsamples=4,
    timestep_idx=None,
    max_samples=500,
    score_batch_size=32,
    out_dir="./results/channel_theory",
    cpu=False,
):
    device = torch.device("cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(out_dir, exist_ok=True)

    dataset, metadata = get_dataset(dataset_name, root="./data")
    in_channels = metadata["num_channels"]
    image_size = metadata["image_size"]
    schedule = cosine_noise_schedule

    # Optional: load scale calibration (median k per step)
    # median tensor shape: (nsteps,) with step index 0 = first reverse step (t ~ 1/nsteps)
    median_ls = load_scale_calibration(scale_file_ls, tld)
    median_els = load_scale_calibration(scale_file_els, tld)

    if timestep_idx is None:
        timestep_idx = nsteps // 2
    t = (timestep_idx + 1) * torch.ones(1, device=device) / nsteps

    k_vals = list(kernelsizes)

    # Build LS and ELS modules for each k
    ls_modules = []
    for k in k_vals:
        mod = LocalScoreModule(
            dataset,
            kernel_size=k,
            image_size=image_size,
            batch_size=min(score_batch_size, len(dataset)),
            schedule=schedule,
            max_samples=max_samples,
        )
        mod.to(device)
        mod.eval()
        ls_modules.append((k, mod))

    els_modules = []
    for k in k_vals:
        mod = LocalEquivScoreModule(
            dataset,
            kernel_size=k,
            batch_size=score_batch_size,
            image_size=image_size,
            channels=in_channels,
            schedule=schedule,
            max_samples=max_samples,
        )
        mod.to(device)
        mod.eval()
        els_modules.append((k, mod))

    # Fixed noise batch at timestep t
    torch.manual_seed(42)
    x_batch = torch.randn(nsamples, in_channels, image_size, image_size, device=device)
    label = None  # unconditional

    # Collect stats vs k for LS and ELS (using src.utils.channel_theory)
    stats_ls = collect_stats_vs_k(ls_modules, t, x_batch, device, label)
    stats_els = collect_stats_vs_k(els_modules, t, x_batch, device, label)
    k_vals = stats_ls["k_vals"]
    ls_entropy = stats_ls["avg_entropy"]
    els_entropy = stats_els["avg_entropy"]
    ls_center_var = stats_ls["center_variance"]
    els_center_var = stats_els["center_variance"]
    ls_center_binder = stats_ls["center_binder"]
    els_center_binder = stats_els["center_binder"]
    ls_total_var = stats_ls["total_variance"]
    els_total_var = stats_els["total_variance"]
    ls_total_binder = stats_ls["total_binder"]
    els_total_binder = stats_els["total_binder"]

    # ---- Plots at this timestep: stats vs k ----
    snr_current = float(t_to_snr(t.cpu(), schedule))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"Channel theory at SNR = α_t/β_t = {snr_current:.3g} — {dataset_name}")

    ax = axes[0, 0]
    ax.plot(k_vals, ls_entropy, "o-", label="LS")
    ax.plot(k_vals, els_entropy, "s-", label="ELS")
    ax.set_xlabel("kernel size k")
    ax.set_ylabel("Average entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(k_vals, ls_center_var, "o-", label="LS")
    ax.plot(k_vals, els_center_var, "s-", label="ELS")
    ax.set_xlabel("kernel size k")
    ax.set_ylabel("Center pixel variance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(k_vals, ls_center_binder, "o-", label="LS")
    ax.plot(k_vals, els_center_binder, "s-", label="ELS")
    ax.set_xlabel("kernel size k")
    ax.set_ylabel("Center pixel Binder cumulant")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(k_vals, ls_total_var, "o-", label="LS")
    ax.plot(k_vals, els_total_var, "s-", label="ELS")
    ax.set_xlabel("kernel size k")
    ax.set_ylabel("Total patch variance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(k_vals, ls_total_binder, "o-", label="LS")
    ax.plot(k_vals, els_total_binder, "s-", label="ELS")
    ax.set_xlabel("kernel size k")
    ax.set_ylabel("Total patch Binder cumulant")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cost = β×⟨d⟩ − S: plot cost vs k for a few betas (LS only as example)
    ax = axes[1, 2]
    betas_plot = [0.0, 0.5, 1.0, 2.0]
    for beta in betas_plot:
        cost = cost_at_beta(ls_entropy, ls_total_var, beta)
        ax.plot(k_vals, cost, "o-", label=f"LS cost (β={beta}, total var)")
    ax.set_xlabel("kernel size k")
    ax.set_ylabel("Cost = β×⟨d⟩ − S (total variance)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Cost vs k (LS)")

    # Info text: calibrated k if available
    if median_ls is not None or median_els is not None:
        text = ""
        if median_ls is not None and timestep_idx < median_ls.numel():
            text += f"Calibrated k (LS): {median_ls[timestep_idx].item()}\n"
        if median_els is not None and timestep_idx < median_els.numel():
            text += f"Calibrated k (ELS): {median_els[timestep_idx].item()}\n"
        ax.text(0.98, 0.02, text.strip(), transform=ax.transAxes, fontsize=9, verticalalignment="bottom", horizontalalignment="right")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"channel_theory_stats_vs_k_{dataset_name}_t{timestep_idx}.png"), dpi=150)
    plt.close()

    # ---- Beta inference and beta vs timestep ----
    # We need stats at multiple timesteps to plot beta vs timestep. If we only have one timestep,
    # we still infer beta at this timestep using calibrated k (if available).
    betas = {}  # key: (score_type, reg_type), value: list of (step_idx, beta)
    for score_name, entropy_arr, cvar, cbinder, tvar, tbinder in [
        ("LS", ls_entropy, ls_center_var, ls_center_binder, ls_total_var, ls_total_binder),
        ("ELS", els_entropy, els_center_var, els_center_binder, els_total_var, els_total_binder),
    ]:
        median = median_ls if score_name == "LS" else median_els
        if median is not None and timestep_idx < median.numel():
            k_star = int(median[timestep_idx].item())
            for reg_name, reg_arr in [
                ("center_variance", cvar),
                ("center_binder", cbinder),
                ("total_variance", tvar),
                ("total_binder", tbinder),
            ]:
                beta = infer_beta_at_k_star(entropy_arr, reg_arr, k_vals, k_star)
                key = (score_name, reg_name)
                betas[key] = [(timestep_idx, beta)]

    # Multi-step beta: recompute stats for a grid of timesteps, then plot β vs SNR
    step_indices = list(np.linspace(0, nsteps - 1, min(8, nsteps), dtype=int))
    step_indices = sorted(set(step_indices))
    for step_idx in step_indices:
        if step_idx == timestep_idx:
            continue  # already have stats from main loop
        t_cur = (step_idx + 1) * torch.ones(1, device=device) / nsteps
        x_cur = torch.randn(nsamples, in_channels, image_size, image_size, device=device)
        st_ls = collect_stats_vs_k(ls_modules, t_cur, x_cur, device, label)
        st_els = collect_stats_vs_k(els_modules, t_cur, x_cur, device, label)
        for score_name, e, cv, cb, tv, tb in [
            ("LS", st_ls["avg_entropy"], st_ls["center_variance"], st_ls["center_binder"], st_ls["total_variance"], st_ls["total_binder"]),
            ("ELS", st_els["avg_entropy"], st_els["center_variance"], st_els["center_binder"], st_els["total_variance"], st_els["total_binder"]),
        ]:
            median = median_ls if score_name == "LS" else median_els
            if median is not None and step_idx < median.numel():
                k_star = int(median[step_idx].item())
                for reg_name, reg_arr in [
                    ("center_variance", cv),
                    ("center_binder", cb),
                    ("total_variance", tv),
                    ("total_binder", tb),
                ]:
                    beta = infer_beta_at_k_star(e, reg_arr, k_vals, k_star)
                    key = (score_name, reg_name)
                    if key not in betas:
                        betas[key] = []
                    betas[key].append((step_idx, beta))

    # Plot beta vs SNR (α_t / β_t) for different rate-distortion choices
    if betas:
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
        for (score_name, reg_name), points in betas.items():
            points = sorted(points, key=lambda p: p[0])
            snr_vals = [float(t_to_snr((s + 1) / nsteps, schedule)) for s in [p[0] for p in points]]
            vals = [p[1] for p in points]
            snr_vals = [s for s, v in zip(snr_vals, vals) if not np.isnan(v)]
            vals = [v for v in vals if not np.isnan(v)]
            if snr_vals and vals:
                ax2.plot(snr_vals, vals, "o-", label=f"{score_name} — {reg_name}")
        ax2.set_xlabel("Signal-to-noise ratio α_t / β_t")
        ax2.set_ylabel("Inferred β (cost = β×⟨d⟩ − S)")
        ax2.set_title("β vs SNR (from calibrated scale)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(os.path.join(out_dir, f"channel_theory_beta_vs_snr_{dataset_name}.png"), dpi=150)
        plt.close()

    return {
        "k_vals": k_vals,
        "ls_entropy": ls_entropy,
        "els_entropy": els_entropy,
        "ls_center_variance": ls_center_var,
        "els_center_variance": els_center_var,
        "ls_center_binder": ls_center_binder,
        "els_center_binder": els_center_binder,
        "ls_total_variance": ls_total_var,
        "els_total_variance": els_total_var,
        "ls_total_binder": ls_total_binder,
        "els_total_binder": els_total_binder,
        "betas": betas,
    }


def main():
    parser = argparse.ArgumentParser(description="Channel theory of scale calibration (pixel-space ELS/LS)")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--kernelsizes", type=int, nargs="*", default=[3, 5, 7, 9, 11])
    parser.add_argument("--scale_file_ls", type=str, default=None, help="e.g. scales_MNIST_ResNet_zeros_LS_median.pt")
    parser.add_argument("--scale_file_els", type=str, default=None, help="e.g. scales_MNIST_ResNet_zeros_ELS_median.pt")
    parser.add_argument("--tld", type=str, default="./checkpoints/")
    parser.add_argument("--nsteps", type=int, default=20)
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--timestep_idx", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--score_batch_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="./results/channel_theory")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    run_channel_theory(
        dataset_name=args.dataset,
        kernelsizes=tuple(args.kernelsizes),
        scale_file_ls=args.scale_file_ls,
        scale_file_els=args.scale_file_els,
        tld=args.tld,
        nsteps=args.nsteps,
        nsamples=args.nsamples,
        timestep_idx=args.timestep_idx,
        max_samples=args.max_samples,
        score_batch_size=args.score_batch_size,
        out_dir=args.out_dir,
        cpu=args.cpu,
    )


if __name__ == "__main__":
    main()
