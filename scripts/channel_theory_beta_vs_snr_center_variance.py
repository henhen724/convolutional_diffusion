"""
Collect statistics at each SNR and infer β (center variance only) for channel theory.

Intended to be run as a Slurm job: loops over all timesteps, at each step collects
posterior stats vs k, infers β from calibrated k* for center_variance only (LS and ELS),
and saves (SNR, beta_ls, beta_els) so the notebook can load and plot β vs SNR.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.channel_theory import (
    collect_stats_vs_k,
    infer_beta_at_k_star,
    infer_beta_range_at_k_star,
    load_scale_calibration,
    t_to_snr,
)
from src.utils.data import get_dataset
from src.utils.idealscore import LocalEquivScoreModule, LocalScoreModule
from src.utils.noise_schedules import cosine_noise_schedule


def main():
    parser = argparse.ArgumentParser(
        description="Collect β vs SNR for center variance only (for Slurm / notebook)"
    )
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument(
        "--kernelsizes",
        type=int,
        nargs="*",
        default=[3, 5, 7, 9, 11, 13, 15],
        help="All window sizes to use (include calibration set; 3–15 for 32x32)",
    )
    parser.add_argument("--scale_file_ls", type=str, default=None)
    parser.add_argument("--scale_file_els", type=str, default=None)
    parser.add_argument("--tld", type=str, default="./checkpoints/")
    parser.add_argument("--nsteps", type=int, default=20)
    parser.add_argument("--nsamples", type=int, default=64)
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=8,
        help="Process nsamples in rounds of this size to limit GPU memory.",
    )
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--score_batch_size", type=int, default=32)
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Only check imports and args, then exit 0.")
    args = parser.parse_args()

    if args.dry_run:
        # Lightweight check: imports and scale-file requirement
        median_ls = load_scale_calibration(args.scale_file_ls, args.tld)
        median_els = load_scale_calibration(args.scale_file_els, args.tld)
        if median_ls is None and median_els is None:
            raise SystemExit("Dry run: need at least one of --scale_file_ls or --scale_file_els.")
        print("Dry run: imports and scale calibration load OK.")
        return

    if args.out_file is None:
        args.out_file = os.path.join(
            "./results/channel_theory",
            f"beta_vs_snr_center_variance_{args.dataset}.pt",
        )
    out_dir = os.path.dirname(args.out_file)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset, metadata = get_dataset(args.dataset, root="./data")
    in_channels = metadata["num_channels"]
    image_size = metadata["image_size"]
    schedule = cosine_noise_schedule
    k_vals = list(args.kernelsizes)

    median_ls = load_scale_calibration(args.scale_file_ls, args.tld)
    median_els = load_scale_calibration(args.scale_file_els, args.tld)
    if median_ls is None and median_els is None:
        raise SystemExit("Need at least one of --scale_file_ls or --scale_file_els")

    ls_modules = []
    for k in k_vals:
        mod = LocalScoreModule(
            dataset,
            kernel_size=k,
            image_size=image_size,
            batch_size=min(args.score_batch_size, len(dataset)),
            schedule=schedule,
            max_samples=args.max_samples,
        )
        mod.to(device).eval()
        ls_modules.append((k, mod))

    els_modules = []
    for k in k_vals:
        mod = LocalEquivScoreModule(
            dataset,
            kernel_size=k,
            batch_size=args.score_batch_size,
            image_size=image_size,
            channels=in_channels,
            schedule=schedule,
            max_samples=args.max_samples,
        )
        mod.to(device).eval()
        els_modules.append((k, mod))

    snr_list = []
    step_list = []
    beta_ls_list = []
    beta_els_list = []
    beta_ls_err_lo_list = []
    beta_ls_err_hi_list = []
    beta_els_err_lo_list = []
    beta_els_err_hi_list = []
    entropy_ls_list = []
    reg_ls_list = []
    entropy_els_list = []
    reg_els_list = []

    def collect_stats_vs_k_in_rounds(modules_by_k, t_cur):
        k_local = [k for k, _ in modules_by_k]
        keys = [
            "avg_entropy",
            "center_variance",
            "center_binder",
            "total_variance",
            "total_binder",
        ]
        accum = {k: np.zeros(len(k_local), dtype=np.float64) for k in keys}
        total = 0
        bs_round = max(1, int(args.sample_batch_size))
        for start in range(0, args.nsamples, bs_round):
            bs = min(bs_round, args.nsamples - start)
            x_round = torch.randn(bs, in_channels, image_size, image_size, device=device)
            st = collect_stats_vs_k(modules_by_k, t_cur, x_round, device, None)
            for kk in keys:
                accum[kk] += bs * st[kk].astype(np.float64)
            total += bs
            del x_round
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for kk in keys:
            accum[kk] /= max(1, total)
        return {"k_vals": k_local, **accum}

    for step_idx in range(args.nsteps):
        t_cur = (step_idx + 1) * torch.ones(1, device=device) / args.nsteps
        torch.manual_seed(42 + step_idx)
        st_ls = collect_stats_vs_k_in_rounds(ls_modules, t_cur)
        st_els = collect_stats_vs_k_in_rounds(els_modules, t_cur)

        snr = float(t_to_snr((step_idx + 1) / args.nsteps, schedule))
        snr_list.append(snr)
        step_list.append(step_idx)

        # Save per-timestep per-k entropy and center variance (LS / ELS)
        entropy_ls_list.append(st_ls["avg_entropy"].astype(np.float64).tolist())
        reg_ls_list.append(st_ls["center_variance"].astype(np.float64).tolist())
        entropy_els_list.append(st_els["avg_entropy"].astype(np.float64).tolist())
        reg_els_list.append(st_els["center_variance"].astype(np.float64).tolist())

        beta_ls = np.nan
        err_ls_lo = err_ls_hi = np.nan
        if median_ls is not None and step_idx < median_ls.numel():
            k_star = int(median_ls[step_idx].item())
            beta_ls = infer_beta_at_k_star(
                st_ls["avg_entropy"],
                st_ls["center_variance"],
                k_vals,
                k_star,
            )
            bl, bh = infer_beta_range_at_k_star(
                st_ls["avg_entropy"],
                st_ls["center_variance"],
                k_vals,
                k_star,
            )
            if not np.isnan(beta_ls):
                err_ls_lo = beta_ls - bl if np.isfinite(bl) and not np.isnan(bl) else np.nan
                err_ls_hi = bh - beta_ls if np.isfinite(bh) and not np.isnan(bh) else np.nan
        beta_ls_list.append(float(beta_ls) if not np.isnan(beta_ls) else None)
        beta_ls_err_lo_list.append(float(err_ls_lo) if not np.isnan(err_ls_lo) else None)
        beta_ls_err_hi_list.append(float(err_ls_hi) if not np.isnan(err_ls_hi) else None)

        beta_els = np.nan
        err_els_lo = err_els_hi = np.nan
        if median_els is not None and step_idx < median_els.numel():
            k_star = int(median_els[step_idx].item())
            beta_els = infer_beta_at_k_star(
                st_els["avg_entropy"],
                st_els["center_variance"],
                k_vals,
                k_star,
            )
            bl, bh = infer_beta_range_at_k_star(
                st_els["avg_entropy"],
                st_els["center_variance"],
                k_vals,
                k_star,
            )
            if not np.isnan(beta_els):
                err_els_lo = beta_els - bl if np.isfinite(bl) and not np.isnan(bl) else np.nan
                err_els_hi = bh - beta_els if np.isfinite(bh) and not np.isnan(bh) else np.nan
        beta_els_list.append(float(beta_els) if not np.isnan(beta_els) else None)
        beta_els_err_lo_list.append(float(err_els_lo) if not np.isnan(err_els_lo) else None)
        beta_els_err_hi_list.append(float(err_els_hi) if not np.isnan(err_els_hi) else None)

    result = {
        "dataset": args.dataset,
        "nsteps": args.nsteps,
        "nsamples": args.nsamples,
        "k_vals": k_vals,
        "snr": snr_list,
        "step_idx": step_list,
        "beta_ls": beta_ls_list,
        "beta_els": beta_els_list,
        "beta_ls_err_lo": beta_ls_err_lo_list,
        "beta_ls_err_hi": beta_ls_err_hi_list,
        "beta_els_err_lo": beta_els_err_lo_list,
        "beta_els_err_hi": beta_els_err_hi_list,
        "entropy_ls": entropy_ls_list,
        "reg_ls": reg_ls_list,
        "entropy_els": entropy_els_list,
        "reg_els": reg_els_list,
    }
    torch.save(result, args.out_file)
    json_path = args.out_file.replace(".pt", ".json")
    with open(json_path, "w") as f:
        meta = {k: v for k, v in result.items() if k not in (
            "snr", "step_idx", "beta_ls", "beta_els",
            "beta_ls_err_lo", "beta_ls_err_hi", "beta_els_err_lo", "beta_els_err_hi",
        )}
        json.dump(
            {
                **meta,
                "snr": snr_list,
                "step_idx": step_list,
                "beta_ls": beta_ls_list,
                "beta_els": beta_els_list,
                "beta_ls_err_lo": beta_ls_err_lo_list,
                "beta_ls_err_hi": beta_ls_err_hi_list,
                "beta_els_err_lo": beta_els_err_lo_list,
                "beta_els_err_hi": beta_els_err_hi_list,
            },
            f,
            indent=2,
        )
    print(f"Saved to {args.out_file} and {json_path}")


if __name__ == "__main__":
    main()
