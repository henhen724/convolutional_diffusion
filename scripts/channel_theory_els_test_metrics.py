"""
Measure and record ELS or LS test L2 error (MSE), entropy, and center pixel variance
at each timestep and scale. Intended for Slurm.

Uses train set for posterior (training patches) and test set for computing
MSE between E[x0] and true x0. Saves results to a .pt file for notebook plotting.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.channel_theory import t_to_snr
from src.utils.data import get_dataset
from src.utils.idealscore import LocalEquivScoreModule, LocalScoreModule
from src.utils.noise_schedules import cosine_noise_schedule


def main():
    parser = argparse.ArgumentParser(
        description="ELS or LS test MSE, entropy, and center variance at each (timestep, scale)"
    )
    parser.add_argument(
        "--score_module",
        type=str,
        choices=("els", "ls"),
        default="els",
        help="Score module: els (LocalEquivScoreModule) or ls (LocalScoreModule)",
    )
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument(
        "--kernelsizes",
        type=int,
        nargs="*",
        default=[3, 5, 7, 9, 11, 13, 15],
    )
    parser.add_argument("--tld", type=str, default="./checkpoints/")
    parser.add_argument("--nsteps", type=int, default=20)
    parser.add_argument("--ntest", type=int, default=200, help="Number of test samples to use")
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=500, help="Max training samples per forward")
    parser.add_argument("--score_batch_size", type=int, default=32)
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        get_dataset(args.dataset, root="./data", train=False)
        print("Dry run: imports and test dataset load OK.")
        return

    if args.out_file is None:
        args.out_file = os.path.join(
            "./results/channel_theory",
            f"{args.score_module}_test_metrics_{args.dataset}.pt",
        )
    out_dir = os.path.dirname(args.out_file)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    schedule = cosine_noise_schedule
    k_vals = list(args.kernelsizes)

    # Train set for ELS (posterior over training patches); test set for L2 error
    dataset_train, metadata = get_dataset(args.dataset, root="./data", train=True)
    dataset_test, _ = get_dataset(args.dataset, root="./data", train=False)
    in_channels = metadata["num_channels"]
    image_size = metadata["image_size"]

    # Cap ntest to test set size
    ntest = min(args.ntest, len(dataset_test))
    test_loader = DataLoader(
        dataset_test,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Build one module per scale (ELS or LS)
    score_modules = []
    for k in k_vals:
        if args.score_module == "els":
            mod = LocalEquivScoreModule(
                dataset_train,
                kernel_size=k,
                batch_size=args.score_batch_size,
                image_size=image_size,
                channels=in_channels,
                schedule=schedule,
                max_samples=args.max_samples,
            )
        else:
            mod = LocalScoreModule(
                dataset_train,
                kernel_size=k,
                image_size=image_size,
                batch_size=min(args.score_batch_size, len(dataset_train)),
                schedule=schedule,
                max_samples=args.max_samples,
            )
        mod.to(device).eval()
        score_modules.append((k, mod))

    # Output arrays: [nsteps, nk]
    nk = len(k_vals)
    entropy_arr = np.zeros((args.nsteps, nk), dtype=np.float64)
    center_var_arr = np.zeros((args.nsteps, nk), dtype=np.float64)
    test_mse_arr = np.zeros((args.nsteps, nk), dtype=np.float64)
    snr_list = []

    for step_idx in range(args.nsteps):
        t_val = (step_idx + 1) / args.nsteps
        t_cur = torch.tensor([t_val], device=device, dtype=torch.get_default_dtype())
        snr_list.append(float(t_to_snr(t_val, schedule)))

        bt = schedule(t_cur).sqrt().to(device)
        at = (1 - schedule(t_cur)).sqrt().to(device)

        for k_idx, (k, mod) in enumerate(score_modules):
            mse_sum = 0.0
            entropy_sum = 0.0
            center_var_sum = 0.0
            n_pixels = 0
            n_entropy = 0   # number of entropy map elements (b * h * w)
            n_center_var = 0   # number of center_variance map elements (b * c * h * w)
            n_done = 0

            for batch in test_loader:
                if n_done >= ntest:
                    break
                x_0 = batch[0].to(device)
                if isinstance(x_0, list):
                    x_0 = x_0[0]
                b = x_0.shape[0]
                if n_done + b > ntest:
                    x_0 = x_0[: ntest - n_done]
                    b = x_0.shape[0]
                n_done += b

                with torch.no_grad():
                    eps = torch.randn_like(x_0, device=device)
                    x_t = at * x_0 + bt * eps

                    out = mod.forward_with_posterior_stats(t_cur, x_t, device=device, k=k)
                    score, E_x0, entropy_map, center_variance_map, center_binder_map, patch_var, patch_binder = out

                    mse_sum += ((E_x0 - x_0) ** 2).sum().item()
                    entropy_sum += entropy_map.sum().item()
                    center_var_sum += center_variance_map.sum().item()
                    n_pixels += x_0.numel()
                    n_entropy += entropy_map.numel()
                    n_center_var += center_variance_map.numel()

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if n_pixels > 0:
                test_mse_arr[step_idx, k_idx] = mse_sum / n_pixels
                entropy_arr[step_idx, k_idx] = entropy_sum / n_entropy if n_entropy > 0 else np.nan
                center_var_arr[step_idx, k_idx] = center_var_sum / n_center_var if n_center_var > 0 else np.nan
            else:
                test_mse_arr[step_idx, k_idx] = np.nan
                entropy_arr[step_idx, k_idx] = np.nan
                center_var_arr[step_idx, k_idx] = np.nan

    sm = args.score_module
    result = {
        "dataset": args.dataset,
        "score_module": sm,
        "nsteps": args.nsteps,
        "ntest": ntest,
        "k_vals": k_vals,
        "snr": snr_list,
        "step_idx": list(range(args.nsteps)),
        f"entropy_{sm}": entropy_arr.tolist(),
        f"reg_{sm}": center_var_arr.tolist(),
        "test_mse": test_mse_arr.tolist(),
    }
    torch.save(result, args.out_file)
    json_path = args.out_file.replace(".pt", ".json")
    with open(json_path, "w") as f:
        skip_keys = (f"entropy_{sm}", f"reg_{sm}", "test_mse")
        meta = {k: v for k, v in result.items() if k not in skip_keys}
        json.dump(meta, f, indent=2)
    print(f"Saved to {args.out_file} and {json_path}")


if __name__ == "__main__":
    main()
