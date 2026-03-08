import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import DDIM
from src.utils.data import get_dataset


def sample_subset(dataset, n, seed=0):
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=rng)[:n]
    xs = []
    for i in idx.tolist():
        x, *_ = dataset[i]
        xs.append(x)
    return torch.stack(xs, dim=0)


def compute_input_gradients_single_pixel(
    model: DDIM,
    xs: torch.Tensor,
    t_scalar: float,
    pixel_i: int,
    pixel_j: int,
    channel_mode: str = "mean",
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    For each input x in xs, compute ∇_x f(x) where f(x) is the scalar
    score at a single pixel (i,j), aggregated over channels by `channel_mode`.

    Returns:
        grads: tensor of shape (N, C*H*W).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    N, C, H, W = xs.shape
    grads = []

    for n in range(N):
        x = xs[n : n + 1].to(device).detach().clone()
        x.requires_grad_(True)

        t = torch.full((1,), float(t_scalar), device=device)
        score = model(t, x)  # (1, C, H, W)

        if channel_mode == "mean":
            f = score[:, :, pixel_i, pixel_j].mean()
        elif channel_mode == "first":
            f = score[:, 0, pixel_i, pixel_j].sum()
        else:
            raise ValueError(f"Unknown channel_mode: {channel_mode}")

        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        f.backward()
        g = x.grad.detach().cpu().reshape(-1)
        grads.append(g)

    return torch.stack(grads, dim=0)  # (N, C*H*W)


def extract_patch_vectors(xs: torch.Tensor, pixel_i: int, pixel_j: int, patch_size: int) -> torch.Tensor:
    """
    Extract (C, patch_size, patch_size) around (pixel_i, pixel_j) for each x and flatten.
    Uses zero-padding if the patch would go out of bounds.
    """
    assert patch_size % 2 == 1, "patch_size must be odd"
    N, C, H, W = xs.shape
    r = patch_size // 2
    # Pad on all sides
    xs_pad = torch.nn.functional.pad(xs, (r, r, r, r), mode="constant", value=0.0)
    # Shift indices due to padding
    pi = pixel_i + r
    pj = pixel_j + r
    patches = xs_pad[:, :, pi - r : pi + r + 1, pj - r : pj + r + 1]  # (N, C, k, k)
    return patches.reshape(N, -1)


def pairwise_sq_dists(X: torch.Tensor) -> torch.Tensor:
    """
    X: (N, D) -> D_ij = ||x_i - x_j||^2, shape (N, N)
    """
    with torch.no_grad():
        XX = (X * X).sum(dim=1, keepdim=True)
        D = XX + XX.t() - 2.0 * (X @ X.t())
        return D


def main():
    parser = argparse.ArgumentParser(
        description="Compute input-gradient kernel (NTK proxy) at a single pixel for a DDIM ResNet."
    )
    parser.add_argument("--dataset", type=str, default="celeba")
    parser.add_argument("--modelfile", type=str, required=True, help="Path to DDIM checkpoint (ResNet backbone).")
    parser.add_argument("--n_train", type=int, default=32)
    parser.add_argument("--n_test", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pixel_i", type=int, default=16, help="Row index of the pixel (0-based).")
    parser.add_argument("--pixel_j", type=int, default=16, help="Column index of the pixel (0-based).")
    parser.add_argument("--patch_size", type=int, default=11, help="Odd patch size around (i,j).")
    parser.add_argument(
        "--time_step",
        type=int,
        default=10,
        help="Discrete timestep index (1..nsteps) used in training; converted to t = time_step/nsteps.",
    )
    parser.add_argument("--nsteps", type=int, default=20)
    parser.add_argument("--channel_mode", type=str, default="mean", choices=["mean", "first"])
    parser.add_argument(
        "--out_file",
        type=str,
        default=None,
        help="Where to save results (.pt). Default uses results/ntk/ based on checkpoint name.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load dataset
    train_set, metadata = get_dataset(args.dataset, root="./data", train=True)
    test_set, _ = get_dataset(args.dataset, root="./data", train=False)

    # Sample subsets
    xs_train = sample_subset(train_set, args.n_train, seed=args.seed)
    xs_test = sample_subset(test_set, args.n_test, seed=args.seed + 1)

    # Load model
    ckpt_path = args.modelfile
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(model, DDIM):
        # In case checkpoint only has state_dict, create a DDIM wrapper
        raise TypeError("Expected a DDIM checkpoint with a ResNet backbone.")

    # Time scalar t in [0,1], using the same discretization as training
    t_scalar = float(args.time_step) / float(args.nsteps)

    # Compute input gradients for train and test subsets
    grads_train = compute_input_gradients_single_pixel(
        model,
        xs_train,
        t_scalar=t_scalar,
        pixel_i=args.pixel_i,
        pixel_j=args.pixel_j,
        channel_mode=args.channel_mode,
        device=device,
    )
    grads_test = compute_input_gradients_single_pixel(
        model,
        xs_test,
        t_scalar=t_scalar,
        pixel_i=args.pixel_i,
        pixel_j=args.pixel_j,
        channel_mode=args.channel_mode,
        device=device,
    )

    # Input-gradient "NTK" (inner product of ∇_x f(x))
    K_train_train = grads_train @ grads_train.t()
    K_test_test = grads_test @ grads_test.t()
    K_train_test = grads_train @ grads_test.t()

    # Patch-restricted squared distances ||x_patch - x'_patch||^2
    patches_train = extract_patch_vectors(xs_train, args.pixel_i, args.pixel_j, args.patch_size)
    patches_test = extract_patch_vectors(xs_test, args.pixel_i, args.pixel_j, args.patch_size)

    D2_train_train = pairwise_sq_dists(patches_train)
    D2_test_test = pairwise_sq_dists(patches_test)
    # Cross distances
    with torch.no_grad():
        X = patches_train
        Y = patches_test
        XX = (X * X).sum(dim=1, keepdim=True)
        YY = (Y * Y).sum(dim=1, keepdim=True)
        D2_train_test = XX + YY.t() - 2.0 * (X @ Y.t())

    # Prepare output path
    if args.out_file is None:
        ckpt_name = Path(ckpt_path).stem
        out_dir = Path("results") / "ntk"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"patch_ntk_{args.dataset}_{ckpt_name}_t{args.time_step}_k{args.patch_size}.pt"
    else:
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "dataset": args.dataset,
        "metadata": metadata,
        "modelfile": ckpt_path,
        "time_step": args.time_step,
        "nsteps": args.nsteps,
        "t_scalar": t_scalar,
        "pixel_i": args.pixel_i,
        "pixel_j": args.pixel_j,
        "patch_size": args.patch_size,
        "channel_mode": args.channel_mode,
        "K_train_train": K_train_train.cpu(),
        "K_test_test": K_test_test.cpu(),
        "K_train_test": K_train_test.cpu(),
        "D2_train_train": D2_train_train.cpu(),
        "D2_test_test": D2_test_test.cpu(),
        "D2_train_test": D2_train_test.cpu(),
    }
    torch.save(result, out_path)
    print(f"Saved NTK and patch distances to {out_path}")


if __name__ == "__main__":
    main()

