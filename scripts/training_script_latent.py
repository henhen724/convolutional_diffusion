"""
Training script for diffusion models on VAE-encoded latent datasets.

Mirrors training_script.py but loads pre-encoded latents instead of raw images.
The latent space from SDXL VAE is 4-channel, 32x32 for 256x256 input images.

Usage:
    # Train ResNet on CelebA-HQ latents
    python scripts/training_script_latent.py \
        --latent_path /scratch/users/hshunt/celeba_hq_latents/celeba_hq_latents.pt \
        --resnet --epochs 500 --batchsize 128

    # Train UNet on CelebA-HQ latents
    python scripts/training_script_latent.py \
        --latent_path /scratch/users/hshunt/celeba_hq_latents/celeba_hq_latents.pt \
        --epochs 500 --batchsize 128
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.models import DDIM, MinimalResNet, MinimalUNet
from src.utils.latent_data import LatentDataset
from src.utils.noise_schedules import cosine_noise_schedule
from src.utils.train import train_diffusion


def main():
    parser = argparse.ArgumentParser(description='DDIM training on VAE latents')
    parser.add_argument('--latent_path', type=str, required=True,
                        help='Path to the .pt file with encoded latents')
    parser.add_argument('--stats_path', type=str, default=None,
                        help='Path to latent normalization stats (auto-detected if not set)')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--mode', type=str, default='zeros')
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--mult', type=int, default=2)
    parser.add_argument('--nonorm', action="store_true", default=True)
    parser.add_argument('--saveinterval', type=int, default=25)
    parser.add_argument('--layers', type=int, default=None,
                        help='Number of layers (default: 8 for ResNet, 3 for UNet)')
    parser.add_argument('--resnet', action="store_true", default=False)
    parser.add_argument('--homedir', type=str, default='./checkpoints')
    parser.add_argument('--gamma', type=float, default=0.999965)

    args = parser.parse_args()

    # Auto-detect stats path
    if args.stats_path is None:
        stats_path = os.path.join(os.path.dirname(args.latent_path),
                                   'celeba_hq_latent_stats.pt')
        if os.path.exists(stats_path):
            args.stats_path = stats_path

    # Load latent dataset
    dataset = LatentDataset(args.latent_path, stats_path=args.stats_path, normalize=True)
    metadata = dataset.get_metadata()

    # Set default layers to match existing CelebA checkpoints
    if args.layers is None:
        args.layers = 8 if args.resnet else 3

    train_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True,
                              num_workers=0, pin_memory=True)

    # Build filename
    arch = 'ResNet' if args.resnet else 'UNet'
    fname = os.path.join(args.homedir,
                         f'backbone_CelebAHQ_latent_{arch}_{args.mode}')
    if args.nonorm:
        fname += '_nonorm'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    normal = None if args.nonorm else 'GroupNorm'

    num_channels = metadata['num_channels']  # 4 for SDXL VAE latents
    image_size = metadata['image_size']      # 32

    print(f"\n{'='*60}")
    print(f"Training {arch} on CelebA-HQ latents")
    print(f"  Latent shape: ({num_channels}, {image_size}, {image_size})")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: {args.batchsize}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {device}")
    print(f"  Output prefix: {fname}")
    print(f"{'='*60}\n")

    if args.resnet:
        backbone = MinimalResNet(
            channels=num_channels,
            emb_dim=128 * args.mult,  # 256, matching CelebA checkpoint
            mode=args.mode,
            conditional=False,
            num_classes=metadata['num_classes'],
            kernel_size=3,
            num_layers=args.layers,  # 8, matching CelebA checkpoint
            normalization=normal,
            lastksize=3,
        )
    else:
        backbone = MinimalUNet(
            channels=num_channels,
            fsizes=[args.mult * 32 * (2 ** i) for i in range(args.layers)],  # [64, 128, 256]
            mode=args.mode,
            conditional=False,
            num_classes=metadata['num_classes'],
            normalization=normal,
            lastksize=3,
        )

    model = DDIM(
        pretrained_backbone=backbone,
        default_imsize=image_size,
        in_channels=num_channels,
        noise_schedule=cosine_noise_schedule,
    )
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.2f}M")

    train_diffusion(
        model, train_loader, cosine_noise_schedule, device,
        max_t=1000,
        num_epochs=args.epochs,
        lr=args.lr,
        in_channels=num_channels,
        gamma=args.gamma,
        fname=fname,
        conditional=False,
        save_interval=args.saveinterval,
        wd=args.wd,
    )

    # Save final model
    final_path = fname + '_final.pt'
    torch.save(model, final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
