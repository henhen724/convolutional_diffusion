"""
Dataset utilities for loading pre-encoded VAE latent datasets.
"""

import os
import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    """
    Dataset of pre-encoded VAE latents.

    Loads a .pt file containing:
        - 'latents': tensor of shape (N, C, H, W)
        - 'metadata': dict with dataset info

    Applies per-channel normalization to zero mean, unit variance
    using precomputed statistics.
    """

    def __init__(self, latent_path, stats_path=None, normalize=True):
        data = torch.load(latent_path, map_location='cpu', weights_only=False)
        self.latents = data['latents']
        self.metadata = data['metadata']
        self.normalize = normalize

        if normalize and stats_path is not None:
            stats = torch.load(stats_path, map_location='cpu', weights_only=False)
            self.channel_mean = stats['channel_mean']  # (C,)
            self.channel_std = stats['channel_std']      # (C,)
        elif normalize:
            # Compute stats from the data itself
            self.channel_mean = self.latents.mean(dim=(0, 2, 3))
            self.channel_std = self.latents.std(dim=(0, 2, 3))
        else:
            self.channel_mean = None
            self.channel_std = None

        if normalize:
            # Normalize in-place to save memory
            self.latents = (
                (self.latents - self.channel_mean[None, :, None, None])
                / self.channel_std[None, :, None, None]
            )

        print(f"Loaded {len(self.latents)} latents, shape {self.latents.shape}")
        if normalize:
            print(f"  Normalized: mean={self.latents.mean():.4f}, std={self.latents.std():.4f}")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        # Return (latent, dummy_label) to match the (image, label) convention
        # used by the existing training loop
        return self.latents[idx], 0

    def get_metadata(self):
        """Return metadata dict compatible with the training script."""
        return {
            'name': self.metadata.get('name', 'CelebAHQ_latent'),
            'image_size': self.metadata.get('latent_size', 32),
            'num_classes': 1,
            'train_images': len(self.latents),
            'val_images': 0,
            'num_channels': self.metadata.get('num_channels', 4),
            'mean': [0.0] * self.metadata.get('num_channels', 4),
            'std': [1.0] * self.metadata.get('num_channels', 4),
        }
