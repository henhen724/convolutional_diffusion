"""
Encode CelebA-HQ 256x256 images through the SDXL VAE to create a latent dataset.

The SDXL VAE downsamples by 8x and produces 4-channel latents:
    256x256x3 (pixel) -> 32x32x4 (latent)

Output: A single .pt file containing:
    - 'latents': tensor of shape (N, 4, 32, 32)
    - 'metadata': dict with dataset info

Usage:
    python scripts/encode_celebahq_latents.py --outdir /scratch/users/hshunt/celeba_hq_latents
"""

import argparse
import os
import glob

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL


class CelebAHQImageFolder(Dataset):
    """Simple image folder dataset for CelebA-HQ."""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # Collect all image files
        self.image_paths = sorted(
            glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
            + glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
        )
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root}")
        print(f"Found {len(self.image_paths)} images in {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


def download_celebahq(data_dir):
    """Download CelebA-HQ 256x256 from HuggingFace."""
    from huggingface_hub import snapshot_download

    img_dir = os.path.join(data_dir, "celebahq256")
    if os.path.exists(img_dir) and len(glob.glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True)) > 1000:
        print(f"CelebA-HQ already downloaded at {img_dir}")
        return img_dir

    print("Downloading CelebA-HQ 256x256 from HuggingFace...")
    snapshot_download(
        repo_id="datasets/tglcourse/CelebA-faces-cropped-128",
        repo_type="dataset",
        local_dir=os.path.join(data_dir, "celebahq_tmp"),
        allow_patterns=["*.jpg", "*.png"],
    )
    # Check if it downloaded properly
    tmp_dir = os.path.join(data_dir, "celebahq_tmp")
    n = len(glob.glob(os.path.join(tmp_dir, "**", "*.jpg"), recursive=True))
    if n > 0:
        print(f"Downloaded {n} images to {tmp_dir}")
        return tmp_dir

    # Fallback: try a different dataset
    print("Trying alternative: HuggingFace AFHQ or another CelebA-HQ source...")
    raise RuntimeError("Could not download CelebA-HQ. Please download manually.")


def download_celebahq_kaggle(data_dir):
    """Alternative: use gdown for the commonly-shared CelebA-HQ 256x256 zip."""
    import gdown

    # Check common extraction paths (the zip may extract to different directory names)
    candidate_dirs = [
        os.path.join(data_dir, "celeba_hq_256"),
        os.path.join(data_dir, "CelebAMask-HQ", "CelebA-HQ-img"),
        os.path.join(data_dir, "CelebA-HQ-img"),
    ]
    for candidate in candidate_dirs:
        if os.path.exists(candidate) and len(glob.glob(os.path.join(candidate, "**", "*.jpg"), recursive=True)) > 1000:
            print(f"CelebA-HQ already present at {candidate}")
            return candidate

    # CelebA-HQ 256x256 on Google Drive (commonly shared)
    url = "https://drive.google.com/uc?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv"
    zip_path = os.path.join(data_dir, "celeba_hq_256.zip")

    os.makedirs(data_dir, exist_ok=True)
    print("Downloading CelebA-HQ 256x256...")
    gdown.download(url, zip_path, quiet=False)

    print("Extracting...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)

    os.remove(zip_path)

    # Find the actual image directory after extraction
    for candidate in candidate_dirs:
        if os.path.exists(candidate) and len(glob.glob(os.path.join(candidate, "**", "*.jpg"), recursive=True)) > 100:
            print(f"Found images at {candidate}")
            return candidate

    # Fallback: search recursively for any directory with many .jpg files
    for root, dirs, files in os.walk(data_dir):
        jpgs = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
        if len(jpgs) > 1000:
            print(f"Found {len(jpgs)} images at {root}")
            return root

    raise RuntimeError(f"Could not find extracted images under {data_dir}")


@torch.no_grad()
def encode_dataset(vae, dataloader, device):
    """Encode all images through the VAE encoder."""
    all_latents = []
    vae.eval()

    for batch in tqdm(dataloader, desc="Encoding images"):
        batch = batch.to(device)
        # VAE expects images in [-1, 1]
        latent_dist = vae.encode(batch).latent_dist
        # Use the mean (deterministic encoding)
        latents = latent_dist.mean
        # Apply the VAE scaling factor (0.13025 for SDXL)
        latents = latents * vae.config.scaling_factor
        all_latents.append(latents.cpu())

    return torch.cat(all_latents, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Encode CelebA-HQ through SDXL VAE")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory for latent dataset")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory to download/find CelebA-HQ images")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(args.outdir, "raw")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Get the images
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    try:
        img_dir = download_celebahq(args.data_dir)
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        print("Trying gdown fallback...")
        img_dir = download_celebahq_kaggle(args.data_dir)

    # Step 2: Load VAE
    print("Loading SDXL VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=torch.float32,
    ).to(device)
    print(f"VAE loaded. Scaling factor: {vae.config.scaling_factor}")

    # Step 3: Create dataset and dataloader
    # VAE expects images normalized to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),           # [0, 1]
        transforms.Normalize([0.5]*3, [0.5]*3),  # [-1, 1]
    ])

    dataset = CelebAHQImageFolder(img_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Step 4: Encode
    print(f"Encoding {len(dataset)} images at {args.image_size}x{args.image_size}...")
    latents = encode_dataset(vae, dataloader, device)
    print(f"Encoded latents shape: {latents.shape}")
    print(f"Latent stats: mean={latents.mean():.4f}, std={latents.std():.4f}, "
          f"min={latents.min():.4f}, max={latents.max():.4f}")

    # Step 5: Save
    out_path = os.path.join(args.outdir, "celeba_hq_latents.pt")
    save_dict = {
        "latents": latents,
        "metadata": {
            "name": "CelebAHQ_latent",
            "source_image_size": args.image_size,
            "latent_size": latents.shape[-1],  # 32
            "num_channels": latents.shape[1],   # 4
            "num_images": latents.shape[0],
            "vae": "stabilityai/sdxl-vae",
            "scaling_factor": vae.config.scaling_factor,
            "encoding": "mean",  # deterministic, not sampled
        },
    }
    torch.save(save_dict, out_path)
    print(f"Saved latent dataset to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1e6:.1f} MB")

    # Also compute and save normalization statistics for training
    latent_mean = latents.mean(dim=0)  # (4, 32, 32)
    latent_std = latents.std(dim=0)    # (4, 32, 32)
    channel_mean = latents.mean(dim=(0, 2, 3))  # (4,)
    channel_std = latents.std(dim=(0, 2, 3))    # (4,)
    print(f"Per-channel mean: {channel_mean}")
    print(f"Per-channel std:  {channel_std}")

    stats_path = os.path.join(args.outdir, "celeba_hq_latent_stats.pt")
    torch.save({
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "spatial_mean": latent_mean,
        "spatial_std": latent_std,
    }, stats_path)
    print(f"Saved normalization stats to {stats_path}")


if __name__ == "__main__":
    main()
