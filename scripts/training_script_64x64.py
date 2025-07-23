import argparse
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.models import DDIM, MinimalResNet, MinimalUNet
from src.utils.data import get_metadata
from src.utils.noise_schedules import cosine_noise_schedule
from src.utils.train import train_diffusion


def get_dataset_64x64(name, root='./data', train=True):
    """Modified dataset loader for 64x64 resolution."""
    name_lower = name.lower()
    
    # Get base metadata but override image size
    metadata = get_metadata(name)
    metadata['image_size'] = 64  # Override to 64x64
    
    # Create transform for 64x64
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=metadata['mean'], std=metadata['std'])
    ])
    
    if name_lower == 'celeba':
        dataset = datasets.CelebA(
            root=root,
            split='train' if train else 'valid',
            download=True,
            transform=transform
        )
    elif name_lower == 'cifar10':
        dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    elif name_lower == 'mnist':
        dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    elif name_lower == 'fashionmnist' or name_lower == 'fashion_mnist':
        dataset = datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset for 64x64: {name}")
    
    return dataset, metadata


def main():

	parser = argparse.ArgumentParser(description='DDIM training for 64x64 resolution')
	parser.add_argument('--epochs', type=int, default=300)
	parser.add_argument('--batchsize', type=int, default=64)  # Reduced for 64x64 (more memory)
	parser.add_argument('--dataset', type=str)
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--conditional', action="store_true", default=False)
	parser.add_argument('--mode', type=str, default='zeros')  # Match existing CelebA checkpoints
	parser.add_argument('--wd', type=float, default=0)
	parser.add_argument('--mult', type=int, default=2)
	parser.add_argument('--nonorm', action="store_true", default=True)
	parser.add_argument('--saveinterval', type=int, default=5)
	parser.add_argument('--layers', type=int, default=4)  # Max 4 layers for 64x64 (64->32->16->8->4)
	parser.add_argument('--resnet', action="store_true", default=False)
	parser.add_argument('--homedir', type=str, default='./checkpoints')  # Use checkpoints dir
	parser.add_argument('--suppress', action="store_true", default=False)
	parser.add_argument('--gamma', type=float, default=0.999965)
	parser.add_argument('--maxsamps', type=int, default=100000)

	args = parser.parse_args()

	# Use 64x64 dataset loader
	dataset, metadata = get_dataset_64x64(args.dataset, root='./data')

	subset_flag = args.maxsamps < len(dataset)
	factor = 1
	if subset_flag:
		factor = len(dataset)//args.maxsamps
		dataset = torch.utils.data.Subset(dataset, [i for i in range(args.maxsamps)])

	train_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)

	# Create filename matching existing convention but with 64x64 suffix
	if args.resnet:
		fname = os.path.join(args.homedir, f'backbone_{metadata["name"].upper()}_ResNet_{args.mode}_64x64')
	else:
		fname = os.path.join(args.homedir, f'backbone_{metadata["name"].upper()}_UNet_{args.mode}_64x64')

	if args.conditional:
		fname += '_conditional'
	if args.nonorm:
		fname += '_nonorm'
	if args.mult != 2:
		fname += '_mult_' + str(args.mult)
	if subset_flag:
		fname += '_maxsamps_' + str(args.maxsamps)

	fname += '.pt'

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	normal = None if args.nonorm else 'GroupNorm'

	print(f"Training {args.dataset.upper()} {'ResNet' if args.resnet else 'UNet'} at 64x64 resolution")
	print(f"Output file: {fname}")
	print(f"Device: {device}")
	print(f"Dataset size: {len(dataset)} samples")
	print(f"Batch size: {args.batchsize}")
	print(f"Epochs: {args.epochs}")

	if args.resnet:
		backbone = MinimalResNet(channels=metadata['num_channels'],
								emb_dim=128*args.mult,
								mode=args.mode,
								conditional=args.conditional,
								num_classes=metadata['num_classes'],
								kernel_size=3,
								num_layers=args.layers,
								normalization=normal,
								lastksize=3)
	else: # UNET
		# More reasonable feature sizes for 64x64: [64, 128, 256, 512] instead of exponential growth
		if args.layers <= 4:
			fsizes = [64, 128, 256, 512][:args.layers]
		else:
			fsizes = [args.mult*32*(2**i) for i in range(args.layers)]
		
		backbone = MinimalUNet(channels=metadata['num_channels'],
								fsizes=fsizes,
								mode=args.mode,
								conditional=args.conditional,
								num_classes=metadata['num_classes'],
								normalization=normal,
								lastksize=3)

	model = DDIM(pretrained_backbone=backbone,
				default_imsize=64,  # 64x64 resolution
				in_channels=metadata['num_channels'],
				noise_schedule=cosine_noise_schedule)

	model.to(device)
	
	print("\nModel created successfully!")
	print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
	print("\nStarting training...")

	train_diffusion(model, train_loader, cosine_noise_schedule, device,
					max_t=1000,
					num_epochs=args.epochs*factor,
					lr=args.lr,
					in_channels=metadata['num_channels'],
					gamma=args.gamma,
					fname=fname,
					conditional=args.conditional,
					save_interval=args.saveinterval*factor,
					wd=args.wd)

	print(f"\nTraining completed! Model saved to {fname}")


if __name__ == "__main__":
	main() 