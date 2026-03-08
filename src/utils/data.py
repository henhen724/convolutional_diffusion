import json
import os

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class ToyGaussianFieldDataset(Dataset):
	def __init__(self, root='./data', train=True):
		base_dir = os.path.join(root, 'toy_gaussian_field')
		metadata_path = os.path.join(base_dir, 'metadata.json')
		if not os.path.isfile(metadata_path):
			raise FileNotFoundError(
				f"Toy Gaussian metadata not found at {metadata_path}. "
				"Generate it first with scripts/generate_toy_gaussian_field_dataset.py"
			)

		with open(metadata_path, 'r', encoding='utf-8') as f:
			self.metadata = json.load(f)

		split_name = 'train' if train else 'valid'
		filename = self.metadata.get(f'{split_name}_file', f'{split_name}_data.npy')
		self.path = os.path.join(base_dir, filename)
		if not os.path.isfile(self.path):
			raise FileNotFoundError(
				f"Toy Gaussian split file not found at {self.path}. "
				"Regenerate with scripts/generate_toy_gaussian_field_dataset.py"
			)
		self.data = np.load(self.path, mmap_mode='r')

	def __len__(self):
		return int(self.data.shape[0])

	def __getitem__(self, idx):
		# Copy to materialize a standalone tensor for DataLoader workers.
		x = torch.from_numpy(np.array(self.data[idx], dtype=np.float32, copy=True))
		label = torch.tensor(0, dtype=torch.long)
		return x, label


def get_dataset(name, root='./data', dirname=None, train=True):
	# Normalize name to lowercase for consistent matching
	name_lower = name.lower()
	
	metadata = get_metadata(name)

	transform = transforms.Compose([
		transforms.Resize((metadata['image_size'], metadata['image_size'])),
		transforms.ToTensor(),
		transforms.Normalize(mean=metadata['mean'], std=metadata['std'])  # Normalize the images
	])

	if name_lower == 'mnist':
		train_set = datasets.MNIST(
			root=root,
			train=train,
			download=True,
			transform=transform
		)
	elif name_lower == 'cifar10':
		train_set = datasets.CIFAR10(
			root=root,
			train=train,
			download=True,
			transform=transform
		)
	elif name_lower == 'fashionmnist' or name_lower == 'fashion_mnist':
		train_set = datasets.FashionMNIST(
			root=root,
			train=train,
			download=True,
			transform=transform
		)
	elif name_lower == 'celeba':
		train_set = datasets.CelebA(
			root=root,
			split='train' if train else 'valid',
			download=True,
			transform=transforms.Compose([
				transforms.Resize((32, 32)),
				transforms.ToTensor(),
				transforms.Normalize(mean=metadata['mean'], std=metadata['std'])
			])
		)
	elif name_lower in ('toy_gaussian_field', 'toygaussianfield', 'toy_gaussian'):
		train_set = ToyGaussianFieldDataset(root=root, train=train)
		# Prefer generated metadata at runtime for exact sample counts/config.
		try:
			metadata_path = os.path.join(root, 'toy_gaussian_field', 'metadata.json')
			with open(metadata_path, 'r', encoding='utf-8') as f:
				generated = json.load(f)
			metadata.update({
				'name': generated.get('name', metadata['name']),
				'image_size': int(generated.get('image_size', metadata['image_size'])),
				'num_channels': int(generated.get('num_channels', metadata['num_channels'])),
				'train_images': int(generated.get('n_train', metadata['train_images'])),
				'val_images': int(generated.get('n_valid', metadata['val_images'])),
				'mean': generated.get('mean', metadata['mean']),
				'std': generated.get('std', metadata['std']),
				'amplitude': generated.get('amplitude', None),
				'alpha': generated.get('alpha', None),
			})
		except Exception:
			# Fall back to static metadata if metadata file is unavailable/unreadable.
			pass
	else:
		raise ValueError(f"Unknown dataset: {name}")
		
	return train_set, metadata


def get_metadata(name):
	# Normalize name to lowercase for consistent matching
	name = name.lower()
	
	if name == "mnist":
		metadata = {
				"name":'mnist',
				"image_size": 32, # resized MNIST to be 32 instead of 28
				"num_classes": 10,
				"train_images": 60000,
				"val_images": 10000,
				"num_channels": 1,
				"mean": [0.5],
				"std": [0.5]
			}
		
	elif name == "cifar10":
		metadata = {
				"name": 'cifar10',
				"image_size": 32,
				"num_classes": 10,
				"train_images": 60000,
				"val_images": 10000,
				"num_channels": 3,
				"mean": [0.5, 0.5, 0.5],
				"std": [0.5, 0.5, 0.5]
			}
		
	elif name == "fashionmnist" or name == "fashion_mnist":
		metadata = {
				"name": 'fashion_mnist',
				"image_size": 32,
				"num_classes": 10,
				"train_images": 60000,
				"val_images": 10000,
				"num_channels": 1,
				"mean": [0.5],
				"std": [0.5]
			}
	elif name == "celeba":
		metadata = {
				"name": 'celeba',
				"image_size": 32,
				"num_classes": 1,
				"train_images": 200000,
				"val_images": 0,
				"num_channels": 3,
				"mean": [0.5, 0.5, 0.5],
				"std": [0.5, 0.5, 0.5]
		}
	elif name in ("toy_gaussian_field", "toygaussianfield", "toy_gaussian"):
		metadata = {
				"name": 'toy_gaussian_field',
				"image_size": 32,
				"num_classes": 1,
				"train_images": 200000,
				"val_images": 10000,
				"num_channels": 3,
				# Data are generated in model-space already; no extra torchvision normalization.
				"mean": [0.0, 0.0, 0.0],
				"std": [1.0, 1.0, 1.0],
				"alpha": 3.0,
				"amplitude": None
		}
	else:
		# Default metadata for unknown datasets
		metadata = {
				"name": name,
				"image_size": 32,
				"num_classes": 1,
				"train_images": 0,
				"val_images": 0,
				"num_channels": 3,
				"mean": [0.5, 0.5, 0.5],
				"std": [0.5, 0.5, 0.5]
		}

	return metadata
