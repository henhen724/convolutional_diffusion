import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def get_dataset(name, root='./data', dirname=None, train=True):

	metadata = get_metadata(name)

	transform = transforms.Compose([
		transforms.Resize((metadata['image_size'], metadata['image_size'])),
		transforms.ToTensor(),
		transforms.Normalize(mean=metadata['mean'], std=metadata['std'])  # Normalize the images
	])

	if name == 'mnist':
		train_set = datasets.MNIST(
			root=root,
			train=train,
			download=True,
			transform=transform
		)
	elif name == 'cifar10':
		train_set = datasets.CIFAR10(
			root=root,
			train=train,
			download=True,
			transform=transform
		)
	elif name == 'fashion_mnist':
		train_set = datasets.FashionMNIST(
			root=root,
			train=train,
			download=True,
			transform=transform
		)
	elif name == 'celeba':
		if dirname is not None:
			celeba_dir = dirname
		else:
			celeba_dir = prepare_celeba_32x32(root, train, transform)
			dirname = celeba_dir

		train_set = datasets.ImageFolder(
			root=celeba_dir,
			transform=transform
		)
	return train_set, metadata

def prepare_celeba_32x32(root, train, transform):
			import os

			from torchvision.datasets import CelebA
			from torchvision.transforms import ToPILImage

			save_dir = os.path.join(root, "celeba32")
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
				class_dir = os.path.join(save_dir, "all")
				os.makedirs(class_dir)
				celeba_raw = datasets.CelebA(
					root=root,
					split='train' if train else 'test',
					download=True,
					transform=transforms.Compose([
						transforms.Resize((32, 32)),
						transforms.ToTensor()
					])
				)
				for idx in range(len(celeba_raw)):
					img, _ = celeba_raw[idx]
					img_pil = ToPILImage()(img)
					img_pil.save(os.path.join(class_dir, f"{idx:06d}.png"))
			return save_dir

def get_metadata(name):
	
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
		
	elif name == "fashion_mnist":
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

	return metadata
