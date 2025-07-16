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
	return train_set, metadata


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
