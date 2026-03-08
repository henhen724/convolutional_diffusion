import math
import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.distributions import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .channel_theory import t_to_snr


def denormalize(image, means, stds):
	if len(image.shape) == 3:
		return (image*torch.tensor(stds)[:,None,None]) + torch.tensor(means)[:,None,None]
	return (image*torch.tensor(stds)[None,:,None,None]) + torch.tensor(means)[None,:,None,None]

def denormalize_imshow(image, means, stds):
	image2 = denormalize(image, means, stds)
	if len(image.shape) == 4:
		plt.imshow(image2.detach().numpy()[0,:,:,:].transpose(1,2,0), cmap='gray_r')
	else:
		plt.imshow(image2.detach().numpy().transpose(1,2,0), cmap='gray_r')
	plt.axis('off')
	plt.show()

def circular_convolution_native(input_signal, kernel):
	pad_h = kernel.size(2) // 2
	pad_w = kernel.size(3) // 2
	
	input_padded = F.pad(input_signal, (pad_w, pad_w, pad_h, pad_h), mode='circular')
	
	result = F.conv2d(input_padded, kernel, padding=0)
	
	return result

def exponential_schedule(t):
	return 1 - torch.exp(-2*t)

def linear_noise_schedule(t):
	# returns beta
	return 0.01+0.97*t

def cosine_noise_schedule(t, mode='legacy'):
	# returns beta
	if mode == 'legacy':
		return 1-torch.cos((t) / 1.008 * math.pi / 2) ** 2
	return 1-torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2


class ScheduledScoreMachine(nn.Module):

	def __init__(self, backbone,
					in_channels=3,
					imsize=32,
					default_time_steps=20,
					noise_schedule=cosine_noise_schedule,
					score_backbone=True,
					scales=None,
					**kwargs):

		super().__init__()

		self.backbone = backbone
		self.default_time_steps = default_time_steps
		self.noise_schedule = noise_schedule
		self.in_channels = in_channels
		self.imsize = imsize
		self.score_backbone = score_backbone
		self.scales = scales

	def forward(self, x, nsteps=None, label=None, device=None, visualize=False):
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		x = x.clone()

		if nsteps is None:
			if self.scales is None:
				nsteps = self.default_time_steps
			else:
				nsteps = len(self.scales)

		for i in range(nsteps-1, 0, -1):

			batch_size = x.shape[0]
			t = i*torch.ones(batch_size)/nsteps
			beta_t = self.noise_schedule(t)  # Determine the noise level for the current step
			beta_t = beta_t.to(device)

			k = None if self.scales is None else self.scales[i]
			if label is not None:
				eps = self.backbone(t,x,label=label,device=device,k=k)
			else:
				eps = self.backbone(t,x,device=device,k=k)

			if self.score_backbone:
				eps *= -beta_t**0.5

			if visualize:
				imputed = (x-eps*((beta_t)**0.5))/((1 - beta_t)**0.5)
				denormalize_imshow(imputed,[0.5 for q in range(x.shape[1])], [0.5 for q in range(x.shape[1])])
			
			alpha_t = 1 - beta_t
			beta_t_prev = self.noise_schedule(t - 1/nsteps)
			beta_t_prev = beta_t_prev.to(device)
			alpha_t_prev = 1 - beta_t_prev


			x *= ((alpha_t_prev/alpha_t)**0.5)[:,None,None,None]
			score_correction = ((beta_t_prev[:,None,None,None]**0.5)-((alpha_t_prev/alpha_t)**0.5)[:,None,None,None]*(beta_t[:,None,None,None]**0.5))*eps
			x += score_correction

		return x

	def sample(self, nsteps=None, label=None, device=None):
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		x = torch.randn(1,self.in_channels,self.imsize,self.imsize, device=device)
		return self(x.clone(), nsteps=nsteps, label=label, device=device)


class LocalEquivBordersScoreModule(nn.Module):

	def __init__(self, dataset,
				kernel_size=3,
				batch_size=64,
				image_size=32,
				channels=3,
				schedule=cosine_noise_schedule,
				max_samples=None,
				shuffle=False,
				**kwargs):

		super().__init__()

		self.dataset = dataset
		self.trainloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
		self.batch_size = batch_size
		self.kernel_size = kernel_size
		self.image_size = image_size
		self.schedule = schedule
		self.max_samples = max_samples
		self.local_module = LocalScoreModule(dataset,
							kernel_size=kernel_size,
							image_size=32,
							batch_size=batch_size,
							mode='zeros',
							schedule=schedule,
							max_samples=max_samples)

	def forward(self, t, x, label=None, device=None, k=None):
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		b,c,h,w = x.shape
		if k is None:
			k = self.kernel_size
		if k >= h:
			return self.local_module(t, x, label=label, device=device, k=k)

		bt = (self.schedule(t))**0.5
		at = (1-self.schedule(t))**0.5

		at = at.to(device)
		bt = bt.to(device)
		xpadded = F.pad(x,(k//2,k//2,k//2,k//2), value=0)
		xpatches = F.unfold(xpadded, k, stride=1, padding=0) 
		xnorms = torch.norm(xpatches, dim=1)**2
		xnorms = xnorms.reshape(b, h, w)


		numerator = torch.zeros(x.shape, device=device)
		denominator = torch.zeros(b,h,w, device=device)

		subtraction = torch.zeros(b,h,w, device=device)

		q = 0
		updated = False
		for images, labels in self.trainloader:

			if self.max_samples is not None and q > self.max_samples:
				break

			if label is not None:
				images = images[(labels==label).squeeze(),:,:,:]
			if images.shape[0] == 0:
				q += self.batch_size
				continue


			images = images.to(device)
			labels = labels.to(device)

			bsize = images.shape[0]

			# CORNERS
			# topleft, topright, bottomleft, bottomright

			dk = k + k//2

			padded_xcorners = [F.pad(x[:,:,:k-1,:k-1], (k//2,0,k//2,0)),
								F.pad(x[:,:,:k-1,w-k+1:], (0,k//2,k//2,0)),
								F.pad(x[:,:,h-k+1:,:k-1], (k//2,0,0,k//2)),
								F.pad(x[:,:,h-k+1:,w-k+1:], (0,k//2,0,k//2))]
			

			padded_imcorners = [F.pad(images[:,:,:k-1,:k-1], (k//2,0,k//2,0)),
					F.pad(images[:,:,:k-1,-k+1:], (0,k//2,k//2,0)),
					F.pad(images[:,:,-k+1:,:k-1], (k//2,0,0,k//2)),
					F.pad(images[:,:,-k+1:,-k+1:], (0,k//2,0,k//2))]


			corner_args = []
			corner_vals = []
			lpatch = k-1-k//2

			for i in range(4):
				xpad = padded_xcorners[i]
				ipad = padded_imcorners[i]

				pwise_diffs = xpad[:,None,:,:,:]-at*ipad[None,:,:,:,:] # [b, NP, c, dk, dk]
				pwise_normsquares = torch.sum(pwise_diffs**2, dim=2) # sum over channel dimenions [b, NP, dk, dk]
				
				patches = F.unfold(pwise_normsquares, k, stride=1, padding=0)				
				patches = patches.view(b, bsize, k**2, lpatch, lpatch) # [b, NP, k^2, lpatch, lpatch]
				weight_args = -torch.sum(patches, dim=2)/(2*bt**2) # [b, NP, h, w]

				corner_val = pwise_diffs[:,:,:,k//2:k//2+lpatch,k//2:k//2+lpatch]
				corner_args.append(weight_args)
				corner_vals.append(corner_val)



			# MIDDLE
			middle_patches = F.unfold(images, k, stride=1, padding=0)
			middle_patches = torch.permute(middle_patches, (2,0,1)) # [h*w, 64, k^2 *c]
			middle_patches = middle_patches.reshape(middle_patches.shape[0]*middle_patches.shape[1], c, k, k) # [NP, c, k, k]			
			mpnorms = torch.sum(middle_patches**2, dim=(1,2,3)) # [NP]
			mpcenters = middle_patches[:,:,k//2,k//2] # [NP, c]

			mpdotx = F.conv2d(x, middle_patches, padding='valid')


			center_exp_args = -(xnorms[:,None,k//2:-(k//2),k//2:-(k//2)] - 2*at*mpdotx + (at**2)*mpnorms[None,:,None,None])/(2*bt**2) # [b, NP, h,w]

			center_vals = x[:,None,:,k//2:-(k//2),k//2:-(k//2)] - at*mpcenters[None,:,:,None,None]


			# EDGES
			# top, right, bottom, left
			edge_args = [torch.zeros(b,bsize*(h-2*(k//2)),lpatch,(h-2*(k//2)), device=device) for j in range(4)]
			edge_vals = [torch.zeros(b,bsize*(h-2*(k//2)),c,lpatch,(h-2*(k//2)), device=device) for j in range(4)]


			padded_xedges = [F.pad(x[:,:,:k-1,:], (0,0,k//2,0)),
							F.pad(x[:,:,:,-k+1:],(0,k//2,0,0)).transpose(-2,-1),
							F.pad(x[:,:,-k+1:,:],(0,0,0,k//2)),
							F.pad(x[:,:,:,:k-1],(k//2,0,0,0)).transpose(-2,-1)]
			

			xedge_norms = [xnorms[:,:lpatch,k//2:-(k//2)], xnorms[:,k//2:-(k//2),-lpatch:].transpose(-2,-1), xnorms[:,-lpatch:,k//2:-(k//2)], xnorms[:,k//2:-(k//2),:lpatch].transpose(-2,-1)]
			
			padded_iedges = [F.pad(images[:,:,:k-1,:], (0,0,k//2,0)),
							F.pad(images[:,:,:,-k+1:],(0,k//2,0,0)).transpose(-2,-1),
							F.pad(images[:,:,-k+1:,:],(0,0,0,k//2)),
							F.pad(images[:,:,:,:k-1],(k//2,0,0,0)).transpose(-2,-1)]

			for i in range(4):
				xedge = padded_xedges[i]
				iedge = padded_iedges[i]
				for j in range(lpatch):
					xslice = xedge[:,:,j:k+j,:]
					islice = iedge[:,:,j:k+j,:] # [NP, c, k, L]
					filters = torch.cat([islice[:,:,:,a:a+k] for a in range(islice.shape[-1]-k+1)], dim=0) # [bNP, c, k, k]
					fnorms = torch.sum(filters**2, dim=(1,2,3))


					epnorms = torch.sum(filters**2, dim=(1,2,3)) # [NP]
					epdotx = F.conv2d(xslice, filters, padding='valid') # [b, NP, l]
					exnorms = xedge_norms[i][:,j,:] # [b, l]

					edge_args[i][:,:,j,:] = -(exnorms[:,None,:] - 2*at*epdotx[:,:,0,:] + (at**2)*fnorms[None,:,None])/(2*bt**2)  
					edge_vals[i][:,:,:,j,:] = (xslice[:,None,:,k//2,k//2:-(k//2)]-at*filters[None,:,:,k//2,k//2,None])


			if not updated:
				updated = True
				# Center
				subtraction[:,k//2:-(k//2),k//2:-(k//2)] = torch.amax(center_exp_args, dim=1)

				# Corners
				subtraction[:,:k//2,:k//2] = torch.amax(corner_args[0], dim=1)
				subtraction[:,:k//2,-(k//2):] = torch.amax(corner_args[1], dim=1)
				subtraction[:,-(k//2):,:k//2] = torch.amax(corner_args[2], dim=1)
				subtraction[:,-(k//2):,-(k//2):] = torch.amax(corner_args[3], dim=1)

				# Edges
				subtraction[:,:k//2,k//2:-(k//2)] = torch.amax(edge_args[0], dim=1)
				subtraction[:,k//2:-(k//2),-(k//2):] = torch.amax(edge_args[1].transpose(-2,-1), dim=1)
				subtraction[:,-(k//2):,k//2:-(k//2)] = torch.amax(edge_args[2], dim=1)
				subtraction[:,k//2:-(k//2),:k//2] = torch.amax(edge_args[3].transpose(-2,-1), dim=1)

			else:

				new_subtraction = torch.zeros(subtraction.shape, device=device)

				# Center
				new_subtraction[:,k//2:-(k//2),k//2:-(k//2)] = torch.amax(center_exp_args, dim=1)

				# Corners
				new_subtraction[:,:k//2,:k//2] = torch.amax(corner_args[0], dim=1)
				new_subtraction[:,:k//2,-(k//2):] = torch.amax(corner_args[1], dim=1)
				new_subtraction[:,-(k//2):,:k//2] = torch.amax(corner_args[2], dim=1)
				new_subtraction[:,-(k//2):,-(k//2):] = torch.amax(corner_args[3], dim=1)

				# Edges
				new_subtraction[:,:k//2,k//2:-(k//2)] = torch.amax(edge_args[0], dim=1)
				new_subtraction[:,k//2:-(k//2),-(k//2):] = torch.amax(edge_args[1].transpose(-2,-1), dim=1)
				new_subtraction[:,-(k//2):,k//2:-(k//2)] = torch.amax(edge_args[2], dim=1)
				new_subtraction[:,k//2:-(k//2),:k//2] = torch.amax(edge_args[3].transpose(-2,-1), dim=1)


				delta_subtraction = (new_subtraction>subtraction)*new_subtraction+(subtraction>=new_subtraction)*subtraction
				numerator /= torch.exp(delta_subtraction-subtraction)[:,None,:,:]
				denominator /= torch.exp(delta_subtraction-subtraction)[:,:,:]
				subtraction = delta_subtraction


			# Center
			center_exp_vals = torch.exp(center_exp_args-subtraction[:,None,k//2:-(k//2),k//2:-(k//2)])
			numerator[:,:,k//2:-(k//2),k//2:-(k//2)] += torch.sum(center_exp_vals[:,:,None,:,:]*center_vals ,dim=1)
			denominator[:,k//2:-(k//2),k//2:-(k//2)] += torch.sum(center_exp_vals, dim=1)

			# Corners
			corner_subtractions = [subtraction[:,:k//2,:k//2], subtraction[:,:k//2,-(k//2):], subtraction[:,-(k//2):,:k//2], subtraction[:,-(k//2):,-(k//2):]]
			corner_exp_vals = [torch.exp(corner_args[i]-corner_subtractions[i][:,None,:,:]) for i in range(4)]

			numerator[:,:,:k//2,:k//2] += torch.sum(corner_exp_vals[0][:,:,None,:,:]*corner_vals[0], dim=1)
			numerator[:,:,:k//2,-(k//2):] += torch.sum(corner_exp_vals[1][:,:,None,:,:]*corner_vals[1], dim=1)
			numerator[:,:,-(k//2):,:k//2] += torch.sum(corner_exp_vals[2][:,:,None,:,:]*corner_vals[2], dim=1)
			numerator[:,:,-(k//2):,-(k//2):] += torch.sum(corner_exp_vals[3][:,:,None,:,:]*corner_vals[3], dim=1)

			denominator[:,:k//2,:k//2] += torch.sum(corner_exp_vals[0], dim=1)
			denominator[:,:k//2,-(k//2):] += torch.sum(corner_exp_vals[1], dim=1)
			denominator[:,-(k//2):,:k//2] += torch.sum(corner_exp_vals[2], dim=1)
			denominator[:,-(k//2):,-(k//2):] += torch.sum(corner_exp_vals[3], dim=1)

			# Edges
			edge_subtractions = [subtraction[:,:k//2,k//2:-(k//2)], subtraction[:,k//2:-(k//2),-(k//2):], subtraction[:,-(k//2):,k//2:-(k//2)], subtraction[:,k//2:-(k//2),:k//2]]
			edge_args = [edge_args[0], edge_args[1].transpose(-2,-1), edge_args[2], edge_args[3].transpose(-2,-1)]
			edge_vals = [edge_vals[0], edge_vals[1].transpose(-2,-1), edge_vals[2], edge_vals[3].transpose(-2,-1)]

			edge_exp_vals = [torch.exp(edge_args[i]-edge_subtractions[i][:,None,:,:]) for i in range(4)]

			numerator[:,:,:k//2,k//2:-(k//2)] += torch.sum(edge_exp_vals[0][:,:,None,:,:]*edge_vals[0], dim=1)
			numerator[:,:,k//2:-(k//2),-(k//2):] += torch.sum(edge_exp_vals[1][:,:,None,:,:]*edge_vals[1], dim=1)
			numerator[:,:,-(k//2):,k//2:-(k//2)] += torch.sum(edge_exp_vals[2][:,:,None,:,:]*edge_vals[2], dim=1)
			numerator[:,:,k//2:-(k//2),:k//2] += torch.sum(edge_exp_vals[3][:,:,None,:,:]*edge_vals[3], dim=1)

			denominator[:,:k//2,k//2:-(k//2)] += torch.sum(edge_exp_vals[0], dim=1)
			denominator[:,k//2:-(k//2),-(k//2):] += torch.sum(edge_exp_vals[1], dim=1)
			denominator[:,-(k//2):,k//2:-(k//2)] += torch.sum(edge_exp_vals[2], dim=1)
			denominator[:,k//2:-(k//2),:k//2] += torch.sum(edge_exp_vals[3], dim=1)

			q += self.batch_size

		return -numerator/denominator[:,None,:,:]/bt**2

	def forward_with_posterior_stats(self, t, x, label=None, device=None, k=None):
		"""
		Return bbELS posterior statistics.

		For boundary-broken ELS, we pair the bbELS score (self.forward) with
		posterior statistics computed from the corresponding zero-padding local
		model (self.local_module), which already exposes a numerically stable
		forward_with_posterior_stats implementation.

		entropy_map is in nats (natural log).
		"""
		if k is None:
			k = self.kernel_size
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		x = x.to(device)
		score = self.forward(t, x, label=label, device=device, k=k)
		(
			_,
			E_x0,
			entropy_map,
			center_variance_map,
			center_binder_map,
			patch_variance_map,
			patch_binder_map,
		) = self.local_module.forward_with_posterior_stats(t, x, label=label, device=device, k=k)
		return (
			score,
			E_x0,
			entropy_map,
			center_variance_map,
			center_binder_map,
			patch_variance_map,
			patch_binder_map,
		)


class LocalEquivScoreModule(nn.Module):

	def __init__(self, dataset,
				kernel_size=3,
				batch_size=64,
				image_size=32,
				channels=3,
				schedule=cosine_noise_schedule,
				max_samples=None,
				shuffle=False,
				**kwargs):

		super().__init__()

		self.dataset = dataset
		self.trainloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
		self.batch_size = batch_size
		self.kernel_size = kernel_size
		self.image_size = image_size
		self.schedule = schedule
		self.max_samples = max_samples

	def forward(self, t, x, label=None, device=None, k=None):
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


		b,c,h,w = x.shape
		if k is None:
			k = self.kernel_size

		bt = (self.schedule(t))**0.5
		at = (1-self.schedule(t))**0.5

		bt = bt.to(device)
		at = at.to(device)

		d = k//2

		xpadded = F.pad(x, (d, d, d, d), mode='circular')

		xpatches = F.unfold(xpadded, k, stride=1, padding=0) 
		xnorms = torch.norm(xpatches, dim=1)**2
		xnorms = xnorms.reshape(b, h, w) # [b, h, w] lol

		numerator = torch.zeros(x.shape, device=device)
		denominator = torch.zeros(b,h,w, device=device)

		subtraction = None

		i = 0
		samps = 0
		max_exp_args = None
		next_exp_args = None

		for images, labels in self.trainloader:

			i += images.shape[0]
			if self.max_samples is not None and i > self.max_samples:
				break

			if label is not None:
				images = images[(labels==label).squeeze(),:,:,:]
			if images.shape[0] == 0:
				continue

			images = images.to(device)
			labels = labels.to(device)

			samps += images.shape[0]

			bsize = images.shape[0]
			patches = F.unfold(images, k, stride=1, padding=0)

			patches = torch.permute(patches, (2,0,1)) # [h*w, 64, k^2 *c]
			patches = patches.reshape(patches.shape[0]*patches.shape[1], c, k, k) # [NP, c, k, k]			
			pnorms = torch.sum(patches**2, dim=(1,2,3)) # [NP]
			pcenters = patches[:,:,k//2,k//2] # [NP, c]
			
			pdotx = circular_convolution_native(x, patches)

			exp_args = -(xnorms[:,None,:,:] - 2*at*pdotx + (at**2)*pnorms[None,:,None,None])/(2*bt**2) # [b, NP, h,w]

			if subtraction is None:
				subtraction = torch.amax(exp_args, dim=(0,1), keepdim=True)
			else:
				new_subtraction = torch.amax(exp_args, dim=(0,1), keepdim=True)
				delta_subtraction = (new_subtraction>subtraction)*new_subtraction+(subtraction>=new_subtraction)*subtraction
				numerator /= torch.exp(delta_subtraction-subtraction)
				denominator /= torch.exp(delta_subtraction-subtraction)[:,0,:,:]
				subtraction = delta_subtraction

			exp_vals = torch.exp(exp_args - subtraction) #[b, NP, h, w]
			num_vals = (x[:,None,:,:,:] - at*pcenters[None,:,:,None,None]) #[b,NP,c,h,w]

			numerator += torch.mean(exp_vals[:,:,None,:,:]*num_vals, dim=1)
			denominator += torch.mean(exp_vals, dim=1)

		return -numerator/denominator[:,None,:,:]/bt**2

	def forward_with_posterior_stats(self, t, x, label=None, device=None, k=None):
		"""Returns center-pixel and patch posterior stats.

		Accumulates sums over training samples (not means) so denominator = Z and
		entropy_map = ln Z - (1/Z)*sum(w_n*(ℓ_n - M)) is the discrete posterior entropy in nats,
		0 <= entropy <= ln(N) at each pixel.

		center_variance_map: at each (h,w), posterior variance of the *center pixel* of
		the matching k×k patch. At each k the posterior is over a *different* set of
		patches (all k×k from the training set), so center variance is not guaranteed
		to decrease with k; it can increase if the larger-patch posterior is more
		bimodal or puts mass on more diverse center values.

		Output tuple:
		(score, E_x0, entropy_map,
		 center_variance_map, center_binder_map,
		 patch_variance_map, patch_binder_map)
		"""
		if k is None:
			k = self.kernel_size
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		b, c, h, w = x.shape
		x = x.to(device)
		bt = (self.schedule(t))**0.5
		at = (1 - self.schedule(t))**0.5
		bt, at = bt.to(device), at.to(device)
		d = k // 2
		xpadded = F.pad(x, (d, d, d, d), mode='circular')
		xpatches = F.unfold(xpadded, k, stride=1, padding=0)
		xnorms = torch.norm(xpatches, dim=1)**2
		xnorms = xnorms.reshape(b, h, w)
		numerator = torch.zeros(x.shape, device=device)
		denominator = torch.zeros(b, h, w, device=device)
		num_sq = torch.zeros(x.shape, device=device)
		num_quartic = torch.zeros(x.shape, device=device)
		patch_dim = c * k * k
		num_patch_vec = torch.zeros(b, patch_dim, h, w, device=device)
		num_patch_sqnorm = torch.zeros(b, h, w, device=device)
		num_patch_sqnorm2 = torch.zeros(b, h, w, device=device)
		sum_exp_log = torch.zeros(b, h, w, device=device)
		subtraction = None
		i = 0
		for images, labels in self.trainloader:
			i += images.shape[0]
			if self.max_samples is not None and i > self.max_samples:
				break
			if label is not None:
				images = images[(labels == label).squeeze(), :, :, :]
			if images.shape[0] == 0:
				continue
			images = images.to(device)
			bsize = images.shape[0]
			patches = F.unfold(images, k, stride=1, padding=0)
			# After permute(2,0,1): (L, bsize, c*k*k). Number of patches NP = L*bsize = shape[2]*shape[0] of original
			NP = patches.shape[2] * patches.shape[0]
			patches = patches.permute(2, 0, 1).reshape(NP, c, k, k)
			pnorms = torch.sum(patches**2, dim=(1, 2, 3))
			# Center of each k×k patch: index (k//2, k//2) in layout [NP, c, k, k]
			pcenters = patches[:, :, k//2, k//2]
			pdotx = circular_convolution_native(x, patches)
			exp_args = -(xnorms[:, None, :, :] - 2*at*pdotx + (at**2)*pnorms[None, :, None, None]) / (2*bt**2)
			if subtraction is None:
				subtraction = torch.amax(exp_args, dim=(0, 1), keepdim=True)
			else:
				new_sub = torch.amax(exp_args, dim=(0, 1), keepdim=True)
				delta = torch.maximum(subtraction, new_sub)
				shift = (delta - subtraction)[:, 0, :, :]  # (1, h, w)
				scale = torch.exp(shift)
				z_old = denominator.clone()
				numerator /= scale.unsqueeze(1)
				denominator /= scale
				num_sq /= scale.unsqueeze(1)
				num_quartic /= scale.unsqueeze(1)
				num_patch_vec /= scale.unsqueeze(1)
				num_patch_sqnorm /= scale
				num_patch_sqnorm2 /= scale
				# A_new = (A_old - shift*Z_old)/scale; use Z_old before rescale for stability
				sum_exp_log = (sum_exp_log - shift * z_old) / scale
				subtraction = delta
			exp_vals = torch.exp(exp_args - subtraction)
			patch_vec = patches.reshape(NP, patch_dim)
			patch_sqnorm = torch.sum(patch_vec**2, dim=1)
			num_vals = (x[:, None, :, :, :] - at * pcenters[None, :, :, None, None])
			# Sum over training samples so denominator = Z, entropy = log Z - (1/Z)*sum_exp_log
			numerator += torch.sum(exp_vals[:, :, None, :, :] * num_vals, dim=1)
			denominator += torch.sum(exp_vals, dim=1)
			num_sq += torch.sum(exp_vals[:, :, None, :, :] * (pcenters[None, :, :, None, None]**2), dim=1)
			num_quartic += torch.sum(exp_vals[:, :, None, :, :] * (pcenters[None, :, :, None, None]**4), dim=1)
			num_patch_vec += torch.sum(exp_vals[:, :, None, :, :] * patch_vec[None, :, :, None, None], dim=1)
			num_patch_sqnorm += torch.sum(exp_vals * patch_sqnorm[None, :, None, None], dim=1)
			num_patch_sqnorm2 += torch.sum(exp_vals * (patch_sqnorm[None, :, None, None]**2), dim=1)
			sum_exp_log += torch.sum(exp_vals * (exp_args - subtraction), dim=1)
		denom = denominator[:, None, :, :].clamp(min=1e-8)
		E_x0 = (x - numerator / denom) / at
		score = -numerator / denom / bt**2
		entropy_map = (torch.log(denominator.clamp(min=1e-8)) - sum_exp_log / denominator.clamp(min=1e-8)).clamp(min=0.0)
		E_x0_sq = num_sq / denom
		E_x0_quartic = num_quartic / denom
		center_variance_map = (E_x0_sq - E_x0**2).clamp(min=0)
		center_binder_map = 1 - E_x0_quartic / (3 * (E_x0_sq.clamp(min=1e-8)**2))
		denom_scalar = denominator.clamp(min=1e-8)
		E_patch_vec = num_patch_vec / denom[:, :, :, :]
		E_patch_sqnorm = num_patch_sqnorm / denom_scalar
		E_patch_sqnorm2 = num_patch_sqnorm2 / denom_scalar
		patch_variance_map = (E_patch_sqnorm - torch.sum(E_patch_vec**2, dim=1)).clamp(min=0)
		patch_binder_map = 1 - E_patch_sqnorm2 / (3 * (E_patch_sqnorm.clamp(min=1e-8)**2))
		return (
			score,
			E_x0,
			entropy_map,
			center_variance_map,
			center_binder_map,
			patch_variance_map,
			patch_binder_map,
		)


class LocalScoreModule(nn.Module):

	def __init__(self, dataset,
				kernel_size=3,
				image_size=32,
				batch_size=256,
				show_plots=False,
				schedule=exponential_schedule,
				max_samples=None,
				**kwargs):

		super().__init__()
		self.dataset = dataset
		self.trainloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
		self.batch_size = batch_size
		self.kernel_size = kernel_size
		self.image_size = image_size
		self.show_plots = show_plots
		self.schedule = schedule
		self.max_samples = max_samples

	def forward(self, t, x, label=None, device=None, k=None):
		if k is None:
			k = self.kernel_size

		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		x = x.to(device)

		b,c,h,w = x.shape
		bt = (self.schedule(t))**0.5
		at = (1-self.schedule(t))**0.5

		at = at.to(device)
		bt = bt.to(device)

		numerator = torch.zeros(x.shape, device=device)
		denominator = torch.zeros(b,h,w, device=device)

		subtraction = None

		i = 0


		for images, labels in self.trainloader:

			if label is not None:
				images = images[(labels==label).squeeze(),:,:,:]

			if images.shape[0] == 0:
				continue

			images = images.to(device)
			labels = labels.to(device)
			bsize = images.shape[0]

			i += bsize
			if self.max_samples is not None and i > self.max_samples:
				break

			# b = number of input images in a batch, NT = number of training images, c,h,w - standard image dimensions
            # The index None cause pytorch to broad cast along that dimension, so this take the parwise difference between the input image and every training image.
			pwise_diffs = x[:,None,:,:,:]-at*images[None,:,:,:,:] # [b, NT, c, h, w]
            # take the square sum of the color channel
			pwise_normsquares = torch.sum(pwise_diffs**2, dim=2) # [b, NT, h, w]
            # This is the critcal step. F.unfold does not all do what you expect. See docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html
            # cutting to the point, the final index becomes each pixel,
            # and the second to last index becomes pixels from training images in a k x k patch 
            # around the pixel listed in the final index
			patches = F.unfold(pwise_normsquares, k, stride=1, padding=k//2) # [b, NT*k^2, h*w]
            # This command actually "folds" the trainset patches, so NT is a trainset image 
            # and the k^2 index represents the relative pixel position to the center pixel (h,w).
			patches = patches.view(b, bsize, k**2, h, w) # [b, NT, k^2, h, w]
            # now we take the sum of the square difference between the patches.
			exp_args = -torch.sum(patches, dim=2)/(2*bt**2) # [b, NT, h, w]

			if subtraction is None:
				subtraction = torch.amax(exp_args, dim=(0,1), keepdim=True)
			else:
				new_subtraction = torch.amax(exp_args, dim=(0,1), keepdim=True)
				delta_subtraction = (new_subtraction>subtraction)*new_subtraction+(subtraction>=new_subtraction)*subtraction
				numerator /= torch.exp(delta_subtraction-subtraction)
				denominator /= torch.exp(delta_subtraction-subtraction)[:,0,:,:]
				subtraction = delta_subtraction

			exp_vals = torch.exp(exp_args - subtraction) #[b, NP, h, w]
			numerator += torch.mean(exp_vals[:,:,None,:,:]*pwise_diffs, dim=1)
			denominator += torch.mean(exp_vals, dim=1)

		return -numerator/denominator/bt**2

	def forward_with_posterior_stats(self, t, x, label=None, device=None, k=None):
		"""Returns center-pixel and patch posterior stats.

		Accumulates sums over training samples (not means) so denominator = Z and
		entropy_map = ln Z - (1/Z)*sum(w_n*(ℓ_n - M)) is the discrete posterior entropy in nats,
		0 <= entropy <= ln(N) at each pixel.

		Output tuple:
		(score, E_x0, entropy_map,
		 center_variance_map, center_binder_map,
		 patch_variance_map, patch_binder_map)
		"""
		if k is None:
			k = self.kernel_size
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		x = x.to(device)
		b, c, h, w = x.shape
		bt = (self.schedule(t))**0.5
		at = (1 - self.schedule(t))**0.5
		at, bt = at.to(device), bt.to(device)
		numerator = torch.zeros(x.shape, device=device)
		denominator = torch.zeros(b, h, w, device=device)
		num_sq = torch.zeros(x.shape, device=device)
		num_quartic = torch.zeros(x.shape, device=device)
		patch_dim = c * k * k
		num_patch_vec = torch.zeros(b, patch_dim, h, w, device=device)
		num_patch_sqnorm = torch.zeros(b, h, w, device=device)
		num_patch_sqnorm2 = torch.zeros(b, h, w, device=device)
		sum_exp_log = torch.zeros(b, h, w, device=device)
		subtraction = None
		i = 0
		for images, labels in self.trainloader:
			if label is not None:
				images = images[(labels == label).squeeze(), :, :, :]
			if images.shape[0] == 0:
				continue
			images = images.to(device)
			bsize = images.shape[0]
			i += bsize
			if self.max_samples is not None and i > self.max_samples:
				break
			pwise_diffs = x[:, None, :, :, :] - at * images[None, :, :, :, :]
			pwise_normsquares = torch.sum(pwise_diffs**2, dim=2)
			patches = F.unfold(pwise_normsquares, k, stride=1, padding=k//2)
			patches = patches.view(b, bsize, k**2, h, w)
			exp_args = -torch.sum(patches, dim=2) / (2 * bt**2)
			if subtraction is None:
				subtraction = torch.amax(exp_args, dim=(0, 1), keepdim=True)
			else:
				new_sub = torch.amax(exp_args, dim=(0, 1), keepdim=True)
				delta = torch.maximum(subtraction, new_sub)
				shift = (delta - subtraction)[:, 0, :, :]  # (1, h, w)
				scale = torch.exp(shift)
				z_old = denominator.clone()
				numerator /= scale.unsqueeze(1)
				denominator /= scale
				num_sq /= scale.unsqueeze(1)
				num_quartic /= scale.unsqueeze(1)
				num_patch_vec /= scale.unsqueeze(1)
				num_patch_sqnorm /= scale
				num_patch_sqnorm2 /= scale
				# A_new = (A_old - shift*Z_old)/scale; use Z_old before rescale for stability
				sum_exp_log = (sum_exp_log - shift * z_old) / scale
				subtraction = delta
			exp_vals = torch.exp(exp_args - subtraction)
			image_patches = F.unfold(images, k, stride=1, padding=k//2)
			image_patches = image_patches.view(bsize, patch_dim, h, w)
			image_patch_sqnorm = torch.sum(image_patches**2, dim=1)
			# Sum over training samples so denominator = Z, entropy = log Z - (1/Z)*sum_exp_log
			numerator += torch.sum(exp_vals[:, :, None, :, :] * pwise_diffs, dim=1)
			denominator += torch.sum(exp_vals, dim=1)
			# E[x0] = (x - num/denom)/at  so x0 = images; num = sum w_n * (x - at*images)
			num_sq += torch.sum(exp_vals[:, :, None, :, :] * (images[None, :, :, :, :]**2), dim=1)
			num_quartic += torch.sum(exp_vals[:, :, None, :, :] * (images[None, :, :, :, :]**4), dim=1)
			num_patch_vec += torch.sum(exp_vals[:, :, None, :, :] * image_patches[None, :, :, :, :], dim=1)
			num_patch_sqnorm += torch.sum(exp_vals * image_patch_sqnorm[None, :, :, :], dim=1)
			num_patch_sqnorm2 += torch.sum(exp_vals * (image_patch_sqnorm[None, :, :, :]**2), dim=1)
			sum_exp_log += torch.sum(exp_vals * (exp_args - subtraction), dim=1)
		denom = denominator[:, None, :, :].clamp(min=1e-8)
		E_x0 = (x - numerator / denom) / at
		score = -numerator / denom / bt**2
		entropy_map = (torch.log(denominator.clamp(min=1e-8)) - sum_exp_log / denominator.clamp(min=1e-8)).clamp(min=0.0)
		E_x0_sq = num_sq / denom
		E_x0_quartic = num_quartic / denom
		center_variance_map = (E_x0_sq - E_x0**2).clamp(min=0)
		center_binder_map = 1 - E_x0_quartic / (3 * (E_x0_sq.clamp(min=1e-8)**2))
		denom_scalar = denominator.clamp(min=1e-8)
		E_patch_vec = num_patch_vec / denom[:, :, :, :]
		E_patch_sqnorm = num_patch_sqnorm / denom_scalar
		E_patch_sqnorm2 = num_patch_sqnorm2 / denom_scalar
		patch_variance_map = (E_patch_sqnorm - torch.sum(E_patch_vec**2, dim=1)).clamp(min=0)
		patch_binder_map = 1 - E_patch_sqnorm2 / (3 * (E_patch_sqnorm.clamp(min=1e-8)**2))
		return (
			score,
			E_x0,
			entropy_map,
			center_variance_map,
			center_binder_map,
			patch_variance_map,
			patch_binder_map,
		)


def _boltzmann_weights_from_energy(energy_per_scale: np.ndarray) -> np.ndarray:
	"""Boltzmann weights w_k = exp(-E_k) / Z from energies E_k (numpy)."""
	emax = np.max(energy_per_scale)
	w = np.exp(-(energy_per_scale - emax))
	return w / (w.sum() + 1e-12)


class AutoscalingLSModule(nn.Module):
	"""
	Autoscaling LS (ASLS): computes LS at all scales in k_vals, then Boltzmann-weights
	score outputs by energy E_k = beta_scale * SNR * Var_k - S_k, where Var_k is the
	center-pixel variance and S_k is the posterior entropy at scale k.
	"""

	def __init__(
		self,
		dataset,
		k_vals=(3, 5, 7, 9, 11),
		beta_scale=1.0,
		image_size=32,
		batch_size=256,
		schedule=cosine_noise_schedule,
		max_samples=None,
		**kwargs,
	):
		super().__init__()
		self.base = LocalScoreModule(
			dataset,
			kernel_size=k_vals[0],
			image_size=image_size,
			batch_size=batch_size,
			schedule=schedule,
			max_samples=max_samples,
			**kwargs,
		)
		self.k_vals = list(k_vals)
		self.beta_scale = float(beta_scale)
		self.schedule = schedule

	def forward(self, t, x, label=None, device=None, k=None):
		# Ignore k; use autoscaling over k_vals
		if device is None:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		scores = []
		energies = []
		t_val = t[0].item() if t.numel() >= 1 else t.item()
		snr = float(t_to_snr(t_val, self.schedule))
		for kk in self.k_vals:
			out = self.base.forward_with_posterior_stats(t, x, label=label, device=device, k=kk)
			score_k = out[0]
			entropy_map = out[2]
			center_variance_map = out[3]
			Var_k = center_variance_map.mean().item()
			S_k = entropy_map.mean().item()
			E_k = self.beta_scale * snr * Var_k - S_k
			scores.append(score_k)
			energies.append(E_k)
		energies_arr = np.array(energies, dtype=np.float64)
		weights = _boltzmann_weights_from_energy(energies_arr)
		# Keep weighted score on same device as scores
		weighted = sum(float(w) * s for w, s in zip(weights, scores))
		return weighted


class AutoscalingELSModule(nn.Module):
	"""
	Autoscaling ELS (ASELS): same as ASLS but wraps LocalEquivScoreModule; Boltzmann
	weights by E_k = beta_scale * SNR * Var_k - S_k (center variance and entropy).
	"""

	def __init__(
		self,
		dataset,
		k_vals=(3, 5, 7, 9, 11),
		beta_scale=1.0,
		image_size=32,
		channels=3,
		batch_size=64,
		schedule=cosine_noise_schedule,
		max_samples=None,
		shuffle=False,
		**kwargs,
	):
		super().__init__()
		self.base = LocalEquivScoreModule(
			dataset,
			kernel_size=k_vals[0],
			batch_size=batch_size,
			image_size=image_size,
			channels=channels,
			schedule=schedule,
			max_samples=max_samples,
			shuffle=shuffle,
			**kwargs,
		)
		self.k_vals = list(k_vals)
		self.beta_scale = float(beta_scale)
		self.schedule = schedule

	def forward(self, t, x, label=None, device=None, k=None):
		if device is None:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		scores = []
		energies = []
		t_val = t[0].item() if t.numel() >= 1 else t.item()
		snr = float(t_to_snr(t_val, self.schedule))
		for kk in self.k_vals:
			out = self.base.forward_with_posterior_stats(t, x, label=label, device=device, k=kk)
			score_k = out[0]
			entropy_map = out[2]
			center_variance_map = out[3]
			Var_k = center_variance_map.mean().item()
			S_k = entropy_map.mean().item()
			E_k = self.beta_scale * snr * Var_k - S_k
			scores.append(score_k)
			energies.append(E_k)
		energies_arr = np.array(energies, dtype=np.float64)
		weights = _boltzmann_weights_from_energy(energies_arr)
		weighted = sum(float(w) * s for w, s in zip(weights, scores))
		return weighted


class AutoscalingBBELSModule(nn.Module):
	"""
	Boundary broken auto scaling ELS (bbASELS): same pattern as ASELS but wraps
	LocalEquivBordersScoreModule (ELS with boundary zeros). Uses uniform weighting
	over k_vals (boundary module does not expose forward_with_posterior_stats
	for Boltzmann weighting).
	"""

	def __init__(
		self,
		dataset,
		k_vals=(3, 5, 7, 9, 11),
		image_size=32,
		channels=3,
		batch_size=64,
		schedule=cosine_noise_schedule,
		max_samples=None,
		shuffle=False,
		**kwargs,
	):
		super().__init__()
		self.base = LocalEquivBordersScoreModule(
			dataset,
			kernel_size=k_vals[0],
			batch_size=batch_size,
			image_size=image_size,
			channels=channels,
			schedule=schedule,
			max_samples=max_samples,
			shuffle=shuffle,
			**kwargs,
		)
		self.k_vals = list(k_vals)
		self.schedule = schedule

	def forward(self, t, x, label=None, device=None, k=None):
		if device is None:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		scores = []
		for kk in self.k_vals:
			score_k = self.base(t, x, label=label, device=device, k=kk)
			scores.append(score_k)
		# Uniform weighting (no posterior stats on border module)
		weighted = sum(scores) / len(scores)
		return weighted


class IdealScoreModule(nn.Module):

	def __init__(self, dataset,
					image_size=32,
					batch_size=128,
					schedule=cosine_noise_schedule,
					max_samples=None,
					shuffle=False,
					**kwargs):

		super().__init__()
		self.dataset = dataset
		self.trainloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
		self.batch_size = batch_size
		self.image_size = image_size
		self.schedule = schedule
		self.max_samples = max_samples

	def forward(self, t, x, label=None, device=None, **kwargs):

		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		x = x.to(device)

		b,c,h,w = x.shape
		bt = (self.schedule(t))**0.5
		at = (1-self.schedule(t))**0.5

		at = at.to(device)
		bt = bt.to(device)

		numerator = torch.zeros(x.shape, device=device)
		denominator = torch.zeros(b, device=device)

		subtraction = None

		i = 0


		for images, labels in self.trainloader:

			if label is not None:
				images = images[(labels==label).squeeze(),:,:,:]

			if images.shape[0] == 0:
				continue

			images = images.to(device)
			labels = labels.to(device)

			bsize = images.shape[0]

			i += bsize
			if self.max_samples is not None and i > self.max_samples:
				break

			pwise_diffs = x[:,None,:,:,:]-at*images[None,:,:,:,:]
			exp_args = -torch.sum(pwise_diffs**2, dim=(2,3,4))/(2*bt**2) 
			

			if subtraction is None:
				subtraction = torch.amax(exp_args, dim=(0,1), keepdim=False)
			else:
				new_subtraction = torch.amax(exp_args, dim=(0,1), keepdim=False)
				delta_subtraction = (new_subtraction>subtraction)*new_subtraction+(subtraction>=new_subtraction)*subtraction
				numerator /= torch.exp(delta_subtraction-subtraction)
				denominator /= torch.exp(delta_subtraction-subtraction)
				subtraction = delta_subtraction

			exp_vals = torch.exp(exp_args - subtraction) 

			numerator += torch.mean(exp_vals[:,:,None,None,None]*pwise_diffs, dim=1)
			denominator += torch.mean(exp_vals, dim=1)


		return -numerator/denominator/bt**2
