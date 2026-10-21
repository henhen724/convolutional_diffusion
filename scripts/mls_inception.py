#!/usr/bin/env python
"""Generate samples from LS / multilayer LS score machines and score them with Inception Score.

Methods:
  LS       plain LocalScoreModule with the per-step scales file (the baseline)
  MLS      fixed two-layer MLS, layer-2 window = layer-1 window + --mls_offset
  EBMLS    entropy-barrier MLS: layer-2 window at each step is the largest k whose
           layer-2 posterior entropy is still >= --barrier (EntropyBarrierMLSModule)
  real     no generation; scores --nsamples training images (the IS ceiling)

Samples are written in chunks to --outdir so an interrupted job resumes where it
stopped. Results (IS mean/std, memorization stats, chosen layer-2 windows per step)
go to --outdir/metrics.json.

Memorization: a higher IS can come from copying training images, which is
exactly what a too-large window does, so every run also reports each sample's
nearest-neighbour RMS distance to the training set (same class if conditional),
and the fraction of samples within --copy_rms of a training image.

Example:
  python scripts/mls_inception.py --dataset cifar10 --conditional --method EBMLS --barrier 2.0 \
      --nsamples 1000 --outdir $SCRATCH/convolutional_diffusion/results/mls_inception/cifar10_eb2
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.utils.data import get_dataset
from src.utils.idealscore import (
	EntropyBarrierMLSModule,
	LocalScoreModule,
	MultilayerLocalScoreModule,
	ScheduledScoreMachine,
)
from src.utils.noise_schedules import cosine_noise_schedule

DEFAULT_SCALES = {
	'mnist': 'scales_MNIST_ResNet_zeros.pt',
	'cifar10': 'scales_CIFAR10_ResNet_zeros_conditional.pt',
	'celeba': 'scales_CelebA_ResNet_zeros.pt',
}


class _KLogger(torch.nn.Module):
	"""Passes calls through to the backbone and records its chosen layer-2 window per step."""

	def __init__(self, backbone):
		super().__init__()
		self.backbone = backbone
		self.log = {}

	def forward(self, t, x, **kw):
		out = self.backbone(t, x, **kw)
		last_k = getattr(self.backbone, 'last_k', None)
		if last_k is not None:
			self.log.setdefault(round(float(t[0]), 4), []).extend(last_k.float().flatten().tolist() if last_k.dim() == 1 else [float(last_k.float().mean())])
		return out


class _Uint8(Dataset):
	def __init__(self, x):
		self.x = x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)

	def __len__(self):
		return self.x.shape[0]

	def __getitem__(self, i):
		return self.x[i]


class _TensorBatches:
	"""Drop-in for a module's trainloader: yields (images, labels) slices of tensors held on the GPU.

	The score modules loop over their whole training set at every denoising step;
	through a torchvision dataset that re-decodes and re-transforms every image on
	the CPU each time, which dominates runtime.
	"""

	def __init__(self, images, labels, batch_size):
		self.images, self.labels, self.batch_size = images, labels, batch_size

	def __iter__(self):
		for i in range(0, self.images.shape[0], self.batch_size):
			yield self.images[i:i + self.batch_size], self.labels[i:i + self.batch_size]


def to_uint8(x, mean, std):
	m = torch.tensor(mean)[None, :, None, None]
	s = torch.tensor(std)[None, :, None, None]
	return ((x.cpu() * s + m).clamp(0, 1) * 255).round().to(torch.uint8)


def nearest_neighbour_rms(samples, labels, loader, device):
	"""Per-sample RMS distance (normalized pixel units) to the closest training image."""
	s = samples.to(device).flatten(1)
	best = torch.full((s.shape[0],), float('inf'), device=device)
	for images, ylab in loader:
		images = images.flatten(1)
		d2 = (s**2).sum(1, keepdim=True) - 2 * s @ images.T + (images**2).sum(1)[None]
		if labels is not None:
			d2 = d2.masked_fill(labels.to(device)[:, None] != ylab.to(device)[None], float('inf'))
		best = torch.minimum(best, d2.min(1).values)
	return (best.clamp(min=0) / s.shape[1]).sqrt().cpu()


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--dataset', default='cifar10')
	p.add_argument('--method', choices=['LS', 'MLS', 'EBMLS', 'real'], default='EBMLS')
	p.add_argument('--conditional', action='store_true')
	p.add_argument('--scalesfile', default=None)
	# A larger window raises IS by moving samples closer to the training set, so plain LS
	# with its scales scaled up traces the single-layer IS-vs-memorization frontier. That is
	# the control EBMLS has to beat: higher IS at the same nearest-neighbour distance.
	p.add_argument('--scale_multiplier', type=float, default=1.0)
	p.add_argument('--barrier', type=float, default=2.0)
	p.add_argument('--barrier_mode', choices=['image', 'pixel'], default='image')
	p.add_argument('--temperature', type=float, default=1.0)
	p.add_argument('--mls_offset', type=int, default=4)
	p.add_argument('--nsamples', type=int, default=1000)
	p.add_argument('--genbatch', type=int, default=50)
	p.add_argument('--scorebatchsize', type=int, default=128)
	p.add_argument('--max_samples', type=int, default=None)
	p.add_argument('--isc_splits', type=int, default=10)
	p.add_argument('--copy_rms', type=float, default=0.05)
	p.add_argument('--seed', type=int, default=0)
	p.add_argument('--outdir', required=True)
	args = p.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	os.makedirs(args.outdir, exist_ok=True)
	dataset, meta = get_dataset(args.dataset, root=str(ROOT / 'data'))
	c, size, ncls = meta['num_channels'], meta['image_size'], meta['num_classes']
	conditional = args.conditional and ncls > 1
	scales_path = args.scalesfile or str(ROOT / 'checkpoints' / DEFAULT_SCALES[args.dataset.lower()])
	scales = [int(s) for s in torch.load(scales_path, weights_only=False)]
	if args.scale_multiplier != 1.0:
		scales = [min(2 * size - 1, int(round(s * args.scale_multiplier)) | 1) for s in scales]
	print(f'dataset={args.dataset} method={args.method} conditional={conditional} scales={scales}', flush=True)

	common = dict(image_size=size, batch_size=args.scorebatchsize, schedule=cosine_noise_schedule, max_samples=args.max_samples)
	if args.method == 'LS':
		mod = LocalScoreModule(dataset, **common)
	elif args.method == 'MLS':
		mod = MultilayerLocalScoreModule(dataset, kernel_sizes=(3, 3 + args.mls_offset), temperature=args.temperature, **common)
	elif args.method == 'EBMLS':
		mod = EntropyBarrierMLSModule(dataset, entropy_barrier=args.barrier, barrier_mode=args.barrier_mode, temperature=args.temperature, **common)
	if args.method != 'real':
		cached = list(DataLoader(dataset, batch_size=1024, num_workers=4))
		images = torch.cat([b[0] for b in cached]).to(device)
		labels = torch.cat([b[1] for b in cached]).to(device)
		if labels.dim() > 1:  # CelebA attribute vectors; unconditional anyway
			labels = torch.zeros(labels.shape[0], dtype=torch.long, device=device)
		for m in (mod, getattr(mod, 'layer1', None)):
			if m is not None:
				m.trainloader = _TensorBatches(images, labels, args.scorebatchsize)
	logger = _KLogger(mod) if args.method != 'real' else None

	labels_all = (torch.arange(args.nsamples) * ncls // args.nsamples) if conditional else None
	t0 = time.time()
	if args.method == 'real':
		idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:args.nsamples]
		samples = torch.stack([dataset[int(i)][0] for i in idx])
		labels_all = None
	else:
		machine = ScheduledScoreMachine(logger, in_channels=c, imsize=size, noise_schedule=cosine_noise_schedule, scales=scales)
		chunks = []
		# generate batches of a single label: the local modules condition on one scalar label per call
		groups = [(labels_all == y).nonzero().flatten() for y in range(ncls)] if conditional else [torch.arange(args.nsamples)]
		ranges = [(int(g[0]) + a, min(int(g[0]) + a + args.genbatch, int(g[-1]) + 1)) for g in groups if len(g) for a in range(0, len(g), args.genbatch)]
		for s, e in ranges:
			path = os.path.join(args.outdir, f'chunk_{s:06d}.pt')
			if os.path.exists(path):
				chunks.append(torch.load(path, weights_only=False))
				continue
			g = torch.Generator().manual_seed(args.seed * 1_000_003 + s)
			seed = torch.randn(e - s, c, size, size, generator=g).to(device)
			label = labels_all[s].to(device) if conditional else None
			with torch.no_grad():
				out = machine(seed, label=label, device=device).cpu()
			torch.save(out, path)
			chunks.append(out)
			print(f'  samples {s}:{e} done, {time.time() - t0:.0f}s elapsed', flush=True)
		samples = torch.cat(chunks)
	gen_time = time.time() - t0

	import torch_fidelity
	u8 = to_uint8(samples, meta['mean'], meta['std'])
	isc = torch_fidelity.calculate_metrics(input1=_Uint8(u8), isc=True, isc_splits=args.isc_splits, cuda=device.type == 'cuda', verbose=False)
	nn_rms = nearest_neighbour_rms(samples, labels_all, _TensorBatches(images, labels, 2048), device) if args.method != 'real' else None

	metrics = {
		'dataset': args.dataset, 'method': args.method, 'conditional': conditional,
		'barrier': args.barrier if args.method == 'EBMLS' else None,
		'barrier_mode': args.barrier_mode if args.method == 'EBMLS' else None,
		'temperature': args.temperature, 'nsamples': int(samples.shape[0]),
		'scale_multiplier': args.scale_multiplier,
		'IS_mean': float(isc['inception_score_mean']), 'IS_std': float(isc['inception_score_std']),
		'nn_rms_median': None if nn_rms is None else float(nn_rms.median()),
		'copy_fraction': None if nn_rms is None else float((nn_rms < args.copy_rms).float().mean()),
		'gen_seconds': gen_time, 'scales': scales,
		'layer2_k_by_t': None if logger is None else {t: sum(v) / len(v) for t, v in sorted(logger.log.items())},
	}
	if nn_rms is not None:
		torch.save(nn_rms, os.path.join(args.outdir, 'nn_rms.pt'))
	torch.save(u8, os.path.join(args.outdir, 'samples_uint8.pt'))
	with open(os.path.join(args.outdir, 'metrics.json'), 'w') as f:
		json.dump(metrics, f, indent=1)
	print(json.dumps({k: v for k, v in metrics.items() if k not in ('scales', 'layer2_k_by_t')}), flush=True)


if __name__ == '__main__':
	main()
