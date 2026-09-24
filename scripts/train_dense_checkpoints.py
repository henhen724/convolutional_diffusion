#!/usr/bin/env python
"""Train with log-spaced checkpoints in *step* time, to resolve the early NTK drift.

The existing trajectories in checkpoints/ are saved every 10 (ResNet) or 20 (DiT)
epochs.  If the empirical NTK moves quickly -- which is the thing under test -- all
of that motion happens inside the first saved interval and the epoch-level series
cannot see it.  This trains the same architectures while checkpointing at
1, 2, 4, 8, ... optimizer steps, so scripts/ntk_kernel_evolution.py can measure the
kernel before, during and after the feature-learning transient.

The ResNet config defaults match checkpoints/backbone_CelebA_ResNet_zeros.pt (the
network the NTK comparison is built on), so the drift measured here applies to that
architecture rather than to a lookalike.

Example:
  python scripts/train_dense_checkpoints.py --arch resnet --max_steps 4096
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch import optim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.models import DDIM, MinimalResNet
from src.models_dit import MinimalDiT, VPredToEps
from src.utils.data import get_dataset
from src.utils.noise_schedules import cosine_noise_schedule


def log_spaced_steps(max_steps, per_decade=8):
	"""{0, 1, 2, ...} roughly log-spaced up to max_steps, always including both ends."""
	steps = {0, max_steps}
	s = 1.0
	while s <= max_steps:
		steps.add(int(round(s)))
		s *= 10 ** (1.0 / per_decade)
	return sorted(steps)


def main():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument('--arch', type=str, default='resnet', choices=['resnet', 'dit'])
	p.add_argument('--dataset', type=str, default='celeba')
	p.add_argument('--data_root', type=str, default='./data')
	p.add_argument('--max_steps', type=int, default=4096)
	p.add_argument('--per_decade', type=int, default=8, help='Checkpoints per decade of step count.')
	p.add_argument('--batchsize', type=int, default=128)
	p.add_argument('--lr', type=float, default=2e-4)
	p.add_argument('--wd', type=float, default=0.0)
	p.add_argument('--max_t', type=int, default=1000)
	p.add_argument('--maxsamps', type=int, default=100000)
	p.add_argument('--num_workers', type=int, default=8)
	p.add_argument('--seed', type=int, default=0)
	# ResNet config: matches checkpoints/backbone_CelebA_ResNet_zeros.pt.
	p.add_argument('--emb_dim', type=int, default=256)
	p.add_argument('--num_layers', type=int, default=8)
	p.add_argument('--mode', type=str, default='zeros')
	p.add_argument('--lastksize', type=int, default=3)
	# DiT config: matches scripts/train_dit_vpred.py defaults.
	p.add_argument('--patch_size', type=int, default=4)
	p.add_argument('--hidden', type=int, default=256)
	p.add_argument('--depth', type=int, default=6)
	p.add_argument('--heads', type=int, default=4)
	p.add_argument('--homedir', type=str, default='./checkpoints/dense')
	p.add_argument('--savename', type=str, default=None)
	p.add_argument('--dry_run', action='store_true')
	args = p.parse_args()

	if args.dry_run:
		args.max_steps, args.maxsamps, args.per_decade = 8, 2048, 4

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.manual_seed(args.seed)

	dataset, metadata = get_dataset(args.dataset, root=args.data_root)
	if args.maxsamps < len(dataset):
		dataset = torch.utils.data.Subset(dataset, range(args.maxsamps))
	loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True,
	                    num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
	                    drop_last=True)

	if args.arch == 'resnet':
		inner = MinimalResNet(channels=metadata['num_channels'], emb_dim=args.emb_dim,
		                      mode=args.mode, normalization=None, conditional=False,
		                      num_layers=args.num_layers, lastksize=args.lastksize)
		backbone, v_pred = inner, False
	else:
		inner = MinimalDiT(channels=metadata['num_channels'], image_size=metadata['image_size'],
		                   patch_size=args.patch_size, hidden=args.hidden, depth=args.depth,
		                   heads=args.heads, conditional=False, adaln_zero=False)
		backbone, v_pred = VPredToEps(inner, cosine_noise_schedule), True

	model = DDIM(backbone=backbone, in_channels=metadata['num_channels'],
	             noise_schedule=cosine_noise_schedule,
	             default_imsize=metadata['image_size']).to(device)

	n_params = sum(q.numel() for q in inner.parameters())
	savename = args.savename or f'backbone_CelebA_{args.arch}_dense'
	os.makedirs(args.homedir, exist_ok=True)
	fname = os.path.join(args.homedir, savename)
	targets = log_spaced_steps(args.max_steps, args.per_decade)
	print(f'device={device}  arch={args.arch}  params={n_params:,}  v_pred={v_pred}', flush=True)
	print(f'checkpoint steps: {targets}', flush=True)

	optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

	# Step 0 is the initialization itself -- the reference point every drift metric
	# is measured against, so it has to be saved before any optimizer step.
	torch.save(model, f'{fname}_step0.pt')
	pending = [s for s in targets if s > 0]

	step, t_start = 0, time.time()
	running, nb = 0.0, 0
	done = False
	while not done:
		for images, _ in loader:
			optimizer.zero_grad(set_to_none=True)
			images = images.to(device, non_blocking=True)
			b = images.shape[0]
			t = torch.randint(0, args.max_t, (b,), device=device).float() / args.max_t
			beta_t = cosine_noise_schedule(t)[:, None, None, None]
			noise = torch.randn_like(images)
			noised = (1 - beta_t) ** 0.5 * images + beta_t ** 0.5 * noise

			if v_pred:
				pred = inner(t, noised)
				target = VPredToEps.v_target(images, noise, beta_t)
			else:
				pred = model(t, noised)
				target = noise
			loss = mse_loss(pred, target)
			loss.backward()
			optimizer.step()

			step += 1
			running += loss.item()
			nb += 1
			while pending and step >= pending[0]:
				s = pending.pop(0)
				torch.save(model, f'{fname}_step{s}.pt')
				print(f'  step {s:6d}  loss={running / max(nb, 1):.5f}  '
				      f'{time.time() - t_start:.0f}s', flush=True)
				running, nb = 0.0, 0
			if step >= args.max_steps:
				done = True
				break

	print(f'Saved {len(targets)} checkpoints to {fname}_step*.pt', flush=True)


if __name__ == '__main__':
	main()
