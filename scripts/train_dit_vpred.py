#!/usr/bin/env python
"""Train the MinimalDiT backbone on the v-prediction objective.

Kept separate from scripts/training_script.py / src/utils/train.py, which train the
convolutional backbones on the eps objective and which the rest of the repo depends
on.  The only substantive differences are the target (v instead of eps) and the fact
that the checkpoint is saved with a VPredToEps wrapper already in place.

That wrapper matters: everything downstream -- DDIM.sample, ScheduledScoreMachine,
scripts/scales_calibration.py, scripts/fit_ntk_score.py -- expects a backbone that
returns eps.  Saving `DDIM(backbone=VPredToEps(dit, schedule))` means none of them
need a v-prediction code path, while the loss here is still computed against the raw
v output of the inner DiT.

Example:
  python scripts/train_dit_vpred.py --dataset celeba --epochs 200 \
      --savename backbone_CelebA_DiT_vpred --homedir ./checkpoints
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

from src.models import DDIM
from src.models_dit import MinimalDiT, VPredToEps
from src.utils.data import get_dataset
from src.utils.noise_schedules import cosine_noise_schedule


def main():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument('--dataset', type=str, default='celeba')
	p.add_argument('--data_root', type=str, default='./data')
	p.add_argument('--epochs', type=int, default=200)
	p.add_argument('--batchsize', type=int, default=256)
	p.add_argument('--lr', type=float, default=2e-4)
	p.add_argument('--wd', type=float, default=0.0)
	p.add_argument('--gamma', type=float, default=0.999995, help='ExponentialLR decay, applied per step.')
	p.add_argument('--max_t', type=int, default=1000, help='Training-time discretization of t in [0,1).')
	p.add_argument('--maxsamps', type=int, default=100000)
	p.add_argument('--num_workers', type=int, default=8)
	p.add_argument('--patch_size', type=int, default=4)
	p.add_argument('--hidden', type=int, default=256)
	p.add_argument('--depth', type=int, default=6)
	p.add_argument('--heads', type=int, default=4)
	p.add_argument('--adaln_zero', action='store_true',
	               help='Reference adaLN-Zero init. Off by default: it makes the NTK at init degenerate '
	                    '(zero output gates kill every block parameter gradient). See src/models_dit.py.')
	p.add_argument('--seed', type=int, default=0)
	p.add_argument('--saveinterval', type=int, default=20)
	p.add_argument('--homedir', type=str, default='./checkpoints')
	p.add_argument('--savename', type=str, default='backbone_CelebA_DiT_vpred')
	p.add_argument('--dry_run', action='store_true')
	args = p.parse_args()

	if args.dry_run:
		args.epochs, args.maxsamps, args.saveinterval = 2, 2048, 1

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.manual_seed(args.seed)
	print(f'device={device}', flush=True)

	dataset, metadata = get_dataset(args.dataset, root=args.data_root)
	if args.maxsamps < len(dataset):
		dataset = torch.utils.data.Subset(dataset, range(args.maxsamps))
	loader = DataLoader(
		dataset, batch_size=args.batchsize, shuffle=True,
		num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True,
	)
	print(f'dataset={metadata["name"]} n={len(dataset)} batches/epoch={len(loader)}', flush=True)

	dit = MinimalDiT(
		channels=metadata['num_channels'],
		image_size=metadata['image_size'],
		patch_size=args.patch_size,
		hidden=args.hidden,
		depth=args.depth,
		heads=args.heads,
		conditional=False,
		adaln_zero=args.adaln_zero,
	)
	n_params = sum(q.numel() for q in dit.parameters())
	print(
		f'MinimalDiT: patch={args.patch_size} tokens={dit.num_patches} hidden={args.hidden} '
		f'depth={args.depth} heads={args.heads} adaln_zero={args.adaln_zero} params={n_params:,}',
		flush=True,
	)

	model = DDIM(
		backbone=VPredToEps(dit, cosine_noise_schedule),
		in_channels=metadata['num_channels'],
		noise_schedule=cosine_noise_schedule,
		default_imsize=metadata['image_size'],
	).to(device)

	optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

	os.makedirs(args.homedir, exist_ok=True)
	fname = os.path.join(args.homedir, args.savename)

	for epoch in range(args.epochs):
		model.train()
		t_epoch, running, nb = time.time(), 0.0, 0
		for images, _ in loader:
			optimizer.zero_grad(set_to_none=True)
			images = images.to(device, non_blocking=True)
			b = images.shape[0]

			t = torch.randint(0, args.max_t, (b,), device=device).float() / args.max_t
			beta_t = cosine_noise_schedule(t)[:, None, None, None]
			noise = torch.randn_like(images)
			noised = (1 - beta_t) ** 0.5 * images + beta_t ** 0.5 * noise

			# Loss on the raw v output of the inner DiT, not the eps-wrapped view.
			pred_v = dit(t, noised)
			target_v = VPredToEps.v_target(images, noise, beta_t)
			loss = mse_loss(pred_v, target_v)

			loss.backward()
			optimizer.step()
			scheduler.step()

			running += loss.item()
			nb += 1

		if epoch % args.saveinterval == 0 or epoch == args.epochs - 1:
			torch.save(model, f'{fname}_epoch{epoch}.pt')
			torch.save(model, f'{fname}.pt')
		print(
			f'epoch {epoch + 1:4d}/{args.epochs}  v_mse={running / max(nb, 1):.5f}  '
			f'lr={scheduler.get_last_lr()[0]:.2e}  {time.time() - t_epoch:.1f}s',
			flush=True,
		)

	print(f'Saved final model to {fname}.pt', flush=True)


if __name__ == '__main__':
	main()
