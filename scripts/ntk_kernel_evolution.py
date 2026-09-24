#!/usr/bin/env python
"""Measure how far the empirical NTK moves from its value at initialization during training.

For a training trajectory theta_0, theta_1, ... we form the empirical NTK over a
fixed set of held-out noisy images, using the same scalar pixel readout the NTK
regression uses (scripts/fit_ntk_score.py):

    K_t[r, s] = < grad_theta f_{theta_t}(x_r) , grad_theta f_{theta_t}(x_s) >

and report, per noise level,

    l2         = mean_{r,s} ( K_0[r,s] - K_t[r,s] )^2        (the requested data-averaged L2)
    rel_fro    = ||K_0 - K_t||_F / ||K_0||_F
    alignment  = <K_0, K_t>_F / (||K_0||_F ||K_t||_F)        (scale-invariant: has the kernel turned?)
    norm_ratio = ||K_t||_F / ||K_0||_F                       (has it just grown?)

The last two separate the two ways a kernel can move: alignment falls only if the
kernel's *shape* changes, while norm_ratio catches a pure rescaling, which the raw
L2 alone cannot distinguish.

If the lazy/NTK regime held, alignment would stay near 1 for the whole run. The
point of the measurement is the timescale on which it does not.

The evaluation images, their noise draws and their readout pixels are all seeded
and shared across checkpoints, so K_t entries are directly comparable along the
trajectory.
"""

import argparse
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.utils.data import get_dataset
from src.utils.noise_schedules import cosine_noise_schedule

# Legacy backbone_*.pt checkpoints were pickled with class module "models"/"utils"
# (pre-refactor layout); register aliases so torch.load's unpickler can find them.
import src.models
sys.modules['models'] = sys.modules['src.models']
import src.utils
sys.modules['utils'] = sys.modules['src.utils']
import src.models_dit  # noqa: F401  (needed to unpickle DiT checkpoints)


def checkpoint_series(pattern):
	"""Expand a glob into (step, path) pairs sorted by the integer in the filename."""
	paths = sorted(Path(ROOT).glob(pattern))
	out = []
	for p in paths:
		m = re.search(r'(?:epoch|step)(\d+)', p.name)
		if m:
			out.append((int(m.group(1)), p))
	if not out:
		raise FileNotFoundError(f'No checkpoints matching {pattern!r} under {ROOT}')
	return sorted(out)


def gram_at_checkpoint(backbone, xt, pixels, t_val, channels, device, row_dtype=torch.float32):
	"""Empirical NTK over the readouts, K = G G^T, with G built one row at a time."""
	backbone = backbone.to(device).eval()
	params = [p for p in backbone.parameters()]
	for p in params:
		p.requires_grad_(True)
	n_params = sum(p.numel() for p in params)
	rows = xt.shape[0] * len(channels)
	G = torch.empty((rows, n_params), dtype=row_dtype, device=device)
	for k in range(xt.shape[0]):
		x = xt[k:k + 1].to(device)
		t = torch.full((1,), float(t_val), device=device)
		out = backbone(t, x)
		i0, j0 = int(pixels[k, 0]), int(pixels[k, 1])
		for ci, c in enumerate(channels):
			f = out[0, c, i0, j0]
			grads = torch.autograd.grad(f, params, retain_graph=(ci < len(channels) - 1))
			G[k * len(channels) + ci] = torch.cat([g.reshape(-1) for g in grads]).to(row_dtype)
		del out
	K = (G @ G.T).double().cpu()
	del G
	if device.type == 'cuda':
		torch.cuda.empty_cache()
	return K


def compare(K0, Kt):
	diff = K0 - Kt
	n0 = K0.norm()
	nt = Kt.norm()
	return {
		'l2': float((diff ** 2).mean()),
		'rel_fro': float(diff.norm() / n0.clamp_min(1e-30)),
		'alignment': float((K0 * Kt).sum() / (n0 * nt).clamp_min(1e-30)),
		'norm_ratio': float(nt / n0.clamp_min(1e-30)),
	}


def main():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument('--pattern', type=str, required=True,
	               help="Glob for the trajectory, e.g. 'checkpoints/backbone_CelebA_ResNet_fresh_epoch*.pt'")
	p.add_argument('--label', type=str, required=True, help='Short name for this trajectory, e.g. ResNet.')
	p.add_argument('--dataset', type=str, default='celeba')
	p.add_argument('--data_root', type=str, default='./data')
	p.add_argument('--n_eval', type=int, default=64, help='Held-out images defining the kernel.')
	p.add_argument('--time_steps', type=int, nargs='+', default=[1, 5, 10, 20],
	               help='Reverse-step indices (t = i/nsteps) at which to evaluate the kernel.')
	p.add_argument('--nsteps', type=int, default=20)
	p.add_argument('--readout', type=str, default='random', choices=['center', 'random'])
	p.add_argument('--pixel_i', type=int, default=16)
	p.add_argument('--pixel_j', type=int, default=16)
	p.add_argument('--seed', type=int, default=0)
	p.add_argument('--max_checkpoints', type=int, default=None,
	               help='Subsample the trajectory to at most this many checkpoints (endpoints kept).')
	p.add_argument('--out_file', type=str, default=None)
	p.add_argument('--cpu', action='store_true')
	args = p.parse_args()

	device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
	series = checkpoint_series(args.pattern)
	if args.max_checkpoints and len(series) > args.max_checkpoints:
		idx = torch.linspace(0, len(series) - 1, args.max_checkpoints).round().long().unique().tolist()
		series = [series[i] for i in idx]
	print(f'device={device}  trajectory={args.label}  {len(series)} checkpoints '
	      f'({series[0][0]} .. {series[-1][0]})', flush=True)

	dataset, metadata = get_dataset(args.dataset, root=args.data_root, train=False)
	imsize = metadata['image_size']
	channels = list(range(metadata['num_channels']))

	g = torch.Generator().manual_seed(args.seed)
	idx = torch.randperm(len(dataset), generator=g)[:args.n_eval]
	x0 = torch.stack([dataset[i][0] for i in idx.tolist()])

	if args.readout == 'center':
		pixels = torch.stack([torch.full((args.n_eval,), args.pixel_i),
		                      torch.full((args.n_eval,), args.pixel_j)], dim=1)
	else:
		pixels = torch.randint(0, imsize, (args.n_eval, 2),
		                       generator=torch.Generator().manual_seed(args.seed + 1))

	# One fixed noise draw per (image, noise level), shared by every checkpoint, so the
	# kernel is evaluated at literally the same inputs all along the trajectory.
	noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(args.seed + 2))

	results = {}
	for ts in args.time_steps:
		t_val = ts / args.nsteps
		beta_t = float(cosine_noise_schedule(torch.tensor(t_val)))
		xt = (1 - beta_t) ** 0.5 * x0 + beta_t ** 0.5 * noise
		print(f'\n=== noise level: step {ts}/{args.nsteps}, beta={beta_t:.4f}, sigma={beta_t**0.5:.4f} ===',
		      flush=True)
		K0 = None
		rows = []
		for step, path in series:
			model = torch.load(path, map_location='cpu', weights_only=False)
			backbone = model.backbone if hasattr(model, 'backbone') else model
			K = gram_at_checkpoint(backbone, xt, pixels, t_val, channels, device)
			if K0 is None:
				K0 = K
			m = compare(K0, K)
			m['step'] = step
			rows.append(m)
			print(f'  {step:5d}  l2={m["l2"]:.4e}  rel_fro={m["rel_fro"]:.4f}  '
			      f'alignment={m["alignment"]:.4f}  norm_ratio={m["norm_ratio"]:.4f}', flush=True)
			del model, backbone, K
		results[ts] = {'beta_t': beta_t, 'rows': rows, 'K0_fro': float(K0.norm())}

	out = Path(args.out_file) if args.out_file else (
		ROOT / 'results' / 'ntk_score' / f'kernel_evolution_{args.dataset}_{args.label}.pt')
	out.parent.mkdir(parents=True, exist_ok=True)
	torch.save({
		'kind': 'ntk_kernel_evolution',
		'label': args.label,
		'dataset': args.dataset,
		'pattern': args.pattern,
		'n_eval': args.n_eval,
		'readout': args.readout,
		'nsteps': args.nsteps,
		'checkpoints': [s for s, _ in series],
		'results': results,
	}, out)
	print(f'\nSaved kernel evolution to {out}', flush=True)


if __name__ == '__main__':
	main()
