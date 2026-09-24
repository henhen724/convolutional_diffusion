#!/usr/bin/env python
"""Generate matched samples from a trained network, its NTK prediction, and a local score machine.

Three score functions -- the trained backbone (ResNet / UNet / DiT), the NTK
kernel-regression prediction of that same backbone (scripts/fit_ntk_score.py), and
an analytic local score machine (LS / ELS / bbELS) -- are driven through the *same*
reverse loop (ScheduledScoreMachine) from the *same* seed noise, so any visual
difference is attributable to the score and not to the sampler.  DDIM.sample and
ScheduledScoreMachine.forward do not iterate over the same range of steps, so
mixing them would confound the comparison.

Also records how closely the scores agree pointwise: at every noise level each
model's eps-prediction is evaluated on one shared batch of noised held-out images
and the pairwise cosine similarities are logged.  That is the quantitative version
of "how similar are the NTK predictions to the ELS".

A fourth, optional machine -- the same local score machine restricted to exactly
the images the NTK regression saw -- is included by default, because the two
theories are otherwise compared at very different data budgets (the local score
machines normally stream ~1e4 images, while the NTK fit is capped by the cost of
per-sample parameter gradients).

Companion compute for notebooks/ntk_vs_els_samples.ipynb; run via
scripts/slurm_ntk_vs_els.sh.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.utils.data import get_dataset
from src.utils.idealscore import (
	LocalEquivBordersScoreModule,
	LocalEquivScoreModule,
	LocalScoreModule,
	ScheduledScoreMachine,
)
from src.utils.noise_schedules import cosine_noise_schedule
from src.utils.ntk_score import NTKScoreModule, EpsBackboneAdapter, build_theta0

# Legacy backbone_*.pt checkpoints were pickled with class module "models"/"utils"
# (pre-refactor layout); register aliases so torch.load's unpickler can find them.
import src.models
sys.modules['models'] = sys.modules['src.models']
import src.utils
sys.modules['utils'] = sys.modules['src.utils']

LOCAL_MODULES = {
	'LS': LocalScoreModule,
	'ELS': LocalEquivScoreModule,
	'bbELS': LocalEquivBordersScoreModule,
}


def seeded_indices(dataset_len, n, seed):
	"""Seeded index draw; scripts/fit_ntk_score.py uses the same call for its training set."""
	g = torch.Generator().manual_seed(seed)
	return torch.randperm(dataset_len, generator=g)[:n]


def make_local_machine(kind, dataset, indices, in_channels, imsize, nsteps, scales, batch_size):
	"""Build a local score machine over a fixed subset of the data.

	The subset is passed as a Subset with `max_samples=None` rather than relying on
	the modules' own `max_samples` cap: LocalScoreModule's DataLoader shuffles, so
	capping by count would draw a different training set on every call and make the
	comparison irreproducible.
	"""
	cls = LOCAL_MODULES[kind]
	backbone = cls(
		Subset(dataset, indices.tolist()),
		image_size=imsize,
		channels=in_channels,
		batch_size=batch_size,
		schedule=cosine_noise_schedule,
		max_samples=None,
		shuffle=False,
	)
	return ScheduledScoreMachine(
		backbone, in_channels=in_channels, imsize=imsize, default_time_steps=nsteps,
		noise_schedule=cosine_noise_schedule, score_backbone=True, scales=scales,
	)


def batch_chunk(machine):
	"""Largest batch a machine's backbone can take.

	The local score modules in src/utils/idealscore.py only work one image at a
	time: they broadcast the scalar noise levels `at`/`bt` against a tensor whose
	trailing axes are the training-patch dimensions, so a batch axis of size > 1
	collides with the patch axis. The network and NTK backbones have no such limit.
	"""
	return 1 if machine.score_backbone else None


def run_machine(machine, x, nsteps, device, chunk):
	"""Run the reverse loop over x, in chunks the backbone can handle."""
	step = chunk or x.shape[0]
	outs = []
	for s in range(0, x.shape[0], step):
		outs.append(machine(x[s:s + step].clone().to(device), nsteps=nsteps, device=device).detach().cpu())
	return torch.cat(outs, dim=0)


def eps_at_step(machine, t_val, x, step_index, chunk):
	"""Evaluate a machine's backbone at one reverse step, in eps units.

	ScheduledScoreMachine applies `eps *= -sqrt(beta_t)` when score_backbone=True, so
	the local score machines (which return the score) and the network backbones
	(which return eps) have to be put in the same units before comparison.
	"""
	k = None if machine.scales is None else machine.scales[step_index]
	step = chunk or x.shape[0]
	outs = []
	for s in range(0, x.shape[0], step):
		xb = x[s:s + step]
		t = torch.full((xb.shape[0],), float(t_val))
		with torch.no_grad():
			out = machine.backbone(t, xb, device=xb.device, k=k)
			if machine.score_backbone:
				beta_t = machine.noise_schedule(t).to(xb.device)
				out = out * -(beta_t[:, None, None, None] ** 0.5)
		outs.append(out.cpu())
	return torch.cat(outs, dim=0)


def cosine_per_sample(a, b):
	"""Per-image cosine similarity between two score fields."""
	af = a.reshape(a.shape[0], -1).double()
	bf = b.reshape(b.shape[0], -1).double()
	return (af * bf).sum(1) / (af.norm(dim=1) * bf.norm(dim=1)).clamp_min(1e-12)


def main():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument('--dataset', type=str, default='celeba')
	p.add_argument('--modelfile', type=str, default='checkpoints/backbone_CelebA_ResNet_zeros.pt')
	p.add_argument('--scalesfile', type=str, default='checkpoints/scales_CelebA_ResNet_zeros.pt')
	p.add_argument('--ntk_fit', type=str, required=True, help='Artifact from scripts/fit_ntk_score.py.')
	p.add_argument('--score_module_type', type=str, default='bbELS', choices=list(LOCAL_MODULES))
	p.add_argument('--network_label', type=str, default=None, help='Display label, e.g. "ResNet". Defaults to the class name.')
	p.add_argument('--data_root', type=str, default='./data')
	p.add_argument('--n_samples', type=int, default=8)
	p.add_argument('--sample_seed', type=int, default=1234)
	p.add_argument('--local_max_samples', type=int, default=10000)
	p.add_argument('--local_batch_size', type=int, default=64)
	p.add_argument('--no_matched_local', action='store_true',
	               help='Skip the local score machine restricted to the NTK regression training set.')
	p.add_argument('--n_agreement', type=int, default=8,
	               help='Held-out images for the score-agreement panel. Each one costs a full '
	                    'local-score evaluation per step, so this dominates runtime.')
	p.add_argument('--out_file', type=str, required=True)
	p.add_argument('--cpu', action='store_true')
	p.add_argument('--dry_run', action='store_true')
	args = p.parse_args()

	if args.dry_run:
		args.n_samples, args.local_max_samples, args.n_agreement = 2, 256, 2

	device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
	print(f'device={device}', flush=True)

	fit = torch.load(args.ntk_fit, map_location='cpu', weights_only=False)
	nsteps = int(fit['nsteps'])
	in_channels = int(fit['in_channels'])
	imsize = int(fit['image_size'])
	print(
		f'NTK fit: backbone={fit["backbone_class"]} nsteps={nsteps} n_train={fit["n_train"]} '
		f'readout={fit.get("readout", "center")} ridge_grid={fit["ridge_grid"]}',
		flush=True,
	)

	scales = torch.load(
		args.scalesfile if os.path.isabs(args.scalesfile) else str(ROOT / args.scalesfile),
		map_location='cpu', weights_only=False,
	)
	# scales_calibration.py saves an int *tensor*; calibrate_els_scales.py saves a
	# list. The local score modules index with k as a Python int (F.unfold rejects a
	# 0-dim tensor), so normalise either form here.
	scales = [int(k) for k in scales]
	if len(scales) != nsteps:
		raise ValueError(
			f'scales has {len(scales)} entries but the NTK fit assumes nsteps={nsteps}; the NTK '
			'dtheta are indexed by step, so the two schedules have to agree.'
		)
	print(f'scales={list(scales)}', flush=True)

	dataset, metadata = get_dataset(args.dataset, root=args.data_root, train=True)

	# --- trained network ------------------------------------------------------
	modelfile = args.modelfile if os.path.isabs(args.modelfile) else str(ROOT / args.modelfile)
	trained = torch.load(modelfile, map_location=device, weights_only=False)
	trained.eval()
	network_label = args.network_label or type(trained.backbone).__name__.replace('Minimal', '')
	machine_network = ScheduledScoreMachine(
		EpsBackboneAdapter(trained.backbone).to(device).eval(),
		in_channels=in_channels, imsize=imsize, default_time_steps=nsteps,
		noise_schedule=cosine_noise_schedule, score_backbone=False, scales=None,
	)

	# --- NTK prediction of that network --------------------------------------
	theta0, _ = build_theta0(trained.backbone, seed=fit['theta0_seed'])
	theta0.load_state_dict(fit['theta0_state_dict'])
	ntk_backbone = NTKScoreModule(theta0, fit['delta_theta'], nsteps=nsteps,
	                              jvp_mode=fit.get('jvp_mode', 'exact'),
	                              fd_step_rel=fit['fd_step_rel'])
	ntk_backbone.theta0.to(device)
	machine_ntk = ScheduledScoreMachine(
		ntk_backbone, in_channels=in_channels, imsize=imsize, default_time_steps=nsteps,
		noise_schedule=cosine_noise_schedule, score_backbone=False, scales=None,
	)

	# --- local score machine --------------------------------------------------
	local_idx = seeded_indices(len(dataset), args.local_max_samples, seed=fit['data_seed'] + 99)
	machine_local = make_local_machine(
		args.score_module_type, dataset, local_idx, in_channels, imsize, nsteps, scales, args.local_batch_size,
	)

	machines = {'network': machine_network, 'ntk': machine_ntk, 'local': machine_local}
	labels = {
		'network': network_label,
		'ntk': f'NTK {network_label}',
		'local': f'{args.score_module_type} (n={args.local_max_samples})',
	}

	# --- local score machine on exactly the NTK regression's training images ---
	if not args.no_matched_local:
		matched_idx = seeded_indices(len(dataset), fit['n_train'], seed=fit['data_seed'])
		machines['local_matched'] = make_local_machine(
			args.score_module_type, dataset, matched_idx, in_channels, imsize, nsteps, scales, args.local_batch_size,
		)
		labels['local_matched'] = f'{args.score_module_type} (n={fit["n_train"]}, matched)'

	# --- shared seed noise ----------------------------------------------------
	g = torch.Generator().manual_seed(args.sample_seed)
	seeds = torch.randn(args.n_samples, in_channels, imsize, imsize, generator=g)
	print(f'\ngenerating {args.n_samples} samples per model, nsteps={nsteps}', flush=True)

	chunks = {name: batch_chunk(m) for name, m in machines.items()}
	samples = {}
	for name, machine in machines.items():
		t0 = time.time()
		samples[name] = run_machine(machine, seeds, nsteps, device, chunks[name])
		print(f'  {labels[name]:34s} {time.time() - t0:7.1f}s', flush=True)

	# --- pointwise score agreement -------------------------------------------
	print('\nscore agreement on shared noised held-out images', flush=True)
	hold_idx = seeded_indices(len(dataset), fit['n_train'] + args.n_agreement, fit['data_seed'])[fit['n_train']:]
	x0 = torch.stack([dataset[i][0] for i in hold_idx.tolist()])
	gh = torch.Generator().manual_seed(args.sample_seed + 7)
	noise = torch.randn(x0.shape, generator=gh)

	names = list(machines)
	agreement = {f'{a}|{b}': [] for i, a in enumerate(names) for b in names[i + 1:]}
	agreement_steps = []
	for i in range(1, nsteps):
		t_val = i / nsteps
		beta_t = float(cosine_noise_schedule(torch.tensor(t_val)))
		xt = ((1 - beta_t) ** 0.5 * x0 + beta_t ** 0.5 * noise).to(device)
		eps = {n: eps_at_step(m, t_val, xt, i, chunks[n]) for n, m in machines.items()}
		agreement_steps.append(i)
		row = []
		for a_i, a in enumerate(names):
			for b in names[a_i + 1:]:
				c = cosine_per_sample(eps[a], eps[b])
				agreement[f'{a}|{b}'].append(c)
				row.append(f'{a}|{b}={c.mean():+.3f}')
		print(f'  step {i:2d} (beta={beta_t:.4f}, k={scales[i]})  ' + '  '.join(row), flush=True)

	agreement = {k: torch.stack(v) for k, v in agreement.items()}  # (n_steps, n_images)

	out = Path(args.out_file)
	out.parent.mkdir(parents=True, exist_ok=True)
	torch.save({
		'kind': 'score_comparison',
		'dataset': args.dataset,
		'metadata': metadata,
		'modelfile': modelfile,
		'network_label': network_label,
		'score_module_type': args.score_module_type,
		'labels': labels,
		'ntk_fit': str(args.ntk_fit),
		'ntk_n_train': fit['n_train'],
		'ntk_readout': fit.get('readout', 'center'),
		'ntk_pixel': fit['pixel'],
		'ntk_diagnostics': fit['diagnostics'],
		'scales': list(scales),
		'nsteps': nsteps,
		'sample_seed': args.sample_seed,
		'seeds': seeds,
		'samples': samples,
		'local_max_samples': args.local_max_samples,
		'agreement': agreement,
		'agreement_steps': agreement_steps,
		'agreement_pairs': list(agreement),
	}, out)
	print(f'\nSaved samples and agreement to {out}', flush=True)


if __name__ == '__main__':
	main()
