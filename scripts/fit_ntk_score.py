#!/usr/bin/env python
"""Fit the NTK-regression denoiser for a trained convolutional diffusion ResNet.

For each discrete noise level t_i = i/nsteps we solve the kernel-ridge problem that
NTK theory says an infinitely-wide copy of this architecture converges to under
gradient flow from its initialization theta0:

    G_i[r, :]   = grad_theta  eps_theta0(x_r, t_i)[c_r, i0, j0]     (one row per readout r)
    Theta_i     = G_i G_i^T                                          (exact empirical NTK)
    y_i[r]      = eps_r[c_r, i0, j0] - eps_theta0(x_r, t_i)[c_r, i0, j0]
    a_i         = (Theta_i + lam I)^{-1} y_i
    dtheta_i    = G_i^T a_i

lam is chosen per noise level from --ridge_grid by held-out R^2 at the readout.
Theta = G G^T squares the conditioning of G, so a fixed tiny ridge leaves dtheta
poorly determined at the low-noise steps; selecting on validation both fixes that
and gives the NTK model its best shot rather than a hand-picked lambda.

The readout is a single pixel (i0, j0), taken per colour channel -- the same
single-pixel reduction scripts/resnet_patch_ntk.py uses for its input-gradient
kernel.  Only dtheta_i is saved: at sampling time the NTK prediction is a
directional derivative of the network at theta0, so the N x N kernel never has to
be revisited (see src/utils/ntk_score.py).

Companion compute for notebooks/ntk_vs_els_samples.ipynb; run it via
scripts/slurm_ntk_vs_els.sh.

Example:
  python scripts/fit_ntk_score.py --dataset celeba \
      --modelfile checkpoints/backbone_CelebA_ResNet_zeros.pt --n_train 1024
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.utils.data import get_dataset
from src.utils.noise_schedules import cosine_noise_schedule
from src.utils.ntk_score import NTKScoreModule, assert_same_parameter_layout, build_theta0

# Legacy backbone_*.pt checkpoints were pickled with class module "models"/"utils"
# (pre-refactor layout); register aliases so torch.load's unpickler can find them.
import src.models
sys.modules['models'] = sys.modules['src.models']
import src.utils
sys.modules['utils'] = sys.modules['src.utils']


def sample_images(dataset, n, seed):
	"""Draw n images from `dataset` by a seeded permutation (labels discarded)."""
	g = torch.Generator().manual_seed(seed)
	idx = torch.randperm(len(dataset), generator=g)[:n]
	xs = []
	for i in idx.tolist():
		x, *_ = dataset[i]
		xs.append(x)
	return torch.stack(xs, dim=0)


def noise_batch(x0, beta_t, seed, reps=1):
	"""VP forward corruption matching src/utils/train.py: x = sqrt(1-b) x0 + sqrt(b) eps.

	With reps > 1 each clean image is repeated and given independent noise, so the
	regression sees several realizations of x_t per image rather than one.
	"""
	g = torch.Generator(device='cpu').manual_seed(seed)
	x0r = x0 if reps == 1 else x0.repeat(reps, *([1] * (x0.dim() - 1)))
	eps = torch.randn(x0r.shape, generator=g)
	xt = (1 - beta_t) ** 0.5 * x0r + beta_t ** 0.5 * eps
	return xt, eps


def readout_pixels(n, image_size, mode, seed, pixel_i, pixel_j):
	"""Pick the readout location for each image: fixed centre, or one random pixel each.

	'center' reproduces the convention in scripts/resnet_patch_ntk.py.  It is a bad
	choice for *generation*, though: the regression then only constrains the score at
	one location, so everywhere else the sample is driven by the raw theta0 output,
	which is untrained noise -- and worst at the borders, where zero padding breaks
	the translation equivariance that would otherwise carry the centre fit outward.
	'random' spreads the same number of readout rows over all locations, so the fit
	covers the interior and the borders, at identical cost.
	"""
	if mode == 'center':
		return torch.full((n, 2), 0, dtype=torch.long).index_put_(
			(torch.arange(n), torch.zeros(n, dtype=torch.long)), torch.tensor(pixel_i)
		).index_put_((torch.arange(n), torch.ones(n, dtype=torch.long)), torch.tensor(pixel_j))
	g = torch.Generator().manual_seed(seed)
	return torch.randint(0, image_size, (n, 2), generator=g)


def gather_readout(out, pixels):
	"""out (b,C,H,W) and pixels (b,2) -> (b,C), each image read at its own pixel."""
	b = out.shape[0]
	return out[torch.arange(b, device=out.device), :, pixels[:, 0], pixels[:, 1]]


def build_gradient_matrix(model, params, xt, t_val, pixels, channels, G, device):
	"""Fill G[r] with grad_theta of the (channel, pixel) readout, one row per readout.

	One forward per image is reused across the channel readouts via retain_graph, so
	the cost is N forwards + N*len(channels) backwards.  Rows land in G's dtype
	because the full stack is N*len(channels) x n_params.
	"""
	n = xt.shape[0]
	nc = len(channels)
	for k in range(n):
		x = xt[k:k + 1].to(device)
		t = torch.full((1,), float(t_val), device=device)
		out = model(t, x)
		i0, j0 = int(pixels[k, 0]), int(pixels[k, 1])
		for ci, c in enumerate(channels):
			f = out[0, c, i0, j0]
			grads = torch.autograd.grad(f, params, retain_graph=(ci < nc - 1))
			flat = torch.cat([g.reshape(-1) for g in grads])
			G[k * nc + ci].copy_(flat.to(G.dtype))
		del out
	return G


def autotune_chunk(n_params, device, requested, safety=0.35):
	"""Largest row-block that keeps two fp32 blocks resident on the GPU.

	The Gram assembly moves O((rows/chunk)^2) blocks across PCIe, so an
	unnecessarily small chunk makes it transfer-bound rather than compute-bound: at
	24576 rows with chunk=384 it shifts ~8 TB host->device and spends ~14 min per
	noise level moving data to do ~3 min of arithmetic. Sizing the chunk to the free
	GPU memory instead cuts the block count quadratically.
	"""
	if device.type != 'cuda':
		return requested
	free, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
	fit = int(safety * free / (n_params * 4 * 2))  # two fp32 blocks live at once
	return max(requested, min(fit, 4096))


def _to_device_fp32(block, device):
	"""Move a G block to the GPU in its stored dtype, then widen there.

	`block.to(device, torch.float32)` would widen on the host first and push twice
	the bytes over PCIe whenever G is stored in float16, which is exactly the case
	where the Gram assembly is transfer-bound.
	"""
	return block.to(device, non_blocking=True).float()


def gram_matrix(G, rows, device, chunk):
	"""Theta = G G^T for the first `rows` rows of G, chunked through the GPU in fp32."""
	theta = torch.empty((rows, rows), dtype=torch.float32)
	for a in range(0, rows, chunk):
		ea = min(a + chunk, rows)
		Ga = _to_device_fp32(G[a:ea], device)
		for b in range(a, rows, chunk):
			eb = min(b + chunk, rows)
			Gb = Ga if b == a else _to_device_fp32(G[b:eb], device)
			block = Ga @ Gb.T
			theta[a:ea, b:eb] = block.cpu()
			if b != a:
				theta[b:eb, a:ea] = block.T.cpu()
			del Gb, block
		del Ga
		torch.cuda.empty_cache() if device.type == 'cuda' else None
	return theta


def contract_delta(G, rows, coeffs, n_params, device, chunk):
	"""dtheta = G^T a, accumulated in fp32."""
	delta = torch.zeros(n_params, dtype=torch.float32, device=device)
	for a in range(0, rows, chunk):
		ea = min(a + chunk, rows)
		Ga = _to_device_fp32(G[a:ea], device)
		delta += Ga.T @ coeffs[a:ea].to(device)
		del Ga
		torch.cuda.empty_cache() if device.type == 'cuda' else None
	return delta.cpu()


def ridge_solve(theta, y, ridge_rel, device=None):
	"""Solve (Theta + lam I) a = y in float64; lam is relative to the mean NTK diagonal.

	Run on the GPU when the float64 copy fits: the solve is O(rows^3) and is called
	once per ridge candidate, so at tens of thousands of rows a CPU Cholesky would
	cost more than building the kernel in the first place.
	"""
	th = theta.double()
	lam = ridge_rel * float(th.diagonal().mean())
	th.diagonal().add_(lam)
	rhs = y.double().unsqueeze(1)

	if device is not None and device.type == 'cuda':
		need = th.numel() * 8 * 2  # the matrix plus Cholesky working space
		# mem_get_info needs an indexed device; torch.device('cuda') has no index.
		free, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
		if need < 0.7 * free:
			th, rhs = th.to(device), rhs.to(device)

	try:
		L = torch.linalg.cholesky(th)
		a = torch.cholesky_solve(rhs, L).squeeze(1)
	except RuntimeError:
		# Theta can lose positive-definiteness when rows are near-duplicates and the
		# gradients were stored in fp16; lstsq still gives the ridge solution.
		a = torch.linalg.lstsq(th, rhs).solution.squeeze(1)
	del th
	if device is not None and device.type == 'cuda':
		torch.cuda.empty_cache()
	return a.float().cpu(), lam


def main():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument('--dataset', type=str, default='celeba')
	p.add_argument('--modelfile', type=str, default='checkpoints/backbone_CelebA_ResNet_zeros.pt')
	p.add_argument('--data_root', type=str, default='./data')
	p.add_argument('--n_train', type=int, default=1024, help='Images in the NTK regression (x n_channels readouts).')
	p.add_argument('--noise_reps', type=int, default=1,
	               help='Independent noise draws per clean image. Each draw is a separate x_t and '
	                    'so a separate set of readout rows; raising it averages the regression over '
	                    'the noise distribution instead of a single realization per image.')
	p.add_argument('--freeze', type=str, nargs='*', default=[],
	               help='Substrings of parameter names to exclude from the NTK (their dtheta is '
	                    "held at zero). E.g. --freeze embs. embedding. t_embed. adaLN  to test "
	                    'whether the time-conditioning pathway is what the fit depends on.')
	p.add_argument('--only_steps', type=int, nargs='*', default=None,
	               help='Fit only these reverse-step indices. For diagnostics that only care about '
	                    'the low-noise end; the artifact is then not samplable.')
	p.add_argument('--n_holdout', type=int, default=256, help='Held-out images for the readout R^2 diagnostic.')
	p.add_argument('--nsteps', type=int, default=20, help='Reverse-diffusion steps; one regression is fit per step.')
	p.add_argument('--readout', type=str, default='random', choices=['center', 'random'],
	               help="Where the scalar readout is taken. 'random' draws one pixel per image "
	                    "and is the default: 'center' constrains the score at a single location, "
	                    'which leaves the rest of the generated image driven by untrained theta0 '
	                    'output. --pixel_i/--pixel_j apply only to center.')
	p.add_argument('--pixel_i', type=int, default=16)
	p.add_argument('--pixel_j', type=int, default=16)
	p.add_argument('--ridge_grid', type=float, nargs='+', default=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
	               help='Candidate ridges (relative to the mean NTK diagonal). One is chosen per '
	                    'noise level by held-out R^2.')
	p.add_argument('--theta0_seed', type=int, default=0)
	p.add_argument('--data_seed', type=int, default=0)
	p.add_argument('--grad_dtype', type=str, default='float32', choices=['float16', 'float32'])
	p.add_argument('--chunk', type=int, default=512, help='Rows of G moved to the GPU at once.')
	p.add_argument('--jvp_mode', type=str, default='exact', choices=['exact', 'fd'],
	               help="How to take <grad f, dtheta>. 'exact' is a double-backward jvp; 'fd' is a "
	                    'central difference and is only kept as a cross-check (it is badly '
	                    'conditioned, see src/utils/ntk_score.py).')
	p.add_argument('--fd_step_rel', type=float, default=1e-3, help='Central-difference step for the sampling-time JVP, relative to ||theta0||.')
	p.add_argument('--out_file', type=str, default=None)
	p.add_argument('--cpu', action='store_true')
	p.add_argument('--dry_run', action='store_true', help='Tiny training set, all steps, to shake out crashes.')
	args = p.parse_args()

	if args.dry_run:
		args.n_train, args.n_holdout, args.chunk = 16, 8, 16

	device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
	print(f'device={device}', flush=True)

	modelfile = args.modelfile if os.path.isabs(args.modelfile) else str(ROOT / args.modelfile)
	trained = torch.load(modelfile, map_location='cpu', weights_only=False)
	backbone = trained.backbone
	print(f'trained backbone: {type(backbone).__name__}', flush=True)
	if getattr(backbone, 'conditional', False):
		raise NotImplementedError(
			'This fit assumes an unconditional backbone; a conditional one needs the class '
			'embedding carried through the readout and the label threaded into the sampler.'
		)

	theta0, n_resets = build_theta0(backbone, seed=args.theta0_seed)
	assert_same_parameter_layout(backbone, theta0)
	theta0.to(device)

	# The NTK is taken over the *unfrozen* parameters only. dtheta is still stored at
	# full length with zeros in the frozen slots, so NTKScoreModule (which unflattens
	# against every parameter) stays valid and the frozen directions contribute
	# nothing to the JVP.
	named = list(theta0.named_parameters())
	full_len = sum(q.numel() for _, q in named)
	params, fit_slices, frozen_names = [], [], []
	off = 0
	for n_, q in named:
		if any(k in n_ for k in args.freeze):
			frozen_names.append(n_)
		else:
			params.append(q)
			fit_slices.append((off, q.numel()))
		off += q.numel()
	for p_ in params:
		p_.requires_grad_(True)
	n_params = sum(p_.numel() for p_ in params)
	if frozen_names:
		print(f'frozen from the NTK: {len(frozen_names)} tensors, '
		      f'{full_len - n_params:,} of {full_len:,} params', flush=True)

	def scatter_full(delta_fit):
		out = torch.zeros(full_len, dtype=torch.float32)
		pos = 0
		for start, size in fit_slices:
			out[start:start + size] = delta_fit[pos:pos + size]
			pos += size
		return out

	dataset, metadata = get_dataset(args.dataset, root=args.data_root, train=True)
	channels = list(range(metadata['num_channels']))
	rows = args.n_train * args.noise_reps * len(channels)
	print(f'dataset={metadata["name"]} n={len(dataset)}', flush=True)
	imsize = metadata['image_size']
	print(
		f'theta0: re-init {n_resets} submodules, n_params={n_params:,}  '
		f'readout_rows={rows}  readout={args.readout}'
		+ (f'  pixel=({args.pixel_i},{args.pixel_j})' if args.readout == 'center' else ''),
		flush=True,
	)
	x0_train = sample_images(dataset, args.n_train, seed=args.data_seed)
	x0_hold = sample_images(dataset, args.n_train + args.n_holdout, seed=args.data_seed)[args.n_train:]

	chunk = autotune_chunk(n_params, device, args.chunk)
	if chunk != args.chunk:
		print(f'chunk auto-raised {args.chunk} -> {chunk} rows to keep the Gram '
		      f'compute-bound rather than PCIe-bound', flush=True)
	args.chunk = chunk

	gdtype = getattr(torch, args.grad_dtype)
	gb = rows * n_params * (2 if gdtype == torch.float16 else 4) / 1e9
	print(f'allocating G: {rows} x {n_params:,} {args.grad_dtype} ({gb:.1f} GB)', flush=True)
	G = torch.empty((rows, n_params), dtype=gdtype)

	# Every reverse step the sampler visits needs its own dtheta, so the dry run
	# shrinks n_train rather than the step count.
	steps = list(range(1, args.nsteps + 1))
	if args.only_steps:
		steps = [s_ for s_ in steps if s_ in set(args.only_steps)]

	# One NTKScoreModule, reused across steps and ridge candidates: evaluating a
	# candidate means swapping its dtheta, which also means every held-out number
	# below is produced by the exact code path the notebook will sample through.
	ntk = NTKScoreModule(theta0, {}, nsteps=args.nsteps,
	                     jvp_mode=args.jvp_mode, fd_step_rel=args.fd_step_rel)
	ntk.theta0.to(device)
	# NTKScoreModule wraps theta0 in place and switches its parameters to
	# requires_grad=False; build_gradient_matrix needs them back on. The exact JVP
	# saves and restores the flag itself, so leaving them on is safe.
	for p_ in params:
		p_.requires_grad_(True)

	def readout(model_fn, xt, pixels, t_val, batch=32):
		outs = []
		for k in range(0, xt.shape[0], batch):
			xb = xt[k:k + batch].to(device)
			tb = torch.full((xb.shape[0],), t_val, device=device)
			pb = pixels[k:k + batch].to(device)
			outs.append(gather_readout(model_fn(tb, xb), pb).cpu())
		return torch.cat(outs)

	delta_theta, diagnostics = {}, {}
	for i in steps:
		t0 = time.time()
		t_val = i / args.nsteps
		beta_t = float(cosine_noise_schedule(torch.tensor(t_val)))
		xt, eps = noise_batch(x0_train, beta_t, seed=args.data_seed * 1000 + i, reps=args.noise_reps)
		xt_hold, eps_hold = noise_batch(x0_hold, beta_t, seed=args.data_seed * 1000 + i + 500000)
		# Re-drawn per step, so across the 20 regressions the readouts cover the frame
		# many times over rather than repeating one fixed set of locations.
		px = readout_pixels(xt.shape[0], imsize, args.readout,
		                    args.data_seed * 7919 + i, args.pixel_i, args.pixel_j)
		px_hold = readout_pixels(x0_hold.shape[0], imsize, args.readout,
		                         args.data_seed * 7919 + i + 500000, args.pixel_i, args.pixel_j)

		build_gradient_matrix(theta0, params, xt, t_val, px, channels, G, device)
		t_grad = time.time()

		# Residual targets: what gradient flow still has to fit after theta0's own output.
		with torch.no_grad():
			base = readout(theta0, xt, px, t_val, batch=64)          # (n_train, C)
			base_hold = readout(theta0, xt_hold, px_hold, t_val)     # (n_holdout, C)
		target = gather_readout(eps, px)
		y = (target - base).reshape(-1)  # row-major matches G's k*nc+ci ordering
		target_hold = gather_readout(eps_hold, px_hold)
		ss_tot = (target_hold ** 2).sum().clamp_min(1e-12)
		r2_theta0 = 1.0 - float(((base_hold - target_hold) ** 2).sum() / ss_tot)

		theta = gram_matrix(G, rows, device, args.chunk)
		sub = min(16, xt.shape[0])

		# Pick the ridge by held-out R^2. Theta = G G^T squares the conditioning of G,
		# so at a fixed tiny ridge the coefficients blow up at the low-noise steps and
		# dtheta stops being determined (visible as train_reconstruction_rel, below).
		# Selecting on validation is the standard remedy and gives the NTK model its
		# best shot rather than handicapping it with an arbitrary lambda.
		candidates = {}
		for r in args.ridge_grid:
			coeffs, lam = ridge_solve(theta, y, r, device=device)
			delta = scatter_full(contract_delta(G, rows, coeffs, n_params, device, args.chunk))
			ntk.delta_theta[i] = delta
			ntk._loaded_step = None

			pred_hold = readout(lambda tb, xb: ntk(tb, xb), xt_hold, px_hold, t_val)
			r2 = 1.0 - float(((pred_hold - target_hold) ** 2).sum() / ss_tot)

			fit = theta @ coeffs  # pre-ridge Theta a, i.e. the fitted residuals
			train_r2 = 1.0 - float(((fit - y) ** 2).sum() / (y ** 2).sum().clamp_min(1e-12))

			# End-to-end check: on its own training readouts the JVP must reproduce the
			# kernel system's fitted values, Theta a. This is the diagnostic that would
			# catch a mismatch between the parameter flattening used to build G and the
			# unflattening used to apply dtheta -- a silent corruption of everything else
			# -- and it is also where ill-conditioning shows up.
			pred_tr = readout(lambda tb, xb: ntk(tb, xb), xt[:sub], px[:sub], t_val)
			want = base[:sub].reshape(-1) + fit[:sub * len(channels)]
			recon = float((pred_tr.reshape(-1) - want).abs().max() /
			              fit[:sub * len(channels)].abs().max().clamp_min(1e-12))
			candidates[r] = {
				'delta': delta, 'lambda': lam, 'holdout_r2': r2,
				'train_r2': train_r2, 'train_reconstruction_rel': recon,
				'delta_norm': float(delta.norm()),
			}

		best_r = max(candidates, key=lambda r: candidates[r]['holdout_r2'])
		best = candidates[best_r]
		delta_theta[i] = best['delta']
		diagnostics[i] = {
			'beta_t': beta_t,
			'ridge_rel': best_r,
			'lambda': best['lambda'],
			'ntk_diag_mean': float(theta.diagonal().mean()),
			'delta_norm': best['delta_norm'],
			'train_r2': best['train_r2'],
			'holdout_r2': best['holdout_r2'],
			'holdout_r2_theta0': r2_theta0,
			'train_reconstruction_rel': best['train_reconstruction_rel'],
			'ridge_sweep': {r: {k: v for k, v in c.items() if k != 'delta'}
			                for r, c in candidates.items()},
			'seconds': time.time() - t0,
		}
		sweep = '  '.join(f'{r:.0e}:R2={c["holdout_r2"]:+.3f}' for r, c in candidates.items())
		edge = '  [EDGE: widen --ridge_grid]' if best_r in (min(candidates), max(candidates)) else ''
		print(
			f'step {i:2d}/{args.nsteps}  beta={beta_t:.4f}  ridge={best_r:.0e}  '
			f'|dtheta|={best["delta_norm"]:.3e}  R2_hold={best["holdout_r2"]:+.4f} '
			f'(theta0 {r2_theta0:+.3f})  recon={best["train_reconstruction_rel"]:.2e}  '
			f'grad={t_grad - t0:.0f}s total={time.time() - t0:.0f}s\n'
			f'          sweep: {sweep}{edge}',
			flush=True,
		)
		del candidates, theta

	del G
	torch.cuda.empty_cache() if device.type == 'cuda' else None

	out = Path(args.out_file) if args.out_file else (
		ROOT / 'results' / 'ntk_score' / f'ntk_score_{args.dataset}_{Path(modelfile).stem}_N{args.n_train}.pt'
	)
	out.parent.mkdir(parents=True, exist_ok=True)
	torch.save({
		'kind': 'ntk_score_fit',
		'dataset': args.dataset,
		'metadata': metadata,
		'modelfile': modelfile,
		'backbone_class': type(backbone).__name__,
		'in_channels': metadata['num_channels'],
		'image_size': metadata['image_size'],
		'theta0_seed': args.theta0_seed,
		'theta0_state_dict': {k: v.cpu() for k, v in theta0.state_dict().items()},
		'delta_theta': delta_theta,
		'nsteps': args.nsteps,
		'readout': args.readout,
		'noise_reps': args.noise_reps,
		'frozen': list(args.freeze),
		'rows': rows,
		'pixel': (args.pixel_i, args.pixel_j),
		'n_train': args.n_train,
		'n_holdout': args.n_holdout,
		'ridge_grid': list(args.ridge_grid),
		'grad_dtype': args.grad_dtype,
		'jvp_mode': args.jvp_mode,
		'fd_step_rel': args.fd_step_rel,
		'data_seed': args.data_seed,
		'diagnostics': diagnostics,
	}, out)
	print(f'\nSaved NTK score fit to {out}', flush=True)


if __name__ == '__main__':
	main()
