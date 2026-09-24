#!/usr/bin/env python
"""Test the low-noise identity: averaging over noise == a data-dependent ridge.

The claim being checked (kambm): at asymptotically low noise, analytically averaging
the per-clean-datapoint outer product grad f(x) grad f(x)^T over the noise is
equivalent to regularizing the *clean* outer product by sigma^2 Hf Hf^T, where
Hf = d^2 f / dtheta dx is the mixed parameter/input Hessian.

Writing g(x) = grad_theta f(x) and x = phi + sigma z, expanding to second order,

    E_z[ g g^T ]  =  E_z[g] E_z[g]^T  +  Cov_z(g),      Cov_z(g) -> sigma^2 Hf Hf^T.

Taking the trace gives a scalar identity that can be checked without ever forming a
P x P matrix, which is the whole point -- P is millions here:

    LHS(phi) = E_z||g(phi + sigma z)||^2 - ||E_z g(phi + sigma z)||^2
    RHS(phi) = sigma^2 * sum_i || grad_theta (df/dx_i) ||^2

LHS is a plain Monte-Carlo variance over noise draws.  RHS is estimated by
Hutchinson: for v ~ N(0, I_D),

    E_v || grad_theta ( v . grad_x f ) ||^2 = sum_i || grad_theta df/dx_i ||^2

since the cross terms carry E[v_i v_j] = delta_ij.

Two things come out of this beyond the identity itself:

*   whether the induced regularizer is really *data-dependent*.  If
    RHS(phi)/||g(phi)||^2 is nearly constant across phi, an isotropic lambda*I ridge
    is a fine stand-in and nothing is lost by sweeping lambda.  If it varies a lot,
    noise averaging is doing something a scalar ridge cannot.

*   the implied ridge fraction, RHS / mean_phi ||g(phi)||^2, which is directly
    comparable to the --ridge_grid value that scripts/fit_ntk_score.py selects on
    held-out R^2 at the same noise level.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.utils.data import get_dataset
from src.utils.noise_schedules import cosine_noise_schedule
from src.utils.ntk_score import build_theta0

import src.models
sys.modules['models'] = sys.modules['src.models']
import src.utils
sys.modules['utils'] = sys.modules['src.utils']
import src.models_dit  # noqa: F401


def theta_grad(model, params, t, x, c, i, j, create_graph=False):
	out = model(t, x)
	f = out[0, c, i, j]
	g = torch.autograd.grad(f, params, create_graph=create_graph)
	return torch.cat([q.reshape(-1) for q in g])


def hutchinson_mixed_hessian_sq(model, params, t, phi, c, i, j, n_probe, generator, device):
	"""Estimate sum_i ||grad_theta df/dx_i||^2 by Hutchinson probing over the input.

	`allow_unused=True` is required and is not a technicality: any parameter that
	enters f additively with respect to the input has d^2 f / dtheta dx = 0 exactly,
	so autograd reports it as unused.  Every bias, and the entire time-conditioning
	pathway (which is added to the activations and never multiplies the input), is in
	that null space.  The noise-induced regularizer therefore does not touch those
	directions at all, whereas an isotropic lambda*I ridge does -- which is precisely
	where the two regularizers stop being interchangeable.  Returns the estimate and
	the number of parameter tensors in that null space.
	"""
	total = 0.0
	n_unused = 0
	for probe in range(n_probe):
		x = phi.clone().requires_grad_(True)
		out = model(t, x)
		f = out[0, c, i, j]
		(gx,) = torch.autograd.grad(f, x, create_graph=True)
		v = torch.randn(gx.shape, generator=generator, device=device)
		d = (gx * v).sum()
		g = torch.autograd.grad(d, params, retain_graph=False, allow_unused=True)
		if probe == 0:
			n_unused = sum(1 for q in g if q is None)
		total += float(sum((q ** 2).sum() for q in g if q is not None))
	return total / n_probe, n_unused


def main():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument('--modelfile', type=str, default='checkpoints/backbone_CelebA_ResNet_zeros.pt')
	p.add_argument('--dataset', type=str, default='celeba')
	p.add_argument('--data_root', type=str, default='./data')
	p.add_argument('--n_points', type=int, default=12, help='Clean images phi to test.')
	p.add_argument('--n_noise', type=int, default=64, help='Noise draws for the Monte-Carlo LHS.')
	p.add_argument('--n_probe', type=int, default=16, help='Hutchinson probes for the RHS.')
	p.add_argument('--time_steps', type=int, nargs='+', default=[1, 2, 3, 5, 10])
	p.add_argument('--nsteps', type=int, default=20)
	p.add_argument('--sigma_override', type=float, nargs='*', default=None,
	               help='Probe these perturbation sizes instead of the schedule sigma, holding the '
	                    "network's t input at --override_step. The identity is an asymptotic "
	                    'statement in sigma, so this is how to check it actually holds as sigma->0 '
	                    'and then locate where it stops describing the real noise levels.')
	p.add_argument('--override_step', type=int, default=1)
	p.add_argument('--theta0_seed', type=int, default=0)
	p.add_argument('--seed', type=int, default=0)
	p.add_argument('--out_file', type=str, default=None)
	p.add_argument('--cpu', action='store_true')
	args = p.parse_args()

	device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
	trained = torch.load(args.modelfile, map_location='cpu', weights_only=False)
	theta0, _ = build_theta0(trained.backbone, seed=args.theta0_seed)
	theta0.to(device).eval()
	params = [q for q in theta0.parameters()]
	for q in params:
		q.requires_grad_(True)
	print(f'device={device}  backbone={type(trained.backbone).__name__}  '
	      f'n_params={sum(q.numel() for q in params):,}', flush=True)

	dataset, metadata = get_dataset(args.dataset, root=args.data_root, train=False)
	imsize, nch = metadata['image_size'], metadata['num_channels']
	g_cpu = torch.Generator().manual_seed(args.seed)
	idx = torch.randperm(len(dataset), generator=g_cpu)[:args.n_points]
	phis = torch.stack([dataset[k][0] for k in idx.tolist()]).to(device)
	pix = torch.randint(0, imsize, (args.n_points, 2), generator=g_cpu)
	chans = torch.randint(0, nch, (args.n_points,), generator=g_cpu)

	gclean_len = sum(q.numel() for q in params)
	gdev = torch.Generator(device=device).manual_seed(args.seed + 1)
	if args.sigma_override:
		# Perturbation size is set by hand; the network still sees the t it was asked
		# for, and the expansion point stays the clean image (alpha = 1).
		probes = [(args.override_step, sig ** 2, sig, 1.0) for sig in args.sigma_override]
	else:
		probes = []
		for ts in args.time_steps:
			b = float(cosine_noise_schedule(torch.tensor(ts / args.nsteps)))
			probes.append((ts, b, b ** 0.5, (1 - b) ** 0.5))

	results = {}
	for ts, beta, sigma, alpha in probes:
		t_val = ts / args.nsteps
		t = torch.full((1,), t_val, device=device)
		rel_pert = sigma * (imsize ** 2 * nch) ** 0.5 / float(phis.flatten(1).norm(dim=1).mean())
		print(f'\n=== step {ts}/{args.nsteps}  sigma={sigma:.5g}  '
		      f'||sigma z|| / ||phi|| = {rel_pert:.3g} ===', flush=True)
		print(f'  {"pt":>3}{"LHS (MC var)":>15}{"RHS (sig^2 Hf)":>16}{"ratio":>9}'
		      f'{"||g(phi)||^2":>15}{"RHS/||g||^2":>13}')
		rows, n_unused = [], 0
		for n in range(args.n_points):
			phi = phis[n:n + 1]
			c, i, j = int(chans[n]), int(pix[n, 0]), int(pix[n, 1])

			# LHS: Monte-Carlo variance of the parameter gradient over noise draws.
			#
			# Computed two-pass as mean_m ||g_m - gbar||^2, in float64, NOT as
			# E||g||^2 - ||E g||^2. The one-pass form is a difference of two numbers of
			# size ||g||^2 ~ 5e4 whose true difference falls below 1e-3 as sigma -> 0;
			# in float32 that is pure cancellation noise and it fabricates a spurious
			# variance floor, which makes the identity look like it fails at small sigma
			# when it is only the estimator failing.
			gs = torch.empty((args.n_noise, gclean_len), dtype=torch.float64, device=device)
			for m in range(args.n_noise):
				z = torch.randn(phi.shape, generator=gdev, device=device)
				x = alpha * phi + sigma * z
				gs[m] = theta_grad(theta0, params, t, x, c, i, j).double()
			gbar = gs.mean(dim=0)
			lhs = float(((gs - gbar) ** 2).sum(dim=1).mean())
			del gs, gbar

			# RHS: sigma^2 * sum_i ||grad_theta df/dx_i||^2 at the clean point.
			hf, n_unused = hutchinson_mixed_hessian_sq(theta0, params, t, alpha * phi, c, i, j,
			                                           args.n_probe, gdev, device)
			rhs = sigma ** 2 * hf

			gclean = theta_grad(theta0, params, t, alpha * phi, c, i, j).double()
			gnorm2 = float((gclean ** 2).sum())
			rows.append({'lhs': lhs, 'rhs': rhs, 'g_norm_sq': gnorm2})
			print(f'  {n:>3}{lhs:>15.4e}{rhs:>16.4e}{rhs / max(lhs, 1e-30):>9.3f}'
			      f'{gnorm2:>15.4e}{rhs / max(gnorm2, 1e-30):>13.4e}', flush=True)

		lhs_all = torch.tensor([r['lhs'] for r in rows])
		rhs_all = torch.tensor([r['rhs'] for r in rows])
		gn_all = torch.tensor([r['g_norm_sq'] for r in rows])
		frac = rhs_all / gn_all.clamp_min(1e-30)
		print(f'  identity RHS/LHS: median={float((rhs_all / lhs_all.clamp_min(1e-30)).median()):.3f}')
		print(f'  implied ridge fraction RHS/||g||^2: median={float(frac.median()):.3e}  '
		      f'min={float(frac.min()):.3e}  max={float(frac.max()):.3e}  '
		      f'spread(max/min)={float(frac.max() / frac.min().clamp_min(1e-30)):.1f}x')
		print(f'  parameter tensors with exactly zero mixed derivative (Hf null space): '
		      f'{n_unused} of {len(params)}')
		results[(ts, sigma)] = {'beta': beta, 'sigma': sigma, 'rows': rows, 'n_unused': n_unused,
		                        'rel_pert': rel_pert}

	out = Path(args.out_file) if args.out_file else (
		ROOT / 'results' / 'ntk_score' / 'noise_averaging_identity.pt')
	out.parent.mkdir(parents=True, exist_ok=True)
	torch.save({'kind': 'noise_averaging_identity',
	            # Stamped so consumers can tell this apart from the earlier one-pass
	            # float32 estimator, whose LHS was dominated by cancellation at small sigma.
	            'estimator': 'two_pass_float64',
	            'modelfile': args.modelfile,
	            'n_noise': args.n_noise, 'n_probe': args.n_probe, 'results': results}, out)
	print(f'\nSaved to {out}', flush=True)


if __name__ == '__main__':
	main()
