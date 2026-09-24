#!/usr/bin/env python
"""Where does the NTK weight perturbation live, and how did the ridge sweep behave?

Two read-outs from an existing scripts/fit_ntk_score.py artifact, no new training:

1. `dtheta` decomposed by parameter group.  The hypothesis under test is that the
   NTK regression is forced to make disproportionately large changes to the
   time-embedding pathway.  Reported two ways, because they answer different
   questions:
     share    -- fraction of ||dtheta||^2 in the group (where the perturbation is)
     rel      -- ||dtheta_g|| / ||theta0_g||   (how hard the group is being pushed
                 relative to its own scale; this is the one that speaks to "it will
                 want to make big changes to that parameter")

2. The held-out R^2 vs ridge curve recorded at each noise level, so the choice of
   lambda and how sharply it matters are both visible.

Example:
  python scripts/analyze_ntk_fit.py --fit results/ntk_score/ntk_fit_celeba_ResNet_N1024.pt
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.utils.ntk_score import build_theta0

import src.models
sys.modules['models'] = sys.modules['src.models']
import src.utils
sys.modules['utils'] = sys.modules['src.utils']
import src.models_dit  # noqa: F401

# Substrings that identify the time-conditioning pathway in each backbone.
# MinimalResNet: `embs.*` are the per-layer MLPs that process the sinusoidal time
# embedding, `embedding.*` is the embedding itself.
# MinimalDiT: `t_embed.*` is the time MLP and every `adaLN.*` head is the point at
# which time modulates a block, so both belong to the time pathway.
TIME_KEYS = ('embs.', 'embedding.', 't_embed.', 'adaLN')


def group_of(name):
	# DiT checkpoints are wrapped in VPredToEps, so every parameter is prefixed
	# 'backbone.'; strip it so the startswith rules below fire for both backbones.
	name = name[len('backbone.'):] if name.startswith('backbone.') else name
	if any(k in name for k in TIME_KEYS):
		return 'time_conditioning'
	if 'pos_embed' in name:
		return 'pos_embed'
	if 'patch_embed' in name or 'up_projection' in name:
		return 'input_proj'
	if 'down_projection' in name or name.startswith('final.'):
		return 'output_proj'
	if 'attn' in name:
		return 'attention'
	if '.mlp' in name or name.startswith('mlp'):
		return 'mlp'
	if 'convs.' in name:
		return 'convs'
	return 'other'


def main():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument('--fit', type=str, required=True)
	p.add_argument('--steps', type=int, nargs='+', default=[1, 2, 5, 10, 20])
	p.add_argument('--out_file', type=str, default=None)
	args = p.parse_args()

	fit = torch.load(args.fit, map_location='cpu', weights_only=False)
	trained = torch.load(fit['modelfile'], map_location='cpu', weights_only=False)
	theta0, _ = build_theta0(trained.backbone, seed=fit['theta0_seed'])
	theta0.load_state_dict(fit['theta0_state_dict'])

	names, sizes, t0norms = [], [], []
	for n, q in theta0.named_parameters():
		names.append(n)
		sizes.append(q.numel())
		t0norms.append(float(q.norm()))
	total = sum(sizes)
	print(f'{fit["backbone_class"]}  n_params={total:,}  readout={fit.get("readout", "center")}  '
	      f'n_train={fit["n_train"]}\n')

	summary = {}
	for i in args.steps:
		if i not in fit['delta_theta']:
			continue
		d = fit['delta_theta'][i]
		if d.numel() != total:
			raise ValueError(f'dtheta has {d.numel()} entries, theta0 has {total}')
		groups, per_param = {}, []
		off = 0
		for n, sz, t0n in zip(names, sizes, t0norms):
			seg = d[off:off + sz]
			off += sz
			sq = float((seg ** 2).sum())
			g = groups.setdefault(group_of(n), {'sq': 0.0, 't0sq': 0.0, 'n': 0})
			g['sq'] += sq
			g['t0sq'] += t0n ** 2
			g['n'] += sz
			per_param.append((n, sq ** 0.5, t0n))

		dn2 = sum(g['sq'] for g in groups.values())
		diag = fit['diagnostics'][i]
		print(f'=== step {i}  beta={diag["beta_t"]:.4f}  ridge={diag["ridge_rel"]:.0e}  '
		      f'R2_hold={diag["holdout_r2"]:+.4f}  |dtheta|={dn2 ** 0.5:.4g} ===')
		print(f'  {"group":<20}{"params":>12}{"share":>9}{"|dth_g|":>12}{"|th0_g|":>12}{"rel":>10}')
		for gname, g in sorted(groups.items(), key=lambda kv: -kv[1]['sq']):
			rel = (g['sq'] ** 0.5) / max(g['t0sq'] ** 0.5, 1e-12)
			print(f'  {gname:<20}{g["n"]:>12,}{g["sq"] / dn2:>9.3f}'
			      f'{g["sq"] ** 0.5:>12.3g}{g["t0sq"] ** 0.5:>12.3g}{rel:>10.3g}')
		# Ranked by absolute |dtheta|, not by the ratio: parameters that are
		# zero-initialized (GroupNorm biases, and every adaLN head under adaln_zero)
		# have ||theta0_p|| = 0, so the ratio is unbounded for them and would other-
		# wise dominate this list without meaning anything.
		scale = max(t0norms)
		top = sorted(per_param, key=lambda r: -r[1])[:6]
		print('  most-perturbed individual tensors (by |dtheta|):')
		for n, dnorm, t0n in top:
			tag = '   (zero-init)' if t0n < 1e-8 * scale else f'   rel={dnorm / t0n:.3g}'
			print(f'    {n:<40}|dtheta|={dnorm:>10.3g}{tag}')
		summary[i] = {'groups': groups, 'delta_norm': dn2 ** 0.5,
		              'ridge_sweep': diag.get('ridge_sweep', {})}

		sweep = diag.get('ridge_sweep', {})
		if sweep:
			print('  ridge sweep (held-out R^2):  ' +
			      '  '.join(f'{r:.0e}:{c["holdout_r2"]:+.3f}' for r, c in sorted(sweep.items())))
		print()

	if args.out_file:
		out = Path(args.out_file)
		out.parent.mkdir(parents=True, exist_ok=True)
		torch.save({'kind': 'ntk_fit_analysis', 'fit': args.fit,
		            'backbone_class': fit['backbone_class'], 'summary': summary}, out)
		print(f'Saved to {out}')


if __name__ == '__main__':
	main()
