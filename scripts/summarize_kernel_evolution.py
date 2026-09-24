#!/usr/bin/env python
"""Print the NTK drift tables produced by scripts/ntk_kernel_evolution.py.

`alignment` is the one to read first: it is scale-invariant, so it isolates whether
the kernel's *shape* changed, as opposed to `norm_ratio`, which only says the kernel
grew or shrank.  Lazy/NTK training predicts alignment stays at 1.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def main():
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument('--labels', type=str, nargs='+',
	               default=['ResNet_steps', 'DiT_steps', 'ResNet_epochs', 'DiT_epochs'])
	p.add_argument('--dataset', type=str, default='celeba')
	p.add_argument('--compact', action='store_true',
	               help='One table per trajectory: alignment vs training time, columns = noise level.')
	args = p.parse_args()

	if args.compact:
		for lab in args.labels:
			f = ROOT / 'results' / 'ntk_score' / f'kernel_evolution_{args.dataset}_{lab}.pt'
			if not f.exists():
				print(f'(missing {f.name})')
				continue
			a = torch.load(f, map_location='cpu', weights_only=False)
			unit = 'opt step' if lab.endswith('_steps') else 'epoch'
			tss = list(a['results'])
			print(f'##### {lab}: kernel ALIGNMENT with theta_0 (1.0 = lazy) #####')
			print(f'{unit:>10}' + ''.join(f'{f"t={t}":>10}' for t in tss))
			steps = [r['step'] for r in a['results'][tss[0]]['rows']]
			for k, st in enumerate(steps):
				print(f'{st:>10}' + ''.join(f'{a["results"][t]["rows"][k]["alignment"]:>10.4f}' for t in tss))
			print(f'{"":>10}' + ''.join(f'{"":>10}' for t in tss))
			print(f'##### {lab}: kernel NORM RATIO ||K_t||/||K_0|| #####')
			print(f'{unit:>10}' + ''.join(f'{f"t={t}":>10}' for t in tss))
			for k, st in enumerate(steps):
				print(f'{st:>10}' + ''.join(f'{a["results"][t]["rows"][k]["norm_ratio"]:>10.4f}' for t in tss))
			print()
		return

	for lab in args.labels:
		f = ROOT / 'results' / 'ntk_score' / f'kernel_evolution_{args.dataset}_{lab}.pt'
		if not f.exists():
			print(f'(missing {f.name})')
			continue
		a = torch.load(f, map_location='cpu', weights_only=False)
		unit = 'opt step' if lab.endswith('_steps') else 'epoch'
		print(f'########## {lab}   n_eval={a["n_eval"]}  readout={a["readout"]}  x-axis: {unit} ##########')
		for ts, res in a['results'].items():
			sigma = res['beta_t'] ** 0.5
			print(f'-- noise level {ts}/{a["nsteps"]}   beta={res["beta_t"]:.4f}  sigma={sigma:.3f}')
			print(f'   {unit:>10}{"alignment":>11}{"rel_fro":>10}{"norm_ratio":>12}{"l2":>13}')
			for r in res['rows']:
				print(f'   {r["step"]:>10}{r["alignment"]:>11.4f}{r["rel_fro"]:>10.4f}'
				      f'{r["norm_ratio"]:>12.4f}{r["l2"]:>13.3e}')
			print()


if __name__ == '__main__':
	main()
