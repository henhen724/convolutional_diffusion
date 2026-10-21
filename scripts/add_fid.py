#!/usr/bin/env python
"""Add FID (and KID) to every run under results/mls_inception/, from the saved samples.

Inception Score has almost no dynamic range on CelebA (real images score 3.31) or
MNIST (1.92), because an ImageNet classifier has nothing useful to say about faces
or digits. That makes "no IS change" unreadable on those two datasets: it could be
the model or it could be the metric. FID compares feature *distributions* against
the real data instead, so it stays informative where IS does not.

Every run already saved samples_uint8.pt, so this needs no regeneration. Results are
merged into each run's metrics.json under 'FID' / 'KID_mean'.

  python scripts/add_fid.py --dataset cifar10 mnist celeba
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / 'scripts')):
	if _p not in sys.path:
		sys.path.insert(0, _p)

from torch.utils.data import DataLoader

from mls_inception import _Uint8, to_uint8
from src.utils.data import get_dataset

RES = ROOT / 'results' / 'mls_inception'


def real_uint8(name, max_images):
	dataset, meta = get_dataset(name, root=str(ROOT / 'data'))
	cached = list(DataLoader(dataset, batch_size=1024, num_workers=4))
	x = torch.cat([b[0] for b in cached])[:max_images]
	return _Uint8(to_uint8(x, meta['mean'], meta['std']))


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--dataset', nargs='+', default=['cifar10', 'mnist', 'celeba'])
	p.add_argument('--glob', default='2026-10-21*/*/metrics.json')
	p.add_argument('--max_real', type=int, default=50000)
	p.add_argument('--kid_subset', type=int, default=100)
	p.add_argument('--force', action='store_true')
	args = p.parse_args()

	import torch_fidelity
	cuda = torch.cuda.is_available()
	for name in args.dataset:
		runs = [f for f in sorted(RES.glob(args.glob))
		        if json.load(open(f))['dataset'] == name
		        and (args.force or 'FID' not in json.load(open(f)))
		        and (f.parent / 'samples_uint8.pt').exists()]
		if not runs:
			print(f'{name}: nothing to do', flush=True)
			continue
		real = real_uint8(name, args.max_real)
		print(f'{name}: {len(runs)} runs vs {len(real)} real images', flush=True)
		for f in runs:
			m = json.load(open(f))
			fake = _Uint8(torch.load(f.parent / 'samples_uint8.pt'))
			out = torch_fidelity.calculate_metrics(
				input1=fake, input2=real, fid=True, kid=True,
				kid_subset_size=min(args.kid_subset, len(fake)), cuda=cuda, verbose=False)
			m['FID'] = float(out['frechet_inception_distance'])
			m['KID_mean'] = float(out['kernel_inception_distance_mean'])
			m['FID_nsamples'] = int(len(fake))
			json.dump(m, open(f, 'w'), indent=1)
			tag = (f"b={m['barrier']:g}" if m['method'] == 'EBMLS'
			       else f"x{m.get('scale_multiplier') or 1:g}{'' if m.get('mult_half', 'all') == 'all' else '-' + m['mult_half']}")
			print(f"  {m['method']:5s} {tag:9s} IS={m['IS_mean']:6.3f}  FID={m['FID']:8.2f}  KID={m['KID_mean']:.4f}", flush=True)


if __name__ == '__main__':
	main()
