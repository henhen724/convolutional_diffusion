#!/usr/bin/env python
"""Build the landscape slide PDF for the entropy-barrier MLS sweep.

  python scripts/build_mls_slides.py --out results/mls_inception/2026-10-21_mls_slides.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results' / 'mls_inception'
FIGSIZE = (11, 6.2)
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 13})


def load():
	runs = []
	for f in sorted(RES.glob('2026-10-21*/*/metrics.json')):
		m = json.load(open(f))
		m['dir'] = f.parent
		m['mult'] = m.get('scale_multiplier') or 1.0
		m['config'] = ('real images' if m['method'] == 'real' else
		               ('LS' if m['mult'] == 1.0 else f"LS (windows x{m['mult']:g})") if m['method'] == 'LS' else
		               f"EBMLS barrier {m['barrier']:g}")
		runs.append(m)
	return runs


def title_page(pdf, runs, headline):
	fig = plt.figure(figsize=FIGSIZE)
	fig.text(.5, .80, 'Multilayer local score with an entropy-barrier window',
	         ha='center', size=22, weight='bold')
	fig.text(.5, .72, 'Does a second, larger-window LS layer raise Inception Score?',
	         ha='center', size=15, color='.3')
	fig.text(.5, .55, headline, ha='center', size=16, weight='bold', wrap=True)
	fig.text(.5, .22,
	         'Layer 2 sees only layer 1\'s estimate, forms a posterior over all training images in a\n'
	         'larger window, and adds the expected error of that estimate. The window at each step is the\n'
	         'largest whose posterior entropy is still above a barrier — i.e. has not collapsed onto one image.\n\n'
	         '2026-10-21 · commit 2c2dc90 · seed 0 · 500 samples/config · IS over 5 splits',
	         ha='center', size=11.5, color='.25', linespacing=1.7)
	pdf.savefig(fig); plt.close(fig)


def table_page(pdf, runs):
	fig = plt.figure(figsize=FIGSIZE)
	fig.text(.5, .94, 'Metrics', ha='center', size=18, weight='bold')
	fig.text(.5, .89, 'NN-RMS: median distance to nearest training image — higher = less memorized',
	         ha='center', size=11, color='.35')
	cols = ['dataset', 'config', 'IS', 'NN-RMS', 'copy frac']
	order = {'real': 0, 'LS': 1, 'MLS': 2, 'EBMLS': 3}
	body, colours = [], []
	for r in sorted(runs, key=lambda r: (r['dataset'], order[r['method']], r['mult'], r['barrier'] or 0)):
		body.append([r['dataset'], r['config'], f"{r['IS_mean']:.3f} ± {r['IS_std']:.3f}",
		             '—' if r['nn_rms_median'] is None else f"{r['nn_rms_median']:.3f}",
		             '—' if r['copy_fraction'] is None else f"{r['copy_fraction']:.3f}"])
		colours.append(['#eef5ff' if r['method'] == 'LS' and r['mult'] == 1.0 else
		                '#f0f0f0' if r['method'] == 'real' else 'white'] * len(cols))
	ax = fig.add_axes([.06, .05, .88, .80]); ax.set_axis_off()
	t = ax.table(cellText=body, colLabels=cols, cellLoc='center', loc='upper center', cellColours=colours)
	t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1, 1.35)
	pdf.savefig(fig); plt.close(fig)


def barrier_page(pdf, runs, dsets):
	fig, axes = plt.subplots(1, len(dsets), figsize=FIGSIZE, squeeze=False)
	fig.suptitle('Inception Score vs entropy barrier', weight='bold')
	for ax, d in zip(axes[0], dsets):
		s = [r for r in runs if r['dataset'] == d]
		eb = sorted([r for r in s if r['method'] == 'EBMLS'], key=lambda r: r['barrier'])
		if eb:
			ax.errorbar([r['barrier'] for r in eb], [r['IS_mean'] for r in eb],
			            yerr=[r['IS_std'] for r in eb], marker='o', capsize=3, c='C3', label='EBMLS')
		ls = [r for r in s if r['method'] == 'LS' and r['mult'] == 1.0]
		if ls:
			ax.axhline(ls[0]['IS_mean'], ls='--', c='k', label=f"LS baseline {ls[0]['IS_mean']:.2f}")
		rl = [r for r in s if r['method'] == 'real']
		if rl:
			ax.axhline(rl[0]['IS_mean'], ls=':', c='g', label=f"real {rl[0]['IS_mean']:.2f}")
		ax.set_xscale('log'); ax.set_xlabel('barrier (nats)'); ax.set_title(d)
		ax.legend(fontsize=9)
	axes[0][0].set_ylabel('Inception Score')
	fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def frontier_page(pdf, runs, dsets):
	fig, axes = plt.subplots(1, len(dsets), figsize=FIGSIZE, squeeze=False)
	fig.suptitle('The test that counts: IS at matched distance to the training set', weight='bold')
	for ax, d in zip(axes[0], dsets):
		s = [r for r in runs if r['dataset'] == d and r['method'] != 'real']
		for meth, c, mk, lab in [('LS', 'k', 'o-', 'LS (window multiplier)'),
		                         ('EBMLS', 'C3', 's-', 'EBMLS (barrier)')]:
			sel = sorted([r for r in s if r['method'] == meth], key=lambda r: r['nn_rms_median'])
			if not sel:
				continue
			ax.plot([r['nn_rms_median'] for r in sel], [r['IS_mean'] for r in sel], mk, c=c, label=lab)
			for r in sel:
				tag = f"x{r['mult']:g}" if meth == 'LS' else f"b={r['barrier']:g}"
				ax.annotate(tag, (r['nn_rms_median'], r['IS_mean']), fontsize=9, color=c,
				            xytext=(4, 5), textcoords='offset points')
		rl = [r for r in runs if r['dataset'] == d and r['method'] == 'real']
		if rl:
			ax.axhline(rl[0]['IS_mean'], ls=':', c='g', label='real images')
		ax.set_xlabel('NN-RMS  (right = less memorized)'); ax.set_title(d); ax.legend(fontsize=9)
	axes[0][0].set_ylabel('Inception Score')
	fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def schedule_page(pdf, runs, dsets):
	fig, axes = plt.subplots(1, len(dsets), figsize=FIGSIZE, squeeze=False)
	fig.suptitle('The barrier produces a coarse-to-fine schedule on its own', weight='bold')
	for ax, d in zip(axes[0], dsets):
		for r in sorted([r for r in runs if r['dataset'] == d and r['method'] == 'EBMLS'],
		                key=lambda r: r['barrier']):
			kk = r['layer2_k_by_t'] or {}
			if kk:
				ax.plot([float(t) for t in kk], list(kk.values()), marker='.', label=f"barrier {r['barrier']:g}")
		ax.set_xlabel('t  (noise level)'); ax.set_title(d); ax.legend(fontsize=9)
	axes[0][0].set_ylabel('mean layer-2 window k')
	fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def samples_page(pdf, runs, dsets):
	picks = []
	for d in dsets:
		s = [r for r in runs if r['dataset'] == d]
		ls = [r for r in s if r['method'] == 'LS' and r['mult'] == 1.0]
		eb = sorted([r for r in s if r['method'] == 'EBMLS'], key=lambda r: -r['IS_mean'])
		picks += ls[:1] + eb[:1]
	if not picks:
		return
	fig, axes = plt.subplots(len(picks), 1, figsize=FIGSIZE, squeeze=False)
	fig.suptitle('Samples: baseline vs best barrier', weight='bold')
	for ax, r in zip(axes[:, 0], picks):
		u8 = torch.load(r['dir'] / 'samples_uint8.pt')[:14]
		ax.imshow(np.concatenate(list(u8.permute(0, 2, 3, 1).numpy()), axis=1), rasterized=True)
		ax.set_axis_off()
		ax.set_title(f"{r['dataset']} — {r['config']}  (IS {r['IS_mean']:.2f}, NN-RMS {r['nn_rms_median']:.3f})",
		             fontsize=10)
	fig.tight_layout(); pdf.savefig(fig, dpi=150); plt.close(fig)


def verdict_page(pdf, text):
	fig = plt.figure(figsize=FIGSIZE)
	fig.text(.5, .92, 'Where this leaves us', ha='center', size=18, weight='bold')
	fig.text(.07, .80, text, size=12.5, va='top', linespacing=1.8)
	pdf.savefig(fig); plt.close(fig)


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--out', required=True)
	p.add_argument('--headline', required=True)
	p.add_argument('--verdict', required=True, help='path to a text file with the closing slide')
	args = p.parse_args()
	runs = load()
	dsets = [d for d in ['cifar10', 'mnist', 'celeba'] if any(r['dataset'] == d for r in runs)]
	Path(args.out).parent.mkdir(parents=True, exist_ok=True)
	with PdfPages(args.out) as pdf:
		title_page(pdf, runs, args.headline)
		table_page(pdf, runs)
		barrier_page(pdf, runs, dsets)
		frontier_page(pdf, runs, dsets)
		schedule_page(pdf, runs, dsets)
		samples_page(pdf, runs, dsets)
		verdict_page(pdf, Path(args.verdict).read_text())
	print(f'wrote {args.out}  ({Path(args.out).stat().st_size / 1e6:.2f} MB)')


if __name__ == '__main__':
	main()
