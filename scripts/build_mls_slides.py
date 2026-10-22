#!/usr/bin/env python
"""Build the landscape slide PDF for the entropy-barrier MLS sweep.

  python scripts/build_mls_slides.py --out results/mls_inception/2026-10-21_mls_slides.pdf
"""

import argparse
import json
import subprocess
from datetime import date
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


def _commit():
	try:
		# cwd=, not `git -C`: the stock Sherlock git is 1.8.3.1 and has no -C.
		return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
		                               cwd=str(ROOT), stderr=subprocess.DEVNULL).decode().strip()
	except Exception:
		return 'unknown'


def load():
	runs = []
	for f in sorted(RES.glob('2026-10-2*/*/metrics.json')):
		m = json.load(open(f))
		m['dir'] = f.parent
		m['mult'] = m.get('scale_multiplier') or 1.0
		m['seed'] = 1 if f.parent.name.endswith('seed1') else 0
		m['half'] = m.get('mult_half') or 'all'
		m['hi'] = m.get('scale_multiplier_hi')
		m['flat'] = m.get('flat_k')
		m['gamma'] = m.get('scale_gamma')
		# Every schedule *shape* is its own family: a shape is not a different point on the
		# uniform frontier, so each gets its own curve.  Flat and gamma runs in particular
		# carry scale_multiplier 1.0 and 2.0, so without this they would be plotted and
		# tabulated as if they were plain uniform runs at those multipliers.
		# Up- and down-ramps are separate families too: they move in opposite directions, so a
		# single curve through both would zig-zag rather than trace a frontier.
		m['family'] = ('LSflat' if m['flat'] else 'LSgamma' if m['gamma'] is not None else
		               ('LSramp_down' if m['mult'] > m['hi'] else 'LSramp_up') if m['hi'] else m['method'])
		h = '' if m['half'] == 'all' else f", {m['half']}-noise half"
		m['config'] = ('real images' if m['method'] == 'real' else
		               f"LS (flat k={m['flat']})" if m['flat'] else
		               f"LS (γ={m['gamma']:g} at x{m['mult']:g})" if m['gamma'] is not None else
		               f"LS (ramp x{m['mult']:g}→x{m['hi']:g})" if m['hi'] else
		               ('LS' if m['mult'] == 1.0 else f"LS (windows x{m['mult']:g}{h})") if m['method'] == 'LS' else
		               f"EBMLS barrier {m['barrier']:g}")
		runs.append(m)
	return runs


def curve(runs, dataset, family):
	"""Full-trajectory runs only, so the split-half runs stay off the frontier curves."""
	return sorted([r for r in runs if r['dataset'] == dataset and r['family'] == family
	               and r['half'] == 'all' and r['seed'] == 0],
	              key=lambda r: r['nn_rms_median'] if r['nn_rms_median'] is not None else 0)


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
	         f'{date.today()} · commit {_commit()} · seed 0 · 500 samples/config · IS over 5 splits · '
	         'FID vs the full training set',
	         ha='center', size=11.5, color='.25', linespacing=1.7)
	pdf.savefig(fig); plt.close(fig)


def table_page(pdf, runs):
	fig = plt.figure(figsize=FIGSIZE)
	fig.text(.5, .94, 'Metrics', ha='center', size=18, weight='bold')
	fig.text(.5, .89, 'IS higher is better, FID lower is better; NN-RMS higher = less memorized',
	         ha='center', size=11, color='.35')
	cols = ['dataset', 'config', 'seed', 'IS', 'FID', 'NN-RMS', 'copy frac']
	order = {'real': 0, 'LS': 1, 'LSramp_down': 2, 'LSramp_up': 3, 'LSgamma': 4, 'LSflat': 5,
	         'MLS': 6, 'EBMLS': 7}
	body, colours = [], []
	for r in sorted(runs, key=lambda r: (r['dataset'], order[r['family']], r['mult'], r['barrier'] or 0,
	                                     r['half'], r['seed'])):
		body.append([r['dataset'], r['config'], str(r['seed']), f"{r['IS_mean']:.3f} ± {r['IS_std']:.3f}",
		             '—' if r.get('FID') is None else f"{r['FID']:.1f}",
		             '—' if r['nn_rms_median'] is None else f"{r['nn_rms_median']:.3f}",
		             '—' if r['copy_fraction'] is None else f"{r['copy_fraction']:.3f}"])
		best = (r['method'] == 'LS' and (r['copy_fraction'] or 1) < 0.01 and r['mult'] > 1
		        and r['half'] == 'all')
		colours.append(['#d8f0d8' if best else
		                '#eef5ff' if r['family'] == 'LS' and r['mult'] == 1.0 else
		                '#f0f0f0' if r['method'] == 'real' else 'white'] * len(cols))
	ax = fig.add_axes([.05, .04, .90, .82]); ax.set_axis_off()
	t = ax.table(cellText=body, colLabels=cols, cellLoc='center', loc='upper center', cellColours=colours)
	n = len(body) + 1
	t.auto_set_font_size(False); t.set_fontsize(min(10.5, 220 / n)); t.scale(1, min(1.35, 26 / n))
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
		ls = [r for r in s if r['family'] == 'LS' and r['mult'] == 1.0]
		if ls:
			ax.axhline(ls[0]['IS_mean'], ls='--', c='k', label=f"LS baseline {ls[0]['IS_mean']:.2f}")
		rl = [r for r in s if r['method'] == 'real']
		if rl:
			ax.axhline(rl[0]['IS_mean'], ls=':', c='g', label=f"real {rl[0]['IS_mean']:.2f}")
		ax.set_xscale('log'); ax.set_xlabel('barrier (nats)'); ax.set_title(d)
		ax.legend(fontsize=9)
	axes[0][0].set_ylabel('Inception Score')
	fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def frontier_page(pdf, runs, dsets, ykey, ylab, title):
	fig, axes = plt.subplots(1, len(dsets), figsize=FIGSIZE, squeeze=False)
	fig.suptitle(title, weight='bold')
	for ax, d in zip(axes[0], dsets):
		for meth, c, mk, lab in [('LS', 'k', 'o-', 'LS (uniform multiplier)'),
		                         ('LSramp_down', 'C0', 'D-', 'LS (ramp down, wide at low noise)'),
		                         ('LSramp_up', 'C9', 'v-', 'LS (ramp up)'),
		                         ('LSgamma', 'C4', '^-', 'LS (γ: compressed range)'),
		                         ('LSflat', 'C1', 'x-', 'LS (flat, γ=0)'),
		                         ('EBMLS', 'C3', 's-', 'EBMLS (barrier)')]:
			s = [r for r in curve(runs, d, meth) if r.get(ykey) is not None]
			if not s:
				continue
			ax.plot([r['nn_rms_median'] for r in s], [r[ykey] for r in s], mk, c=c, label=lab)
			for r in s:
				tag = (f"b={r['barrier']:g}" if meth == 'EBMLS' else
				       f"k={r['flat']}" if r['flat'] else
				       f"γ={r['gamma']:g}" if r['gamma'] is not None else
				       f"{r['mult']:g}→{r['hi']:g}" if r['hi'] else f"x{r['mult']:g}")
				ax.annotate(tag, (r['nn_rms_median'], r[ykey]), fontsize=9, color=c,
				            xytext=(4, 5), textcoords='offset points')
		rl = [r for r in runs if r['dataset'] == d and r['method'] == 'real']
		if rl and rl[0].get(ykey) is not None:
			ax.axhline(rl[0][ykey], ls=':', c='g', label='real images')
		ax.set_xlabel('NN-RMS  (right = less memorized)'); ax.set_title(d); ax.legend(fontsize=9)
	axes[0][0].set_ylabel(ylab)
	fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def superadd_page(pdf, runs):
	split = [r for r in runs if r['half'] != 'all']
	base = [r for r in runs if r['dataset'] == 'cifar10' and r['family'] == 'LS' and r['mult'] == 1.0
	        and r['half'] == 'all' and r['seed'] == 0]
	full = [r for r in runs if r['dataset'] == 'cifar10' and r['family'] == 'LS' and r['mult'] == 2.0
	        and r['half'] == 'all' and r['seed'] == 0]
	if not (split and base and full):
		return
	vals = base + sorted(split, key=lambda r: r['half']) + full
	labels = ['baseline (x1)'] + [f"x2 on {r['half']}-noise half" for r in sorted(split, key=lambda r: r['half'])] \
	         + ['x2 everywhere']
	fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
	fig.suptitle('Widening half the schedule barely helps; widening all of it helps a lot',
	             weight='bold')
	for ax, (key, lab) in zip(axes, [('IS_mean', 'Inception Score  (higher better)'),
	                                 ('FID', 'FID  (lower better)')]):
		y = [r.get(key) for r in vals]
		if any(v is None for v in y):
			ax.set_axis_off(); continue
		ax.bar(range(len(y)), y, color=['.6'] + ['C0'] * len(split) + ['C2'])
		ax.axhline(y[0], ls='--', c='k', lw=1)
		ax.set_xticks(range(len(y))); ax.set_xticklabels(labels, rotation=18, ha='right', fontsize=9)
		ax.set_ylabel(lab)
	fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def shape_page(pdf, runs):
	"""Coherent widening is not the same as uniform widening: the tilt of the schedule matters."""
	down = curve(runs, 'cifar10', 'LSramp_down')
	if not down:
		return
	colour = {'LS': 'k', 'LSramp_down': 'C0', 'LSramp_up': 'C9', 'LSgamma': 'C4', 'LSflat': 'C1'}
	show = ([r for r in runs if r['dataset'] == 'cifar10' and r['family'] == 'LS'
	         and r['seed'] == 0 and r['half'] == 'all' and r['mult'] in (1.0, 2.0)]
	        + down + curve(runs, 'cifar10', 'LSramp_up')
	        + curve(runs, 'cifar10', 'LSgamma') + curve(runs, 'cifar10', 'LSflat'))
	fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
	fig.suptitle('Tilting the schedule beats widening it uniformly', weight='bold')
	for r in show:
		k = r['scales']
		axes[0].plot(np.arange(len(k)) / (len(k) - 1), k, marker='.',
		             c='.6' if r['family'] == 'LS' and r['mult'] == 1.0 else colour[r['family']],
		             label=r['config'].replace('LS ', '').strip('()'))
	axes[0].set_xlabel('t  (low noise → high noise)'); axes[0].set_ylabel('window k')
	axes[0].legend(fontsize=8); axes[0].set_title('window schedules')
	for fam, mk, lab in [('LS', 'o-', 'uniform'), ('LSramp_down', 'D-', 'ramp down'),
	                     ('LSramp_up', 'v-', 'ramp up'), ('LSgamma', '^-', 'γ-compressed'),
	                     ('LSflat', 'x-', 'flat (γ=0)')]:
		s = [r for r in curve(runs, 'cifar10', fam) if r.get('FID') is not None]
		if not s:
			continue
		axes[1].plot([r['nn_rms_median'] for r in s], [r['FID'] for r in s], mk, c=colour[fam], label=lab)
		for r in s:
			axes[1].annotate(f"k={r['flat']}" if r['flat'] else f"γ={r['gamma']:g}" if r['gamma'] is not None
			                 else f"{r['mult']:g}→{r['hi']:g}" if r['hi'] else f"x{r['mult']:g}",
			                 (r['nn_rms_median'], r['FID']), fontsize=9, color=colour[fam],
			                 xytext=(4, 5), textcoords='offset points')
	axes[1].set_xlabel('NN-RMS  (right = less memorized)'); axes[1].set_ylabel('FID  (lower better)')
	axes[1].legend(fontsize=9); axes[1].set_title('CIFAR-10 FID frontier by schedule shape')
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
	d = dsets[0]
	s = [r for r in runs if r['dataset'] == d and r['method'] != 'real']
	nocopy = [r for r in s if r['method'] == 'LS' and r['copy_fraction'] < 0.01]
	picks = ([r for r in s if r['family'] == 'LS' and r['mult'] == 1.0][:1]
	         + sorted(nocopy, key=lambda r: -r['IS_mean'])[:1]
	         + sorted([r for r in s if r['method'] == 'EBMLS'], key=lambda r: -r['IS_mean'])[:1]
	         + sorted([r for r in s if r['method'] == 'LS'], key=lambda r: -r['copy_fraction'])[:1])
	picks = [r for i, r in enumerate(picks) if r['dir'] not in [q['dir'] for q in picks[:i]]]
	if not picks:
		return
	fig, axes = plt.subplots(len(picks), 1, figsize=FIGSIZE, squeeze=False)
	fig.suptitle(f'{d} samples: baseline, best non-copying, best EBMLS, and the memorizing end',
	             weight='bold')
	for ax, r in zip(axes[:, 0], picks):
		full = torch.load(r['dir'] / 'samples_uint8.pt')
		u8 = full[torch.linspace(0, len(full) - 1, 14).long()]  # spread over classes, not the first 14
		ax.imshow(np.concatenate(list(u8.permute(0, 2, 3, 1).numpy()), axis=1), rasterized=True)
		ax.set_axis_off()
		ax.set_title(f"{r['config']}  —  IS {r['IS_mean']:.2f},  NN-RMS {r['nn_rms_median']:.3f},  "
		             f"copies {r['copy_fraction']:.1%}", fontsize=11)
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
	p.add_argument('--png_dir', default=None, help='also dump each page as a PNG, to eyeball the layout')
	args = p.parse_args()
	if args.png_dir:
		Path(args.png_dir).mkdir(parents=True, exist_ok=True)
		_orig, _n = PdfPages.savefig, [0]

		def savefig(self, figure=None, **kw):
			_n[0] += 1
			(figure or plt.gcf()).savefig(f'{args.png_dir}/page{_n[0]}.png', dpi=70)
			return _orig(self, figure, **kw)

		PdfPages.savefig = savefig
	runs = load()
	dsets = [d for d in ['cifar10', 'mnist', 'celeba'] if any(r['dataset'] == d for r in runs)]
	Path(args.out).parent.mkdir(parents=True, exist_ok=True)
	with PdfPages(args.out) as pdf:
		title_page(pdf, runs, args.headline)
		table_page(pdf, runs)
		barrier_page(pdf, runs, dsets)
		frontier_page(pdf, runs, dsets, 'IS_mean', 'Inception Score  (higher better)',
		              'The test that counts: IS at matched distance to the training set')
		frontier_page(pdf, runs, dsets, 'FID', 'FID  (lower better)',
		              'Same frontier under FID — the metric that still works on CelebA and MNIST')
		superadd_page(pdf, runs)
		shape_page(pdf, runs)
		schedule_page(pdf, runs, dsets)
		samples_page(pdf, runs, dsets)
		verdict_page(pdf, Path(args.verdict).read_text())
	print(f'wrote {args.out}  ({Path(args.out).stat().st_size / 1e6:.2f} MB)')


if __name__ == '__main__':
	main()
