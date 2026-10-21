#!/usr/bin/env python
"""Generate notebooks/mls_entropy_barrier_results.ipynb from results/mls_inception/.

Built from a script, like build_ntk_analysis_notebook.py, so the prose stays in
version control as readable Python and the notebook can be regenerated after a
rerun. Execute it with:

  jupyter nbconvert --to notebook --execute --inplace \
      notebooks/mls_entropy_barrier_results.ipynb
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'notebooks' / 'mls_entropy_barrier_results.ipynb'


def md(text):
	return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n').splitlines(keepends=True)}


def code(text):
	return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
	        'source': text.strip('\n').splitlines(keepends=True)}


CELLS = []

CELLS.append(md(r'''
# Multilayer local score with an entropy-barrier window: does it raise Inception Score?

**Run:** 2026-10-21, commit `2c2dc90` (+ `--scale_multiplier` control, commit noted in the
config cell below). Seed 0, 500 samples per configuration, Inception Score over 5 splits.

### The model

`MultilayerLocalScoreModule` adds a second layer on top of the existing `LocalScoreModule`:

1. Layer 1 is plain LS: a per-pixel Gaussian posterior over training patches in a $k_1\times k_1$
   window gives $\hat{x}_1$.
2. Layer 2 sees **only** $\hat{x}_1$. Using a larger window $k_2$ it forms a posterior
   $w_n(p)$ over all training images $n$, and adds the expected error of the layer-1 estimate,
   $\;\delta(p) = \sum_n w_n(p)\,\bigl(x_n(p) - \hat{x}_1(p)\bigr)$.
3. The full estimate is $\hat{x} = \hat{x}_1 + \delta$, converted to a score by the usual
   convention $s = (a_t\hat{x} - x)/b_t^2$.

`EntropyBarrierMLSModule` chooses $k_2$ at every denoising step: the **largest** candidate window
whose layer-2 posterior entropy is still $\ge$ `barrier` nats. Entropy falls as the window grows
(a big window matches one training image and the posterior collapses), so the barrier is a floor on
how much of the training set the layer is still averaging over.

> **Wording check for hshunt.** The request was "the large length scale that stays *below* some
> posterior entropy barrier". Entropy *decreases* with $k$ here, so "largest $k$ below the barrier"
> is degenerate — it always selects the biggest candidate. This implements the non-degenerate
> reading: the largest scale that has *not yet* collapsed onto a single training image. If you
> meant the other one, it is a one-line flip of the comparison.

### What this notebook has to settle

IS can be bought outright with memorization: real CIFAR-10 images score 8.75 on this eval, so any
model that copies the training set wins. Every run therefore also reports the median RMS distance
from each sample to its nearest training image (same class when conditional), and the fraction of
samples within 0.05 RMS of one. **The claim to test is not "higher IS" but "higher IS at matched
distance to the training set"**, against plain LS with its windows scaled up as the control.
'''))

CELLS.append(md('## Load every run'))

CELLS.append(code(r'''
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
RES = ROOT / 'results' / 'mls_inception'

rows = []
for f in sorted(RES.glob('2026-10-21*/*/metrics.json')):
    m = json.load(open(f))
    m['run'] = f.parent.name
    m['dir'] = str(f.parent)
    rows.append(m)
df = pd.DataFrame(rows)
df['mult'] = (df['scale_multiplier'] if 'scale_multiplier' in df else np.nan)
df['mult'] = df['mult'].fillna(1.0)


def label(r):
    if r.method == 'real':
        return 'real images'
    if r.method == 'LS':
        return 'LS' if r.mult == 1.0 else f'LS (windows x{r.mult:g})'
    return f'EBMLS barrier {r.barrier:g}'


df['config'] = df.apply(label, axis=1)
print(f'{len(df)} runs; commit:')
!cd {ROOT} && git rev-parse --short HEAD && git status --porcelain scripts/ src/ | head
'''))

CELLS.append(md(r'''
## Metrics

`IS` is the Inception Score (higher = better, 5 splits; the $\pm$ is the split spread, not a seed
spread — with 500 samples it understates run-to-run variation). `NN-RMS` is the median distance to
the nearest training image in normalized pixel units (**higher = less like the training set**);
`copy` is the fraction of samples within 0.05 RMS of a training image.
'''))

CELLS.append(code(r'''
tab = (df.sort_values(['dataset', 'method', 'mult', 'barrier'])
         [['dataset', 'config', 'IS_mean', 'IS_std', 'nn_rms_median', 'copy_fraction', 'gen_seconds']]
         .rename(columns={'IS_mean': 'IS', 'IS_std': '+-', 'nn_rms_median': 'NN-RMS',
                          'copy_fraction': 'copy', 'gen_seconds': 'gen_s'}))
tab.round({'IS': 3, '+-': 3, 'NN-RMS': 3, 'copy': 3, 'gen_s': 0}).to_string(index=False)
'''))

CELLS.append(md(r'''
## Figure 1 — IS against the barrier

The LS baseline (dashed) is the previous best on this eval; the real-image ceiling (dotted) is what
pure memorization would score.
'''))

CELLS.append(code(r'''
dsets = [d for d in ['cifar10', 'mnist', 'celeba'] if d in set(df.dataset)]
fig, axes = plt.subplots(1, len(dsets), figsize=(5 * len(dsets), 4), squeeze=False)
for ax, d in zip(axes[0], dsets):
    s = df[df.dataset == d]
    eb = s[s.method == 'EBMLS'].sort_values('barrier')
    if len(eb):
        ax.errorbar(eb.barrier, eb.IS_mean, yerr=eb.IS_std, marker='o', capsize=3, label='EBMLS')
    ls = s[(s.method == 'LS') & (s.mult == 1.0)]
    if len(ls):
        ax.axhline(ls.IS_mean.iloc[0], ls='--', c='k', label=f'LS baseline ({ls.IS_mean.iloc[0]:.2f})')
        ax.axhspan(*(ls.IS_mean.iloc[0] + np.array([-1, 1]) * ls.IS_std.iloc[0]), color='k', alpha=.08)
    rl = s[s.method == 'real']
    if len(rl):
        ax.axhline(rl.IS_mean.iloc[0], ls=':', c='g', label=f'real images ({rl.IS_mean.iloc[0]:.2f})')
    ax.set_xscale('log'); ax.set_xlabel('entropy barrier (nats)'); ax.set_ylabel('Inception Score')
    ax.set_title(d); ax.legend(fontsize=8)
fig.tight_layout()
'''))

CELLS.append(md(r'''
## Figure 2 — the figure that decides it: IS against distance to the training set

Every point is one configuration. Right = samples are far from the training set (not copying);
up = higher IS. A method only beats the baseline if it sits **above the single-layer LS curve at the
same x**. The LS points at several window multipliers trace that single-layer frontier.
'''))

CELLS.append(code(r'''
fig, axes = plt.subplots(1, len(dsets), figsize=(5 * len(dsets), 4.2), squeeze=False)
for ax, d in zip(axes[0], dsets):
    s = df[(df.dataset == d) & (df.method != 'real')]
    ls = s[s.method == 'LS'].sort_values('nn_rms_median')
    eb = s[s.method == 'EBMLS'].sort_values('nn_rms_median')
    if len(ls):
        ax.plot(ls.nn_rms_median, ls.IS_mean, 'o-', c='k', label='LS (window multiplier)')
        for _, r in ls.iterrows():
            ax.annotate(f'x{r.mult:g}', (r.nn_rms_median, r.IS_mean), fontsize=8,
                        xytext=(3, -10), textcoords='offset points')
    if len(eb):
        ax.errorbar(eb.nn_rms_median, eb.IS_mean, yerr=eb.IS_std, fmt='s-', c='C3',
                    capsize=3, label='EBMLS (barrier)')
        for _, r in eb.iterrows():
            ax.annotate(f'b={r.barrier:g}', (r.nn_rms_median, r.IS_mean), fontsize=8,
                        xytext=(3, 6), textcoords='offset points', color='C3')
    rl = df[(df.dataset == d) & (df.method == 'real')]
    if len(rl):
        ax.axhline(rl.IS_mean.iloc[0], ls=':', c='g', label='real images')
    ax.set_xlabel('median NN-RMS to training set  (right = less memorized)')
    ax.set_ylabel('Inception Score'); ax.set_title(d); ax.legend(fontsize=8)
fig.tight_layout()
'''))

CELLS.append(md(r'''
## Figure 3 — the window schedule the barrier produces

`layer2_k_by_t` is the average layer-2 window the barrier selected at each denoising step
($k=0$ means no candidate passed, so the step is plain LS). Nothing imposes a schedule; it falls
out of the entropy constraint.
'''))

CELLS.append(code(r'''
fig, axes = plt.subplots(1, len(dsets), figsize=(5 * len(dsets), 3.6), squeeze=False)
for ax, d in zip(axes[0], dsets):
    for _, r in df[(df.dataset == d) & (df.method == 'EBMLS')].sort_values('barrier').iterrows():
        kk = r.layer2_k_by_t or {}
        if kk:
            t = [float(x) for x in kk]
            ax.plot(t, [kk[x] for x in kk], marker='.', label=f'barrier {r.barrier:g}')
    ax.set_xlabel('t (noise level)'); ax.set_ylabel('mean layer-2 window k')
    ax.set_title(d); ax.legend(fontsize=8)
fig.tight_layout()
'''))

CELLS.append(md('## Figure 4 — how close the samples get to the training set'))

CELLS.append(code(r'''
fig, axes = plt.subplots(1, len(dsets), figsize=(5 * len(dsets), 3.6), squeeze=False)
for ax, d in zip(axes[0], dsets):
    for _, r in df[(df.dataset == d) & (df.method != 'real')].sort_values(['method', 'barrier']).iterrows():
        p = Path(r['dir']) / 'nn_rms.pt'
        if p.exists():
            ax.hist(torch.load(p).numpy(), bins=40, histtype='step', label=r.config)
    ax.axvline(0.05, c='r', ls=':', label='copy threshold')
    ax.set_xlabel('NN-RMS to training set'); ax.set_ylabel('samples')
    ax.set_title(d); ax.legend(fontsize=7)
fig.tight_layout()
'''))

CELLS.append(md('## Figure 5 — samples'))

CELLS.append(code(r'''
show = [(d, c) for d in dsets for c in ['LS', 'EBMLS']]
picks = []
for d in dsets:
    s = df[df.dataset == d]
    for m, sel in [('LS', s[(s.method == 'LS') & (s.mult == 1.0)]),
                   ('EBMLS', s[s.method == 'EBMLS'].sort_values('IS_mean', ascending=False))]:
        if len(sel):
            picks.append(sel.iloc[0])
fig, axes = plt.subplots(len(picks), 1, figsize=(12, 1.7 * len(picks)), squeeze=False)
for ax, r in zip(axes[:, 0], picks):
    u8 = torch.load(Path(r['dir']) / 'samples_uint8.pt')[:16]
    grid = np.concatenate(list(u8.permute(0, 2, 3, 1).numpy()), axis=1)
    ax.imshow(grid); ax.set_axis_off()
    ax.set_title(f"{r.dataset} — {r.config}  (IS {r.IS_mean:.2f}, NN-RMS {r.nn_rms_median:.3f})", fontsize=9)
fig.tight_layout()
'''))

CELLS.append(md(r'''
## Config and provenance
'''))

CELLS.append(code(r'''
print(json.dumps({
    'commit': '2c2dc90 (+ scale_multiplier control)',
    'seed': 0, 'nsamples': 500, 'isc_splits': 5, 'genbatch': 50, 'scorebatchsize': 512,
    'schedule': 'cosine', 'temperature': 1.0, 'barrier_mode': 'image',
    'cifar10': 'conditional, scales_CIFAR10_ResNet_zeros_conditional.pt, all 50k train images',
    'mnist': 'unconditional, scales_MNIST_ResNet_zeros.pt',
    'celeba': 'unconditional, --max_samples 50000, genbatch 10 / scorebatch 96 (OOM at 50/512)',
    'candidates': 'geomspace(k1+2, 2*image_size-1, 10), odd; image-size independent',
}, indent=1))
df[['dataset', 'config', 'nsamples', 'scales']].drop_duplicates('config').to_string(index=False)
'''))

CELLS.append(md(r'''
## Reading of the result

Filled in from the numbers above when the control runs land. The three things to keep honest:

- The barrier response is **non-monotonic** on CIFAR-10: barrier 1.0 is well above the LS baseline
  while barriers 2.0 and 4.0 are well below it. A single intermediate barrier is not evidence of a
  trend; the low end of the sweep (0.25, 0.5) is what shows whether IS keeps climbing all the way
  into outright copying.
- The barrier-1.0 CIFAR run sits at NN-RMS 0.139 against the baseline's 0.342. Figure 2, not the IS
  column, is what says whether that is a better model or just a closer one.
- MNIST differences are inside the split spread ($\pm$0.06) and come with up to 46% of samples
  within the copy threshold. Nothing there should be called an improvement.
'''))

nb = {
	'cells': CELLS,
	'metadata': {
		'kernelspec': {'display_name': 'diffusion_env (PyTorch 2.4 / Py3.12)',
		               'language': 'python', 'name': 'diffusion_env'},
		'language_info': {'name': 'python', 'version': '3.12.1'},
	},
	'nbformat': 4,
	'nbformat_minor': 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1) + '\n')
print(f'wrote {OUT}  ({len(CELLS)} cells)')
