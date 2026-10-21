"""Tests for MultilayerLocalScoreModule (multilayer LS)."""

import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.idealscore import (
	EntropyBarrierMLSModule,
	LocalScoreModule,
	_box_sum,
	MultilayerLocalScoreModule,
	ScheduledScoreMachine,
)
from src.utils.noise_schedules import cosine_noise_schedule

CPU = torch.device("cpu")


def _dataset(n=12, c=2, h=8, seed=0):
	g = torch.Generator().manual_seed(seed)
	images = torch.randn(n, c, h, h, generator=g)
	labels = torch.arange(n) % 2
	return TensorDataset(images, labels)


def _x0_from_score(score, x, t):
	bt2 = cosine_noise_schedule(t)[:, None, None, None]
	at = (1 - bt2)**0.5
	return (x + bt2 * score) / at


def _brute_force_delta(x_hat, images, k, sigma2):
	"""Per-pixel loop: sum_n w_n(p) (x0_n(p) - x_hat(p)), zero-padded k x k windows."""
	_, c, h, w = x_hat.shape
	d = k // 2
	xp = F.pad(x_hat[0], (d, d, d, d))
	ip = F.pad(images, (d, d, d, d))
	out = torch.zeros(c, h, w, dtype=torch.float64)
	for r in range(h):
		for s in range(w):
			dist = ((ip[:, :, r:r + k, s:s + k] - xp[None, :, r:r + k, s:s + k])**2).sum(dim=(1, 2, 3))
			wts = torch.softmax(-dist / (2 * sigma2), dim=0)
			out[:, r, s] = (wts[:, None] * (images[:, :, r, s] - x_hat[0, :, r, s][None])).sum(0)
	return out[None]


@pytest.mark.parametrize("b", [1, 3])
def test_layer1_matches_ls(b):
	ds = _dataset()
	t = torch.full((b,), 0.4)
	x = torch.randn(b, 2, 8, 8)
	ls = LocalScoreModule(ds, kernel_size=3, batch_size=4, schedule=cosine_noise_schedule)
	mls = MultilayerLocalScoreModule(ds, kernel_sizes=(3, 7), batch_size=4, schedule=cosine_noise_schedule)
	x_hat_1 = mls.forward_layers(t, x, device=CPU)[0]
	torch.testing.assert_close(x_hat_1, _x0_from_score(ls(t, x, device=CPU), x, t), rtol=1e-4, atol=1e-5)


def test_single_layer_equals_ls():
	ds = _dataset()
	t = torch.full((1,), 0.6)
	x = torch.randn(1, 2, 8, 8)
	ls = LocalScoreModule(ds, kernel_size=5, batch_size=4, schedule=cosine_noise_schedule)
	mls = MultilayerLocalScoreModule(ds, kernel_sizes=(5,), batch_size=4, schedule=cosine_noise_schedule)
	torch.testing.assert_close(mls(t, x, device=CPU), ls(t, x, device=CPU), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("temperature", [1.0, 0.3])
def test_layer2_matches_brute_force(temperature):
	ds = _dataset()
	t = torch.full((1,), 0.5)
	x = torch.randn(1, 2, 8, 8)
	# batch_size 5 doesn't divide 12, so the streaming log-sum-exp crosses a ragged batch
	mls = MultilayerLocalScoreModule(ds, kernel_sizes=(3, 5), temperature=temperature, batch_size=5, schedule=cosine_noise_schedule)
	x_hat_1, delta_2 = mls.forward_layers(t, x, device=CPU)
	bt2 = cosine_noise_schedule(t).item()
	sigma2 = temperature * bt2 / (1 - bt2)
	expected = _brute_force_delta(x_hat_1.double(), ds.tensors[0].double(), 5, sigma2)
	torch.testing.assert_close(delta_2.double(), expected, rtol=1e-4, atol=1e-5)


def test_full_estimate_is_sum_and_score_convention():
	ds = _dataset()
	t = torch.full((1,), 0.5)
	x = torch.randn(1, 2, 8, 8)
	mls = MultilayerLocalScoreModule(ds, kernel_sizes=(3, 5, 7), batch_size=4, schedule=cosine_noise_schedule)
	layers = mls.forward_layers(t, x, device=CPU)
	assert len(layers) == 3
	torch.testing.assert_close(_x0_from_score(mls(t, x, device=CPU), x, t), sum(layers), rtol=1e-4, atol=1e-4)


def test_batched_matches_individual():
	ds = _dataset()
	t = torch.tensor([0.2, 0.5, 0.8])
	x = torch.randn(3, 2, 8, 8)
	mls = MultilayerLocalScoreModule(ds, kernel_sizes=(3, 7), batch_size=4, schedule=cosine_noise_schedule)
	batched = mls(t, x, device=CPU)
	for j in range(3):
		single = mls(t[j:j + 1], x[j:j + 1], device=CPU)
		torch.testing.assert_close(batched[j:j + 1], single, rtol=1e-4, atol=1e-4)


def test_k_override_keeps_offsets():
	ds = _dataset()
	mls = MultilayerLocalScoreModule(ds, kernel_sizes=(3, 7), schedule=cosine_noise_schedule)
	assert mls._layer_kernel_sizes(None) == [3, 7]
	assert mls._layer_kernel_sizes(5) == [5, 9]
	assert mls._layer_kernel_sizes([3, 11]) == [3, 11]
	with pytest.raises(ValueError):
		MultilayerLocalScoreModule(ds, kernel_sizes=(7, 3))


def test_label_conditioning_and_sampling_runs():
	ds = _dataset()
	mls = MultilayerLocalScoreModule(ds, kernel_sizes=(3, 5), batch_size=4, schedule=cosine_noise_schedule)
	machine = ScheduledScoreMachine(mls, in_channels=2, imsize=8, noise_schedule=cosine_noise_schedule, default_time_steps=6)
	out = machine(torch.randn(1, 2, 8, 8), label=torch.tensor([1]), device=CPU)
	assert out.shape == (1, 2, 8, 8)
	assert torch.isfinite(out).all()


@pytest.mark.parametrize("k", [1, 3, 7, 15])
def test_box_sum_matches_unfold(k):
	x = torch.randn(2, 3, 8, 8, dtype=torch.float64)
	ref = F.unfold(x.reshape(6, 1, 8, 8), k, padding=k//2).sum(1).reshape(2, 3, 8, 8)
	torch.testing.assert_close(_box_sum(x, k), ref)


def _brute_force_entropy(x_hat, images, k, sigma2):
	_, c, h, w = x_hat.shape
	d = k // 2
	xp = F.pad(x_hat[0], (d, d, d, d))
	ip = F.pad(images, (d, d, d, d))
	out = torch.zeros(h, w, dtype=torch.float64)
	for r in range(h):
		for s in range(w):
			dist = ((ip[:, :, r:r + k, s:s + k] - xp[None, :, r:r + k, s:s + k])**2).sum(dim=(1, 2, 3))
			p = torch.softmax(-dist / (2 * sigma2), dim=0)
			out[r, s] = -(p * torch.log(p.clamp(min=1e-300))).sum()
	return out[None]


def test_barrier_deltas_and_entropy_match_brute_force():
	ds = _dataset()
	t = torch.full((1,), 0.4)
	x = torch.randn(1, 2, 8, 8)
	mod = EntropyBarrierMLSModule(ds, kernel_size=3, image_size=8, batch_size=5, schedule=cosine_noise_schedule)
	x_hat = mod.forward_layers(t, x, device=CPU)[0]
	bt2 = cosine_noise_schedule(t).item()
	sigma2 = bt2 / (1 - bt2)
	ks = [5, 9]
	deltas, entropy = mod._expected_error_and_entropy(x_hat, ks, torch.tensor(sigma2), None, CPU)
	for j, k in enumerate(ks):
		torch.testing.assert_close(deltas[j].double(), _brute_force_delta(x_hat.double(), ds.tensors[0].double(), k, sigma2), rtol=1e-4, atol=1e-5)
		torch.testing.assert_close(entropy[j].double(), _brute_force_entropy(x_hat.double(), ds.tensors[0].double(), k, sigma2), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("mode", ["image", "pixel"])
def test_barrier_extremes(mode):
	ds = _dataset()
	t = torch.full((2,), 0.3)
	x = torch.randn(2, 2, 8, 8)
	ls = LocalScoreModule(ds, kernel_size=3, batch_size=4, schedule=cosine_noise_schedule)
	# barrier above log(N): nothing passes, so the module reduces to plain LS
	high = EntropyBarrierMLSModule(ds, kernel_size=3, entropy_barrier=10.0, barrier_mode=mode, image_size=8, batch_size=4, schedule=cosine_noise_schedule)
	torch.testing.assert_close(high(t, x, device=CPU), ls(t, x, device=CPU), rtol=1e-4, atol=1e-4)
	assert (high.last_k == 0).all()
	# barrier 0: everything passes, so the largest candidate is used
	zero = EntropyBarrierMLSModule(ds, kernel_size=3, entropy_barrier=0.0, barrier_mode=mode, image_size=8, batch_size=4, schedule=cosine_noise_schedule)
	fixed = MultilayerLocalScoreModule(ds, kernel_sizes=(3, 15), batch_size=4, schedule=cosine_noise_schedule)
	torch.testing.assert_close(zero(t, x, device=CPU), fixed(t, x, device=CPU), rtol=1e-4, atol=1e-4)
	assert (zero.last_k == 15).all()


def test_barrier_picks_largest_k_above_barrier():
	ds = _dataset()
	t = torch.full((1,), 0.3)
	x = torch.randn(1, 2, 8, 8)
	mod = EntropyBarrierMLSModule(ds, kernel_size=3, image_size=8, batch_size=4, schedule=cosine_noise_schedule)
	x_hat = mod.forward_layers(t, x, device=CPU)[0]
	bt2 = cosine_noise_schedule(t)[:, None, None, None]
	ks = mod._candidates(3)
	_, entropy = mod._expected_error_and_entropy(x_hat, ks, bt2 / (1 - bt2), None, CPU)
	means = entropy.mean(dim=(2, 3))[:, 0]
	mod.entropy_barrier = float(means[len(ks) // 2])  # between the extremes
	mod.forward_layers(t, x, device=CPU)
	expected = max(k for k, e in zip(ks, means.tolist()) if e >= mod.entropy_barrier)
	assert int(mod.last_k[0]) == expected
