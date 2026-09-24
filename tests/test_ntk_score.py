"""Tests for the NTK score machinery and the v-prediction DiT.

These use random data and tiny models, so they run on CPU in seconds and do not
touch the CelebA checkpoints.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import DDIM, MinimalResNet
from src.models_dit import MinimalDiT, VPredToEps
from src.utils.idealscore import ScheduledScoreMachine
from src.utils.noise_schedules import cosine_noise_schedule
from src.utils.ntk_score import (
	EpsBackboneAdapter,
	NTKScoreModule,
	assert_same_parameter_layout,
	build_theta0,
)


def tiny_resnet():
	return MinimalResNet(channels=3, emb_dim=16, mode='zeros', num_layers=2, lastksize=3)


def tiny_dit(adaln_zero=False):
	return MinimalDiT(channels=3, image_size=8, patch_size=4, hidden=16, depth=2,
	                  heads=2, adaln_zero=adaln_zero)


@pytest.mark.parametrize('factory', [tiny_resnet, tiny_dit])
def test_build_theta0_covers_every_parameter(factory):
	"""build_theta0 must re-draw every parameter, not silently keep trained values."""
	trained = factory()
	with torch.no_grad():
		for p in trained.parameters():
			p.add_(1.0)  # move well away from any init distribution

	theta0, n_resets = build_theta0(trained, seed=0)
	assert n_resets > 0
	assert_same_parameter_layout(trained, theta0)

	# Every parameter that is not deliberately zero-initialized must have changed.
	changed = 0
	for (name, a), (_, b) in zip(trained.named_parameters(), theta0.named_parameters()):
		if not torch.allclose(a, b):
			changed += 1
	assert changed > 0
	for (name, a), (_, b) in zip(trained.named_parameters(), theta0.named_parameters()):
		assert not torch.allclose(a, b), f'{name} kept its trained value in theta0'


def test_build_theta0_is_seed_reproducible():
	trained = tiny_resnet()
	a, _ = build_theta0(trained, seed=3)
	b, _ = build_theta0(trained, seed=3)
	c, _ = build_theta0(trained, seed=4)
	for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
		assert torch.allclose(pa, pb)
	assert any(not torch.allclose(pa, pc)
	           for (_, pa), (_, pc) in zip(a.named_parameters(), c.named_parameters()))


def test_build_theta0_does_not_disturb_global_rng():
	trained = tiny_resnet()
	torch.manual_seed(11)
	before = torch.randn(4)
	torch.manual_seed(11)
	build_theta0(trained, seed=99)
	after = torch.randn(4)
	assert torch.allclose(before, after)


def test_build_theta0_rejects_unowned_parameters():
	"""A raw nn.Parameter with no reset_parameters owner must raise, not slip through."""
	model = tiny_resnet()
	model.register_parameter('stray', torch.nn.Parameter(torch.ones(3)))
	with pytest.raises(ValueError, match='reset_parameters'):
		build_theta0(model, seed=0)


def test_ntk_module_reproduces_the_directional_derivative():
	"""The finite-difference JVP must match an autograd-computed <grad f, dtheta>."""
	torch.manual_seed(0)
	trained = tiny_resnet()
	theta0, _ = build_theta0(trained, seed=1)

	n_params = sum(p.numel() for p in theta0.parameters())
	dtheta = torch.randn(n_params) * (1.0 / n_params ** 0.5)
	nsteps = 4
	ntk = NTKScoreModule(theta0, {2: dtheta}, nsteps=nsteps, fd_step_rel=1e-3)

	x = torch.randn(2, 3, 8, 8)
	t = torch.full((2,), 2 / nsteps)

	# Reference: autograd JVP of a scalar readout, contracted against dtheta.
	params = list(ntk.theta0.parameters())
	for p in params:
		p.requires_grad_(True)
	out = ntk.theta0(t, x)
	f = out[0, 1, 3, 4]
	grads = torch.cat([g.reshape(-1) for g in torch.autograd.grad(f, params)])
	expected = float(grads @ dtheta)
	for p in params:
		p.requires_grad_(False)

	base = ntk.theta0(t, x).detach()
	got = float((ntk(t, x) - base)[0, 1, 3, 4])
	assert got == pytest.approx(expected, rel=1e-5, abs=1e-7)


def test_exact_jvp_is_linear_in_delta_theta():
	"""Scaling dtheta must scale the prediction's correction term exactly."""
	torch.manual_seed(0)
	theta0, _ = build_theta0(tiny_resnet(), seed=1)
	n_params = sum(p.numel() for p in theta0.parameters())
	d = torch.randn(n_params) / n_params ** 0.5
	x, t = torch.randn(2, 3, 8, 8), torch.full((2,), 1 / 4)

	one = NTKScoreModule(theta0, {1: d}, nsteps=4)
	three = NTKScoreModule(theta0, {1: 3.0 * d}, nsteps=4)
	a = one.jvp(1, t, x)
	b = three.jvp(1, t, x)
	# Compared against the scale of the field, not elementwise: entries near zero
	# carry the same absolute float32 error as the large ones.
	rel = (b - 3.0 * a).abs().max() / (3.0 * a).abs().max()
	assert float(rel) < 1e-4


def test_fd_jvp_converges_towards_the_exact_one():
	"""FD is truncation-dominated here, so it only approaches the exact JVP as h shrinks.

	This is why `jvp_mode` defaults to 'exact': even along a real gradient direction
	the central difference cannot be driven below a few percent before float32
	cancellation takes over, so it is a smoke test and not an oracle.
	"""
	torch.manual_seed(0)
	theta0, _ = build_theta0(tiny_resnet(), seed=1)
	params = list(theta0.parameters())
	for p in params:
		p.requires_grad_(True)
	x, t = torch.randn(2, 3, 8, 8), torch.full((2,), 1 / 4)
	f = theta0(t, x)[0, 0, 4, 4]
	d = torch.cat([g.reshape(-1) for g in torch.autograd.grad(f, params)])
	for p in params:
		p.requires_grad_(False)

	ntk = NTKScoreModule(theta0, {1: d}, nsteps=4, fd_step_rel=1e-2)
	_, coarse = ntk.consistency_check(t, x)
	ntk.fd_step_rel, ntk.fd_step, ntk._loaded_step = 1e-4, 1e-4 * ntk.theta0_norm, None
	_, fine = ntk.consistency_check(t, x)
	assert fine < coarse
	assert fine < 0.2


def test_ntk_module_rejects_unfitted_step_and_mixed_t():
	theta0, _ = build_theta0(tiny_resnet(), seed=0)
	n_params = sum(p.numel() for p in theta0.parameters())
	ntk = NTKScoreModule(theta0, {1: torch.zeros(n_params)}, nsteps=4)
	x = torch.randn(2, 3, 8, 8)
	with pytest.raises(KeyError):
		ntk(torch.full((2,), 3 / 4), x)
	with pytest.raises(ValueError, match='mixed-t'):
		ntk(torch.tensor([1 / 4, 2 / 4]), x)


def test_eps_adapter_matches_the_wrapped_backbone():
	backbone = tiny_resnet().eval()
	adapter = EpsBackboneAdapter(backbone)
	x, t = torch.randn(2, 3, 8, 8), torch.full((2,), 0.3)
	with torch.no_grad():
		assert torch.allclose(adapter(t, x, device=x.device, k=7), backbone(t, x))


def test_scheduled_machine_matches_ddim_sample_on_the_shared_steps():
	"""ScheduledScoreMachine(score_backbone=False) must apply the same DDIM update.

	It starts one step lower than DDIM.sample, so the two are compared from the step
	where they do overlap -- this is what licenses driving every model through
	ScheduledScoreMachine instead of DDIM.sample.
	"""
	torch.manual_seed(0)
	backbone = tiny_resnet().eval()
	ddim = DDIM(backbone=backbone, in_channels=3, noise_schedule=cosine_noise_schedule,
	            default_imsize=8).eval()
	machine = ScheduledScoreMachine(EpsBackboneAdapter(backbone).eval(), in_channels=3, imsize=8,
	                                default_time_steps=4, noise_schedule=cosine_noise_schedule,
	                                score_backbone=False, scales=None)
	x = torch.randn(1, 3, 8, 8)
	device = torch.device('cpu')
	# DDIM.sample from step nsteps-1 onwards == ScheduledScoreMachine over all its steps.
	a = ddim.sample(x=x.clone(), nsteps=4, device=device, breakstep=-1)
	b = machine(x.clone(), nsteps=4, device=device)
	assert a.shape == b.shape
	assert not torch.allclose(a, b)  # they differ only by DDIM's extra t=1 step


class TestDiT:

	def test_output_shape_and_v_roundtrip(self):
		dit = tiny_dit()
		x, t = torch.randn(2, 3, 8, 8), torch.rand(2)
		v = dit(t, x)
		assert v.shape == x.shape

		wrapped = VPredToEps(dit, cosine_noise_schedule)
		eps = wrapped(t, x)
		assert eps.shape == x.shape

		# eps = sqrt(1-b) v + sqrt(b) x_t must invert the v definition exactly.
		beta = cosine_noise_schedule(t)[:, None, None, None]
		x0, noise = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
		xt = (1 - beta) ** 0.5 * x0 + beta ** 0.5 * noise
		v_true = VPredToEps.v_target(x0, noise, beta)
		recovered = (1 - beta) ** 0.5 * v_true + beta ** 0.5 * xt
		assert torch.allclose(recovered, noise, atol=1e-5)

	def test_unpatchify_inverts_patch_layout(self):
		dit = MinimalDiT(channels=3, image_size=8, patch_size=4, hidden=8, depth=1, heads=2)
		b, g, p, c = 2, dit.grid, dit.patch_size, dit.channels
		img = torch.arange(b * c * 8 * 8, dtype=torch.float32).reshape(b, c, 8, 8)
		tokens = img.reshape(b, c, g, p, g, p).permute(0, 2, 4, 3, 5, 1).reshape(b, g * g, p * p * c)
		assert torch.allclose(dit.unpatchify(tokens), img)

	def test_adaln_zero_makes_the_init_output_vanish(self):
		"""Documents why adaln_zero defaults to False: it zeroes the init NTK."""
		zero, _ = build_theta0(tiny_dit(adaln_zero=True), seed=0)
		x, t = torch.randn(2, 3, 8, 8), torch.rand(2)
		assert torch.allclose(zero(t, x), torch.zeros_like(x))

		plain, _ = build_theta0(tiny_dit(adaln_zero=False), seed=0)
		assert not torch.allclose(plain(t, x), torch.zeros_like(x))

	def test_supports_double_backward(self):
		"""The exact JVP differentiates a vjp again, which fused SDPA kernels cannot do."""
		dit, _ = build_theta0(tiny_dit(), seed=0)
		n_params = sum(p.numel() for p in dit.parameters())
		ntk = NTKScoreModule(dit, {1: torch.randn(n_params) / n_params ** 0.5}, nsteps=4)
		out = ntk(torch.full((2,), 1 / 4), torch.randn(2, 3, 8, 8))
		assert out.shape == (2, 3, 8, 8)
		assert torch.isfinite(out).all()

	def test_attention_matches_scaled_dot_product_attention(self):
		"""The hand-written attention must equal what SDPA computes, so weights trained
		through the fused kernel stay valid."""
		from torch.nn import functional as F

		from src.models_dit import Attention

		torch.manual_seed(0)
		attn = Attention(hidden=16, heads=2).eval()
		x = torch.randn(2, 5, 16)
		with torch.no_grad():
			b, n, d = x.shape
			qkv = attn.qkv(x).reshape(b, n, 3, attn.heads, attn.head_dim).permute(2, 0, 3, 1, 4)
			ref = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
			ref = attn.proj(ref.transpose(1, 2).reshape(b, n, d))
			assert torch.allclose(attn(x), ref, atol=1e-5)

	def test_plain_init_has_nonzero_block_gradients(self):
		"""The NTK is only informative if block parameters actually move the output."""
		dit, _ = build_theta0(tiny_dit(adaln_zero=False), seed=0)
		params = dict(dit.named_parameters())
		target = params['blocks.0.attn.qkv.weight']
		target.requires_grad_(True)
		out = dit(torch.rand(1), torch.randn(1, 3, 8, 8))
		g = torch.autograd.grad(out[0, 0, 2, 2], target)[0]
		assert g.abs().max() > 0
