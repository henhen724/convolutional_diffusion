"""A small Diffusion Transformer (DiT) backbone, trained with v-prediction.

Kept separate from src/models.py, which holds the convolutional backbones the
analytic local-score theory is written about.  The DiT is here to ask the same
questions of a non-convolutional architecture: how close is its NTK prediction,
and how close is a local score machine.

Two deliberate departures from the reference DiT, both forced by what the NTK
comparison needs:

*   **adaLN, not adaLN-Zero, by default.**  adaLN-Zero zeroes every block's
    output gate at initialization, which makes the network's output identically
    zero *and* kills the parameter gradients of every transformer block --
    d(out)/d(block params) = 0 when the gate is 0.  The empirical NTK at such an
    init is rank-deficient and carries almost no signal, so the NTK prediction
    would be degenerate for reasons that have nothing to do with the question
    being asked.  `adaln_zero=True` restores the reference behaviour.

*   **Every parameter lives in a module that owns a `reset_parameters`.**
    src/utils/ntk_score.build_theta0 rebuilds the initialization by deep-copying
    a trained network and re-drawing through `reset_parameters`, and refuses to
    run if any parameter has no owner.  So attention is written out of plain
    `nn.Linear` rather than `nn.MultiheadAttention` (whose init hook is the
    private `_reset_parameters`), positions use an `nn.Embedding`, and the
    zero-initialized heads are a `ZeroLinear` subclass that re-zeroes itself --
    which also means the zero-init survives a theta0 rebuild instead of being
    silently replaced by a random draw.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ZeroLinear(nn.Linear):
	"""nn.Linear whose `reset_parameters` re-zeroes it, so zero-init is reproducible.

	Using a subclass rather than an after-the-fact `nn.init.zeros_` keeps the
	initialization attached to the module, which is what lets build_theta0
	reconstruct the exact init convention from the architecture alone.
	"""

	def reset_parameters(self):
		nn.init.zeros_(self.weight)
		if self.bias is not None:
			nn.init.zeros_(self.bias)


class TimestepEmbedder(nn.Module):
	"""Sinusoidal embedding of the continuous diffusion time, then an MLP.

	`t` arrives in [0, 1] (the convention throughout this repo); it is rescaled by
	`max_period_scale` before the sinusoids so the frequency bank actually spans
	the interval instead of collapsing onto sin(t) ~ t.
	"""

	def __init__(self, hidden, freq_dim=256, max_period_scale=1000.0):
		super().__init__()
		self.freq_dim = freq_dim
		self.max_period_scale = max_period_scale
		self.mlp = nn.Sequential(
			nn.Linear(freq_dim, hidden),
			nn.SiLU(),
			nn.Linear(hidden, hidden),
		)

	def forward(self, t):
		half = self.freq_dim // 2
		freqs = torch.exp(
			-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
		)
		args = (t.float() * self.max_period_scale)[:, None] * freqs[None, :]
		emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
		return self.mlp(emb)


class Attention(nn.Module):
	"""Multi-head self-attention built from plain nn.Linear (see module docstring).

	Attention is written out rather than calling `F.scaled_dot_product_attention`.
	The fused SDPA kernels have no double backward -- they raise "derivative for
	aten::_scaled_dot_product_efficient_attention_backward is not implemented" --
	and the NTK path in src/utils/ntk_score.py differentiates a vjp a second time
	to take its Jacobian-vector products. This spells out exactly what SDPA's math
	backend computes, so a model trained through the fused kernel is unchanged; at
	64 tokens the speed difference is irrelevant.
	"""

	def __init__(self, hidden, heads):
		super().__init__()
		if hidden % heads != 0:
			raise ValueError(f'hidden={hidden} must be divisible by heads={heads}')
		self.heads = heads
		self.head_dim = hidden // heads
		self.qkv = nn.Linear(hidden, 3 * hidden)
		self.proj = nn.Linear(hidden, hidden)

	def forward(self, x):
		b, n, d = x.shape
		qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]
		attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
		out = attn.softmax(dim=-1) @ v
		out = out.transpose(1, 2).reshape(b, n, d)
		return self.proj(out)


def modulate(x, shift, scale):
	return x * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTBlock(nn.Module):
	"""Transformer block with adaptive layer-norm conditioning on the diffusion time."""

	def __init__(self, hidden, heads, mlp_ratio=4.0, adaln_zero=False):
		super().__init__()
		self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
		self.attn = Attention(hidden, heads)
		self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
		inner = int(hidden * mlp_ratio)
		self.mlp = nn.Sequential(nn.Linear(hidden, inner), nn.GELU(approximate='tanh'), nn.Linear(inner, hidden))
		head = ZeroLinear if adaln_zero else nn.Linear
		self.adaLN = nn.Sequential(nn.SiLU(), head(hidden, 6 * hidden))

	def forward(self, x, c):
		shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(c).chunk(6, dim=-1)
		x = x + gate_a[:, None, :] * self.attn(modulate(self.norm1(x), shift_a, scale_a))
		x = x + gate_m[:, None, :] * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
		return x


class FinalLayer(nn.Module):
	def __init__(self, hidden, patch_size, out_channels, adaln_zero=False):
		super().__init__()
		self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
		head = ZeroLinear if adaln_zero else nn.Linear
		self.adaLN = nn.Sequential(nn.SiLU(), head(hidden, 2 * hidden))
		self.linear = head(hidden, patch_size * patch_size * out_channels)

	def forward(self, x, c):
		shift, scale = self.adaLN(c).chunk(2, dim=-1)
		return self.linear(modulate(self.norm(x), shift, scale))


class MinimalDiT(nn.Module):
	"""Patch-based diffusion transformer.

	`forward(t, x, label=None)` returns the network's raw prediction with the same
	shape as `x`.  Trained here on the **v** target, so wrap it in `VPredToEps`
	before handing it to anything that expects an eps-predicting backbone.
	"""

	def __init__(self, channels=3,
	             image_size=32,
	             patch_size=4,
	             hidden=256,
	             depth=6,
	             heads=4,
	             mlp_ratio=4.0,
	             conditional=False,
	             num_classes=None,
	             adaln_zero=False):
		super().__init__()
		if image_size % patch_size != 0:
			raise ValueError(f'image_size={image_size} must be divisible by patch_size={patch_size}')
		self.channels = channels
		self.image_size = image_size
		self.patch_size = patch_size
		self.hidden = hidden
		self.depth = depth
		self.heads = heads
		self.mlp_ratio = mlp_ratio
		self.conditional = conditional
		self.num_classes = num_classes
		self.adaln_zero = adaln_zero
		self.grid = image_size // patch_size
		self.num_patches = self.grid ** 2

		self.patch_embed = nn.Conv2d(channels, hidden, patch_size, stride=patch_size)
		self.pos_embed = nn.Embedding(self.num_patches, hidden)
		self.t_embed = TimestepEmbedder(hidden)
		if conditional:
			if num_classes is None:
				raise ValueError('num_classes must be specified when conditional is True')
			self.class_embed = nn.Embedding(num_classes, hidden)
		self.blocks = nn.ModuleList([DiTBlock(hidden, heads, mlp_ratio, adaln_zero) for _ in range(depth)])
		self.final = FinalLayer(hidden, patch_size, channels, adaln_zero)

	def unpatchify(self, x):
		b = x.shape[0]
		g, p, c = self.grid, self.patch_size, self.channels
		x = x.reshape(b, g, g, p, p, c).permute(0, 5, 1, 3, 2, 4)
		return x.reshape(b, c, g * p, g * p)

	def forward(self, t, x, label=None, **kwargs):
		device = x.device
		t = t.to(device)
		tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # (b, num_patches, hidden)
		pos = torch.arange(self.num_patches, device=device)
		tokens = tokens + self.pos_embed(pos)[None, :, :]

		c = self.t_embed(t)
		if self.conditional:
			if label is None:
				raise ValueError('conditional DiT called without a label')
			c = c + self.class_embed(label.to(device))

		for block in self.blocks:
			tokens = block(tokens, c)
		return self.unpatchify(self.final(tokens, c))


class VPredToEps(nn.Module):
	"""Present a v-predicting backbone as an eps-predicting one.

	With this repo's VP corruption x_t = sqrt(1-b) x0 + sqrt(b) eps, the v target is
	v = sqrt(1-b) eps - sqrt(b) x0, which inverts to

	    eps = sqrt(1-b) * v + sqrt(b) * x_t.

	Wrapping the DiT at the DDIM-backbone boundary means every existing consumer --
	DDIM.sample, ScheduledScoreMachine, scripts/scales_calibration.py, and the NTK
	fit -- keeps working unchanged against an eps interface, while the training loss
	is still computed on the inner module's raw v output.

	Note for the NTK fit: at a *fixed* noise level the v and eps regressions give the
	same dtheta, since eps and v differ by the constant factor sqrt(1-b) plus a
	theta-independent term, and that factor cancels between the kernel and the
	targets.  So fitting the wrapped (eps) interface is equivalent to fitting the
	v objective the DiT was actually trained on.
	"""

	def __init__(self, backbone, noise_schedule):
		super().__init__()
		self.backbone = backbone
		self.noise_schedule = noise_schedule

	def forward(self, t, x, label=None, **kwargs):
		v = self.backbone(t, x, label=label, **kwargs)
		beta_t = self.noise_schedule(t.to(x.device)).reshape(-1, 1, 1, 1)
		return (1 - beta_t) ** 0.5 * v + beta_t ** 0.5 * x

	@staticmethod
	def v_target(x0, noise, beta_t):
		"""v = sqrt(1-b) * eps - sqrt(b) * x0, with beta_t broadcastable over (b,c,h,w)."""
		return (1 - beta_t) ** 0.5 * noise - beta_t ** 0.5 * x0
