"""NTK-regression score machine for the convolutional diffusion ResNet.

This is the neural-tangent-kernel prediction of the denoiser that the ResNet
converges to under gradient flow from its initialization theta0:

    eps_ntk(x, t) = eps_theta0(x, t) + < grad_theta eps_theta0(x, t) , dtheta_t >

with dtheta_t = G_t^T (Theta_t + lam I)^{-1} y_t the kernel-ridge solution of the
denoising regression at noise level t.  G_t stacks the parameter gradients of a
scalar pixel readout over a training subset, Theta_t = G_t G_t^T is the *exact*
empirical NTK over those readouts, and y_t are the residual denoising targets
eps - eps_theta0 at the readout.

The point that makes this cheap: the kernel-regression coefficients contract into
a single parameter-space vector dtheta_t, so evaluating the prediction needs no
kernel at sampling time -- only a directional derivative of the network at
theta0, taken here by central finite differences (two extra forward passes).
Because that derivative is taken on the full network output, a readout fit at one
pixel yields a prediction at *every* pixel, which is the same weight-sharing /
equivariance structure that the ELS machines assume by construction.

The heavy fit lives in scripts/fit_ntk_score.py.  This module holds the pieces
shared by that script, scripts/gen_three_way_samples.py, and
notebooks/ntk_vs_els_samples.ipynb.
"""

import copy

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def build_theta0(trained_backbone, seed=0):
	"""Fresh init of the *same* architecture as a trained backbone.

	Deep-copies the trained module and re-draws every parameter via each
	submodule's own `reset_parameters`, rather than reconstructing from a config
	dict.  That keeps this architecture-agnostic -- the same call works for
	MinimalResNet, MinimalUNet and a DiT -- and makes the parameter layout
	identical to the trained network by construction, which everything downstream
	(G rows, the flat dtheta, vector_to_parameters) depends on.

	None of the trained runs here kept an epoch-0 checkpoint, so the NTK expansion
	point is a re-draw from the architecture's init ensemble rather than the exact
	init the trained weights came from.  NTK theory only needs *an* init from that
	ensemble -- the kernel concentrates over draws -- and the draw is seeded so
	every consumer of the fitted dtheta reconstructs the identical theta0.
	"""
	model = copy.deepcopy(trained_backbone)

	covered, resets = set(), 0
	gen_state = torch.random.get_rng_state()
	try:
		torch.manual_seed(seed)
		for name, m in model.named_modules():
			if not hasattr(m, 'reset_parameters'):
				continue
			m.reset_parameters()
			resets += 1
			for pname, _ in m.named_parameters(recurse=False):
				covered.add(f'{name}.{pname}' if name else pname)
	finally:
		torch.random.set_rng_state(gen_state)

	# A parameter no submodule resets would silently keep its *trained* value and
	# quietly contaminate theta0, so refuse rather than fit against a hybrid.
	missed = [n for n, _ in model.named_parameters() if n not in covered]
	if missed:
		raise ValueError(
			f'{len(missed)} parameter(s) have no reset_parameters() owner and would keep their '
			f'trained values in theta0: {missed[:8]}. Add an explicit re-init for them.'
		)

	model.eval()
	for p in model.parameters():
		p.requires_grad_(False)
	return model, resets


def assert_same_parameter_layout(a, b):
	"""Guard that two backbones share parameter names, shapes and ordering.

	Everything here -- G rows, dtheta, vector_to_parameters -- assumes one fixed
	flattening of `model.parameters()`, so a silent architecture mismatch between
	the trained checkpoint and the rebuilt theta0 would corrupt the fit rather than
	raise.
	"""
	na = [(n, tuple(p.shape)) for n, p in a.named_parameters()]
	nb = [(n, tuple(p.shape)) for n, p in b.named_parameters()]
	if na != nb:
		only_a = [x for x in na if x not in nb]
		only_b = [x for x in nb if x not in na]
		raise ValueError(
			'Parameter layout mismatch between trained backbone and rebuilt theta0.\n'
			f'  only in trained: {only_a[:8]}\n  only in theta0:  {only_b[:8]}'
		)
	buffers = [n for n, _ in a.named_buffers()]
	if buffers:
		raise ValueError(f'Backbone has buffers, which the flat-vector fit does not carry: {buffers}')


class EpsBackboneAdapter(nn.Module):
	"""Adapt a trained network backbone to the ScheduledScoreMachine signature.

	ScheduledScoreMachine calls `backbone(t, x, label=..., device=..., k=...)`; the
	trained networks take no `device`/`k`.  Wrapping one (rather than calling
	DDIM.sample) lets every model run through the *same* reverse loop, so any
	difference in the samples comes from the score and not from the sampler --
	DDIM.sample and ScheduledScoreMachine.forward do not iterate over the same
	range of steps.
	"""

	def __init__(self, backbone):
		super().__init__()
		self.backbone = backbone

	def forward(self, t, x, label=None, device=None, k=None):
		with torch.no_grad():
			return self.backbone(t, x, label=label)


class NTKScoreModule(nn.Module):
	"""Evaluate the fitted NTK denoiser, in the ScheduledScoreMachine backbone API.

	Returns eps (not the score), so wrap it with `score_backbone=False`.

	`delta_theta` maps a discrete step index i (with t = i / nsteps) to a flat
	parameter-space vector, and the prediction is

	    eps_ntk(x, t_i) = f_theta0(x, t_i) + < grad_theta f_theta0(x, t_i) , dtheta_i >.

	The directional derivative is taken **exactly**, by the double-backward
	Jacobian-vector trick (`jvp_mode='exact'`): a vjp against a dummy cotangent w
	with `create_graph=True`, differentiated again with respect to w.  Two backward
	passes, no step size to tune, and no cancellation.

	Central finite differences are available as `jvp_mode='fd'`, but only as a
	cross-check.  They are badly conditioned here: dtheta from a kernel-ridge solve
	is nearly orthogonal to any single input's gradient, so f(theta0 + h u) and
	f(theta0 - h u) differ in their last few float32 digits and the quotient loses
	most of its precision.  `consistency_check` reports the disagreement between the
	two modes.
	"""

	def __init__(self, theta0, delta_theta, nsteps=20, jvp_mode='exact', fd_step_rel=1e-3):
		super().__init__()
		if jvp_mode not in ('exact', 'fd'):
			raise ValueError(f"jvp_mode must be 'exact' or 'fd', got {jvp_mode!r}")
		self.theta0 = theta0.eval()
		for p in self.theta0.parameters():
			p.requires_grad_(False)

		self.delta_theta = {int(k): v.detach().float().cpu() for k, v in delta_theta.items()}
		self.nsteps = int(nsteps)
		self.jvp_mode = jvp_mode
		self.fd_step_rel = float(fd_step_rel)
		self.theta0_norm = float(parameters_to_vector(self.theta0.parameters()).norm())
		self.fd_step = self.fd_step_rel * self.theta0_norm
		self._plus = None
		self._minus = None
		self._loaded_step = None
		self._scale = 0.0

	def step_index(self, t):
		"""Map the sampler's continuous t back to the discrete index it was fit at."""
		tv = t.reshape(-1)
		i = int(round(float(tv[0]) * self.nsteps))
		if not torch.allclose(tv, tv[0].expand_as(tv)):
			raise ValueError('NTKScoreModule needs one shared t across the batch; got a mixed-t batch.')
		if i not in self.delta_theta:
			raise KeyError(
				f'No dtheta fitted for step index {i} (t={float(tv[0]):.4f}, nsteps={self.nsteps}). '
				f'Fitted steps: {sorted(self.delta_theta)}'
			)
		return i

	def _delta_as_param_list(self, i, device):
		"""Unflatten dtheta_i into tensors shaped like theta0's parameters."""
		flat = self.delta_theta[i].to(device)
		out, off = [], 0
		for p in self.theta0.parameters():
			n = p.numel()
			out.append(flat[off:off + n].view_as(p))
			off += n
		if off != flat.numel():
			raise ValueError(f'dtheta has {flat.numel()} entries but theta0 has {off} parameters.')
		return out

	def _jvp_exact(self, i, t, x, label):
		"""<grad_theta f(x), dtheta> for the whole output, via a double vjp."""
		params = list(self.theta0.parameters())
		prev = [p.requires_grad for p in params]
		for p in params:
			p.requires_grad_(True)
		try:
			with torch.enable_grad():
				out = self.theta0(t, x, label=label)
				w = torch.zeros_like(out, requires_grad=True)
				g = torch.autograd.grad(out, params, grad_outputs=w, create_graph=True)
				jvp = torch.autograd.grad(g, w, grad_outputs=self._delta_as_param_list(i, x.device))
		finally:
			for p, r in zip(params, prev):
				p.requires_grad_(r)
		return jvp[0].detach()

	def _load_fd_step(self, i, device, fd_step=None):
		h = self.fd_step if fd_step is None else float(fd_step)
		if self._plus is None:
			self._plus = copy.deepcopy(self.theta0).eval()
			self._minus = copy.deepcopy(self.theta0).eval()
			for m in (self._plus, self._minus):
				for p in m.parameters():
					p.requires_grad_(False)
		self._plus.to(device)
		self._minus.to(device)
		d = self.delta_theta[i].to(device)
		norm = d.norm()
		base = parameters_to_vector(self.theta0.parameters()).detach()
		if float(norm) == 0.0:
			vector_to_parameters(base, self._plus.parameters())
			vector_to_parameters(base, self._minus.parameters())
			self._scale = 0.0
		else:
			u = d / norm
			vector_to_parameters(base + h * u, self._plus.parameters())
			vector_to_parameters(base - h * u, self._minus.parameters())
			self._scale = float(norm) / (2.0 * h)
		self._loaded_step = (i, h)

	def _jvp_fd(self, i, t, x, label, fd_step=None):
		h = self.fd_step if fd_step is None else float(fd_step)
		if self._loaded_step != (i, h):
			self._load_fd_step(i, x.device, fd_step=h)
		with torch.no_grad():
			if self._scale == 0.0:
				return torch.zeros_like(x)
			return self._scale * (self._plus(t, x, label=label) - self._minus(t, x, label=label))

	def jvp(self, i, t, x, label=None, mode=None):
		mode = mode or self.jvp_mode
		return self._jvp_exact(i, t, x, label) if mode == 'exact' else self._jvp_fd(i, t, x, label)

	def forward(self, t, x, label=None, device=None, k=None):
		i = self.step_index(t)
		with torch.no_grad():
			base = self.theta0(t, x, label=label)
		return base + self.jvp(i, t, x, label)

	def consistency_check(self, t, x, label=None):
		"""Relative disagreement between the exact and finite-difference JVPs.

		The exact path is the reference; this only flags whether the cheap FD path
		would have been usable, and catches a gross error in either.  Returns
		(max_abs_deviation, relative_deviation).
		"""
		i = self.step_index(t)
		a = self._jvp_exact(i, t, x, label)
		b = self._jvp_fd(i, t, x, label)
		dev = (a - b).abs().max()
		denom = a.abs().max().clamp_min(1e-12)
		return float(dev), float(dev / denom)
