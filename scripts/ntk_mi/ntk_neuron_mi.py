"""
Mutual information between a single hidden-layer neuron and individual training
samples, over the course of lazy (NTK-regime) training.

Setup
-----
Data: x in R^d, x ~ N(0, I_d) (isotropic). Target direction v = e_0 (WLOG by
isotropy of x -- any fixed unit vector gives the same statistics).

Network: 2-layer NTK-parameterized MLP,
    z(x)   = x @ W.T / sqrt(d)            hidden pre-activations, W ~ N(0,1) iid
    h(x)   = phi(z(x))                    phi = identity ("linear") or ReLU
    f(x)   = h(x) @ readout / sqrt(m)     readout ~ N(0,1) iid
trained by ordinary full-batch gradient descent (all parameters, autograd) with
a small fixed learning rate. At large width this realizes the "lazy"/NTK
training regime empirically, without hand-deriving the kernel gradient-descent
ODE.

Tasks
-----
  supervised: target y_i = v^T x_i (scalar regression), out_dim = 1.
  denoising : single fixed noise level beta. xtilde = sqrt(1-beta) x + sqrt(beta) eps,
              network predicts eps from xtilde (matches this repo's diffusion
              training convention), out_dim = d. MI is measured against the
              underlying *clean* x_i, not xtilde_i.

Mutual information -- and why a naive pooled estimate is misleading
---------------------------------------------------------------------
We want I(neuron_j(t) ; x_i). MI against a literal constant vector is zero, so
the quantity only makes sense once "a neuron" and "a sample" are embedded in
an ensemble: on every Monte Carlo trial we redraw a fresh training set AND a
fresh network init, train to each checkpoint, and consider the joint law of
(h_j(input_i,t), x_i) over neurons j, samples i, and trials.

A hidden unit's trained activation is, to leading order in the lazy/NTK
regime, h_j(x,t) ~= h_j(x,0) + a_j(0) * g(x,t), where g is a *shared* function
of the data (from the collective, self-averaging dynamics of all m neurons)
and a_j(0) is that neuron's own random, mean-zero outer-layer weight. Naively
pooling raw covariances Cov(h_j, x) over many neurons averages over the
random *sign* of a_j(0) and cancels almost exactly for a linear network (a
strictly odd dependence on a_j(0)), leaving only estimator noise -- not real
dynamics. ReLU breaks this cancellation only weakly (a_j(0) enters *inside*
the nonlinearity's argument, not just as a multiplicative output sign).

So the pooled linear-correlation MI computed here is a real, but genuinely
tiny (down at O(1/width)), quantity -- consistent with the defining feature of
lazy training that an individual neuron's parameters move by only O(1/sqrt(m)).
To resolve it above finite-sample estimation bias we use an OUT-OF-SAMPLE
(split-half) estimator: fit the regression coefficient on one random half of
the trials and evaluate R^2 on the other half (and vice versa), which is
unbiased (can go negative) unlike the trivially-upward-biased in-sample R^2.

    R^2 = Cov(h,x)^T Cov(x,x)^-1 Cov(h,x) / Var(h)     (fit on one half)
    R^2_oos = 1 - [Var(h) - 2 beta.Cov(h,x) + beta^T Cov(x,x) beta] / Var(h)   (eval on the other half)
    I(h;x) = -0.5 * log(1 - max(R^2_oos, 0))   [nats]

Two analyses are produced from the same run:
  1. width scaling: this pooled/OOS MI evaluated on the actual training points,
     as a function of hidden width m (run this script at several --m and
     compare the converged R^2 across runs -- expected to scale like 1/m).
  2. memorization gap: the SAME estimator evaluated on the training points the
     network actually saw vs. a matched, freshly-drawn held-out set from the
     same trained network, at fixed m. A gap between the two curves is
     information specifically about having been *in* the training set, beyond
     the shared population-level structure (e.g. the learned direction v) that
     a generic point would show too.
"""
import argparse
import json
import time

import numpy as np
import torch


def make_checkpoints(n_steps, n_points=18):
    pts = {0, n_steps}
    if n_steps >= 1:
        for x in np.unique(np.round(np.geomspace(1, n_steps, n_points)).astype(int)):
            pts.add(int(x))
    return sorted(pts)


def init_network(d, m, out_dim, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    W = torch.randn(m, d, generator=g, device=device)
    if out_dim == 1:
        readout = torch.randn(m, generator=g, device=device)
    else:
        readout = torch.randn(out_dim, m, generator=g, device=device)
    W.requires_grad_(True)
    readout.requires_grad_(True)
    return W, readout


def forward(x, W, readout, activation, d, m):
    z = x @ W.T / np.sqrt(d)
    h = z if activation == "linear" else torch.relu(z)
    if readout.dim() == 1:
        f = h @ readout / np.sqrt(m)
    else:
        f = h @ readout.T / np.sqrt(m)
    return f, h


def run_trial(task, activation, d, m, n, n_steps, checkpoints, lr, beta, seed, device):
    torch.manual_seed(seed)
    x = torch.randn(n, d, device=device)
    x_test = torch.randn(n, d, device=device)  # matched, never used for training
    v = torch.zeros(d, device=device)
    v[0] = 1.0

    if task == "supervised":
        net_input, test_input = x, x_test
        target = x @ v
        out_dim = 1
    elif task == "denoising":
        eps = torch.randn(n, d, device=device)
        eps_test = torch.randn(n, d, device=device)
        net_input = np.sqrt(1.0 - beta) * x + np.sqrt(beta) * eps
        test_input = np.sqrt(1.0 - beta) * x_test + np.sqrt(beta) * eps_test
        target = eps
        out_dim = d
    else:
        raise ValueError(task)

    W, readout = init_network(d, m, out_dim, seed + 10_000, device)
    opt = torch.optim.SGD([W, readout], lr=lr)

    ckpt_set = set(checkpoints)
    records_train, records_test = {}, {}
    for step in range(n_steps + 1):
        if step in ckpt_set:
            with torch.no_grad():
                _, h = forward(net_input, W, readout, activation, d, m)
                records_train[step] = h.detach().cpu().numpy()  # (n, m)
                _, h_test = forward(test_input, W, readout, activation, d, m)
                records_test[step] = h_test.detach().cpu().numpy()  # (n, m)
        if step == n_steps:
            break
        f, _ = forward(net_input, W, readout, activation, d, m)
        loss = torch.mean((f - target) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return records_train, records_test, x.cpu().numpy(), x_test.cpu().numpy(), float(loss.item())


def trial_stats(x, h):
    """Sufficient statistics for one trial's pooled (h_j, x_i) sample, j=1..m, i=1..n."""
    n, m = h.shape
    colsum = h.sum(axis=1)  # (n,), sum over neurons per sample
    return {
        "sum_x": x.sum(axis=0), "sum_xx": x.T @ x, "n": n,
        "sum_h": float(h.sum()), "sum_hh": float((h ** 2).sum()),
        "sum_hx": x.T @ colsum, "m": m,
    }


def aggregate(stats_list, d):
    agg = {
        "sum_x": np.zeros(d), "sum_xx": np.zeros((d, d)), "count_x": 0,
        "sum_h": 0.0, "sum_hh": 0.0, "sum_hx": np.zeros(d), "count_h": 0,
    }
    for s in stats_list:
        agg["sum_x"] += s["sum_x"]
        agg["sum_xx"] += s["sum_xx"]
        agg["count_x"] += s["n"]  # x-moments: n independent draws per trial, NOT n*m
        agg["sum_h"] += s["sum_h"]
        agg["sum_hh"] += s["sum_hh"]
        agg["sum_hx"] += s["sum_hx"]
        agg["count_h"] += s["n"] * s["m"]
    return agg


def moments(agg):
    mean_x = agg["sum_x"] / agg["count_x"]
    cov_xx = agg["sum_xx"] / agg["count_x"] - np.outer(mean_x, mean_x)
    mean_h = agg["sum_h"] / agg["count_h"]
    var_h = agg["sum_hh"] / agg["count_h"] - mean_h ** 2
    cov_hx = agg["sum_hx"] / agg["count_h"] - mean_h * mean_x
    return mean_x, cov_xx, mean_h, var_h, cov_hx


def oos_r2(fit_agg, eval_agg):
    _, cov_xx_fit, _, _, cov_hx_fit = moments(fit_agg)
    beta = np.linalg.solve(cov_xx_fit, cov_hx_fit)
    _, cov_xx_eval, _, var_h_eval, cov_hx_eval = moments(eval_agg)
    mse = var_h_eval - 2 * beta @ cov_hx_eval + beta @ cov_xx_eval @ beta
    return float(1.0 - mse / var_h_eval)


def split_half_r2(stats_list, d, n_splits, rng):
    """Average bidirectional out-of-sample R^2 over n_splits random 50/50 trial splits."""
    n_trials = len(stats_list)
    vals = []
    for _ in range(n_splits):
        perm = rng.permutation(n_trials)
        half = n_trials // 2
        idx_a, idx_b = perm[:half], perm[half:]
        agg_a = aggregate([stats_list[i] for i in idx_a], d)
        agg_b = aggregate([stats_list[i] for i in idx_b], d)
        vals.append(oos_r2(agg_a, agg_b))
        vals.append(oos_r2(agg_b, agg_a))
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))


class MIAccumulator:
    def __init__(self, d, checkpoints):
        self.d = d
        self.checkpoints = list(checkpoints)
        self.train_stats = {k: [] for k in self.checkpoints}
        self.test_stats = {k: [] for k in self.checkpoints}
        self.final_losses = []

    def add_trial(self, records_train, records_test, x_train, x_test, final_loss):
        self.final_losses.append(final_loss)
        for k in self.checkpoints:
            self.train_stats[k].append(trial_stats(x_train, records_train[k]))
            self.test_stats[k].append(trial_stats(x_test, records_test[k]))

    def finalize(self, n_splits=10, seed=0):
        rng = np.random.default_rng(seed)
        out = {
            "checkpoints": self.checkpoints,
            "R2_train": [], "R2_train_stderr": [], "MI_train_nats": [],
            "R2_test": [], "R2_test_stderr": [], "MI_test_nats": [],
        }
        for k in self.checkpoints:
            r2_tr, se_tr = split_half_r2(self.train_stats[k], self.d, n_splits, rng)
            r2_te, se_te = split_half_r2(self.test_stats[k], self.d, n_splits, rng)
            out["R2_train"].append(r2_tr)
            out["R2_train_stderr"].append(se_tr)
            out["MI_train_nats"].append(-0.5 * np.log(1.0 - max(r2_tr, 0.0)))
            out["R2_test"].append(r2_te)
            out["R2_test_stderr"].append(se_te)
            out["MI_test_nats"].append(-0.5 * np.log(1.0 - max(r2_te, 0.0)))
        out["mean_final_loss"] = float(np.mean(self.final_losses))
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["supervised", "denoising"], required=True)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--m", type=int, default=2000)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--n_steps", type=int, default=3000)
    p.add_argument("--n_checkpoints", type=int, default=18)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.5, help="denoising noise level")
    p.add_argument("--n_splits", type=int, default=10, help="random 50/50 trial splits for OOS R^2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    checkpoints = make_checkpoints(args.n_steps, args.n_checkpoints)

    results = {}
    for activation in ["linear", "relu"]:
        acc = MIAccumulator(args.d, checkpoints)
        t0 = time.time()
        for trial in range(args.n_trials):
            records_train, records_test, x_train, x_test, final_loss = run_trial(
                task=args.task,
                activation=activation,
                d=args.d,
                m=args.m,
                n=args.n,
                n_steps=args.n_steps,
                checkpoints=checkpoints,
                lr=args.lr,
                beta=args.beta,
                seed=args.seed * 10_000 + trial,
                device=args.device,
            )
            acc.add_trial(records_train, records_test, x_train, x_test, final_loss)
        results[activation] = acc.finalize(n_splits=args.n_splits, seed=args.seed)
        elapsed = time.time() - t0
        r = results[activation]
        print(
            f"[{args.task}/{activation}, m={args.m}] {args.n_trials} trials in {elapsed:.1f}s, "
            f"mean final loss={r['mean_final_loss']:.4g}, "
            f"R2_train(t=0)={r['R2_train'][0]:.4g}, R2_train(final)={r['R2_train'][-1]:.4g}, "
            f"R2_test(final)={r['R2_test'][-1]:.4g}"
        )

    meta = vars(args) | {"checkpoints": checkpoints}
    np.savez(
        args.out,
        meta=json.dumps(meta),
        **{f"{act}_{k}": np.array(v) for act, r in results.items() for k, v in r.items()},
    )
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
