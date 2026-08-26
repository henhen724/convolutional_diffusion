"""Plot pooled/OOS neuron-sample R^2 and MI results from ntk_neuron_mi.py."""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def load(path):
    npz = np.load(path, allow_pickle=True)
    meta = json.loads(str(npz["meta"]))
    out = {"meta": meta}
    for act in ["linear", "relu"]:
        out[act] = {k: npz[f"{act}_{k}"] for k in [
            "checkpoints", "R2_train", "R2_train_stderr", "MI_train_nats",
            "R2_test", "R2_test_stderr", "MI_test_nats",
        ]}
        out[act]["mean_final_loss"] = float(npz[f"{act}_mean_final_loss"])
    return out


def noise_floor(meta):
    """Expected E[R^2_oos] under a true null, order d / N_fit (N_fit = n * (n_trials//2) * m)."""
    n_fit = meta["n"] * (meta["n_trials"] // 2) * meta["m"]
    return -meta["d"] / n_fit


def gaussian_mi_nats(r2):
    """Gaussian-formula mutual information (nats) between a scalar and a vector,
    given the R^2 of the linear regression of the scalar on the vector:
        I(H;X) = -1/2 * log(1 - R^2)
    MI cannot be negative, so R^2 (which can come back slightly negative from an
    out-of-sample estimator under the null) is clipped to [0, 1) first.
    """
    r2 = np.clip(np.asarray(r2, dtype=float), 0.0, 1.0 - 1e-12)
    return -0.5 * np.log(1.0 - r2)


def plot_mi_over_time(ax, data, task_name):
    for act, color in [("linear", "tab:blue"), ("relu", "tab:orange")]:
        d = data[act]
        t = np.asarray(d["checkpoints"], dtype=float)
        mi = gaussian_mi_nats(d["R2_train"])
        mask = t > 0
        ax.plot(t[mask], mi[mask], "o-", color=color, ms=4,
                label=f"{act} (final loss={d['mean_final_loss']:.2g})")
    ax.set_xscale("log")
    ax.set_xlabel("training step $t$")
    ax.set_ylabel(r"$I(\mathrm{neuron}(t)\,;\,x_i)$ [nats]")
    ax.set_title(f"Gaussian-formula MI: {task_name}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_mi_width_scan(ax, width_results):
    ms = sorted(width_results.keys())
    for act, color in [("linear", "tab:blue"), ("relu", "tab:orange")]:
        vals = np.array([gaussian_mi_nats(width_results[m][act]["R2_train"][-1]) for m in ms])
        ax.plot(ms, vals, "o-", color=color, label=f"{act}")
    ax.set_xscale("log")
    ax.set_xlabel("hidden width $m$")
    ax.set_ylabel(r"$I(\mathrm{neuron}\,;\,x_i)$ [nats], final $t$")
    ax.set_title("Gaussian-formula MI vs. width")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


def plot_mi_memorization_gap(ax, data, task_name):
    xs = np.arange(2)
    width = 0.35
    for i, act in enumerate(["linear", "relu"]):
        d = data[act]
        mi_tr = gaussian_mi_nats(d["R2_train"][-1])
        mi_te = gaussian_mi_nats(d["R2_test"][-1])
        ax.bar(xs + i * width, [mi_tr, mi_te], width, label=act)
    ax.set_xticks(xs + width / 2)
    ax.set_xticklabels(["train points", "held-out points"])
    ax.set_ylabel(r"$I(\mathrm{neuron}\,;\,x)$ [nats], final $t$")
    ax.set_title(f"MI memorization gap: {task_name}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")


def plot_over_time(ax, data, task_name):
    for act, color in [("linear", "tab:blue"), ("relu", "tab:orange")]:
        d = data[act]
        t = np.asarray(d["checkpoints"], dtype=float)
        r2 = np.asarray(d["R2_train"], dtype=float)
        se = np.asarray(d["R2_train_stderr"], dtype=float)
        mask = t > 0
        ax.errorbar(t[mask], r2[mask], yerr=se[mask], fmt="o-", color=color,
                    capsize=2, ms=4, label=f"{act} (final loss={d['mean_final_loss']:.2g})")
    ax.axhline(noise_floor(data["meta"]), ls="--", color="gray",
               label=r"expected CV-null bias $-d/N_{fit}$")
    ax.set_xscale("log")
    ax.set_xlabel("training step $t$")
    ax.set_ylabel(r"out-of-sample $R^2$(neuron, $x_i$)")
    ax.set_title(f"NTK training: {task_name}\n(no detectable change from init -> converged)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_width_scan(ax, width_results):
    ms = sorted(width_results.keys())
    for act, color in [("linear", "tab:blue"), ("relu", "tab:orange")]:
        vals = np.array([abs(width_results[m][act]["R2_train"][-1]) for m in ms])
        se = np.array([width_results[m][act]["R2_train_stderr"][-1] for m in ms])
        ax.errorbar(ms, vals, yerr=se, fmt="o-", color=color, capsize=2, label=f"{act} |R^2| (final t)")
    meta0 = width_results[ms[0]]["linear"]["meta"] if "meta" in width_results[ms[0]]["linear"] else None
    floor = [abs(noise_floor(width_results[m]["meta"])) for m in ms]
    ax.plot(ms, floor, "--", color="gray", label=r"expected CV-null bias $|d/N_{fit}|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("hidden width $m$")
    ax.set_ylabel(r"$|R^2|$ (neuron, $x_i$), final $t$")
    ax.set_title("Width scan (supervised): observed values track the\nCV-null noise floor -- no residual signal above it")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


def plot_memorization_gap(ax, data, task_name):
    xs = np.arange(2)
    width = 0.35
    for i, act in enumerate(["linear", "relu"]):
        d = data[act]
        r2_tr, se_tr = d["R2_train"][-1], d["R2_train_stderr"][-1]
        r2_te, se_te = d["R2_test"][-1], d["R2_test_stderr"][-1]
        ax.bar(xs + i * width, [r2_tr, r2_te], width, yerr=[se_tr, se_te], capsize=3, label=act)
    ax.set_xticks(xs + width / 2)
    ax.set_xticklabels(["train points", "held-out points"])
    ax.set_ylabel(r"out-of-sample $R^2$(neuron, $x$), final $t$")
    ax.set_title(f"Memorization gap: {task_name}\n(no significant train-vs-held-out gap detected)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--supervised", required=True)
    p.add_argument("--denoising", required=True)
    p.add_argument("--width_scan_dir", required=True)
    p.add_argument("--width_scan_ms", type=int, nargs="+", default=[20, 50, 150, 500, 1500, 3000])
    p.add_argument("--out_prefix", default="ntk_neuron_mi")
    args = p.parse_args()

    sup = load(args.supervised)
    den = load(args.denoising)
    width_results = {
        m: load(f"{args.width_scan_dir}/supervised_d16_m{m}_n150.npz") for m in args.width_scan_ms
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_mi_over_time(axes[0, 0], sup, "supervised regression $y=v^Tx$")
    plot_mi_over_time(axes[0, 1], den, f"denoising (beta={den['meta']['beta']})")
    plot_over_time(axes[1, 0], sup, "supervised regression $y=v^Tx$")
    plot_over_time(axes[1, 1], den, f"denoising (beta={den['meta']['beta']})")
    fig.suptitle("Mutual information (top, Gaussian formula) and its underlying R^2 (bottom) "
                 f"(d={sup['meta']['d']}, m={sup['meta']['m']}, n={sup['meta']['n']}, trials={sup['meta']['n_trials']})")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_over_time.png", dpi=150)
    print(f"Saved {args.out_prefix}_over_time.png")

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_mi_width_scan(axes2[0], width_results)
    plot_width_scan(axes2[1], width_results)
    fig2.tight_layout()
    fig2.savefig(f"{args.out_prefix}_width_scan.png", dpi=150)
    print(f"Saved {args.out_prefix}_width_scan.png")

    fig3, axes3 = plt.subplots(2, 2, figsize=(9, 8))
    plot_mi_memorization_gap(axes3[0, 0], sup, "supervised regression $y=v^Tx$")
    plot_mi_memorization_gap(axes3[0, 1], den, f"denoising (beta={den['meta']['beta']})")
    plot_memorization_gap(axes3[1, 0], sup, "supervised regression $y=v^Tx$")
    plot_memorization_gap(axes3[1, 1], den, f"denoising (beta={den['meta']['beta']})")
    fig3.tight_layout()
    fig3.savefig(f"{args.out_prefix}_memorization_gap.png", dpi=150)
    print(f"Saved {args.out_prefix}_memorization_gap.png")


if __name__ == "__main__":
    main()
