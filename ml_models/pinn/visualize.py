import os
os.environ.setdefault("DDE_BACKEND", "pytorch")

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_prediction(model_dir: str):
    blob = torch.load(os.path.join(model_dir, "prediction.pt"),
                      map_location="cpu", weights_only=False)
    return (blob["u_pred"].numpy(), blob["u_ref"].numpy(),
            blob["x"].numpy(), blob["t"].numpy(), float(blob["t_train_end"]))


def plot_comparison(model_dir: str, out_path: str = None):
    pred, ref, x, t, t_end = _load_prediction(model_dir)
    err = np.abs(pred - ref)
    out_path = out_path or os.path.join(model_dir, "pinn_comparison.png")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    vmax = max(np.max(np.abs(ref)), 1e-9)

    for ax, field, title, cmap, vmn, vmx in [
        (axes[0, 0], ref, "Cole-Hopf (ground truth)", "RdBu_r", -vmax, vmax),
        (axes[0, 1], pred, "PINN prediction", "RdBu_r", -vmax, vmax),
        (axes[0, 2], err, "|error|", "magma", 0, np.max(err) + 1e-12),
    ]:
        im = ax.pcolormesh(x, t, field, cmap=cmap, shading="auto", vmin=vmn, vmax=vmx)
        ax.axhline(t_end, color="k", ls="--", lw=1)
        ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_title(title)
        fig.colorbar(im, ax=ax)

    # snapshots
    ax = axes[1, 0]
    for tt in [0.0, 0.5, 1.0, 1.5, 2.0]:
        idx = int(np.argmin(np.abs(t - tt)))
        ax.plot(x, ref[idx], color="0.6", lw=2)
        ax.plot(x, pred[idx], "--", lw=1.4, label=f"t={t[idx]:.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("u")
    ax.set_title("snapshots (grey=ref, dashed=PINN)")
    ax.legend(fontsize=7)

    # per-time relative L2
    rel = np.linalg.norm(pred - ref, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-12)
    ax = axes[1, 1]
    ax.semilogy(t, rel, color="crimson", lw=1.6)
    ax.axvline(t_end, color="k", ls="--", lw=1, label="train / extrap")
    ax.axvspan(0.25, 0.75, color="orange", alpha=0.15, label="shock window")
    ax.set_xlabel("t"); ax.set_ylabel("relative L2"); ax.set_title("error vs time")
    ax.legend(fontsize=8)

    # energy
    dx = x[1] - x[0]
    e_ref = 0.5 * dx * np.sum(ref ** 2, axis=1)
    e_pred = 0.5 * dx * np.sum(pred ** 2, axis=1)
    ax = axes[1, 2]
    ax.semilogy(t, e_ref, color="0.5", lw=2, label="ref")
    ax.semilogy(t, e_pred, "--", color="steelblue", lw=1.5, label="PINN")
    ax.axvline(t_end, color="k", ls="--", lw=1)
    ax.set_xlabel("t"); ax.set_ylabel("E(t)"); ax.set_title("energy")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    path = plot_comparison(args.model, args.out)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
