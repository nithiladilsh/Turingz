import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from hybrid_pde.trust.signals import residual_signal

FAIL = 0.10
OUT = os.path.join(ROOT, "results", "trust")
os.makedirs(OUT, exist_ok=True)

d = np.load(os.path.join(ROOT, "results", "eval", "predictions.npz"))
x, t, true = d["x"], d["t"], d["u_true_eval"]
models = {"PINN": "PINN", "FNO": "FNO_eval", "DeepONet": "DeepONet_eval"}

for name, key in models.items():
    if key not in d.files:
        continue
    pred = d[key]
    error = np.linalg.norm(pred - true, axis=2) / (np.linalg.norm(true, axis=2) + 1e-12)
    label = (error > FAIL).astype(int)
    signal = residual_signal(pred, x, t)
    corr = float(np.corrcoef(signal.ravel(), error.ravel())[0, 1])
    print(f"{name:9s} corr(residual, true error) = {corr:.3f}")

    np.savez(os.path.join(OUT, f"answerkey_{name}.npz"),
             error=error, label=label, residual=signal, t=t)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, error.mean(0) * 100, color="#d1495b", lw=2)
    ax.set_xlabel("time t")
    ax.set_ylabel("true error vs Cole-Hopf (%)", color="#d1495b")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax2 = ax.twinx()
    ax2.plot(t, signal.mean(0), color="#2563eb", lw=2)
    ax2.set_ylabel("physics residual (no ground truth)", color="#2563eb")
    ax.set_title(f"{name}: warning signal tracks true error  (corr = {corr:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"phase1_{name}.png"), dpi=150)
    plt.close(fig)

print(f"saved answer keys + plots to {OUT}")
