import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from hybrid_pde.trust.signals import residual_signal, shock_coeff, corrected_signal

START = 2
OUT = os.path.join(ROOT, "results", "trust")
os.makedirs(OUT, exist_ok=True)

d = np.load(os.path.join(ROOT, "results", "eval", "predictions.npz"))
x, t = d["x"], d["t"]
true, train_true = d["u_true_eval"], d["u_true_seen"]
coeff = shock_coeff(train_true, x, t)
models = {"PINN": "PINN", "FNO": "FNO_eval", "DeepONet": "DeepONet_eval"}

def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

names, raw_c, cor_c = [], [], []
for name, key in models.items():
    if key not in d.files:
        continue
    pred = d[key]
    err = np.linalg.norm(pred - true, axis=2) / (np.linalg.norm(true, axis=2) + 1e-12)
    raw = residual_signal(pred, x, t)
    fixed = corrected_signal(pred, coeff, x, t)
    m = slice(START, None)
    names.append(name)
    raw_c.append(corr(raw[:, m], err[:, m]))
    cor_c.append(corr(fixed[:, m], err[:, m]))
    print("%-9s raw %.3f  ->  corrected %.3f" % (name, raw_c[-1], cor_c[-1]))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].plot(t, residual_signal(train_true, x, t).mean(0), color="#d1495b", lw=2)
ax[0].set_xlabel("time t"); ax[0].set_ylabel("residual of the CORRECT solution")
ax[0].set_title("The false alarm: a correct solution\nstill shows high residual at shock formation")
ax[0].grid(True, alpha=0.3)

xb = np.arange(len(names))
ax[1].bar(xb - 0.2, raw_c, 0.4, color="#9aa0a6", label="raw residual")
ax[1].bar(xb + 0.2, cor_c, 0.4, color="#2563eb", label="shock-corrected (steepness)")
ax[1].set_xticks(xb); ax[1].set_xticklabels(names)
ax[1].set_ylabel("correlation with true error")
ax[1].set_title("Shock correction helps where the shock\npollutes the signal (FNO, PINN)")
ax[1].legend(); ax[1].grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "phase2_shock_correction.png"), dpi=150)
print("saved to", OUT)
