import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from hybrid_pde.trust.signals import residual_signal, energy_signal, roughness_signal

START = 2
OUT = os.path.join(ROOT, "results", "trust")
os.makedirs(OUT, exist_ok=True)

d = np.load(os.path.join(ROOT, "results", "eval", "predictions.npz"))
x, t = d["x"], d["t"]
true = d["u_true_eval"]
models = {"PINN": "PINN", "FNO": "FNO_eval", "DeepONet": "DeepONet_eval"}
signals = [("physics residual", lambda p: residual_signal(p, x, t)),
           ("energy drift", lambda p: energy_signal(p, x)),
           ("roughness", lambda p: roughness_signal(p, x))]

def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

m = slice(START, None)
scores = {}
for name, key in models.items():
    if key not in d.files:
        continue
    pred = d[key]
    err = np.linalg.norm(pred - true, axis=2) / (np.linalg.norm(true, axis=2) + 1e-12)
    scores[name] = [corr(fn(pred)[:, m], err[:, m]) for _, fn in signals]
    print(name, " ".join(f"{sn}={c:.2f}" for (sn, _), c in zip(signals, scores[name])))

# Figure 1: which signal catches which model
fig, ax = plt.subplots(figsize=(8.5, 5))
labels = list(scores.keys())
xb = np.arange(len(labels))
colors = ["#2563eb", "#2e8b57", "#d1495b"]
for i, (sn, _) in enumerate(signals):
    ax.bar(xb + (i - 1) * 0.26, [scores[l][i] for l in labels], 0.26, color=colors[i], label=sn)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xb); ax.set_xticklabels(labels)
ax.set_ylabel("correlation with true error")
ax.set_title("No single signal wins for every model\n(roughness rescues DeepONet where the residual fails)")
ax.legend(); ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "phase3_signals_per_model.png"), dpi=150); plt.close()

# Figure 2: DeepONet rescue
pred = d["DeepONet_eval"]
err = (np.linalg.norm(pred - true, axis=2) / (np.linalg.norm(true, axis=2) + 1e-12)).mean(0)
res = residual_signal(pred, x, t).mean(0)
rgh = roughness_signal(pred, x).mean(0)
norm = lambda a: a / a.max()
plt.figure(figsize=(8.5, 5))
plt.plot(t, norm(err), color="#d1495b", lw=2.5, label="true error")
plt.plot(t, norm(res), color="#2563eb", lw=2, label="physics residual")
plt.plot(t, norm(rgh), color="#2e8b57", lw=2, label="roughness")
plt.axvline(1.0, color="k", ls="--", lw=1)
plt.xlabel("time t"); plt.ylabel("each curve scaled to its own max")
plt.title("DeepONet: roughness follows the true error,\nwhile the physics residual drops away")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "phase3_deeponet_rescue.png"), dpi=150); plt.close()
print("saved to", OUT)
