import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from hybrid_pde.trust.signals import shock_coeff, corrected_signal, energy_signal, roughness_signal
from hybrid_pde.trust.fuse import fit_fusion, fuse

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

def signals_of(p):
    return [corrected_signal(p, coeff, x, t), energy_signal(p, x), roughness_signal(p, x)]

names = list(models)
res, ene, rou, fus = [], [], [], []
for name, key in models.items():
    p = d[key]
    err = np.linalg.norm(p - true, axis=2) / (np.linalg.norm(true, axis=2) + 1e-12)
    S = signals_of(p)
    h = p.shape[0] // 2
    m = slice(START, None)
    Str = [s[:h][:, m].reshape(-1) for s in S]
    Ste = [s[h:][:, m] for s in S]
    ytr = err[:h][:, m].reshape(-1)
    yte = err[h:][:, m]
    params = fit_fusion(Str, ytr)
    f = fuse(Ste, params)
    res.append(corr(Ste[0], yte)); ene.append(corr(Ste[1], yte))
    rou.append(corr(Ste[2], yte)); fus.append(corr(f, yte))
    print("%-9s resid %.2f energy %.2f rough %.2f FUSED %.2f" % (name, res[-1], ene[-1], rou[-1], fus[-1]))

strat = {"always residual": res, "always energy": ene, "always roughness": rou, "FUSED": fus}
worst = {k: min(v) for k, v in strat.items()}
print("worst-case:", {k: round(v, 2) for k, v in worst.items()})

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

xb = np.arange(len(names))
cols = {"always residual": "#9aa0a6", "always energy": "#f4a261",
        "always roughness": "#e76f51", "FUSED": "#2563eb"}
off = np.linspace(-0.3, 0.3, 4)
for i, (k, v) in enumerate(strat.items()):
    ax[0].bar(xb + off[i], v, 0.18, color=cols[k], label=k)
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_xticks(xb); ax[0].set_xticklabels(names)
ax[0].set_ylabel("correlation with true error")
ax[0].set_title("Each single signal collapses on some model.\nThe fused score stays high everywhere.")
ax[0].legend(fontsize=8); ax[0].grid(True, axis="y", alpha=0.3)

kb = np.arange(len(strat))
wc = [worst[k] for k in strat]
bars = ax[1].bar(kb, wc, 0.6, color=[cols[k] for k in strat])
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xticks(kb); ax[1].set_xticklabels(list(strat), rotation=20, ha="right")
ax[1].set_ylabel("worst-case correlation across all 3 models")
ax[1].set_title("Worst-case reliability:\nfusion is the only strategy that never fails")
for b, w in zip(bars, wc):
    ax[1].text(b.get_x() + b.get_width() / 2, w + (0.02 if w >= 0 else -0.06),
               "%.2f" % w, ha="center", fontsize=9)
ax[1].grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "phase4_fusion_robustness.png"), dpi=150)
print("saved to", OUT)
