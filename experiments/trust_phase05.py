import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from hybrid_pde.trust.signals import shock_coeff, corrected_signal, energy_signal, roughness_signal
from hybrid_pde.trust.fuse import fit_fusion, fuse
from hybrid_pde.trust.horizon import (smooth, fit_calibration, trust_score,
                                      reliable_horizon, true_horizon, select_persistence)

START, FAIL, WIN, CUT = 2, 0.10, 5, 0.5
OUT = os.path.join(ROOT, "results", "trust")
os.makedirs(OUT, exist_ok=True)

d = np.load(os.path.join(ROOT, "results", "eval", "predictions.npz"))
x, t = d["x"], d["t"]
true, train_true = d["u_true_eval"], d["u_true_seen"]
coeff = shock_coeff(train_true, x, t)
te = t[START:]
models = {"PINN": "PINN", "FNO": "FNO_eval", "DeepONet": "DeepONet_eval"}
cols = {"PINN": "#2563eb", "FNO": "#e76f51", "DeepONet": "#2a9d8f"}

def sig(u):
    return [corrected_signal(u, coeff, x, t), energy_signal(u, x), roughness_signal(u, x)]

fits = {}
tr_train, err_train = [], []
for name, key in models.items():
    p = d[key]
    err = np.linalg.norm(p - true, axis=2) / (np.linalg.norm(true, axis=2) + 1e-12)
    S = sig(p)
    h = p.shape[0] // 2
    m = slice(START, None)
    Str = [s[:h][:, m].reshape(-1) for s in S]
    ytr = err[:h][:, m].reshape(-1)
    fp = fit_fusion(Str, ytr)
    fused = smooth(fuse([s[:, m] for s in S], fp), WIN)
    fail = (err[:, m] > FAIL).astype(float)
    cal = fit_calibration(fused[:h], fail[:h])
    tr = np.array([trust_score(f, cal) for f in fused])
    tr_train.append(tr[:h]); err_train.append(err[:h, m])
    fits[name] = dict(err=err[:, m], fused=fused, trust=tr, h=h)

K = select_persistence(np.vstack(tr_train), te, np.vstack(err_train), FAIL, CUT)
print("persistence window K (chosen on training waves) =", K)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for j, (name, f) in enumerate(fits.items()):
    i = f["h"]
    tr, err = f["trust"][i], f["err"][i]
    ph = reliable_horizon(tr, te, CUT, K)
    th = true_horizon(err, te, FAIL)
    a = ax[j]
    a.plot(te, tr, color=cols[name], lw=2, label="trust score (0-1)")
    a.plot(te, np.minimum(err / FAIL, 1.5) / 1.5, color="#999", lw=1.5, ls="--",
           label="true error (scaled)")
    a.axhline(CUT, color="k", lw=0.8, ls=":")
    a.axvline(ph, color=cols[name], lw=1.5, label="predicted horizon")
    a.axvline(th, color="#444", lw=1.5, ls="--", label="true horizon")
    a.set_title("%s   predicted %.2f vs true %.2f" % (name, ph, th))
    a.set_xlabel("time t"); a.set_ylim(-0.05, 1.05)
    if j == 0:
        a.set_ylabel("trust  /  scaled error"); a.legend(fontsize=7, loc="lower left")
    a.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "phase5_trust_curves.png"), dpi=150)

fig, ax = plt.subplots(figsize=(5.6, 5.4))
P, T = [], []
for name, f in fits.items():
    for i in range(f["h"], f["err"].shape[0]):
        ph = reliable_horizon(f["trust"][i], te, CUT, K)
        th = true_horizon(f["err"][i], te, FAIL)
        P.append(ph); T.append(th)
        ax.scatter(th, ph, color=cols[name], s=55, edgecolor="w", zorder=3)
P, T = np.array(P), np.array(T)
lim = [0, 2.05]
ax.plot(lim, lim, "k--", lw=1, label="perfect")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("true horizon (where error first exceeds 10%)")
ax.set_ylabel("predicted horizon (reference-free)")
ax.set_title("Predicted vs true reliable horizon\nMAE = %.2f    correlation = %.2f"
             % (np.abs(P - T).mean(), np.corrcoef(P, T)[0, 1]))
for name in models:
    ax.scatter([], [], color=cols[name], label=name)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "phase5_horizon_accuracy.png"), dpi=150)
print("test MAE=%.2f  corr=%.2f" % (np.abs(P - T).mean(), np.corrcoef(P, T)[0, 1]))
print("saved to", OUT)
