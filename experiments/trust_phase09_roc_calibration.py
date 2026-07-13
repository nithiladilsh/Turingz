import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from hybrid_pde.trust.signals import shock_coeff, corrected_signal, energy_signal, roughness_signal
from hybrid_pde.trust.fuse import fit_fusion, fuse
from hybrid_pde.trust.horizon import smooth, fit_calibration, trust_score

START, FAIL, WIN = 2, 0.10, 5
OUT = os.path.join(ROOT, "results", "trust")
d = np.load(os.path.join(ROOT, "results", "eval", "predictions.npz"))
x, t = d["x"], d["t"]
true, train_true = d["u_true_eval"], d["u_true_seen"]
coeff = shock_coeff(train_true, x, t)
m = slice(START, None)
models = {"PINN": "PINN", "FNO": "FNO_eval", "DeepONet": "DeepONet_eval"}
cols = {"PINN": "#2563eb", "FNO": "#e76f51", "DeepONet": "#2a9d8f"}

def sig(u):
    return [corrected_signal(u, coeff, x, t), energy_signal(u, x), roughness_signal(u, x)]

def auc(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    alls = np.concatenate([pos, neg]); order = alls.argsort()
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(alls) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def roc(score, label):
    th = np.unique(score)[::-1]
    P, N = (label == 1).sum(), (label == 0).sum()
    tpr, fpr = [0.0], [0.0]
    for c in th:
        pred = score >= c
        tpr.append((pred & (label == 1)).sum() / P)
        fpr.append((pred & (label == 0)).sum() / N)
    return np.array(fpr), np.array(tpr)

warn_all, lab_all, trust_all = [], [], []
per = {}
for name, key in models.items():
    p = d[key]
    err = np.linalg.norm(p - true, axis=2) / (np.linalg.norm(true, axis=2) + 1e-12)
    S = sig(p); h = p.shape[0] // 2
    fp = fit_fusion([s[:h][:, m].reshape(-1) for s in S], err[:h][:, m].reshape(-1))
    fused = smooth(fuse([s[:, m] for s in S], fp), WIN)
    cal = fit_calibration(fused[:h], (err[:, m][:h] > FAIL).astype(float))
    trm = np.array([trust_score(f, cal) for f in fused])
    warn = (1 - trm[h:]).ravel(); lab = (err[h:, m] > FAIL).astype(int).ravel()
    per[name] = (warn, lab, trm[h:].ravel())
    warn_all.append(warn); lab_all.append(lab); trust_all.append(trm[h:].ravel())
warn_all = np.concatenate(warn_all); lab_all = np.concatenate(lab_all); trust_all = np.concatenate(trust_all)

fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))

for name in models:
    warn, lab, _ = per[name]
    a = auc(warn, lab)
    if np.isnan(a):
        continue
    f, tp = roc(warn, lab)
    ax[0].plot(f, tp, color=cols[name], lw=2, label="%s  (AUC %.2f)" % (name, a))
fa, ta = roc(warn_all, lab_all)
ax[0].plot(fa, ta, color="k", lw=2.5, label="Combined  (AUC %.2f)" % auc(warn_all, lab_all))
ax[0].plot([0, 1], [0, 1], "k:", lw=1, label="random (AUC 0.50)")
ax[0].set_xlabel("false alarm rate"); ax[0].set_ylabel("failures caught")
ax[0].set_title("ROC: how well the trust score separates\ngood moments from failed ones")
ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(True, alpha=0.3)

bins = np.linspace(0, 1, 11)
mids, obs, cnt = [], [], []
for i in range(10):
    sel = (trust_all >= bins[i]) & (trust_all < bins[i + 1] if i < 9 else trust_all <= bins[i + 1])
    if sel.sum() == 0:
        continue
    mids.append(trust_all[sel].mean())
    obs.append((lab_all[sel] == 0).mean())
    cnt.append(sel.sum())
ax[1].plot([0, 1], [0, 1], "k:", lw=1.2, label="perfectly honest")
ax[1].plot(mids, obs, "o-", color="#6a1b9a", lw=2, ms=7, label="our trust score")
for xm, ym, c in zip(mids, obs, cnt):
    ax[1].annotate(str(c), (xm, ym), textcoords="offset points", xytext=(4, 6), fontsize=7, color="#555")
ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
ax[1].set_xlabel("trust score the module reported")
ax[1].set_ylabel("fraction that were actually fine")
ax[1].set_title("Reliability: is the trust score honest?\n(points near the line = honest; numbers = frame counts)")
ax[1].legend(fontsize=9, loc="upper left"); ax[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "phase9_roc_calibration.png"), dpi=150)
print("AUC  PINN %.3f  FNO %.3f  combined %.3f" %
      (auc(*per["PINN"][:2]), auc(*per["FNO"][:2]), auc(warn_all, lab_all)))
print("saved to", OUT)
