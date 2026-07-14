import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from hybrid_pde.trust.signals import shock_coeff, residual_signal
from hybrid_pde.trust.monitor import TrustMonitor, load_params, FAIL

START = 2; m = slice(START, None)
OUT = os.path.join(ROOT, "results", "trust")
d = np.load(os.path.join(ROOT, "results", "eval", "predictions_extended.npz"))
x, t = d["x"], d["t"]
inw = t[START:] <= 1.0
mods = {"FNO": "#e76f51", "DeepONet": "#2a9d8f"}

def relerr(p, u):
    return np.linalg.norm(p - u, axis=2) / (np.linalg.norm(u, axis=2) + 1e-12)
def sens(p, pp):
    return np.linalg.norm(p - pp, axis=2) / (np.linalg.norm(p, axis=2) + 1e-12)
def corr(a, b):
    a, b = a.ravel(), b.ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

trust_mean = {}
for M in mods:
    pr = load_params(os.path.join(OUT, "trust_params_%s.npz" % M))
    for tag in ["id", "ood"]:
        P = d["%s_%s" % (M, tag)]
        tr = np.zeros((len(P), P.shape[1]))
        for i in range(len(P)):
            mon = TrustMonitor(pr)
            for n in range(P.shape[1]):
                tr[i, n] = mon.update(P[i, n], float(t[n]))["trust"]
        trust_mean[(M, tag)] = tr[:, m][:, inw].mean()

pert = {}
for M in mods:
    for tag in ["id", "ood"]:
        P = d["%s_%s" % (M, tag)]; PP = d["%s_%s_pert" % (M, tag)]; U = d["u_true_%s" % tag]
        err = relerr(P, U)
        pert[(M, tag, "res")] = corr(residual_signal(P, x, t)[:, m], err[:, m])
        pert[(M, tag, "pert")] = corr(sens(P, PP)[:, m], err[:, m])

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
xb = np.arange(len(mods))
ax[0].bar(xb - 0.2, [trust_mean[(M, "id")] for M in mods], 0.4, color="#9aa0a6", label="in-distribution inputs")
ax[0].bar(xb + 0.2, [trust_mean[(M, "ood")] for M in mods], 0.4, color="#c1121f", label="OOD inputs (unfamiliar waves)")
ax[0].axhline(0.5, color="k", ls=":", lw=1)
ax[0].set_xticks(xb); ax[0].set_xticklabels(list(mods))
ax[0].set_ylabel("mean trust score (in-window)")
ax[0].set_title("Reference-free OOD detection:\ntrust collapses on unfamiliar inputs")
ax[0].legend(fontsize=9); ax[0].grid(True, axis="y", alpha=0.3)
for i, M in enumerate(mods):
    ax[0].text(i - 0.2, trust_mean[(M, "id")] + 0.02, "%.2f" % trust_mean[(M, "id")], ha="center", fontsize=9)
    ax[0].text(i + 0.2, trust_mean[(M, "ood")] + 0.02, "%.2f" % trust_mean[(M, "ood")], ha="center", fontsize=9)

groups = [("FNO", "id"), ("FNO", "ood"), ("DeepONet", "id"), ("DeepONet", "ood")]
gx = np.arange(len(groups))
ax[1].bar(gx - 0.2, [pert[(M, tg, "res")] for M, tg in groups], 0.4, color="#9aa0a6", label="physics residual")
ax[1].bar(gx + 0.2, [pert[(M, tg, "pert")] for M, tg in groups], 0.4, color="#2563eb", label="perturbation (model-unsure)")
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xticks(gx); ax[1].set_xticklabels(["FNO\nin-dist", "FNO\nOOD", "DeepONet\nin-dist", "DeepONet\nOOD"], fontsize=8)
ax[1].set_ylabel("correlation with true error")
ax[1].set_title("Perturbation signal is the OOD specialist:\nit works where the physics residual fails")
ax[1].legend(fontsize=9); ax[1].grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "ext_ood_perturbation.png"), dpi=140)
print("trust id->ood:", {M: (round(trust_mean[(M,'id')],2), round(trust_mean[(M,'ood')],2)) for M in mods})
print("saved")
