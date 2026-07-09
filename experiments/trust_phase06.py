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

START, FAIL, WIN, CUT, TW = 2, 0.10, 5, 0.5, 1.0
OUT = os.path.join(ROOT, "results", "trust")
os.makedirs(OUT, exist_ok=True)

d = np.load(os.path.join(ROOT, "results", "eval", "predictions.npz"))
x, t = d["x"], d["t"]
true, train_true = d["u_true_eval"], d["u_true_seen"]
coeff = shock_coeff(train_true, x, t)
te = t[START:]
inw, ood = te <= TW, te > TW
models = {"PINN": "PINN", "FNO": "FNO_eval", "DeepONet": "DeepONet_eval"}
cols = {"PINN": "#2563eb", "FNO": "#e76f51", "DeepONet": "#2a9d8f"}

def sig(u):
    return [corrected_signal(u, coeff, x, t), energy_signal(u, x), roughness_signal(u, x)]

def safecorr(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

rows = {}
for name, key in models.items():
    p = d[key]
    err = np.linalg.norm(p - true, axis=2) / (np.linalg.norm(true, axis=2) + 1e-12)
    S = sig(p)
    h = p.shape[0] // 2
    m = slice(START, None)
    Str = [s[:h][:, m].reshape(-1) for s in S]
    fp = fit_fusion(Str, err[:h][:, m].reshape(-1))
    fused = smooth(fuse([s[:, m] for s in S], fp), WIN)
    fail = (err[:, m] > FAIL).astype(float)
    cal = fit_calibration(fused[:h], fail[:h])
    tr = np.array([trust_score(f, cal) for f in fused])
    E, W = err[h:, m], 1.0 - tr[h:]
    flag = W > (1 - CUT)
    fine_in = E[:, inw] <= FAIL
    bad_ood = E[:, ood] > FAIL
    rows[name] = dict(
        err_in=E[:, inw].mean(), err_ood=E[:, ood].mean(),
        warn_in=W[:, inw].mean(), warn_ood=W[:, ood].mean(),
        false_alarm=(flag[:, inw] & fine_in).sum() / max(fine_in.sum(), 1),
        detect=(flag[:, ood] & bad_ood).sum() / max(bad_ood.sum(), 1),
        corr=safecorr(W.ravel(), E.ravel()))
    r = rows[name]
    print("%-9s err %.3f->%.3f  warn %.2f->%.2f  FA %.0f%%  detect %.0f%%  corr %.2f" %
          (name, r["err_in"], r["err_ood"], r["warn_in"], r["warn_ood"],
           r["false_alarm"] * 100, r["detect"] * 100, r["corr"]))

names = list(models)
xb = np.arange(len(names))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].bar(xb - 0.2, [rows[n]["err_in"] for n in names], 0.4, color="#9aa0a6", label="in-window  (t<=1, trained)")
ax[0].bar(xb + 0.2, [rows[n]["err_ood"] for n in names], 0.4, color="#c1121f", label="extrapolation  (t>1, OOD)")
ax[0].axhline(FAIL, color="k", ls=":", lw=1)
ax[0].set_xticks(xb); ax[0].set_xticklabels(names)
ax[0].set_ylabel("true relative error")
ax[0].set_title("The models are accurate in-distribution\nand fail out-of-distribution (past t=1)")
ax[0].legend(fontsize=8); ax[0].grid(True, axis="y", alpha=0.3)

ax[1].bar(xb - 0.2, [rows[n]["warn_in"] for n in names], 0.4, color="#9aa0a6", label="in-window  (t<=1)")
ax[1].bar(xb + 0.2, [rows[n]["warn_ood"] for n in names], 0.4, color="#c1121f", label="extrapolation  (t>1, OOD)")
ax[1].axhline(1 - CUT, color="k", ls=":", lw=1)
ax[1].set_xticks(xb); ax[1].set_xticklabels(names)
ax[1].set_ylabel("trust module warning level (1 - trust)")
ax[1].set_title("Reference-free warning rises in step:\nlow in-distribution, high out-of-distribution")
ax[1].legend(fontsize=8); ax[1].grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "phase6_in_vs_ood.png"), dpi=150)

fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.bar(xb - 0.2, [rows[n]["detect"] * 100 for n in names], 0.4, color="#2a9d8f",
       label="OOD failures caught (higher is better)")
ax.bar(xb + 0.2, [rows[n]["false_alarm"] * 100 for n in names], 0.4, color="#e9c46a",
       label="in-distribution false alarms (lower is better)")
ax.set_xticks(xb); ax.set_xticklabels(names)
ax.set_ylabel("percent of frames")
ax.set_title("Phase 6 scorecard: reference-free failure detection\ncatches most OOD failures with few false alarms")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
for i, n in enumerate(names):
    ax.text(i - 0.2, rows[n]["detect"] * 100 + 1, "%.0f%%" % (rows[n]["detect"] * 100), ha="center", fontsize=9)
    ax.text(i + 0.2, rows[n]["false_alarm"] * 100 + 1, "%.0f%%" % (rows[n]["false_alarm"] * 100), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "phase6_scorecard.png"), dpi=150)
print("saved to", OUT)
