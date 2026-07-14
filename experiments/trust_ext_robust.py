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
from hybrid_pde.trust.monitor import TrustMonitor, load_params, FAIL

START, WIN = 2, 5
OUT = os.path.join(ROOT, "results", "trust")
d = np.load(os.path.join(ROOT, "results", "eval", "predictions_extended.npz"))
x, t = d["x"], d["t"]
END = float(t[-1]); MARGIN = 0.10; m = slice(START, None)
mods = {"FNO": "#e76f51", "DeepONet": "#2a9d8f"}

def relerr(p, u):
    return np.linalg.norm(p - u, axis=2) / (np.linalg.norm(u, axis=2) + 1e-12)
def sig(u, coeff):
    return [corrected_signal(u, coeff, x, t), energy_signal(u, x), roughness_signal(u, x)]
def auc(s, l):
    pos, neg = s[l == 1], s[l == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty_like(o, float); r[o] = np.arange(1, len(a) + 1)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def roc(s, l):
    th = np.unique(s)[::-1]; P, N = (l == 1).sum(), (l == 0).sum(); tp, fp = [0.0], [0.0]
    for c in th:
        pr = s >= c; tp.append((pr & (l == 1)).sum() / P); fp.append((pr & (l == 0)).sum() / N)
    return np.array(fp), np.array(tp)
def stime(pr, w):
    mon = TrustMonitor(pr)
    for n in range(w.shape[0]):
        if not mon.update(w[n], float(t[n]))["ok"]:
            return float(t[n])
    return END

warn_all, lab_all, trust_all = [], [], []
sw_all, tf_all, swc = [], [], []
aucs = {}
for M in mods:
    P = d["%s_id" % M]; U = d["u_true_id"]; err = relerr(P, U); h = 50
    coeff = shock_coeff(U[:h], x, t)
    fp = fit_fusion([s[:h][:, m].reshape(-1) for s in sig(P, coeff)], err[:h][:, m].reshape(-1))
    fused = smooth(fuse([s[:, m] for s in sig(P, coeff)], fp), WIN)
    cal = fit_calibration(fused[:h], (err[:, m][:h] > FAIL).astype(float))
    trm = np.array([trust_score(f, cal) for f in fused])
    warn = (1 - trm[h:]).ravel(); lab = (err[h:, m] > FAIL).astype(int).ravel()
    aucs[M] = auc(warn, lab)
    warn_all.append(warn); lab_all.append(lab); trust_all.append(trm[h:].ravel())
    pr = load_params(os.path.join(OUT, "trust_params_%s.npz" % M))
    for i in range(h, 100):
        e = err[i]; sw_all.append(stime(pr, P[i]))
        tf_all.append(float(t[np.argmax(e > FAIL)]) if (e > FAIL).any() else END); swc.append(M)
warn_all = np.concatenate(warn_all); lab_all = np.concatenate(lab_all); trust_all = np.concatenate(trust_all)
sw_all = np.array(sw_all); tf_all = np.array(tf_all)

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
# ROC
f, tp = roc(warn_all, lab_all)
ax[0].plot(f, tp, "k", lw=2.5, label="Combined (AUC %.2f)" % auc(warn_all, lab_all))
fw, lw2 = (1 - np.array([])), None
Pf = d["FNO_id"]; Uf = d["u_true_id"]; ef = relerr(Pf, Uf); h = 50
cf = shock_coeff(Uf[:h], x, t)
fpp = fit_fusion([s[:h][:, m].reshape(-1) for s in sig(Pf, cf)], ef[:h][:, m].reshape(-1))
fu = smooth(fuse([s[:, m] for s in sig(Pf, cf)], fpp), WIN)
ca = fit_calibration(fu[:h], (ef[:, m][:h] > FAIL).astype(float))
trf = np.array([trust_score(v, ca) for v in fu])
wf = (1 - trf[h:]).ravel(); lf = (ef[h:, m] > FAIL).astype(int).ravel()
ff, tf2 = roc(wf, lf)
ax[0].plot(ff, tf2, color="#e76f51", lw=2, label="FNO (AUC %.2f)" % auc(wf, lf))
ax[0].plot([0, 1], [0, 1], "k:", lw=1, label="random")
ax[0].set_xlabel("false alarm rate"); ax[0].set_ylabel("failures caught")
ax[0].set_title("ROC on 50 test waves per model\n(higher AUC = better separation)")
ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(True, alpha=0.3)
# reliability
bins = np.linspace(0, 1, 11); mids, obs, cnt = [], [], []
for i in range(10):
    sel = (trust_all >= bins[i]) & ((trust_all < bins[i + 1]) if i < 9 else (trust_all <= bins[i + 1]))
    if sel.sum() == 0:
        continue
    mids.append(trust_all[sel].mean()); obs.append((lab_all[sel] == 0).mean()); cnt.append(int(sel.sum()))
ax[1].plot([0, 1], [0, 1], "k:", lw=1.2, label="perfectly honest")
ax[1].plot(mids, obs, "o-", color="#6a1b9a", lw=2, ms=7, label="our trust score")
ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
ax[1].set_xlabel("trust score reported"); ax[1].set_ylabel("fraction actually fine")
ax[1].set_title("Reliability on ~20,000 test frames\n(points near line = honest score)")
ax[1].legend(fontsize=9, loc="upper left"); ax[1].grid(True, alpha=0.3)
# safety
miss = int(((sw_all >= END - 1e-9) & (tf_all < END - 1e-9)).sum())
late = int(((sw_all > tf_all + MARGIN) & (sw_all < END - 1e-9)).sum())
safe = len(sw_all) - late - miss
ax[2].fill_between([0, 2.05], [0, 2.05], [2.05, 2.05], color="#f8d7da", zorder=0)
ax[2].text(1.25, 1.93, "kept trusting a\nfailed model (unsafe)", color="#a01722", fontsize=8.5, ha="center", va="top")
ax[2].text(1.7, 0.10, "switched early\n(safe)", color="#1b5e20", fontsize=8.5, ha="center")
for M in mods:
    idx = [i for i, c in enumerate(swc) if c == M]
    ax[2].scatter(tf_all[idx], sw_all[idx], color=mods[M], s=32, edgecolor="w", lw=0.4, zorder=3, label=M)
ax[2].plot([0, 2.05], [0, 2.05], "k--", lw=1)
ax[2].set_xlim(0, 2.05); ax[2].set_ylim(0, 2.05)
ax[2].set_xlabel("true failure time"); ax[2].set_ylabel("actual switch time (streaming)")
ax[2].set_title("Safety on 100 test waves\nsafe/early %d   late %d   missed %d" % (safe, late, miss))
ax[2].legend(fontsize=8, loc="upper left"); ax[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "ext_robust.png"), dpi=140)
print("AUC:", {k: round(v, 3) for k, v in aucs.items()}, " combined", round(auc(warn_all, lab_all), 3))
print("safety: safe/early=%d late=%d miss=%d of 100" % (safe, late, miss))
print("saved")
