import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from hybrid_pde.trust.coarse_reference import CoarseReferenceMonitor

FAIL = 0.10
OUT = os.path.join(ROOT, "results", "trust")
os.makedirs(OUT, exist_ok=True)
d = np.load(os.path.join(ROOT, "results", "eval", "predictions_extended.npz"))
x, t = d["x"], d["t"]
P, U, ICs = d["FNO_id"], d["u_true_id"], d["ics_id"]
END = float(t[-1])

def run_wave(i):
    mon = CoarseReferenceMonitor(ICs[i], x, n=256)
    trust, sw = [], END
    for n in range(P.shape[1]):
        out = mon.update(P[i, n], float(t[n]))
        trust.append(out["trust"])
        if sw == END and not out["ok"]:
            sw = float(t[n])
    return np.array(trust), sw

sw, tf = [], []
for i in range(50, 100):
    tr, s = run_wave(i)
    e = np.linalg.norm(P[i] - U[i], axis=1) / (np.linalg.norm(U[i], axis=1) + 1e-12)
    sw.append(s)
    tf.append(float(t[np.argmax(e > FAIL)]) if (e > FAIL).any() else END)
sw, tf = np.array(sw), np.array(tf)
near = int((np.abs(sw - tf) <= 0.20).sum())
late = int((sw > tf + 0.10).sum())
print("FNO via coarse-reference (50 test waves): mean|switch-true|=%.2f  near=%d/50  late=%d" %
      (np.abs(sw - tf).mean(), near, late))

fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
# demo trust curve for one wave
i = 55
tr, s = run_wave(i)
e = np.linalg.norm(P[i] - U[i], axis=1) / (np.linalg.norm(U[i], axis=1) + 1e-12)
tfi = t[np.argmax(e > FAIL)] if (e > FAIL).any() else END
ax[0].plot(t, tr, color="#e76f51", lw=2, label="trust (coarse-reference)")
ax[0].plot(t, np.minimum(e / FAIL, 1.5) / 1.5, color="#999", ls="--", lw=1.5, label="true error (scaled)")
ax[0].axhline(0.5, color="k", ls=":", lw=0.8)
ax[0].axvline(s, color="#e76f51", lw=1.5, label="switch fires")
ax[0].axvline(tfi, color="#444", ls="--", lw=1.5, label="true failure")
ax[0].set_title("FNO with coarse-reference monitor\nswitch %.2f  vs true failure %.2f" % (s, tfi))
ax[0].set_xlabel("time t"); ax[0].set_ylim(-0.05, 1.05); ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)

ax[1].fill_between([0, 2.05], [0, 2.05], [2.05, 2.05], color="#f8d7da", zorder=0)
ax[1].text(1.25, 1.93, "switched late\n(unsafe)", color="#a01722", fontsize=9, ha="center", va="top")
ax[1].scatter(tf, sw, color="#e76f51", s=45, edgecolor="w", zorder=3)
ax[1].plot([0, 2.05], [0, 2.05], "k--", lw=1, label="perfect")
ax[1].set_xlim(0, 2.05); ax[1].set_ylim(0, 2.05)
ax[1].set_xlabel("true failure time"); ax[1].set_ylabel("switch time (coarse-reference)")
ax[1].set_title("FNO switches AT its true error, all 50 waves\nmean gap = %.2f    (was ~1.3 reference-free)" % np.abs(sw - tf).mean())
ax[1].legend(fontsize=9); ax[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "coarse_reference_fno.png"), dpi=140)
print("saved")
