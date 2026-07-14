import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from hybrid_pde.trust.monitor import TrustMonitor, load_params, FAIL

MARGIN = 0.10
OUT = os.path.join(ROOT, "results", "trust")
d = np.load(os.path.join(ROOT, "results", "eval", "predictions.npz"))
x, t = d["x"], d["t"]
true = d["u_true_eval"]
END = float(t[-1])
models = {"PINN": "PINN", "FNO": "FNO_eval", "DeepONet": "DeepONet_eval"}
cols = {"PINN": "#2563eb", "FNO": "#e76f51", "DeepONet": "#2a9d8f"}

def switch_time(params, wave):
    mon = TrustMonitor(params)
    for n in range(wave.shape[0]):
        if not mon.update(wave[n], float(t[n]))["ok"]:
            return float(t[n])
    return END

def collect(setting):
    P, T, mods = [], [], []
    for name, key in models.items():
        pr = load_params(os.path.join(OUT, "trust_params_%s.npz" % name))
        if setting == "before":
            pr = dict(pr); pr["CUT"] = 0.5; pr["K"] = 6
        p = d[key]
        h = p.shape[0] // 2
        for i in range(h, p.shape[0]):
            err = np.linalg.norm(p[i] - true[i], axis=1) / (np.linalg.norm(true[i], axis=1) + 1e-12)
            P.append(switch_time(pr, p[i]))
            T.append(float(t[np.argmax(err > FAIL)]) if (err > FAIL).any() else END)
            mods.append(name)
    return np.array(P), np.array(T), mods

def stats(P, T):
    miss = int(((P >= END - 1e-9) & (T < END - 1e-9)).sum())
    late = int(((P > T + MARGIN) & (P < END - 1e-9)).sum())
    safe = len(P) - miss - late
    return safe, late, miss

fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))
for j, (setting, ttl) in enumerate([("before", "Before: accuracy-tuned (cutoff 0.5)"),
                                     ("after", "After: safety-tuned (per-model cutoff)")]):
    P, T, mods = collect(setting)
    safe, late, miss = stats(P, T)
    a = ax[j]
    a.fill_between([0, 2.05], [0, 2.05], [2.05, 2.05], color="#f8d7da", zorder=0)
    a.text(1.28, 1.93, "kept trusting a\nfailed model (unsafe)", color="#a01722", fontsize=8.5, ha="center", va="top")
    a.text(1.72, 0.10, "switched early\n(safe)", color="#1b5e20", fontsize=8.5, ha="center")
    for n in models:
        idx = [i for i, mm in enumerate(mods) if mm == n]
        a.scatter(T[idx], P[idx], color=cols[n], s=60, edgecolor="w", zorder=3, label=n)
    a.plot([0, 2.05], [0, 2.05], "k--", lw=1)
    a.set_xlim(0, 2.05); a.set_ylim(0, 2.05)
    a.set_xlabel("true failure time")
    a.set_ylabel("actual switch time (streaming)")
    a.set_title("%s\nsafe/early %d   |   never switched %d" % (ttl, safe, miss))
    a.legend(fontsize=8, loc="upper left"); a.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "phase8_safety.png"), dpi=150)

for setting in ["before", "after"]:
    P, T, mods = collect(setting)
    safe, late, miss = stats(P, T)
    expo = np.maximum(0.0, np.minimum(P, END) - T)
    print("%-7s safe/early=%2d  late>%.2f=%d  never-switched=%d   avg unsafe exposure=%.2f" %
          (setting, safe, MARGIN, late, miss, expo.mean()))
print("saved to", OUT)
