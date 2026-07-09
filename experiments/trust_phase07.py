import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from hybrid_pde.trust.monitor import fit_trust, save_params, load_params, TrustMonitor, FAIL

OUT = os.path.join(ROOT, "results", "trust")
os.makedirs(OUT, exist_ok=True)
d = np.load(os.path.join(ROOT, "results", "eval", "predictions.npz"))
x, t = d["x"], d["t"]
true, train_true = d["u_true_eval"], d["u_true_seen"]
models = {"PINN": "PINN", "FNO": "FNO_eval", "DeepONet": "DeepONet_eval"}
cols = {"PINN": "#2563eb", "FNO": "#e76f51", "DeepONet": "#2a9d8f"}

for name, key in models.items():
    p = d[key]
    err = np.linalg.norm(p - true, axis=2) / (np.linalg.norm(true, axis=2) + 1e-12)
    h = p.shape[0] // 2
    params = fit_trust(p[:h], err[:h], x, t, train_true)
    save_params(os.path.join(OUT, "trust_params_%s.npz" % name), params)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for j, (name, key) in enumerate(models.items()):
    params = load_params(os.path.join(OUT, "trust_params_%s.npz" % name))
    mon = TrustMonitor(params)
    p = d[key]
    h = p.shape[0] // 2
    wave = p[h]
    err = np.linalg.norm(wave - true[h], axis=1) / (np.linalg.norm(true[h], axis=1) + 1e-12)
    trust, switch_t = [], None
    for n in range(wave.shape[0]):
        out = mon.update(wave[n], float(t[n]))
        trust.append(out["trust"])
        if switch_t is None and not out["ok"]:
            switch_t = float(t[n])
    trust = np.array(trust)
    true_fail = t[np.argmax(err > FAIL)] if (err > FAIL).any() else t[-1]
    a = ax[j]
    a.plot(t, trust, color=cols[name], lw=2, label="trust score")
    a.plot(t, np.minimum(err / FAIL, 1.5) / 1.5, color="#999", ls="--", lw=1.5, label="true error (scaled)")
    a.axhline(0.5, color="k", ls=":", lw=0.8)
    if switch_t is not None:
        a.axvline(switch_t, color=cols[name], lw=1.5, label="switch flag fires")
    a.axvline(true_fail, color="#444", ls="--", lw=1.5, label="true failure")
    a.set_title("%s   switch at %.2f  vs true fail %.2f" %
                (name, switch_t if switch_t else t[-1], true_fail))
    a.set_xlabel("time t"); a.set_ylim(-0.05, 1.05)
    if j == 0:
        a.set_ylabel("trust  /  scaled error"); a.legend(fontsize=7, loc="center left")
    a.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "phase7_streaming_demo.png"), dpi=150)
print("saved to", OUT)

print("\n--- streaming API demo: feed one frame at a time ---")
params = load_params(os.path.join(OUT, "trust_params_PINN.npz"))
mon = TrustMonitor(params)
wave = d["PINN"][d["PINN"].shape[0] // 2]
for n in range(wave.shape[0]):
    out = mon.update(wave[n], float(t[n]))
    if n % 30 == 0:
        print("t=%.2f  trust=%.2f  ok=%s" % (t[n], out["trust"], out["ok"]))
