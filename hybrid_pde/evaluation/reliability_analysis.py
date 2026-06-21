import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESHOLD = 0.10
MODELS = ["PINN", "FNO", "DeepONet"]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "eval")

data = np.load(os.path.join(OUT, "predictions.npz"))
x, t, te = data["x"], data["t"], float(data["te"])

def curve(pred, true):
    return (np.linalg.norm(pred - true, axis=2) / (np.linalg.norm(true, axis=2) + 1e-12)).mean(0)

def summarize(c):
    over = np.where(c > THRESHOLD)[0]
    return {"in_window": float(c[t <= te].mean()),
            "extrapolation": float(c[t > te].mean()),
            "reliable_horizon": float(t[over[0]]) if len(over) else float(t[-1])}

# ---------- Experiment 1: fair 3-way comparison on operator-unseen ICs ----------
ut = data["u_true_eval"]
keymap = {"PINN": "PINN", "FNO": "FNO_eval", "DeepONet": "DeepONet_eval"}
eval_pred = {m: data[k] for m, k in keymap.items() if k in data.files}
curves = {m: curve(p, ut) for m, p in eval_pred.items()}
summary = {m: summarize(c) for m, c in curves.items()}

print(f"\nRELIABILITY (10 operator-unseen ICs, all models)")
print(f"{'model':10s}{'in-window':>12}{'extrapolation':>16}{'reliable t':>13}")
for m in eval_pred:
    s = summary[m]
    print(f"{m:10s}{s['in_window']*100:11.2f}%{s['extrapolation']*100:15.2f}%{s['reliable_horizon']:12.2f}")

plt.figure(figsize=(8, 5))
for m in eval_pred:
    plt.plot(t, curves[m] * 100, lw=2, label=f"{m} (reliable to t={summary[m]['reliable_horizon']:.2f})")
plt.axvline(te, color="k", ls="--", lw=1, label="train / extrapolation split")
plt.axhline(THRESHOLD * 100, color="gray", ls=":", lw=1)
plt.xlabel("time t"); plt.ylabel("relative error vs Cole-Hopf (%)")
plt.title("Accuracy over time (10 unseen initial conditions)")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "1_error_over_time.png"), dpi=150); plt.close()

plt.figure(figsize=(7, 5))
xb = np.arange(len(eval_pred))
plt.bar(xb - 0.2, [summary[m]["in_window"] * 100 for m in eval_pred], 0.4, label="in-window (t<=1)")
plt.bar(xb + 0.2, [summary[m]["extrapolation"] * 100 for m in eval_pred], 0.4, label="extrapolation (t>1)")
plt.xticks(xb, list(eval_pred)); plt.ylabel("relative error (%)")
plt.title("In-window vs extrapolation error"); plt.legend(); plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "2_in_vs_extrap.png"), dpi=150); plt.close()

snaps = [0.5, 1.0, 1.5, 2.0]
fig, ax = plt.subplots(1, len(snaps), figsize=(16, 4))
for j, tt in enumerate(snaps):
    k = int(np.argmin(np.abs(t - tt)))
    ax[j].plot(x, ut[0, k], "k", lw=2.5, label="true")
    for m in eval_pred:
        ax[j].plot(x, eval_pred[m][0, k], lw=1.5, label=m)
    ax[j].set_title(f"t = {t[k]:.2f}" + ("  (in-window)" if t[k] <= te else "  (extrapolation)"))
    ax[j].set_xlabel("x"); ax[j].grid(True, alpha=0.3)
ax[0].set_ylabel("u"); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "3_snapshots.png"), dpi=150); plt.close()

# ---------- Experiment 2: operator generalization (seen vs unseen ICs) ----------
us = data["u_true_seen"]
gen = {}
plt.figure(figsize=(8, 5))
for m in ["FNO", "DeepONet"]:
    if f"{m}_seen" in data.files and f"{m}_eval" in data.files:
        cs, ce = curve(data[f"{m}_seen"], us), curve(data[f"{m}_eval"], ut)
        gen[m] = {"seen": summarize(cs), "unseen": summarize(ce)}
        plt.plot(t, cs * 100, lw=2, label=f"{m} seen ICs")
        plt.plot(t, ce * 100, lw=2, ls="--", label=f"{m} unseen ICs")
plt.axvline(te, color="k", ls="--", lw=1)
plt.axhline(THRESHOLD * 100, color="gray", ls=":", lw=1)
plt.xlabel("time t"); plt.ylabel("relative error vs Cole-Hopf (%)")
plt.title("Operator generalization: seen vs unseen initial conditions")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "4_generalization.png"), dpi=150); plt.close()

json.dump({"reliability_unseen": summary, "generalization": gen},
          open(os.path.join(OUT, "reliability_summary.json"), "w"), indent=2)
print(f"\nSaved 4 plots and reliability_summary.json to {OUT}")
