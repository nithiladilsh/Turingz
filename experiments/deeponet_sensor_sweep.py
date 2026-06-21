import os, sys, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from hybrid_pde.common import load, split, evaluate
from hybrid_pde.solvers.ml.deepOnet.deeponet import DeepONetDDE

SENSORS = [64, 100, 128, 256]
SEEDS = [0, 1, 2]
ITERS = 4000
OUT = os.path.join(ROOT, "results", "deeponet", "sensor_sweep.json")

U, ICs, x, t, te, Tmax = load()
train_idx, val_idx, _ = split(U.shape[0])
ds = {"u": U, "ICs": ICs, "x": x, "t": t, "t_train_end": te, "T": Tmax,
      "train_idx": train_idx, "val_idx": val_idx}

summary = []
for m in SENSORS:
    vals = []
    for sd in SEEDS:
        solver = DeepONetDDE(m=m, iterations=ITERS, seed=sd)
        solver.fit(ds)
        vals.append(evaluate(solver, U, ICs, x, t, te, val_idx)["in_dist_mean"])
    vals = np.array(vals)
    summary.append({"n_sensors": m, "val_in_dist_mean": float(vals.mean()),
                    "val_in_dist_std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0)})
    print(f"m={m:3d}  val_in_dist={vals.mean():.4f}±{vals.std(ddof=1):.4f}")

means = np.array([s["val_in_dist_mean"] for s in summary])
best = summary[int(means.argmin())]
thr = best["val_in_dist_mean"] + best["val_in_dist_std"]
rec = next(s["n_sensors"] for s in sorted(summary, key=lambda s: s["n_sensors"])
           if s["val_in_dist_mean"] <= thr)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"sensors": SENSORS, "seeds": SEEDS, "iterations": ITERS, "per_m": summary,
           "best_n_sensors": best["n_sensors"], "recommended_n_sensors": rec,
           "rule": "smallest sensor count within one std of the best validation error"},
          open(OUT, "w"), indent=2)
print(f"\nrecommended n_sensors = {rec}  (saved to {OUT})")
