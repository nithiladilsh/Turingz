from __future__ import annotations
import numpy as np
from .integrate import load_ml_solver, load_trust
from .groundtruth import load_reference


def main():
    ml = load_ml_solver()
    R = load_reference()
    t = np.asarray(R.t, dtype=float)
    keyt = [0.5, 0.9, 1.0, 1.1, 1.3, 1.6]
    print("M1 trust behaviour with REAL FNO (per test IC):")
    for i in [900, 901, 902, 905]:
        u = np.asarray(ml.rollout(R.ICs[i], R.x, R.t), dtype=float)
        tr = load_trust()
        tr.reset()
        pairs = [tr(u[k], float(t[k])) for k in range(len(t))]
        ts = np.array([p[0] for p in pairs])
        fls = [p[1] for p in pairs]
        ff = next((float(t[k]) for k in range(len(t)) if fls[k]), None)
        at = {tt: float(ts[np.abs(t - tt).argmin()]) for tt in keyt}
        print(f"  IC{i}: in-window(t<=1) {ts[t<=1].mean():.3f} | extrap(t>1) {ts[t>1].mean():.3f} | "
              f"min {ts.min():.3f}@t{float(t[ts.argmin()]):.2f} | flag@t={ff}")
        print(f"        trust @ t=" + ", ".join(f"{k}:{v:.2f}" for k, v in at.items()))


if __name__ == "__main__":
    main()
