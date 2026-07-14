from __future__ import annotations
import numpy as np
from .integrate import load_ml_solver, spectral_rollout
from .groundtruth import load_reference, relative_l2


def main():
    ml = load_ml_solver()
    R = load_reference()
    x, t = R.x, R.t
    t = np.asarray(t, dtype=float)
    print("Coarse-drift detection on REAL FNO (reference = spectral solve):")
    print(f"{'IC':>5}{'true_fail_t':>13}{'detector_t':>12}{'corr(div,err)':>15}")
    for i in [900, 901, 902, 905]:
        ic = R.ICs[i]
        u_fno = np.asarray(ml.rollout(ic, x, t), dtype=float)
        u_true = np.asarray(R.u[i], dtype=float)
        u_ref = np.asarray(spectral_rollout(ic, x, t), dtype=float)     # numerical reference
        true_err = np.linalg.norm(u_fno - u_true, axis=1) / (np.linalg.norm(u_true, axis=1) + 1e-12)
        det = np.linalg.norm(u_fno - u_ref, axis=1) / (np.linalg.norm(u_ref, axis=1) + 1e-12)
        tfail = next((float(t[k]) for k in range(len(t)) if true_err[k] > 0.10), None)
        tdet = next((float(t[k]) for k in range(len(t)) if det[k] > 0.10), None)
        m = t > 0.1
        corr = float(np.corrcoef(det[m], true_err[m])[0, 1])
        print(f"{i:>5}{str(round(tfail,2) if tfail else None):>13}{str(round(tdet,2) if tdet else None):>12}{corr:>15.3f}")
    print("\nGood result: detector_t is close to true_fail_t, and corr is high (>0.9).")


if __name__ == "__main__":
    main()
