"""
Day-2 gate + preliminary hybrid sweep, offline from results/eval/predictions.npz.
Real viscosity, real Cole-Hopf truth (u_true_eval), real FNO preds (FNO_eval),
10 held-out ICs. Three trajectories per (IC, switch time):
  UB : restart spectral from TRUE state at t_s (upper bound / gate)
  H  : restart spectral from FNO state  at t_s (real hybrid)
  FNO: pure FNO baseline
Reference = u_true_eval (Cole-Hopf).
"""
import os, json
import numpy as np
from restart_spectral import solve_from, nearest_index, TGRID, NU

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRED = os.path.join(ROOT, "results", "eval", "predictions.npz")
SWITCH_TIMES = [1.0, 1.2, 1.4, 1.6, 1.8]
EPS = 1e-12

def rel_l2_curve(U, Uref):
    return np.sqrt(((U - Uref)**2).sum(-1)) / (np.sqrt((Uref**2).sum(-1)) + EPS)

def integ(curve, times):
    return np.trapz(curve, times) / (times[-1] - times[0] + EPS)

def main():
    d = np.load(PRED)
    t = d["t"]; Utrue = d["u_true_eval"].astype(np.float64); Ufno = d["FNO_eval"].astype(np.float64)
    n_ic = Utrue.shape[0]
    assert np.allclose(t, TGRID), "time grid mismatch"
    print("loaded %d held-out ICs | nu=%.6f | grid ok\n" % (n_ic, NU))
    hdr = ("t_s","idx","e_s(FNO)","UB_tail","FNO_tail","H_tail","B_int","ICs+")
    print("%4s %4s %9s %9s %9s %9s %7s %6s" % hdr)
    print("-"*62)
    gate_ok = True; rows = []
    for ts in SWITCH_TIMES:
        i = nearest_index(ts); tail_t = t[i:]; true_tail = Utrue[:, i:]
        UB = solve_from(Utrue[:, i], i)
        H  = solve_from(Ufno[:,  i], i)
        ub  = np.array([integ(rel_l2_curve(UB[j],       true_tail[j]), tail_t) for j in range(n_ic)])
        he  = np.array([integ(rel_l2_curve(H[j],        true_tail[j]), tail_t) for j in range(n_ic)])
        fe  = np.array([integ(rel_l2_curve(Ufno[j, i:], true_tail[j]), tail_t) for j in range(n_ic)])
        es  = np.linalg.norm(Ufno[:, i]-Utrue[:, i], axis=1) / (np.linalg.norm(Utrue[:, i], axis=1)+EPS)
        b   = (fe - he) / (fe + EPS)
        improved = int((he < fe).sum())
        gate_ok &= float(ub.mean()) < 1e-2
        print("%4.1f %4d %9.4f %9.2e %9.4f %9.4f %7.2f %4d/%d" %
              (ts, i, es.mean(), ub.mean(), fe.mean(), he.mean(), b.mean(), improved, n_ic))
        rows.append(dict(t_s=ts, idx=i, e_s=float(es.mean()), ub_tail=float(ub.mean()),
                         fno_tail=float(fe.mean()), h_tail=float(he.mean()),
                         b_int=float(b.mean()), ics_improved=improved, n_ic=n_ic))
    print("\nGATE (UB restart from TRUE state): %s" %
          ("PASS -- restart tracks Cole-Hopf" if gate_ok else "FAIL -- debug wrapper"))
    out = os.path.join(os.path.dirname(__file__), "day2_sweep_results.json")
    json.dump(rows, open(out, "w"), indent=2)
    print("saved", out)

if __name__ == "__main__":
    main()
