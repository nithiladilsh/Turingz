"""
End-to-end integration test: Module 1 (trust) -> Module 2 (coupling) -> Module 3 (runtime).

Proves the FULL system path, not just the coupling in isolation:
  * the REAL trust monitor (hybrid_pde.trust.monitor.TrustMonitor) decides WHEN to switch,
  * Module 3's load_coupling() supplies the coupling object (RealCoupling wrapping M2Coupling),
  * that coupling performs the physical ML -> numerical hard switch,
  * and the resulting hybrid trajectory is more accurate than the pure ML trajectory.

Runs offline from the committed evaluation predictions (numpy only; no torch/GPU).
Skips cleanly if the prediction / trust-parameter fixtures are absent.

Author: Dharmapala R.D. (214050V)
"""
import os
import sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "hybrid_pde", "coupling - 214050V"))

PRED = os.path.join(ROOT, "results", "eval", "predictions.npz")
TRUST_PARAMS = os.path.join(ROOT, "results", "trust", "trust_params_FNO.npz")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(PRED) and os.path.exists(TRUST_PARAMS)),
    reason="prediction / trust-parameter fixtures not present",
)

EPS = 1e-12
N_WAVES = 2  # small smoke; enough to prove the wired path end-to-end


def _tail_rel_l2(traj, ref, times, i0):
    """Time-integrated relative L2 error of traj vs ref over times[i0:]."""
    c = np.sqrt(((traj[i0:] - ref[i0:]) ** 2).sum(-1)) / (np.sqrt((ref[i0:] ** 2).sum(-1)) + EPS)
    return float(np.trapezoid(c, times[i0:]) / (times[-1] - times[i0] + EPS))


def test_m1_m2_m3_end_to_end_beats_pure_ml():
    from restart_spectral import solve_from, nearest_index
    from hybrid_pde.trust.monitor import TrustMonitor, load_params
    from hybrid_pde.control_214133E.integrate import load_coupling

    d = np.load(PRED)
    t = d["t"]; x = d["x"]
    true = d["u_true_eval"].astype(float)
    fno = d["FNO_eval"].astype(float)
    nt = len(t)
    i1 = nearest_index(1.0)
    params = load_params(TRUST_PARAMS)

    # Module 3 hands us the coupling object it will actually call.
    coupling = load_coupling()
    assert type(coupling).__name__ == "RealCoupling"
    assert hasattr(coupling, "rollout") and hasattr(coupling, "correct")

    class MLStub:
        """Stands in for the ML solver: returns the precomputed FNO trajectory."""
        def __init__(self, pred): self.pred = pred
        def rollout(self, ic, x, t): return self.pred

    class NumSolver:
        """Real numerical continuation: integrate from the handed-over state over local time tau."""
        def rollout(self, u0, x, tau):
            return solve_from(u0, nt - len(tau))  # i_start = switch index

    improved = 0
    for j in range(N_WAVES):
        monitor = TrustMonitor(params)  # Module 1, stateful over the sequence

        def trigger(u, tt, _m=monitor):
            r = _m.update(u, tt)
            return (r["trust"], not r["ok"])   # switch when trust says "not ok"

        # Full path: trust decides -> M3 coupling performs the hard switch.
        hybrid = np.asarray(
            coupling.rollout(fno[j], x, t, MLStub(fno[j]), NumSolver(), trigger), dtype=float
        )

        assert hybrid.shape == true[j].shape
        assert np.isfinite(hybrid).all()

        pure_err = _tail_rel_l2(fno[j], true[j], t, i1)
        hybrid_err = _tail_rel_l2(hybrid, true[j], t, i1)

        # The hybrid must never be worse than pure ML on the extrapolation window.
        assert hybrid_err <= pure_err + 1e-9
        if hybrid_err < pure_err:
            improved += 1

    # With the real (early-firing) trust monitor the hybrid should strictly help.
    assert improved >= 1


def test_m3_correct_primitive_runs():
    """Module 3's per-step re-anchor primitive (used by the adaptive controller) executes."""
    from restart_spectral import solve_from, nearest_index
    from hybrid_pde.control_214133E.integrate import load_coupling

    d = np.load(PRED)
    x = d["x"]; fno = d["FNO_eval"].astype(float)
    nt = len(d["t"])
    coupling = load_coupling()

    class NumSolver:
        def rollout(self, u0, x, tau):
            return solve_from(u0, nt - len(tau))

    i = nearest_index(1.0)
    out = coupling.correct(fno[0, i], x, 1.0, 1.01, NumSolver())
    out = np.asarray(out, dtype=float)
    assert out.shape == fno[0, i].shape
    assert np.isfinite(out).all()
