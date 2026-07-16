"""Verify an M2 coupling object satisfies the Coupling contract before wiring it in.
Run:  python -m hybrid_pde.control_214133E._check_m2
Or:   from hybrid_pde.control_214133E._check_m2 import check_coupling; check_coupling(m2_obj)
"""
from __future__ import annotations
import numpy as np
from .contracts import Coupling
from .runtime import HybridRuntime
from .controller import AdaptiveController
from .trigger import SyntheticTrust
from ._smoke import MLDrift, NumExact, HORIZON


def _check(cond, msg, report):
    report.append(("PASS" if cond else "FAIL", msg))
    return bool(cond)


def check_coupling(coupling, name="M2 coupling", verbose=True):
    report = []
    ok = True

    ok &= _check(hasattr(coupling, "correct") and callable(getattr(coupling, "correct", None)),
                 "has callable correct(state, x, t0, t1, num)", report)
    _check(hasattr(coupling, "rollout") and callable(getattr(coupling, "rollout", None)),
           "has callable rollout(ic, x, t, ml, num, trigger)  [protocol asks for it; runtime does not call it]", report)

    nx = 64
    x = np.linspace(-1.0, 1.0, nx)
    state = np.sin(np.pi * x)
    if hasattr(coupling, "correct"):
        try:
            out = np.asarray(coupling.correct(state, x, 1.0, 1.02, NumExact()), dtype=float)
            ok &= _check(out.shape == (nx,), f"correct() returns a state of shape (nx,)  [got {out.shape}, expected {(nx,)}]", report)
            ok &= _check(np.all(np.isfinite(out)), "correct() output is all finite (no NaN/Inf)", report)
            ok &= _check(float(np.abs(out).max()) < 1e3, "correct() output magnitude is sane (< 1e3)", report)
        except Exception as e:
            ok &= _check(False, f"correct() raised {type(e).__name__}: {e}", report)

    try:
        nx2, nt = 128, 60
        x2 = np.linspace(-1.0, 1.0, nx2)
        t = np.linspace(0.0, 2.0, nt)
        ic = np.sin(np.pi * x2)
        truth = ic[None, :] * np.exp(-t)[:, None]
        rt = HybridRuntime(MLDrift(), NumExact(),
                           SyntheticTrust(horizon=HORIZON, width=0.05, flag_at=0.5),
                           coupling, AdaptiveController(0.4, 0.6))
        res = rt.run(ic, x2, t, 0.05, reference=truth)
        u = np.asarray(res.u, dtype=float)
        ok &= _check(u.shape == (nt, nx2), f"end-to-end run() output shape (nt, nx)  [got {u.shape}]", report)
        ok &= _check(np.all(np.isfinite(u)), "end-to-end output is all finite", report)
        ok &= _check(res.cost.ml_steps + res.cost.correction_steps == nt,
                     f"cost accounting sums to nt  [{res.cost.ml_steps}+{res.cost.correction_steps} vs {nt}]", report)
        ok &= _check(res.cost.correction_steps > 0, "coupling actually got invoked (correction_steps > 0)", report)
    except Exception as e:
        ok &= _check(False, f"end-to-end run raised {type(e).__name__}: {e}", report)

    _check(isinstance(coupling, Coupling), "matches the Coupling protocol (method names present)", report)

    if verbose:
        print(f"\nM2 conformance check -- {name}")
        print("-" * 62)
        for status, msg in report:
            print(f"  [{status}] {msg}")
        print("-" * 62)
        print("RESULT:", "PASS -- safe to wrap in RealCoupling and run the frontier"
              if ok else "FAIL -- fix the items above before integrating")
    return ok


def _broken():
    class BrokenCoupling:
        def correct(self, state, x, t0, t1, num):
            return np.asarray(state, float)[:10]
        def rollout(self, ic, x, t, ml, num, trigger):
            return np.asarray(ml.rollout(ic, x, t), float)
    return BrokenCoupling()


def main():
    from .coupling import CouplingStub
    print("Self-test 1: known-good CouplingStub (expect PASS)")
    good = check_coupling(CouplingStub(), name="CouplingStub (reference)")
    print("\nSelf-test 2: deliberately broken coupling (expect FAIL)")
    bad = check_coupling(_broken(), name="BrokenCoupling (returns wrong shape)")
    print("\nChecker discriminates good from bad:", good and not bad)

    try:
        from .integrate import load_coupling
        m2 = load_coupling()
        print("\nReal M2 detected -- checking it:")
        check_coupling(m2, name="real M2 (from load_coupling)")
    except NotImplementedError:
        print("\nReal M2 not wired yet (load_coupling still raises NotImplementedError).")
        print("Once M2 is ready, re-run:  python -m hybrid_pde.control_214133E._check_m2")
    except Exception as e:
        print(f"\nCould not load real M2: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
