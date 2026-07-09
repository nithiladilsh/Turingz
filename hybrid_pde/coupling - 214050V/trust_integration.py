"""
Trust-triggered hybrid: integration scaffold (Module 2, Coupling).

This wires the trust-detection module (teammate) to the coupling module (this repo).
It runs offline from predictions (numpy only). Set MODULE2_PRED=results/eval/
predictions_ext.npz for n=20.

------------------------------------------------------------------------------
DECISION CONTRACT (agree these field names with the trust module before wiring)
------------------------------------------------------------------------------
For each held-out wave the trust module is expected to provide, per time index,
a decision of the form:

    {
      "trust_score":  float,   # higher = more reliable
      "is_reliable":  bool,    # trust_score >= threshold
      "should_switch": bool,   # True  => hand over to the numerical solver NOW
      "time_index":   int,     # grid index of this decision
      "time":         float,   # physical time of this decision
    }

The coupling module switches at the FIRST index where should_switch is True.
NOTE on semantics: is_reliable=True means "keep using ML"; should_switch is its
negation once the score drops below threshold. Do not conflate the two.

In this demonstration the real (reference-free) trust signal is not present, so a
STAND-IN oracle trigger is used: should_switch fires at the first time the FNO's
true error exceeds a threshold. This is a placeholder for the teammate's signal
and is clearly labelled as such; the real trust output plugs into first_switch().
"""
import os, json
import numpy as np
from restart_spectral import solve_from, nearest_index, TGRID

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRED = os.environ.get("MODULE2_PRED", os.path.join(ROOT, "results", "eval", "predictions.npz"))
EPS = 1e-12
FIXED_TS = 1.4                 # fixed-switch baseline
TRUST_THRESH = 0.10            # stand-in: switch when (oracle) reliability drops below this
VIAB_BOUNDARY = 1.47          # measured viability boundary (Section 6d)
EVAL_LO = 1.0                  # compare all methods on the extrapolation window [1, 2]


def rl2_curve(U, Uref):
    return np.sqrt(((U - Uref) ** 2).sum(-1)) / (np.sqrt((Uref ** 2).sum(-1)) + EPS)

def integ(curve, times):
    return float(np.trapezoid(curve, times) / (times[-1] - times[0] + EPS))

# ---- the coupling side of the contract ----
def first_switch(should_switch):
    """Return the first index where should_switch is True, else None (stay on ML)."""
    idx = np.where(np.asarray(should_switch))[0]
    return int(idx[0]) if idx.size else None

def trust_triggered_hybrid(fno_pred, switch_index):
    """Build the hybrid trajectory given the ML prediction and a switch index."""
    if switch_index is None:
        return fno_pred.copy()
    tail = solve_from(fno_pred[switch_index], switch_index)
    return np.concatenate([fno_pred[:switch_index], tail], axis=0)


def main():
    d = np.load(PRED)
    t = d["t"]; true = d["u_true_eval"].astype(float); fno = d["FNO_eval"].astype(float)
    n, nt, _ = true.shape
    i1 = nearest_index(EVAL_LO); win = t[i1:]
    i_fix = nearest_index(FIXED_TS)
    print(f"trust-integration demo | n={n} waves | window [{EVAL_LO},2] | "
          f"stand-in oracle trigger at true-error>{TRUST_THRESH:.0%}\n")

    Hfix = solve_from(fno[:, i_fix], i_fix)          # fixed switch, batched
    rows = []
    for j in range(n):
        # STAND-IN trust: should_switch fires when the oracle reliability drops
        err_t = rl2_curve(fno[j], true[j])            # (nt,) true error over time
        should = err_t > TRUST_THRESH
        i_trig = first_switch(should)
        t_trig = float(t[i_trig]) if i_trig is not None else float('nan')

        pure = integ(rl2_curve(fno[j, i1:], true[j, i1:]), win)
        fx = np.concatenate([fno[j, :i_fix], Hfix[j]], axis=0)
        fixed = integ(rl2_curve(fx[i1:], true[j, i1:]), win)
        tr = trust_triggered_hybrid(fno[j], i_trig)
        trust = integ(rl2_curve(tr[i1:], true[j, i1:]), win)
        wl = (nt - i_trig) / nt if i_trig is not None else 0.0     # numerical-workload fraction
        rows.append(dict(ic=j, trigger_time=t_trig,
                         inside_viability=bool(i_trig is not None and t_trig <= VIAB_BOUNDARY),
                         pure_fno=pure, fixed_1p4=fixed, trust_triggered=trust,
                         numerical_workload_frac=wl))

    def mean(k): return float(np.nanmean([r[k] for r in rows]))
    print("%-18s %10s" % ("method (mean over waves)", "err[1,2]"))
    print("-"*30)
    print("%-18s %10.4f" % ("Pure FNO", mean("pure_fno")))
    print("%-18s %10.4f" % ("Fixed switch @1.4", mean("fixed_1p4")))
    print("%-18s %10.4f" % ("Trust-triggered", mean("trust_triggered")))
    print("%-18s %10s" % ("Full numerical", "~1e-3 (reference; 100% numerical work)"))
    print("\nmean trigger time: %.3f | mean numerical-workload fraction: %.2f | triggers inside viability window: %d/%d"
          % (mean("trigger_time"), mean("numerical_workload_frac"),
             sum(r["inside_viability"] for r in rows), n))

    out = os.path.join(ROOT, "results", "module2", "figures", "trust_integration_results.json")
    json.dump({"note": "stand-in oracle trigger; real trust output plugs into first_switch()",
               "n": n, "fixed_switch_time": FIXED_TS, "trust_threshold": TRUST_THRESH,
               "viability_boundary": VIAB_BOUNDARY, "eval_window": [EVAL_LO, 2.0],
               "means": {k: mean(k) for k in ["pure_fno","fixed_1p4","trust_triggered",
                         "trigger_time","numerical_workload_frac"]},
               "per_ic": rows}, open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
