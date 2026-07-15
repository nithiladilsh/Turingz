"""
Trust-trigger tuning sweep (Module 2 / integration diagnostic).
Author: Dharmapala R.D. (214050V)

QUESTION (leadership / cross-module):
    Where should the Module 1 trust trigger fire so the hybrid is accurate but NOT
    almost fully numerical?  The current calibration fires very early (mean t ~ 0.21),
    giving high accuracy at ~90% numerical work.  My Module 2 viability boundary
    (t_s ~ 1.47, aligned with the FNO reliable horizon 1.457) is the target the trigger
    should aim for.  This script sweeps the trust threshold CUT and debounce K and
    reports, for each setting: mean trigger time, hybrid tail error, pure-FNO tail error,
    benefit, and numerical-work fraction.

WHAT IT DOES NOT DO:
    It does NOT change the coupling.  It only asks where the switch should be triggered.

DATA DISCIPLINE:
    Threshold selection should use VALIDATION predictions; final numbers are reported on
    the TEST set.  Set MODULE2_VAL_PRED to a validation predictions .npz to select the
    best config on validation and then evaluate it on test.  If no validation file is
    given, the script runs in clearly-labelled DIAGNOSTIC mode on the test set only
    (illustrative, not for final claims).

    Full-fidelity runs need validation predictions regenerated with torch/neuralop on a
    GPU machine (see extend_predictions.py); the offline path is numpy-only.

OUTPUTS:
    results/module2/figures/trust_tuning_sweep.json
    results/module2/figures/trust_tuning_sweep.png   (trigger time vs error vs work)
"""
import os, sys, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from restart_spectral import solve_from, nearest_index, TGRID
from hybrid_pde.trust.monitor import TrustMonitor, load_params

EPS = 1e-12
TEST_PRED = os.environ.get("MODULE2_TEST_PRED", os.path.join(ROOT, "results", "eval", "predictions.npz"))
VAL_PRED  = os.environ.get("MODULE2_VAL_PRED", "")               # optional; enables proper tuning
TRUST_PARAMS = os.path.join(ROOT, "results", "trust", "trust_params_FNO.npz")
FIGDIR = os.path.join(ROOT, "results", "module2", "figures")
VIABILITY = 1.47          # Module 2 measured boundary (~ FNO reliable horizon 1.457)
ERR_TARGET = float(os.environ.get("M2_ERR_TARGET", "0.08"))     # accuracy target for the recommendation

def _grid(env, default):
    v = os.environ.get(env, "")
    return [float(s) for s in v.split(",")] if v else default

CUT_GRID = _grid("M2_CUT_GRID", [0.5, 0.6, 0.7, 0.8, 0.9])
K_GRID   = [int(x) for x in _grid("M2_K_GRID", [4, 8, 12, 18, 25])]
N_WAVES  = int(os.environ.get("M2_N_WAVES", "0"))               # 0 = all available


def tail_rel_l2(traj, ref, t, i0):
    c = np.sqrt(((traj[i0:] - ref[i0:]) ** 2).sum(-1)) / (np.sqrt((ref[i0:] ** 2).sum(-1)) + EPS)
    return float(np.trapezoid(c, t[i0:]) / (t[-1] - t[i0] + EPS))


def raw_trust_trajectory(u_wave, t, base_params):
    """Stream the REAL TrustMonitor once; record trust per frame (independent of CUT/K)."""
    p = dict(base_params); p["K"] = 10 ** 9        # never latch 'failed' -> pure trust signal
    mon = TrustMonitor(p)
    return np.array([mon.update(u_wave[i], float(t[i]))["trust"] for i in range(len(t))])


def switch_index(trust, cut, K):
    """First index where trust<cut for K consecutive frames (mirrors monitor debounce)."""
    run = 0
    for i, v in enumerate(trust):
        run = run + 1 if v < cut else 0
        if run >= K:
            return i
    return None                                    # never fires -> stay on ML


def evaluate(pred_file, base_params, configs, tag):
    d = np.load(pred_file)
    t = d["t"]; x = d["x"]
    true = d["u_true_eval"].astype(float); fno = d["FNO_eval"].astype(float)
    n = fno.shape[0] if N_WAVES == 0 else min(N_WAVES, fno.shape[0])
    true, fno = true[:n], fno[:n]
    nt = len(t); i1 = nearest_index(1.0)

    # trust trajectories once per wave
    trust = np.array([raw_trust_trajectory(fno[j], t, base_params) for j in range(n)])

    # map each (config, wave) -> switch index; collect unique indices for batched solves
    def switches_for(cfg):
        if cfg["kind"] == "trust":
            return [switch_index(trust[j], cfg["CUT"], cfg["K"]) for j in range(n)]
        if cfg["kind"] == "fixed":
            return [nearest_index(cfg["t_s"])] * n
        return [None] * n

    per_cfg_sw = {c["name"]: switches_for(c) for c in configs}
    uniq = sorted({s for sws in per_cfg_sw.values() for s in sws if s is not None})

    # batched numerical continuation from each needed switch index (over all waves)
    Hcache = {s: solve_from(fno[:, s], s) for s in uniq}        # (n, nt-s, NX)

    pure = np.array([tail_rel_l2(fno[j], true[j], t, i1) for j in range(n)])

    rows = []
    for c in configs:
        sws = per_cfg_sw[c["name"]]
        hyb_errs, works, trigs = [], [], []
        for j, s in enumerate(sws):
            if s is None:
                out = fno[j]; work = 0.0; trig = float("nan")
            else:
                out = fno[j].copy(); out[s:] = Hcache[s][j]; work = (nt - s) / nt; trig = float(t[s])
            hyb_errs.append(tail_rel_l2(out, true[j], t, i1)); works.append(work); trigs.append(trig)
        he = float(np.mean(hyb_errs)); pe = float(np.mean(pure))
        rows.append(dict(name=c["name"], kind=c["kind"], CUT=c.get("CUT"), K=c.get("K"),
                         t_s=c.get("t_s"),
                         mean_trigger=float(np.nanmean(trigs)),
                         hybrid_err=he, pure_fno_err=pe,
                         benefit=float((pe - he) / (pe + EPS)),
                         numerical_work=float(np.mean(works))))
    return rows, int(n)


def main():
    base = load_params(TRUST_PARAMS)
    base_cut, base_K = float(base.get("CUT", 0.5)), int(base["K"])

    configs = [dict(name="current_trust", kind="trust", CUT=base_cut, K=base_K),
               dict(name="fixed@1.0", kind="fixed", t_s=1.0),
               dict(name="fixed@1.4", kind="fixed", t_s=1.4)]
    for cut in CUT_GRID:
        for K in K_GRID:
            configs.append(dict(name=f"trust_CUT{cut:.2f}_K{K}", kind="trust", CUT=cut, K=K))

    val_rows, selected = None, None
    if VAL_PRED and os.path.exists(VAL_PRED):
        val_rows, _ = evaluate(VAL_PRED, base, configs, "validation")
        # recommendation: cheapest config meeting the accuracy target on VALIDATION
        cand = [r for r in val_rows if r["kind"] == "trust" and r["hybrid_err"] <= ERR_TARGET]
        pick = min(cand, key=lambda r: r["numerical_work"]) if cand else min(val_rows, key=lambda r: r["hybrid_err"])
        selected = pick["name"]
        mode = "TUNED-ON-VALIDATION, REPORTED-ON-TEST"
    else:
        mode = "DIAGNOSTIC (tuned & reported on TEST set — illustrative only, not final claims)"

    test_rows, n = evaluate(TEST_PRED, base, configs, "test")
    if selected is None:  # diagnostic recommendation from test itself, clearly labelled
        cand = [r for r in test_rows if r["kind"] == "trust" and r["hybrid_err"] <= ERR_TARGET]
        selected = (min(cand, key=lambda r: r["numerical_work"]) if cand
                    else min(test_rows, key=lambda r: r["hybrid_err"]))["name"]

    os.makedirs(FIGDIR, exist_ok=True)
    out = dict(mode=mode, n_waves=n, viability_boundary=VIABILITY, err_target=ERR_TARGET,
               base_calibration=dict(CUT=base_cut, K=base_K),
               recommended=selected, cut_grid=CUT_GRID, k_grid=K_GRID,
               validation=val_rows, test=test_rows)
    with open(os.path.join(FIGDIR, "trust_tuning_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- table ----
    print(f"\nTrust-tuning sweep | mode: {mode} | n={n} waves | viability t_s~{VIABILITY}")
    print("%-22s %8s %10s %9s %8s" % ("config", "trigger", "hyb_err", "benefit", "num_work"))
    print("-" * 62)
    for r in test_rows:
        star = " <= RECOMMENDED" if r["name"] == selected else ""
        print("%-22s %8.3f %10.3f %9.2f %8.2f%s" %
              (r["name"], r["mean_trigger"], r["hybrid_err"], r["benefit"], r["numerical_work"], star))

    # ---- plot: trigger time vs hybrid error (size ~ numerical work) ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        tr = [r for r in test_rows if r["kind"] == "trust"]
        fx = [r for r in test_rows if r["kind"] == "fixed"]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        sc = ax.scatter([r["mean_trigger"] for r in tr], [r["hybrid_err"] for r in tr],
                        s=[30 + 260 * r["numerical_work"] for r in tr],
                        c=[r["numerical_work"] for r in tr], cmap="viridis", alpha=0.85,
                        edgecolor="k", linewidth=0.4, label="trust configs")
        for r in fx:
            ax.scatter(r["mean_trigger"], r["hybrid_err"], marker="D", s=70, color="#d1495b")
            ax.annotate(r["name"], (r["mean_trigger"], r["hybrid_err"]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        ax.axvspan(1.3, VIABILITY, color="green", alpha=0.10)
        ax.axvline(VIABILITY, ls="--", color="green", lw=1)
        ax.text(VIABILITY + 0.01, ax.get_ylim()[1] * 0.9, "viability", color="green", fontsize=8)
        ax.set(xlabel="mean trust trigger time", ylabel="hybrid tail error [1,2]",
               title="Trust tuning: trigger time vs error (bubble/colour = numerical work)")
        fig.colorbar(sc, label="numerical-work fraction"); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "trust_tuning_sweep.png"), dpi=140)
        print("\nwrote trust_tuning_sweep.png and trust_tuning_sweep.json")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
