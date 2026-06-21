"""
Regenerate web/web_data.js for the showcase site — and populate FNO (and,
optionally, DeepONet via torch) so the Live Demo comparison fills in.

Why this exists: the static site ships with the solvers that run without PyTorch
(Cole-Hopf truth, spectral, FDM, cached DeepONet, hybrid). FNO needs the trained
torch model, so its solution field is produced here, on a machine that has torch.

Run on the GPU/host with the training stack installed:
    pip install torch neuralop          # (deepxde too if you wire DeepONet/PINN)
    python generate_web_data.py

It writes web/web_data.js. Re-open web/index.html and the FNO row/line/chip in the
Live Demo are now live alongside the rest. Without torch it still runs and simply
leaves FNO 'pending' (identical to the shipped data), so it is always safe to run.

Note on PINN: PINN is trained PER initial condition (only ICs 0-9). It has no model
for the unseen test IC 900, so it is intentionally left N/A in the IC-900 comparison
— that limitation is a finding, not a gap to paper over.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.join(HERE, "hybrid_pde", "control - 214133E")
sys.path.insert(0, MDIR)
from m3_cost import groundtruth as G, surrogate as S, runtime as RT, config as C

IC = 900
SX, ST = 4, 2                      # spatial / temporal downsampling for the web
UNIT = {"fdm": 0.64, "spectral": 15.6}
TARGETS = [0.50, 0.42, 0.36, 0.30, 0.26, 0.22, 0.20, 0.18, 0.16, 0.14, 0.13]


def ds(U):
    return np.round(U[::ST, ::SX], 3).tolist()


def metrics(u, u_true, steps, unit):
    return {"inDist": round(G.rel_l2(u, u_true, G.TRAIN_MASK), 4),
            "extrap": round(G.rel_l2(u, u_true, G.EXTRAP_MASK), 4),
            "costSteps": steps, "costMs": round(steps * unit) if steps else 0}


def try_fno():
    """Return (field, ok). Loads the trained FNO if torch is available."""
    try:
        sur = S.TorchSurrogate("fno", C.ROOT)
        return sur.predict_rollout(IC), True
    except Exception as ex:
        print(f"[fno] not populated ({type(ex).__name__}: {ex}). Leaving pending.")
        return None, False


def main():
    ic = G.make_ics()[IC]
    u_true = G.colehopf_solve(ic)
    u_ml = S.CachedSurrogate(C.DEEPONET_FIELD).predict_rollout(IC)
    u_fdm = G.fdm_solve(ic)
    u_spec = G.spectral_solve(ic)

    rt = RT.HybridRuntime(S.CachedSurrogate(C.DEEPONET_FIELD), UNIT,
                          solver="spectral", trust_mode="oracle")
    hyb_variants, hyb_meta = {}, {}
    for eps in TARGETS:
        rep, uh = rt.run(IC, eps, u_true=u_true, return_field=True)
        hyb_variants[str(eps)] = ds(uh)
        hyb_meta[str(eps)] = {"switch_time": round(rep["switch_time"], 3),
                              "num_steps": rep["numerical_steps_spent"],
                              "wall_ms": round(rep["wall_ms"]),
                              "achieved": round(rep["achieved_extrap_error"], 3),
                              "target_met": rep["target_met"]}
    u_hyb = rt.run(IC, 0.20, u_true=u_true, return_field=True)[1]

    fields = {"x": np.round(G.X[::SX], 3).tolist(), "t": np.round(G.T_GRID[::ST], 3).tolist(),
              "truth": ds(u_true), "ml": ds(u_ml), "fdm": ds(u_fdm), "spectral": ds(u_spec),
              "hybrid": hyb_variants["0.2"], "hybrid_variants": hyb_variants,
              "hybrid_meta": hyb_meta, "targets": TARGETS,
              "switch_time": hyb_meta["0.2"]["switch_time"], "t_train_end": 1.0}

    u_fno, fno_ok = try_fno()
    if fno_ok:
        fields["fno"] = ds(u_fno)

    def errc(u):
        return np.round(G.rel_l2(u, u_true), 4).tolist()
    err = {"t": np.round(G.T_GRID, 4).tolist(), "ml": errc(u_ml), "hybrid": errc(u_hyb),
           "fdm": errc(u_fdm), "spectral": errc(u_spec)}

    ea = json.load(open(os.path.join(C.RESULTS, "deeponet", "extrapolation_analysis.json")))
    deeponet_curve = {"t": [round(x, 4) for x in ea["time"]],
                      "model": [round(x, 4) for x in ea["error_vs_time_model"]],
                      "persistence": [round(x, 4) for x in ea["error_vs_time_persistence_extrap_baseline"]],
                      "in_dist_mean": round(ea["model_in_dist_mean"], 4),
                      "extrap_mean": round(ea["model_extrap_mean"], 4),
                      "extrap_final": round(ea["model_extrap_final"], 4),
                      "persistence_extrap_mean": round(ea["persistence_extrap_mean"], 4)}
    ti = json.load(open(os.path.join(C.RESULTS, "deeponet", "training_info.json")))
    deeponet_eval = {k: {"mean": round(ti[k]["mean"], 4), "std": round(ti[k]["std"], 4)}
                     for k in ["train_in_dist", "val_in_dist", "test_in_dist", "test_extrap"]}

    solvers_meta = [
        {"key": "truth", "label": "Cole-Hopf (exact)", "family": "Reference", "color": "#e8edf7",
         "inDist": 0.0, "extrap": 0.0, "costSteps": None, "costMs": None, "available": True,
         "note": "Exact analytical solution — the ground truth."},
        {"key": "spectral", "label": "Spectral (numerical)", "family": "Numerical", "color": "#5e9bec",
         **metrics(u_spec, u_true, 199, UNIT["spectral"]), "available": True,
         "note": "4th-order pseudo-spectral. Near-exact but full cost."},
        {"key": "fdm", "label": "Finite-Difference (numerical)", "family": "Numerical", "color": "#f0a060",
         **metrics(u_fdm, u_true, 199, UNIT["fdm"]), "available": True,
         "note": "Cheap per step but artificial diffusion biases the shock."},
        {"key": "ml", "label": "DeepONet (ML)", "family": "Machine Learning", "color": "#5bd97c",
         **metrics(u_ml, u_true, 0, 0), "available": True,
         "note": "Operator network. Fast, accurate in-window, diverges in extrapolation."},
        {"key": "fno", "label": "FNO (ML)", "family": "Machine Learning", "color": "#b07cff",
         **(metrics(u_fno, u_true, 0, 0) if fno_ok else {"inDist": None, "extrap": None, "costSteps": 0, "costMs": 0}),
         "available": bool(fno_ok),
         "note": "Fourier Neural Operator — generalises across ICs." + ("" if fno_ok else " Run with torch to populate.")},
        {"key": "pinn", "label": "PINN (ML)", "family": "Machine Learning", "color": "#f0b760",
         "inDist": None, "extrap": None, "costSteps": 0, "costMs": 0, "available": False,
         "note": "Physics-informed net — PER-IC (trained only on ICs 0-9). No model for the unseen test IC 900 by design."},
        {"key": "hybrid", "label": "Hybrid Engine (M3)", "family": "Hybrid", "color": "#ff6b6b",
         **metrics(u_hyb, u_true, hyb_meta["0.2"]["num_steps"], UNIT["spectral"]), "available": True,
         "note": "ML where trusted, numerics where needed. Best accuracy/cost balance."},
    ]

    out = {"fields": fields, "err": err, "deeponet_curve": deeponet_curve,
           "deeponet_eval": deeponet_eval, "solvers_meta": solvers_meta, "ic_index": IC}
    # carry over frontier/controller/demo if present (produced by the M3 scripts)
    for k, p in [("frontier", "frontier_deeponet-cached.json"),
                 ("controller", "controller_deeponet-cached.json"),
                 ("demo", "demo_deeponet-cached.json")]:
        fp = os.path.join(C.M3_RESULTS, p)
        if os.path.exists(fp):
            d = json.load(open(fp))
            if k == "frontier":
                out["frontier"] = d["per_ic"][0]; out["frontier_units"] = d["unit_num_ms"]
            elif k == "controller":
                out["controller"] = list(d["per_ic"].values())[0]
            else:
                out["demo"] = d

    web = os.path.join(HERE, "web", "web_data.js")
    os.makedirs(os.path.dirname(web), exist_ok=True)
    with open(web, "w") as f:
        f.write("window.TURINGZ_DATA = " + json.dumps(out) + ";\n")
    print(f"wrote {web}  (FNO {'POPULATED' if fno_ok else 'pending'})")


if __name__ == "__main__":
    main()
