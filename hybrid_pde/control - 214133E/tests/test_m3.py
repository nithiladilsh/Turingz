#!/usr/bin/env python3
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.dirname(HERE); ROOT = os.path.abspath(os.path.join(MDIR, "..", ".."))
sys.path.insert(0, MDIR); sys.path.insert(0, ROOT)
from m3_cost import (groundtruth as G, profiler as P, surrogate as S,
                     coupling as CP, accuracy_cost as AC, runtime as RT,
                     controller as CT, config as C)

PASS = 0; FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    print(("  PASS " if cond else "  FAIL ") + name + ("" if cond else f"  <- {detail}"))
    PASS += cond; FAIL += (not cond)

ICs = G.make_ics()
ic = ICs[900]
u_true = G.colehopf_solve(ic)

d = np.load(C.DEEPONET_FIELD, allow_pickle=True)
err = np.linalg.norm(u_true - d["u_true"]) / np.linalg.norm(d["u_true"])
check("ground truth matches stored u_true (<1e-5)", err < 1e-5, f"err={err:.2e}")

e_sp = G.rel_l2(G.spectral_solve(ic), u_true, G.EXTRAP_MASK)
e_fd = G.rel_l2(G.fdm_solve(ic), u_true, G.EXTRAP_MASK)
check("spectral near-exact (<1e-3)", e_sp < 1e-3, f"{e_sp:.2e}")
check("fdm is approximate (>1e-3)", e_fd > 1e-3, f"{e_fd:.2e}")

units = P.profile_numerical_units(G, ic, step_repeats=5, full_repeats=1)
steps = {u["op"]: u["median_ms"] for u in units if u["op"].endswith(".step")}
check("step costs positive", all(v > 0 for v in steps.values()), str(steps))
check("spectral step > fdm step", steps["spectral.step"] > steps["fdm.step"], str(steps))

u_ml = S.CachedSurrogate(C.DEEPONET_FIELD).predict_rollout(900)
_, cost = CP.switch_rollout(ic, u_ml, j_switch=100, solver="spectral", unit_num_ms=15.0)
check("switch cost = NT-1-j numerical steps", cost.n_num_steps == u_ml.shape[0]-1-100,
      f"{cost.n_num_steps}")
check("wall_ms = steps*unit", abs(cost.wall_ms - cost.n_num_steps*15.0) < 1e-6)

cloud = AC.policy_cloud(ic, u_ml, u_true, {"fdm":0.8,"spectral":15.0},
                        solvers=("fdm",), switch_grid=list(range(2,199,20))+[198])
front = AC.pareto_front(cloud)
costs = [r["numerical_step_equivalents"] for r in front]
accs = [r["err_extrap"] for r in front]
check("frontier cost strictly increasing", all(costs[i]<costs[i+1] for i in range(len(costs)-1)))
check("frontier accuracy strictly improving", all(accs[i]>accs[i+1] for i in range(len(accs)-1)))
base = AC.baselines(ic, u_ml, u_true, {"fdm":0.8,"spectral":15.0}, ref_solver="fdm")
check("hybrid floor beats pure-ML", min(accs) < base["pure_ml"]["err_extrap"]/2,
      f"floor={min(accs):.3f} ml={base['pure_ml']['err_extrap']:.3f}")

rt = RT.HybridRuntime(S.CachedSurrogate(C.DEEPONET_FIELD), {"fdm":0.8,"spectral":15.0},
                      solver="spectral", trust_mode="oracle")
rep = rt.run(900, 0.5, u_true=u_true)
check("runtime meets loose target 0.5", rep["target_met"] is True, str(rep["achieved_extrap_error"]))

oc = CT.OracleController({"spectral":15.0}, solver="spectral")
jf = int(np.searchsorted(G.T_GRID, 1.0))
jo,_,_ = oc.plan(0.5, u_ml, u_true)
ca = CT.evaluate_policy(ic,u_ml,u_true,jo,"spectral",15.0,float("nan"))["n_num_steps"]
cf = CT.evaluate_policy(ic,u_ml,u_true,jf,"spectral",15.0,float("nan"))["n_num_steps"]
check("adaptive <= fixed cost at loose target", ca <= cf, f"adaptive={ca} fixed={cf}")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
