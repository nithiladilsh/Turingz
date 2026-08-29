"""
Final integration result: trust-triggered hard switch vs baselines (Module 2).
Compares, over the extrapolation window [1, 2] on the held-out FNO waves:
  Pure FNO | Fixed switch @ t=1.4 | Real trust-triggered switch
using the REAL trust monitor (hybrid_pde.trust) and the verified hard-switch handoff.
Set MODULE2_PRED=results/eval/predictions.npz for n=10 (default) or
    MODULE2_PRED=results/eval/predictions_ext.npz for n=100.
The "Pure numerical (ref)" row below is a DISPLAY-ONLY reference line (not
computed by this script - the Cole-Hopf ground truth is the reference every
other row is measured against, so its own error is ~0 by definition). It is
not written to the output JSON.
"""
import os, sys, json
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(__file__))
from restart_spectral import solve_from, nearest_index, TGRID
from hybrid_pde.trust.monitor import TrustMonitor, load_params
EPS=1e-12; FIXED=1.4
PRED = os.environ.get("MODULE2_PRED", os.path.join(ROOT,"results","eval","predictions.npz"))
def rl2c(U,Ur): return np.sqrt(((U-Ur)**2).sum(-1))/(np.sqrt((Ur**2).sum(-1))+EPS)
def ig(c,tt): return float(np.trapezoid(c,tt)/(tt[-1]-tt[0]+EPS))

d=np.load(PRED); t=d["t"]; true=d["u_true_eval"].astype(float); fno=d["FNO_eval"].astype(float); n=fno.shape[0]
i1=nearest_index(1.0); win=t[i1:]; nt=len(t)
p=load_params(os.path.join(ROOT,"results","trust","trust_params_FNO.npz"))

def hard_switch(pred_j, sw):                       # my validated handoff
    if sw is None or sw>=nt: return pred_j.copy()
    tail=solve_from(pred_j[sw], sw); out=pred_j.copy(); out[sw:]=tail; return out

rows={"pure_fno":[], "fixed_1p4":[], "trust":[]}; trig=[]; wl_fixed=(nt-nearest_index(FIXED))/nt; wl_trust=[]
for j in range(n):
    tj=true[j]
    rows["pure_fno"].append(ig(rl2c(fno[j,i1:], tj[i1:]), win))
    hf=hard_switch(fno[j], nearest_index(FIXED)); rows["fixed_1p4"].append(ig(rl2c(hf[i1:], tj[i1:]), win))
    mon=TrustMonitor(p); sw=None
    for i in range(nt):
        if not mon.update(fno[j,i], t[i])["ok"]: sw=i; break
    trig.append(t[sw] if sw is not None else float('nan')); wl_trust.append((nt-sw)/nt if sw is not None else 0.0)
    ht=hard_switch(fno[j], sw); rows["trust"].append(ig(rl2c(ht[i1:], tj[i1:]), win))

def m(a): return float(np.mean(a))
print("Final integration comparison  (mean over %d held-out waves, error window [1,2])\n"%n)
print("%-26s %10s %14s"%("method","err[1,2]","numerical work"))
print("-"*54)
print("%-26s %10.4f %14s"%("Pure FNO", m(rows["pure_fno"]), "0%"))
print("%-26s %10.4f %13.0f%%"%("Fixed switch @ t=1.4", m(rows["fixed_1p4"]), 100*wl_fixed))
print("%-26s %10.4f %13.0f%%"%("Real trust-triggered", m(rows["trust"]), 100*m(wl_trust)))
print("%-26s %10s %14s"%("Pure numerical (ref)", "~0 (def.)", "100%"))
print("  ^ display-only: this is the Cole-Hopf reference itself, not a measured hybrid")
print("    run, and is not written to the JSON.")
print("\nmean trust trigger time: %.3f   (n=%d; see handoff_sweep_results*.json for the "
      "measured viability boundary at this n; FNO true-fail ~1.50)" % (np.nanmean(trig), n))
json.dump({"n":n,"means":{k:m(v) for k,v in rows.items()},
           "fixed_workload":wl_fixed,"trust_workload":m(wl_trust),"mean_trigger":float(np.nanmean(trig))},
          open(os.path.join(ROOT,"results","module2","figures","trust_hardswitch_compare.json"),"w"), indent=2)
print("saved trust_hardswitch_compare.json")
