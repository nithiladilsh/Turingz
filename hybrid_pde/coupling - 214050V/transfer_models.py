import os, numpy as np
from restart_spectral import solve_from, nearest_index, TGRID

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRED = os.path.join(ROOT, "results", "eval", "predictions.npz")
EPS = 1e-12
SWITCH = [1.0, 1.4, 1.8]

def rl2c(U, Ur): return np.sqrt(((U-Ur)**2).sum(-1))/(np.sqrt((Ur**2).sum(-1))+EPS)
def integ(c, tt): return np.trapezoid(c, tt)/(tt[-1]-tt[0]+EPS)

d = np.load(PRED)
t = d["t"]; Utrue = d["u_true_eval"].astype(np.float64)
models = {"FNO": d["FNO_eval"], "DeepONet": d["DeepONet_eval"], "PINN": d["PINN"]}
n = Utrue.shape[0]
print("Phase 4 - same coupling, three ML models (mean over %d waves)\n" % n)
print("%-9s %5s  %-9s %-11s %-9s %-8s" % ("model","t_s","state_err","pureML_tail","hyb_tail","benefit"))
print("-"*58)
results = {}
for name, Mpred in models.items():
    Mp = Mpred.astype(np.float64)
    for ts in SWITCH:
        i = nearest_index(ts); tail_t = t[i:]; true_tail = Utrue[:, i:]
        H = solve_from(Mp[:, i], i)
        es = np.mean(np.linalg.norm(Mp[:,i]-Utrue[:,i],axis=1)/(np.linalg.norm(Utrue[:,i],axis=1)+EPS))
        pml = np.mean([integ(rl2c(Mp[j, i:], true_tail[j]), tail_t) for j in range(n)])
        hyb = np.mean([integ(rl2c(H[j],      true_tail[j]), tail_t) for j in range(n)])
        ben = (pml - hyb)/(pml+EPS)
        results.setdefault(name, []).append({"t_s": float(ts), "state_err": float(es),
                                             "pureML_tail": float(pml), "hybrid_tail": float(hyb), "benefit": float(ben)})
        print("%-9s %5.1f  %-9.3f %-11.3f %-9.3f %-8.2f" % (name, ts, es, pml, hyb, ben))
    print()
print("Same solve_from() call for all three - only the model array changed => solver-agnostic.")
import json
_OUT = os.path.join(ROOT, "results", "module2", "figures", "transfer_models_results.json")
json.dump({"note": "same coupling, three ML models (predictions.npz) - model/solver-agnostic",
           "n_waves": int(n), "switch_times": [float(s) for s in SWITCH],
           "reference": "Cole-Hopf (u_true_eval)", "metric": "time-integrated relative L2 over [t_s, 2]",
           "models": results}, open(_OUT, "w"), indent=2)
print("wrote", _OUT)

# ---- figure: benefit vs handoff-state error, all three models (the universal law) ----
def make_fig():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIG = os.path.join(ROOT, "results", "module2", "figures")
    SW = [1.0,1.2,1.4,1.6,1.8]
    col = {"FNO":"#2e7d32","PINN":"#1f6feb","DeepONet":"#d1495b"}
    fig, ax = plt.subplots(figsize=(7,4.3))
    for name, Mpred in models.items():
        Mp = Mpred.astype(np.float64); xs=[]; ys=[]
        for ts in SW:
            i=nearest_index(ts); tail_t=t[i:]; tt=Utrue[:,i:]
            H=solve_from(Mp[:,i],i)
            es=np.mean(np.linalg.norm(Mp[:,i]-Utrue[:,i],axis=1)/(np.linalg.norm(Utrue[:,i],axis=1)+EPS))
            pml=np.mean([integ(rl2c(Mp[j,i:],tt[j]),tail_t) for j in range(n)])
            hyb=np.mean([integ(rl2c(H[j],tt[j]),tail_t) for j in range(n)])
            xs.append(es); ys.append((pml-hyb)/(pml+EPS))
        ax.plot(xs, ys, "o-", color=col[name], lw=2, label=name)
    ax.axhline(0.10, ls="--", color="grey")
    ax.set(xlabel="FNO/ML wave error at handoff", ylabel="benefit (1 - hybrid/ML)",
           title="Consistent across models: benefit falls as the handed-over wave degrades")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(FIG,"fig7_transfer_benefit_vs_stateerror.png"), dpi=140)
    print("wrote fig7")
make_fig()
