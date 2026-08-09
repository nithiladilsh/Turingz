import os, json
import numpy as np
import torch
from neuralop.models import FNO

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
import sys
sys.path.insert(0, os.path.join(ROOT, "hybrid_pde", "coupling - 214050V"))
from restart_spectral import solve_from, nearest_index, TGRID   # bit-identical team scheme

PRED = os.path.join(ROOT, "results", "eval", "predictions.npz")
FNO_PT = os.path.join(ROOT, "results", "fno", "fno.pt")
FNO_CFG = os.path.join(ROOT, "results", "fno", "fno_config.pt")
FIGDIR = os.path.join(ROOT, "results", "module2", "figures")

nu = 1.0 / (100 * np.pi); L, nx, T, nt = 2.0, 512, 2.0, 200
x = np.linspace(-1, 1, nx, endpoint=False); dx = L / nx
t = TGRID
SWITCH = [1.0, 1.2, 1.4, 1.6, 1.8]; EPS = 1e-12

# ---- Cole-Hopf reference ----
x_ext = np.concatenate([x - L, x, x + L]); diff = x[:, None] - x_ext
def cole_hopf(ic):
    ICs = ic[None, :]
    cumint = np.concatenate([np.zeros((1, 1)), np.cumsum(0.5*(ICs[:, :-1]+ICs[:, 1:])*dx, axis=1)], axis=1)
    a = -cumint/(2*nu); pe = np.tile(np.exp(a - a.max(axis=1, keepdims=True)), (1, 3))
    U = np.empty((1, nt, nx)); U[:, 0] = ICs
    for j in range(1, nt):
        K = np.exp(-diff**2/(4*nu*t[j])); U[:, j] = (pe @ (diff*K).T)/(pe @ K.T)/t[j]
    return U[0]

# ---- FNO ----
cfg = torch.load(FNO_CFG, map_location="cpu", weights_only=False)
model = FNO(n_modes=(cfg["n_modes"],), hidden_channels=cfg["hidden_channels"],
            in_channels=cfg.get("in_channels", 3), out_channels=1)
model.load_state_dict(torch.load(FNO_PT, map_location="cpu", weights_only=False)); model.eval()
xt = torch.tensor(x, dtype=torch.float32); tt = torch.tensor(t, dtype=torch.float32)
@torch.no_grad()
def fno_predict(ic):
    ic_t = torch.tensor(ic, dtype=torch.float32)
    inp = torch.stack([ic_t[None,:].expand(nt,nx), tt[:,None].expand(nt,nx), xt[None,:].expand(nt,nx)], dim=1)
    return model(inp).squeeze(1).cpu().numpy()

def norm(u): u = u - u.mean(); return u/(np.abs(u).max()+EPS)
def rl2(A,B): return np.sqrt(((A-B)**2).sum(-1))/(np.sqrt((B**2).sum(-1))+EPS)
def ig(c,tt_): return np.trapezoid(c,tt_)/(tt_[-1]-tt_[0]+EPS)

# ---- self-check: reproduce committed FNO prediction for an in-distribution wave ----
d = np.load(PRED)
r = np.random.default_rng(42)
def gen_ic(rr, m=4):
    u = sum(rr.standard_normal()*np.sin(2*np.pi*mm*x/L + rr.uniform(0,2*np.pi)) for mm in range(1,m+1)); return u/(np.abs(u).max()+EPS)
ics = [np.sin(np.pi*x)] + [gen_ic(r) for _ in range(909)]
chk = float(np.linalg.norm(fno_predict(ics[900]) - d["FNO_eval"][0]) / (np.linalg.norm(d["FNO_eval"][0])+EPS))
print(f"self-check (in-distribution IC 900): FNO rel diff = {chk:.2e}  {'OK' if chk<1e-3 else 'FAIL'}")
assert chk < 1e-3, "FNO inference does not match committed pipeline"

# ---- OOD waves ----
ood = {
    "high_freq": norm(np.sin(6*np.pi*x)),                          # frequency shift (beyond trained band)
    "gaussian":  norm(np.exp(-(x**2)/(2*0.10**2))),               # shape shift (localized bump)
}

results = {}
for name, ic in ood.items():
    true = cole_hopf(ic); fno = fno_predict(ic)
    rows = []
    for ts in SWITCH:
        i = nearest_index(ts); tail_t = t[i:]; tr = true[i:]
        H = solve_from(fno[i], i)
        es = float(np.linalg.norm(fno[i]-true[i])/(np.linalg.norm(true[i])+EPS))
        fe = float(ig(rl2(fno[i:], tr), tail_t)); he = float(ig(rl2(H, tr), tail_t))
        rows.append(dict(t_s=ts, e_s=es, fno_tail=fe, hybrid_tail=he, benefit=(fe-he)/(fe+EPS)))
    results[name] = rows
    print(f"\nOOD case: {name}")
    print("%4s %8s %10s %9s %7s" % ("t_s","e_s","FNO_tail","H_tail","B"))
    for rr in rows: print("%4.1f %8.3f %10.3f %9.3f %7.2f" % (rr["t_s"],rr["e_s"],rr["fno_tail"],rr["hybrid_tail"],rr["benefit"]))

json.dump({"note":"OOD test - not in training distribution","cases":results},
          open(os.path.join(FIGDIR,"ood_results.json"),"w"), indent=2)

# ---- figure: error over time for high_freq at t_s=1.0 ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ic = ood["high_freq"]; true = cole_hopf(ic); fno = fno_predict(ic); i = nearest_index(1.0)
H = solve_from(fno[i], i); NUM = solve_from(true[0], 0)
fno_c = rl2(fno, true); num_c = rl2(NUM, true); hyb_c = fno_c.copy(); hyb_c[i:] = rl2(H, true[i:])
fig, ax = plt.subplots(figsize=(7,4.3))
ax.plot(t, fno_c, color="#d1495b", lw=2, label="FNO alone")
ax.plot(t, num_c, color="#1f6feb", lw=2, label="Numerical alone")
ax.plot(t, hyb_c, color="#2e7d32", lw=2, label="Hybrid (handoff)")
ax.axvline(1.0, ls="--", color="k", lw=1)
ax.set(xlabel="time t", ylabel="relative L2 error",
       title="OOD (higher-frequency wave): handoff at t=1.0")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(os.path.join(FIGDIR,"fig8_ood_error_over_time.png"), dpi=140)
print("\nwrote fig8_ood_error_over_time.png and ood_results.json")
