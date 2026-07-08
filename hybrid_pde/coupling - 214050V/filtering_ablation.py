"""
Phase 3 - state-preparation ablation (Module 2, Coupling).
Question: does low-pass filtering (cleaning) the handed-over FNO wave help the
hybrid, or is the raw wave already best? Filter keeps a fraction of the resolved
Fourier band before handing over; we compare hybrid tail error vs raw.
"""
import os, numpy as np
from restart_spectral import solve_from, nearest_index, TGRID, NX

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRED = os.path.join(ROOT, "results", "eval", "predictions.npz")
EPS = 1e-12
BAND = NX // 3                      # resolved band (dealias): modes 0..170
FILTERS = [("raw", 1.00), ("keep90%", 0.90), ("keep75%", 0.75), ("keep50%", 0.50)]
SWITCH = [1.0, 1.4, 1.8]

def lowpass(u, frac):               # u: (B, NX)
    if frac >= 1.0: return u.copy()
    uh = np.fft.rfft(u, axis=-1)
    cut = int(frac * BAND)
    uh[..., cut+1:] = 0.0
    return np.fft.irfft(uh, n=NX, axis=-1)

def rl2c(U, Ur): return np.sqrt(((U-Ur)**2).sum(-1))/(np.sqrt((Ur**2).sum(-1))+EPS)
def integ(c, tt): return np.trapezoid(c, tt)/(tt[-1]-tt[0]+EPS)

d = np.load(PRED)
t = d["t"]; Utrue = d["u_true_eval"].astype(np.float64); Ufno = d["FNO_eval"].astype(np.float64)
n = Utrue.shape[0]
print("Phase 3 - handoff filtering ablation (mean hybrid tail error over %d waves)\n" % n)
print("%5s  %-8s  %-12s  %-14s" % ("t_s","filter","hybrid_tail","state_change"))
print("-"*46)
for ts in SWITCH:
    i = nearest_index(ts); tail_t = t[i:]; true_tail = Utrue[:, i:]
    raw_state = Ufno[:, i]
    for name, frac in FILTERS:
        st = lowpass(raw_state, frac)
        H = solve_from(st, i)
        err = np.mean([integ(rl2c(H[j], true_tail[j]), tail_t) for j in range(n)])
        chg = np.mean(np.linalg.norm(st-raw_state,axis=1)/(np.linalg.norm(raw_state,axis=1)+EPS))
        print("%5.1f  %-8s  %-12.4f  %-14.4f" % (ts, name, err, chg))
    print()
print("Reading: if 'raw' has the lowest hybrid_tail at each t_s, filtering does not help.")
