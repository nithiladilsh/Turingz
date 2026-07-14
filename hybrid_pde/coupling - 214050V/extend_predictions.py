"""
Extend the held-out evaluation set from 10 -> 20 waves (Module 2, Coupling).

RUN ON YOUR MACHINE (needs torch + neuralop; not runnable in the offline sandbox):

    python "hybrid_pde/coupling - 214050V/extend_predictions.py"

What it does:
  1. Regenerates the Cole-Hopf initial conditions with the fixed seed (seed=42,
     exactly as scripts/generate_dataset.py / colehopf.py), and computes the
     Cole-Hopf ground truth for test waves 900-919.
  2. Loads the trained FNO (results/fno/fno.pt + fno_config.pt) and predicts the
     full trajectory for those 20 waves, using the SAME (ic, t, x) input channels
     the model was trained with (see solvers/ml/fno/fno.py `build`).
  3. SELF-CHECK: reproduces predictions for the first 10 waves (900-909) and
     asserts they match results/eval/predictions.npz to a tight tolerance. If this
     fails, STOP - the inference does not match the committed pipeline.
  4. Saves results/eval/predictions_ext.npz with u_true_eval / FNO_eval at n=20.

Then rebuild the figures at n=20:
    MODULE2_PRED=results/eval/predictions_ext.npz python "hybrid_pde/coupling - 214050V/make_figures.py" all
(on Windows PowerShell:  $env:MODULE2_PRED="results/eval/predictions_ext.npz"; python ... )
"""
import os
import numpy as np
import torch
from neuralop.models import FNO

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRED = os.path.join(ROOT, "results", "eval", "predictions.npz")
FNO_PT = os.path.join(ROOT, "results", "fno", "fno.pt")
FNO_CFG = os.path.join(ROOT, "results", "fno", "fno_config.pt")
OUT = os.path.join(ROOT, "results", "eval", "predictions_ext.npz")

# ---- constants (must match colehopf.py / spectral.py exactly) ----
nu = 1.0 / (100 * np.pi)
L, nx = 2.0, 512
T, nt = 2.0, 200
x = np.linspace(-1, 1, nx, endpoint=False)
dx = L / nx
t_start = 0.01
t = np.concatenate([[0.0], np.linspace(t_start, T, nt - 1)])
FIRST, LAST = 900, 920                       # test waves 900..919 (20 held out)

# ---- reproduce the exact initial conditions (seed 42) ----
def random_ic(rng, n_modes=4):
    u = sum(rng.standard_normal() * np.sin(2 * np.pi * m * x / L + rng.uniform(0, 2 * np.pi))
            for m in range(1, n_modes + 1))
    return u / (np.abs(u).max() + 1e-12)

rng = np.random.default_rng(42)
ICs_all = [np.sin(np.pi * x)] + [random_ic(rng) for _ in range(LAST - 1)]  # indices 0..LAST-1
ICs = np.stack(ICs_all[FIRST:LAST])          # (20, nx)

# ---- Cole-Hopf ground truth for these 20 waves (analytic; matches colehopf.py) ----
x_ext = np.concatenate([x - L, x, x + L])
diff = x[:, None] - x_ext
def cole_hopf(ICs):
    cumint = np.concatenate([np.zeros((len(ICs), 1)),
                             np.cumsum(0.5 * (ICs[:, :-1] + ICs[:, 1:]) * dx, axis=1)], axis=1)
    a = -cumint / (2 * nu)
    pe = np.tile(np.exp(a - a.max(axis=1, keepdims=True)), (1, 3))
    U = np.empty((len(ICs), nt, nx)); U[:, 0] = ICs
    for j in range(1, nt):
        K = np.exp(-diff**2 / (4 * nu * t[j]))
        U[:, j] = (pe @ (diff * K).T) / (pe @ K.T) / t[j]
    return U
u_true = cole_hopf(ICs)                       # (20, nt, nx)

# ---- FNO predictions (same (ic,t,x) channels as training) ----
cfg = torch.load(FNO_CFG, map_location="cpu", weights_only=False)
model = FNO(n_modes=(cfg["n_modes"],), hidden_channels=cfg["hidden_channels"],
            in_channels=cfg.get("in_channels", 3), out_channels=1)
model.load_state_dict(torch.load(FNO_PT, map_location="cpu", weights_only=False))
model.eval()

xt = torch.tensor(x, dtype=torch.float32)
tt = torch.tensor(t, dtype=torch.float32)

@torch.no_grad()
def fno_predict(ic):
    ic_t = torch.tensor(ic, dtype=torch.float32)
    ch_ic = ic_t[None, :].expand(nt, nx)
    ch_t = tt[:, None].expand(nt, nx)
    ch_x = xt[None, :].expand(nt, nx)
    inp = torch.stack([ch_ic, ch_t, ch_x], dim=1)   # (nt, 3, nx)
    return model(inp).squeeze(1).cpu().numpy()      # (nt, nx)

fno_pred = np.stack([fno_predict(ic) for ic in ICs])  # (20, nt, nx)

# ---- SELF-CHECK against the committed predictions for waves 900-909 ----
d = np.load(PRED)
def rel(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
err_truth = rel(u_true[:10], d["u_true_eval"])
err_fno = rel(fno_pred[:10], d["FNO_eval"])
print(f"self-check vs predictions.npz  |  truth rel diff = {err_truth:.2e}  |  FNO rel diff = {err_fno:.2e}")
assert err_truth < 1e-4, "Cole-Hopf truth does not match - check constants/seed"
assert err_fno < 1e-3, "FNO inference does not match committed predictions - check channels/checkpoint"

np.savez(OUT, x=x.astype(np.float32), t=t.astype(np.float32),
         u_true_eval=u_true.astype(np.float32), FNO_eval=fno_pred.astype(np.float32))
print(f"OK - wrote {OUT} with n=20 held-out waves. Rebuild figures with MODULE2_PRED set to it.")
