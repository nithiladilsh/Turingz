import os, sys
os.environ.setdefault("DDE_BACKEND", "pytorch")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "colehopf", "burgers_colehopf.pt")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "eval")
os.makedirs(OUT, exist_ok=True)

N_ID = 100
N_OOD = 50
OOD_MODES = 10
PERT_EPS = 0.02 

d = torch.load(DATA, weights_only=False, map_location="cpu")
U = d["u"].numpy(); ICs = d["ICs"].numpy()
x = d["x"].numpy(); t = d["t"].numpy()
nx, nt = U.shape[-1], U.shape[1]
L = 2.0; dx = L / nx; nu = 1.0 / (100 * np.pi)
x_ext = np.concatenate([x - L, x, x + L]); diff = x[:, None] - x_ext

def cole_hopf(ics):
    cumint = np.concatenate([np.zeros((len(ics), 1)),
                             np.cumsum(0.5 * (ics[:, :-1] + ics[:, 1:]) * dx, axis=1)], axis=1)
    a = -cumint / (2 * nu)
    pe = np.tile(np.exp(a - a.max(axis=1, keepdims=True)), (1, 3))
    out = np.empty((len(ics), nt, nx), np.float64); out[:, 0] = ics
    for j in range(1, nt):
        K = np.exp(-diff ** 2 / (4 * nu * t[j]))
        out[:, j] = (pe @ (diff * K).T) / (pe @ K.T) / t[j]
    return out.astype(np.float32)

def ood_ic(rng, n_modes):
    u = sum(rng.standard_normal() * np.sin(2 * np.pi * mm * x / L + rng.uniform(0, 2 * np.pi))
            for mm in range(1, n_modes + 1))
    return u / (np.abs(u).max() + 1e-12)

def fno_pred(ics):
    from neuralop.models import FNO
    torch.set_default_device("cpu")
    cfg = torch.load(os.path.join(RES, "fno", "fno_config.pt"), map_location="cpu", weights_only=False)
    m = FNO(n_modes=(cfg["n_modes"],), hidden_channels=cfg["hidden_channels"], in_channels=3, out_channels=1)
    m.load_state_dict(torch.load(os.path.join(RES, "fno", "fno.pt"), map_location="cpu", weights_only=False))
    m.eval()
    xt = torch.tensor(x, dtype=torch.float32); tt = torch.tensor(t, dtype=torch.float32)
    out = np.zeros((len(ics), nt, nx), np.float32)
    with torch.no_grad():
        for r in range(len(ics)):
            ic = torch.tensor(ics[r], dtype=torch.float32)
            inp = torch.stack([ic.unsqueeze(0).repeat(nt, 1),
                               tt.view(-1, 1).repeat(1, nx),
                               xt.view(1, -1).repeat(nt, 1)], dim=1)
            out[r] = m(inp).squeeze(1).numpy()
    return out

def deeponet_pred(ics):
    from hybrid_pde.solvers.ml.deepOnet.deeponet import load_model
    s = load_model(os.path.join(RES, "deeponet"), x)
    return s.predict_grid(ics, x, t)

id_idx = np.arange(900, 900 + N_ID)
ics_id = ICs[id_idx]
u_true_id = U[id_idx]

rng = np.random.default_rng(2026)
ics_ood = np.stack([ood_ic(rng, OOD_MODES) for _ in range(N_OOD)])
u_true_ood = cole_hopf(ics_ood)

def nudged(ics):
    g = rng.standard_normal(ics.shape); g /= (np.abs(g).max(axis=1, keepdims=True) + 1e-12)
    return ics + PERT_EPS * g

save = dict(x=x, t=t, te=float(d["t_train_end"]),
            u_true_id=u_true_id, ics_id=ics_id,
            u_true_ood=u_true_ood, ics_ood=ics_ood)

for label, fn in [("FNO", fno_pred), ("DeepONet", deeponet_pred)]:
    try:
        save["%s_id" % label] = fn(ics_id)
        save["%s_id_pert" % label] = fn(nudged(ics_id))
        save["%s_ood" % label] = fn(ics_ood)
        save["%s_ood_pert" % label] = fn(nudged(ics_ood))
        print("%s: ok" % label)
    except Exception as e:
        print("%s: FAILED (%s)" % (label, e))

np.savez(os.path.join(OUT, "predictions_extended.npz"), **save)
print("saved", os.path.join(OUT, "predictions_extended.npz"))
