import json, os
import numpy as np
import torch
import torch.nn as nn

CKPT = "results/deeponet"
DATA = "data/colehopf/burgers_colehopf.pt"
OUT_JSON = "results/deeponet/extrapolation_analysis.json"
OUT_FIELDS = "results/deeponet/extrapolation_fields.npz"

cfg = json.load(open(f"{CKPT}/config.json"))
meta = np.load(f"{CKPT}/meta.npz")
M, P, W, D, NFF = cfg["n_sensors"], cfg["latent_dim"], cfg["width"], cfg["depth"], cfg["n_fourier"]
Tmax = cfg["T"]
sidx, test_idx = meta["sensor_idx"], meta["test_idx"]

d = torch.load(DATA, weights_only=False, map_location="cpu")
U, ICs = d["u"].numpy(), d["ICs"].numpy()
x, t = d["x"].numpy(), d["t"].numpy()
te = float(d["t_train_end"])
nt, nx = len(t), len(x)
FF = 2.0 ** np.arange(NFF)
TRUNK_IN = 2 + 2 * NFF

def feats(z):
    xc, tc = z[:, 0:1], z[:, 1:2] / Tmax
    ang = np.pi * xc * FF[None, :]
    return np.concatenate([xc, tc, np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)

def grid(tt):
    Xg, Tg = np.meshgrid(x, tt, indexing="xy")
    return np.stack([Xg.ravel(), Tg.ravel()], 1).astype(np.float32)

def mlp(i, o):
    s = [i] + [W] * D + [o]; layers = []
    for a, b in zip(s[:-1], s[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])

class DeepONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch, self.trunk = mlp(M, P), mlp(TRUNK_IN, P)
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, b, tr):
        return self.branch(b) @ self.trunk(tr).T + self.bias

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = DeepONet().to(dev)
model.load_state_dict(torch.load(f"{CKPT}/model.pt", map_location=dev))
model.eval()

T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
branch = T(ICs[test_idx][:, sidx])
g = T(feats(grid(t)))
with torch.no_grad():
    pred = model(branch, g).cpu().numpy().reshape(len(test_idx), nt, nx)
true = U[test_idx]

i_te = np.where(t <= te)[0][-1]
pers = np.repeat(true[:, i_te:i_te + 1, :], nt, axis=1)

def err_vs_t(p):
    num = np.linalg.norm(p - true, axis=2)
    den = np.linalg.norm(true, axis=2) + 1e-12
    return (num / den).mean(axis=0)

e_model, e_pers = err_vs_t(pred), err_vs_t(pers)
inm, exm = t <= te, t > te

report = {
    "t_train_end": te, "n_test": int(len(test_idx)),
    "time": t.tolist(),
    "error_vs_time_model": e_model.tolist(),
    "error_vs_time_persistence_extrap_baseline": e_pers.tolist(),
    "model_in_dist_mean": float(e_model[inm].mean()),
    "model_extrap_mean": float(e_model[exm].mean()),
    "model_extrap_final": float(e_model[-1]),
    "persistence_extrap_mean": float(e_pers[exm].mean()),
    "note": "persistence freezes the solution at t_train_end; it is the naive baseline for t>1 only",
}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump(report, open(OUT_JSON, "w"), indent=2)
np.savez(OUT_FIELDS, x=x, t=t, t_train_end=te,
         u_true=true[0], u_pred=pred[0], ic_index=int(test_idx[0]))

print("model      : in_dist=%.3f  extrap=%.3f  extrap@T=%.3f" % (
    report["model_in_dist_mean"], report["model_extrap_mean"], report["model_extrap_final"]))
print("persistence: extrap=%.3f  (naive freeze-at-t=1 baseline)" % report["persistence_extrap_mean"])
print("beats persistence in extrap window:", report["model_extrap_mean"] < report["persistence_extrap_mean"])
print("saved", OUT_JSON, "and", OUT_FIELDS)
