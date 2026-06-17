import json, os
import numpy as np
import torch
import torch.nn as nn

DATA = "data/colehopf/burgers_colehopf.pt"
OUT = "results/deeponet/sensor_sweep.json"
SENSORS = [64, 100, 128, 256]
SEEDS = [0, 1, 2, 3, 4]
P, W, D, NFF = 256, 256, 4, 6
LR, ITERS, BATCH, PT = 1e-3, 4000, 64, 8192
N_TRAIN, N_VAL = 800, 100
ACT = nn.ReLU

dev = "cuda" if torch.cuda.is_available() else "cpu"

d = torch.load(DATA, weights_only=False, map_location="cpu")
U, ICs = d["u"].numpy(), d["ICs"].numpy()
x, t = d["x"].numpy(), d["t"].numpy()
te, Tmax = float(d["t_train_end"]), float(d["T"])
N, nt, nx = U.shape
tr_mask, ex_mask = t <= te, t > te
train_idx = np.arange(N_TRAIN)
val_idx = np.arange(N_TRAIN, N_TRAIN + N_VAL)
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
        layers += [nn.Linear(a, b), ACT()]
    return nn.Sequential(*layers[:-1])


class DeepONet(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.branch, self.trunk = mlp(m, P), mlp(TRUNK_IN, P)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, b, tr):
        return self.branch(b) @ self.trunk(tr).T + self.bias


T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
trunk_tr = T(feats(grid(t[tr_mask])))
y_tr = T(U[:, tr_mask, :].reshape(N, -1))
M_tr = trunk_tr.shape[0]


def rel_l2(model, branch, mask, idx):
    g = T(feats(grid(t[mask])))
    y = T(U[idx][:, mask, :].reshape(len(idx), -1))
    with torch.no_grad():
        e = torch.linalg.norm(model(branch[idx], g) - y, dim=1) / torch.linalg.norm(y, dim=1)
    return float(e.mean())


def run(m, seed):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    sidx = np.linspace(0, nx - 1, m).astype(int)
    branch = T(ICs[:, sidx])
    model = DeepONet(m).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(ITERS):
        bi = rng.choice(train_idx, BATCH, replace=False)
        pj = rng.choice(M_tr, PT, replace=False)
        pred = model(branch[bi], trunk_tr[pj])
        yb = y_tr[bi][:, pj]
        loss = (torch.linalg.norm(pred - yb, dim=1) / torch.linalg.norm(yb, dim=1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return rel_l2(model, branch, tr_mask, val_idx), rel_l2(model, branch, ex_mask, val_idx)


summary = []
for m in SENSORS:
    r = np.array([run(m, s) for s in SEEDS])
    iv, ev = r[:, 0], r[:, 1]
    summary.append({"n_sensors": m,
                    "val_in_dist_mean": float(iv.mean()), "val_in_dist_std": float(iv.std(ddof=1)),
                    "val_extrap_mean": float(ev.mean()), "val_extrap_std": float(ev.std(ddof=1)),
                    "val_in_dist_per_seed": iv.tolist(), "val_extrap_per_seed": ev.tolist()})

means = np.array([s["val_in_dist_mean"] for s in summary])
best = summary[int(means.argmin())]
thr = best["val_in_dist_mean"] + best["val_in_dist_std"]
rec = next(s["n_sensors"] for s in sorted(summary, key=lambda s: s["n_sensors"]) if s["val_in_dist_mean"] <= thr)

out = {"sensors": SENSORS, "seeds": SEEDS, "iterations": ITERS, "n_fourier": NFF, "per_m": summary,
       "best_n_sensors": best["n_sensors"], "recommended_n_sensors": rec,
       "rule": "smallest m with mean validation in-distribution rel L2 <= best_mean + best_std"}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=2)
for s in summary:
    print("m=%3d  val_in_dist=%.4f±%.4f  val_extrap=%.4f±%.4f%s" % (
        s["n_sensors"], s["val_in_dist_mean"], s["val_in_dist_std"],
        s["val_extrap_mean"], s["val_extrap_std"],
        "  <- recommended" if s["n_sensors"] == rec else ""))
