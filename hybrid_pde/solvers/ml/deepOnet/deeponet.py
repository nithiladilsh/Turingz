import json, os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn as nn

DATA = "data/colehopf/burgers_colehopf.pt"
OUT = "results/deeponet"
M, P, W, D, NFF = 128, 256, 256, 4, 6
LR, ITERS, BATCH, PT = 1e-3, 30000, 64, 8192
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN, N_VAL = 800, 100
ACT = nn.ReLU
DETERMINISTIC = False

if DETERMINISTIC:
    torch.use_deterministic_algorithms(True, warn_only=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, (torch.cuda.get_device_name(0) if dev == "cuda" else ""), flush=True)

d = torch.load(DATA, weights_only=False, map_location="cpu")
U, ICs = d["u"].numpy(), d["ICs"].numpy()
x, t = d["x"].numpy(), d["t"].numpy()
te, Tmax = float(d["t_train_end"]), float(d["T"])
N, nt, nx = U.shape
sidx = np.linspace(0, nx - 1, M).astype(int)
tr_mask, ex_mask = t <= te, t > te
train_idx = np.arange(N_TRAIN)
val_idx = np.arange(N_TRAIN, N_TRAIN + N_VAL)
test_idx = np.arange(N_TRAIN + N_VAL, N)
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
    def __init__(self):
        super().__init__()
        self.branch, self.trunk = mlp(M, P), mlp(TRUNK_IN, P)
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, b, tr):
        return self.branch(b) @ self.trunk(tr).T + self.bias

T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
branch = T(ICs[:, sidx])
trunk_tr = T(feats(grid(t[tr_mask])))
y_tr = T(U[:, tr_mask, :].reshape(N, -1))
M_tr = trunk_tr.shape[0]

def rel_l2(model, mask, idx):
    g = T(feats(grid(t[mask])))
    y = T(U[idx][:, mask, :].reshape(len(idx), -1))
    with torch.no_grad():
        e = torch.linalg.norm(model(branch[idx], g) - y, dim=1) / torch.linalg.norm(y, dim=1)
    return float(e.mean())

def train_one(seed):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    model = DeepONet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for it in range(ITERS):
        bi = rng.choice(train_idx, BATCH, replace=False)
        pj = rng.choice(M_tr, PT, replace=False)
        pred = model(branch[bi], trunk_tr[pj])
        yb = y_tr[bi][:, pj]
        loss = (torch.linalg.norm(pred - yb, dim=1) / torch.linalg.norm(yb, dim=1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 2000 == 0 or it == ITERS - 1:
            print(f"  seed {seed}  iter {it+1}/{ITERS}  loss {float(loss):.4e}", flush=True)
    return model, float(loss)

runs, best = [], None
for s in SEEDS:
    model, fl = train_one(s)
    r = {"seed": s, "final_train_loss": fl,
         "train_in_dist": rel_l2(model, tr_mask, train_idx),
         "val_in_dist": rel_l2(model, tr_mask, val_idx),
         "test_in_dist": rel_l2(model, tr_mask, test_idx),
         "train_extrap": rel_l2(model, ex_mask, train_idx),
         "test_extrap": rel_l2(model, ex_mask, test_idx)}
    runs.append(r)
    print("  -> seed %d  train_in_dist=%.4f  val_in_dist=%.4f  test_in_dist=%.4f  test_extrap=%.4f" % (
        s, r["train_in_dist"], r["val_in_dist"], r["test_in_dist"], r["test_extrap"]), flush=True)
    if best is None or r["val_in_dist"] < best[0]:
        best = (r["val_in_dist"], model, s)

def agg(key):
    v = np.array([r[key] for r in runs])
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1))}

info = {"seeds": SEEDS, "n_train": N_TRAIN, "n_val": N_VAL, "n_test": int(len(test_idx)),
        "t_train_end": te,
        "train_in_dist": agg("train_in_dist"), "val_in_dist": agg("val_in_dist"),
        "test_in_dist": agg("test_in_dist"),
        "train_extrap": agg("train_extrap"), "test_extrap": agg("test_extrap"),
        "checkpoint_seed": best[2], "runs": runs}
config = {"n_sensors": M, "latent_dim": P, "width": W, "depth": D, "n_fourier": NFF,
          "activation": "relu", "lr": LR, "iterations": ITERS, "batch": BATCH, "points": PT,
          "seeds": SEEDS, "n_train": N_TRAIN, "n_val": N_VAL, "T": Tmax}

os.makedirs(OUT, exist_ok=True)
torch.save(best[1].state_dict(), f"{OUT}/model.pt")
np.savez(f"{OUT}/meta.npz", sensor_idx=sidx, x=x, t=t,
         train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
json.dump(config, open(f"{OUT}/config.json", "w"), indent=2)
json.dump(info, open(f"{OUT}/training_info.json", "w"), indent=2)
print("test in_dist=%.4f±%.4f | test extrap=%.4f±%.4f" % (
    info["test_in_dist"]["mean"], info["test_in_dist"]["std"],
    info["test_extrap"]["mean"], info["test_extrap"]["std"]))
