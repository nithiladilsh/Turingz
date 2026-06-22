import os, json, sys
os.environ.setdefault("DDE_BACKEND", "pytorch")
import numpy as np
import torch
import deepxde as dde

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from hybrid_pde.common import load, split, evaluate

OUT = os.path.join(ROOT, "results", "deeponet")
SWEEP = os.path.join(OUT, "sensor_sweep.json")
M = json.load(open(SWEEP))["recommended_n_sensors"] if os.path.exists(SWEEP) else 100
P, W, D, NFF = 256, 256, 4, 6
LR, ITERS, BATCH, PT = 1e-3, 30000, 64, 8192

def feats(pts, Tmax):
    xc, tc = pts[:, 0:1], pts[:, 1:2] / Tmax
    ang = np.pi * xc * (2.0 ** np.arange(NFF))[None, :]
    return np.concatenate([xc, tc, np.sin(ang), np.cos(ang)], 1).astype(np.float32)

def grid(x, t):
    Xg, Tg = np.meshgrid(x, t, indexing="xy")
    return np.stack([Xg.ravel(), Tg.ravel()], 1).astype(np.float32)

def build_net(m):
    return dde.nn.DeepONetCartesianProd(
        [m] + [W] * D + [P], [2 + 2 * NFF] + [W] * D + [P], "relu", "Glorot normal")

class DeepONet:
    def __init__(self, m, Tmax, x):
        self.m, self.Tmax = m, Tmax
        self.sidx = np.linspace(0, len(x) - 1, m).astype(int)
        self.net = None

    def fit(self, U, ICs, x, t, te, train_idx, val_idx, iterations=ITERS, seed=0):
        tr = t <= te
        trunk = feats(grid(x, t[tr]), self.Tmax)
        y_tr = U[train_idx][:, tr, :].reshape(len(train_idx), -1).astype(np.float32)
        y_va = U[val_idx][:, tr, :].reshape(len(val_idx), -1).astype(np.float32)
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(trunk.shape[0], min(PT, trunk.shape[0]), replace=False))
        br_tr = ICs[train_idx][:, self.sidx].astype(np.float32)
        br_va = ICs[val_idx][:, self.sidx].astype(np.float32)
        dde.config.set_random_seed(seed)
        data = dde.data.TripleCartesianProd((br_tr, trunk[sel]), y_tr[:, sel],
                                            (br_va, trunk[sel]), y_va[:, sel])
        model = dde.Model(data, build_net(self.m))
        model.compile("adam", lr=LR, metrics=["l2 relative error"])
        model.train(iterations=iterations, batch_size=min(BATCH, len(train_idx)),
                    display_every=max(iterations // 10, 1))
        self.net = model.net

    def predict_grid(self, ICs, x, t, chunk=200):
        trunk = torch.tensor(feats(grid(x, t), self.Tmax))
        outs = []
        with torch.no_grad():
            for s in range(0, len(ICs), chunk):
                br = torch.tensor(ICs[s:s + chunk][:, self.sidx].astype(np.float32))
                outs.append(self.net((br, trunk)).cpu().numpy())
        return np.concatenate(outs, 0).reshape(len(ICs), len(t), len(x))

    def num_parameters(self):
        return int(sum(p.numel() for p in self.net.parameters()))

    def save(self, out):
        os.makedirs(out, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(out, "model.pt"))
        json.dump({"m": self.m, "Tmax": self.Tmax, "P": P, "W": W, "D": D, "NFF": NFF},
                  open(os.path.join(out, "config.json"), "w"), indent=2)

def load_model(out, x):
    cfg = json.load(open(os.path.join(out, "config.json")))
    s = DeepONet(cfg["m"], cfg["Tmax"], x)
    net = build_net(cfg["m"])
    net.load_state_dict(torch.load(os.path.join(out, "model.pt"), map_location="cpu", weights_only=False))
    net.eval()
    s.net = net
    return s

def main():
    U, ICs, x, t, te, Tmax = load()
    tr_idx, va_idx, te_idx = split(U.shape[0])
    s = DeepONet(M, Tmax, x)
    s.fit(U, ICs, x, t, te, tr_idx, va_idx)
    s.save(OUT)
    for name, idx in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        e = evaluate(s, U, ICs, x, t, te, idx)
        print(f"{name:5s}  in-window {e['in_dist_mean']*100:6.2f}%   extrapolation {e['extrap_mean']*100:6.2f}%")
    print(f"saved model.pt + config.json (m={M}) to {OUT}")

if __name__ == "__main__":
    main()
