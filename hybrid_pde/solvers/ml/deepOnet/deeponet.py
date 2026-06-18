import os
os.environ.setdefault("DDE_BACKEND", "pytorch")
import json, sys, time
import numpy as np
import torch
import deepxde as dde

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, "..", "..", "..", ".."))
sys.path.insert(0, _ROOT)
from hybrid_pde.common import AbstractSolver, evaluate, load, split

OUT = os.path.join(_ROOT, "results", "deeponet")
M, P, W, D, NFF = 100, 256, 256, 4, 6
LR, ITERS, BATCH, PT = 1e-3, 30000, 64, 8192
SEEDS = [0, 1, 2, 3, 4]


def _grid(x, tt):
    Xg, Tg = np.meshgrid(x, tt, indexing="xy")
    return np.stack([Xg.ravel(), Tg.ravel()], 1).astype(np.float32)


class DeepONetDDE(AbstractSolver):
    name = "DeepONet(DeepXDE)"

    def __init__(self, m=M, p=P, w=W, d=D, nff=NFF, lr=LR, iterations=ITERS,
                 batch=BATCH, points=PT, seed=0):
        self.m, self.p, self.w, self.d = m, p, w, d
        self.lr, self.iterations, self.batch, self.points, self.seed = lr, iterations, batch, points, seed
        self.FF = 2.0 ** np.arange(nff)
        self.Tmax, self.sidx, self.model = None, None, None

    def _feats(self, pts):
        xc, tc = pts[:, 0:1], pts[:, 1:2] / self.Tmax
        ang = np.pi * xc * self.FF[None, :]
        return np.concatenate([xc, tc, np.sin(ang), np.cos(ang)], 1).astype(np.float32)

    def fit(self, dataset):
        U, ICs, x, t = dataset["u"], dataset["ICs"], dataset["x"], dataset["t"]
        te, self.Tmax = float(dataset["t_train_end"]), float(dataset["T"])
        train_idx, val_idx = dataset["train_idx"], dataset["val_idx"]
        nx = len(x)
        self.sidx = np.linspace(0, nx - 1, self.m).astype(int)
        tr = t <= te

        full = _grid(x, t[tr])
        y_tr_full = U[train_idx][:, tr, :].reshape(len(train_idx), -1).astype(np.float32)
        y_va_full = U[val_idx][:, tr, :].reshape(len(val_idx), -1).astype(np.float32)
        rng = np.random.default_rng(self.seed)
        sel = np.sort(rng.choice(full.shape[0], min(self.points, full.shape[0]), replace=False))
        trunk = self._feats(full[sel])
        br_tr = ICs[train_idx][:, self.sidx].astype(np.float32)
        br_va = ICs[val_idx][:, self.sidx].astype(np.float32)

        dde.config.set_random_seed(self.seed)
        data = dde.data.TripleCartesianProd((br_tr, trunk), y_tr_full[:, sel],
                                            (br_va, trunk), y_va_full[:, sel])
        net = dde.nn.DeepONetCartesianProd(
            [self.m] + [self.w] * self.d + [self.p],
            [trunk.shape[1]] + [self.w] * self.d + [self.p],
            "relu", "Glorot normal")
        self.model = dde.Model(data, net)
        self.model.compile("adam", lr=self.lr, metrics=["l2 relative error"])
        t0 = time.perf_counter()
        bs = min(self.batch, br_tr.shape[0])
        self.model.train(iterations=self.iterations, batch_size=bs,
                         display_every=max(self.iterations // 10, 1))
        return {"wall_s": time.perf_counter() - t0, "n_parameters": self.num_parameters()}

    def predict(self, ic, x, t):
        pts = np.stack([np.asarray(x).ravel(), np.asarray(t).ravel()], 1).astype(np.float32)
        br = ic[self.sidx][None, :].astype(np.float32)
        return np.asarray(self.model.predict((br, self._feats(pts)))).reshape(-1)

    def predict_grid(self, ics, x, t, chunk=100):
        trunk = self._feats(_grid(x, t))
        outs = []
        for s in range(0, len(ics), chunk):
            br = ics[s:s + chunk][:, self.sidx].astype(np.float32)
            outs.append(np.asarray(self.model.predict((br, trunk))))
        return np.concatenate(outs, 0).reshape(len(ics), len(t), len(x))

    def num_parameters(self):
        return int(sum(q.numel() for q in self.model.net.parameters())) if self.model else 0


def main(smoke=False):
    print("device:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)
    U, ICs, x, t, te, Tmax = load()
    N = U.shape[0]
    train_idx, val_idx, test_idx = split(N)

    iters, seeds = (300, [0]) if smoke else (ITERS, SEEDS)
    if smoke:
        train_idx, val_idx, test_idx = train_idx[:32], val_idx[:8], test_idx[:8]

    ds = {"u": U, "ICs": ICs, "x": x, "t": t, "t_train_end": te, "T": Tmax,
          "train_idx": train_idx, "val_idx": val_idx}

    runs, best = [], None
    for sd in seeds:
        solver = DeepONetDDE(m=M, iterations=iters, seed=sd)
        fit = solver.fit(ds)
        etr = evaluate(solver, U, ICs, x, t, te, train_idx)
        eva = evaluate(solver, U, ICs, x, t, te, val_idx)
        ete = evaluate(solver, U, ICs, x, t, te, test_idx)
        r = {"seed": sd, "wall_s": fit["wall_s"], "n_parameters": fit["n_parameters"],
             "train_in_dist": etr["in_dist_mean"], "val_in_dist": eva["in_dist_mean"],
             "test_in_dist": ete["in_dist_mean"], "test_extrap": ete["extrap_mean"]}
        runs.append(r)
        print("  seed %d  train_in=%.4f val_in=%.4f test_in=%.4f test_extrap=%.4f" %
              (sd, r["train_in_dist"], r["val_in_dist"], r["test_in_dist"], r["test_extrap"]), flush=True)
        if best is None or r["val_in_dist"] < best[0]:
            best = (r["val_in_dist"], sd)

    def agg(k):
        v = np.array([r[k] for r in runs])
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1) if len(v) > 1 else 0.0)}

    info = {"model": "DeepONet(DeepXDE)", "library": "deepxde", "n_sensors": M, "latent_dim": P,
            "width": W, "depth": D, "n_fourier": NFF, "lr": LR, "iterations": iters,
            "batch": BATCH, "points": PT, "seeds": seeds, "checkpoint_seed": best[1],
            **{k: agg(k) for k in ["train_in_dist", "val_in_dist", "test_in_dist", "test_extrap"]},
            "runs": runs}
    os.makedirs(OUT, exist_ok=True)
    json.dump(info, open(os.path.join(OUT, "training_info.json"), "w"), indent=2)
    print("test in_dist=%.4f±%.4f | test extrap=%.4f±%.4f" %
          (info["test_in_dist"]["mean"], info["test_in_dist"]["std"],
           info["test_extrap"]["mean"], info["test_extrap"]["std"]))


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
