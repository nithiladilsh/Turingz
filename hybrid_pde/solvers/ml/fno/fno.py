import os, json, sys, time
import numpy as np
import torch
from neuralop.models import FNO

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, "..", "..", "..", ".."))
sys.path.insert(0, _ROOT)
from hybrid_pde.common import AbstractSolver, evaluate, load, split

OUT = os.path.join(_ROOT, "results", "fno")
MODES_T, MODES_X = 16, 16
WIDTH, LAYERS = 32, 4
LR, WD = 1e-3, 1e-4
EPOCHS, BATCH = 500, 20
SEEDS = [0, 1, 2, 3, 4]

dev = "cuda" if torch.cuda.is_available() else "cpu"


def _build_input(ic, x, t_block, Tmax):
    B, nx = ic.shape
    K = t_block.shape[0]
    ic_ch = ic[:, None, None, :].expand(B, 1, K, nx)
    x_ch = x[None, None, None, :].expand(B, 1, K, nx)
    t_ch = (t_block / Tmax)[None, None, :, None].expand(B, 1, K, nx)
    return torch.cat([ic_ch, x_ch, t_ch], dim=1).contiguous()


def _interp_time(full_u, full_t, t_target):
    idx = torch.searchsorted(full_t, t_target).clamp(1, full_t.shape[0] - 1)
    t0, t1 = full_t[idx - 1], full_t[idx]
    w = ((t_target - t0) / (t1 - t0).clamp_min(1e-12)).clamp(0, 1)
    u0, u1 = full_u[:, idx - 1, :], full_u[:, idx, :]
    return u0 + (u1 - u0) * w[None, :, None]


class FNONeural(AbstractSolver):
    name = "FNO(neuralop)"

    def __init__(self, mt=MODES_T, mx=MODES_X, width=WIDTH, layers=LAYERS,
                 lr=LR, wd=WD, epochs=EPOCHS, batch=BATCH, seed=0):
        self.mt, self.mx, self.width, self.layers = mt, mx, width, layers
        self.lr, self.wd, self.epochs, self.batch, self.seed = lr, wd, epochs, batch, seed
        self.model, self.norm, self.x, self.t_block, self.Tmax = None, None, None, None, None

    def fit(self, dataset):
        U, ICs, x, t = dataset["u"], dataset["ICs"], dataset["x"], dataset["t"]
        te, self.Tmax = float(dataset["t_train_end"]), float(dataset["T"])
        train_idx = dataset["train_idx"]
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        tr = t <= te
        self.x = torch.tensor(x, dtype=torch.float32, device=dev)
        self.t_block = torch.tensor(t[tr], dtype=torch.float32, device=dev)
        utr = U[train_idx][:, tr, :]
        mean, std = float(utr.mean()), float(utr.std()) + 1e-12
        self.norm = (mean, std)
        ic_tr = torch.tensor(ICs[train_idx], dtype=torch.float32, device=dev)
        y_tr = torch.tensor((utr - mean) / std, dtype=torch.float32, device=dev)

        self.model = FNO(n_modes=(self.mt, self.mx), in_channels=3, out_channels=1,
                         hidden_channels=self.width, n_layers=self.layers).to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        n = len(train_idx)
        t0 = time.perf_counter()
        for ep in range(self.epochs):
            self.model.train()
            order = rng.permutation(n)
            run = 0.0
            for s in range(0, n, self.batch):
                bi = order[s:s + self.batch]
                pred = self.model(_build_input(ic_tr[bi], self.x, self.t_block, self.Tmax))[:, 0]
                yb = y_tr[bi]
                loss = (torch.linalg.norm((pred - yb).reshape(len(bi), -1), dim=1) /
                        torch.linalg.norm(yb.reshape(len(bi), -1), dim=1).clamp_min(1e-12)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                run += float(loss) * len(bi)
            sched.step()
            if ep % max(self.epochs // 10, 1) == 0 or ep == self.epochs - 1:
                print(f"  seed {self.seed}  epoch {ep+1}/{self.epochs}  loss {run/n:.4e}", flush=True)
        return {"wall_s": time.perf_counter() - t0, "n_parameters": self.num_parameters()}

    def predict_grid(self, ics, x, t, chunk=100):
        mean, std = self.norm
        Tblk = float(self.t_block[-1])
        tt = torch.tensor(t, dtype=torch.float32, device=dev)
        n_blocks = max(int(np.ceil(float(t.max()) / Tblk - 1e-9)), 1)
        self.model.eval()
        out_all = []
        for s in range(0, len(ics), chunk):
            cur = torch.tensor(ics[s:s + chunk], dtype=torch.float32, device=dev)
            abs_t, slabs = [], []
            with torch.no_grad():
                for b in range(n_blocks):
                    o = self.model(_build_input(cur, self.x, self.t_block, self.Tmax))[:, 0] * std + mean
                    tb = self.t_block + b * Tblk
                    if b == 0:
                        abs_t.append(tb); slabs.append(o)
                    else:
                        abs_t.append(tb[1:]); slabs.append(o[:, 1:])
                    cur = o[:, -1, :]
            grid = _interp_time(torch.cat(slabs, 1), torch.cat(abs_t), tt)
            out_all.append(grid.cpu().numpy())
        return np.concatenate(out_all, 0)

    def num_parameters(self):
        return int(sum(p.numel() for p in self.model.parameters())) if self.model else 0


def main(smoke=False):
    print("device:", dev, flush=True)
    U, ICs, x, t, te, Tmax = load()
    N = U.shape[0]
    train_idx, val_idx, test_idx = split(N)

    epochs, batch, seeds = (EPOCHS, BATCH, SEEDS)
    if smoke:
        epochs, batch, seeds = 4, 8, [0]
        train_idx, val_idx, test_idx = train_idx[:32], val_idx[:8], test_idx[:8]

    ds = {"u": U, "ICs": ICs, "x": x, "t": t, "t_train_end": te, "T": Tmax, "train_idx": train_idx}

    runs, best = [], None
    for sd in seeds:
        solver = FNONeural(epochs=epochs, batch=batch, seed=sd)
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

    info = {"model": "FNO(neuralop)", "library": "neuraloperator", "n_modes_t": MODES_T,
            "n_modes_x": MODES_X, "width": WIDTH, "n_layers": LAYERS, "lr": LR, "weight_decay": WD,
            "epochs": epochs, "batch": batch, "seeds": seeds, "checkpoint_seed": best[1],
            **{k: agg(k) for k in ["train_in_dist", "val_in_dist", "test_in_dist", "test_extrap"]},
            "runs": runs}
    os.makedirs(OUT, exist_ok=True)
    json.dump(info, open(os.path.join(OUT, "training_info.json"), "w"), indent=2)
    print("test in_dist=%.4f±%.4f | test extrap=%.4f±%.4f" %
          (info["test_in_dist"]["mean"], info["test_in_dist"]["std"],
           info["test_extrap"]["mean"], info["test_extrap"]["std"]))


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
