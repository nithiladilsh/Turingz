import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, "..", "..", "..", ".."))
DATA = os.path.join(_ROOT, "data", "colehopf", "burgers_colehopf.pt")
OUT = os.path.join(_ROOT, "results", "fno")

MODES_T, MODES_X = 16, 16
WIDTH, LAYERS = 32, 4
LR, WD = 1e-3, 1e-4
EPOCHS, BATCH = 500, 20
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN, N_VAL = 800, 100

dev = "cuda" if torch.cuda.is_available() else "cpu"


class SpectralConv2d(nn.Module):
    def __init__(self, in_c, out_c, m1, m2):
        super().__init__()
        self.in_c, self.out_c, self.m1, self.m2 = in_c, out_c, m1, m2
        s = 1.0 / (in_c * out_c)
        self.w1 = nn.Parameter(s * torch.rand(in_c, out_c, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(s * torch.rand(in_c, out_c, m1, m2, dtype=torch.cfloat))

    @staticmethod
    def _mul(inp, w):
        return torch.einsum("bixy,ioxy->boxy", inp, w)

    def forward(self, x):
        B, _, H, W = x.shape
        m1, m2 = min(self.m1, H // 2), min(self.m2, W // 2 + 1)
        x_ft = torch.fft.rfft2(x)
        out = torch.zeros(B, self.out_c, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :m1, :m2] = self._mul(x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out[:, :, -m1:, :m2] = self._mul(x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(H, W))


class FNO2d(nn.Module):
    def __init__(self, m1, m2, width, layers, in_ch=3, out_ch=1):
        super().__init__()
        self.fc0 = nn.Linear(in_ch, width)
        self.convs = nn.ModuleList([SpectralConv2d(width, width, m1, m2) for _ in range(layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_ch)

    def forward(self, x):
        x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for conv, w in zip(self.convs, self.ws):
            x = F.gelu(conv(x) + w(x))
        x = F.gelu(self.fc1(x.permute(0, 2, 3, 1)))
        return self.fc2(x).permute(0, 3, 1, 2)


def load_data():
    d = torch.load(DATA, weights_only=False, map_location="cpu")
    U = d["u"].float()
    ICs = d["ICs"].float() if "ICs" in d else U[:, 0, :].clone()
    x = d["x"].float()
    t = d["t"].float()
    te = float(d["t_train_end"])
    Tmax = float(d["T"]) if "T" in d else float(t.max())
    return U, ICs, x, t, te, Tmax


def build_input(ic, x, t_block, Tmax):
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


def predict_full(model, ic, x, t_block, t_target, Tmax, mean, std):
    Tblk = float(t_block[-1])
    n_blocks = max(int(np.ceil(float(t_target.max()) / Tblk - 1e-9)), 1)
    cur, abs_t, slabs = ic, [], []
    model.eval()
    with torch.no_grad():
        for b in range(n_blocks):
            out = model(build_input(cur, x, t_block, Tmax))[:, 0] * std + mean
            tb = t_block + b * Tblk
            if b == 0:
                abs_t.append(tb); slabs.append(out)
            else:
                abs_t.append(tb[1:]); slabs.append(out[:, 1:])
            cur = out[:, -1, :]
    return _interp_time(torch.cat(slabs, 1), torch.cat(abs_t), t_target)


def rel_l2_masked(model, ICs, U, x, t, t_block, mask, idx, Tmax, mean, std, chunk=40):
    errs = []
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        pred = predict_full(model, ICs[ii].to(dev), x, t_block, t, Tmax, mean, std)
        ref = U[ii].to(dev)
        e = (torch.linalg.norm((pred - ref)[:, mask], dim=(1, 2)) /
             torch.linalg.norm(ref[:, mask], dim=(1, 2)).clamp_min(1e-12))
        errs.append(e.cpu())
    return float(torch.cat(errs).mean())


def train_one(seed, U, ICs, x, t, te, Tmax, train_idx):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    tr_mask = t <= te
    t_block = t[tr_mask].to(dev)
    x_d = x.to(dev)
    mean, std = float(U[train_idx][:, tr_mask].mean()), float(U[train_idx][:, tr_mask].std()) + 1e-12

    ic_tr = ICs[train_idx].to(dev)
    y_tr = ((U[train_idx][:, tr_mask, :] - mean) / std).to(dev)

    model = FNO2d(MODES_T, MODES_X, WIDTH, LAYERS).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    n = len(train_idx)
    final = 0.0
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(n)
        running = 0.0
        for s in range(0, n, BATCH):
            bi = order[s:s + BATCH]
            pred = model(build_input(ic_tr[bi], x_d, t_block, Tmax))[:, 0]
            yb = y_tr[bi]
            loss = (torch.linalg.norm((pred - yb).reshape(len(bi), -1), dim=1) /
                    torch.linalg.norm(yb.reshape(len(bi), -1), dim=1).clamp_min(1e-12)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss) * len(bi)
        sched.step()
        final = running / n
        if ep % max(EPOCHS // 10, 1) == 0 or ep == EPOCHS - 1:
            print(f"  seed {seed}  epoch {ep+1}/{EPOCHS}  loss {final:.4e}", flush=True)
    return model, final, (mean, std)


def main(smoke=False):
    global MODES_T, MODES_X, WIDTH, LAYERS, EPOCHS, BATCH, SEEDS, N_TRAIN, N_VAL
    print("device:", dev, flush=True)
    U, ICs, x, t, te, Tmax = load_data()
    N = U.shape[0]

    if smoke:
        MODES_T = MODES_X = 6; WIDTH, LAYERS = 12, 2; EPOCHS, BATCH = 4, 8
        SEEDS = [0]; N_TRAIN, N_VAL = 16, 8
        U, ICs = U[:32], ICs[:32]; N = 32

    tr_mask, ex_mask = t <= te, t > te
    train_idx = np.arange(N_TRAIN)
    val_idx = np.arange(N_TRAIN, N_TRAIN + N_VAL)
    test_idx = np.arange(N_TRAIN + N_VAL, N)
    t_block = t[tr_mask].to(dev)

    runs, best = [], None
    for sd in SEEDS:
        t0 = time.perf_counter()
        model, fl, (mean, std) = train_one(sd, U, ICs, x, t, te, Tmax, train_idx)
        m = lambda mask, idx: rel_l2_masked(model, ICs, U, x.to(dev), t.to(dev),
                                            t_block, mask, idx, Tmax, mean, std)
        r = {"seed": sd, "final_train_loss": fl, "wall_s": time.perf_counter() - t0,
             "train_in_dist": m(tr_mask, train_idx), "val_in_dist": m(tr_mask, val_idx),
             "test_in_dist": m(tr_mask, test_idx), "train_extrap": m(ex_mask, train_idx),
             "test_extrap": m(ex_mask, test_idx)}
        runs.append(r)
        print("  -> seed %d  train_in=%.4f val_in=%.4f test_in=%.4f test_extrap=%.4f" %
              (sd, r["train_in_dist"], r["val_in_dist"], r["test_in_dist"], r["test_extrap"]), flush=True)
        if best is None or r["val_in_dist"] < best[0]:
            best = (r["val_in_dist"], model, sd, (mean, std))

    def agg(k):
        v = np.array([r[k] for r in runs])
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1) if len(v) > 1 else 0.0)}

    config = {"model": "FNO2d", "n_modes_t": MODES_T, "n_modes_x": MODES_X,
              "width": WIDTH, "n_layers": LAYERS, "lr": LR, "weight_decay": WD,
              "epochs": EPOCHS, "batch": BATCH, "seeds": SEEDS,
              "n_train": N_TRAIN, "n_val": N_VAL, "t_train_end": te, "T": Tmax}
    info = {**{k: agg(k) for k in ["train_in_dist", "val_in_dist", "test_in_dist",
                                   "train_extrap", "test_extrap"]},
            "seeds": SEEDS, "n_train": N_TRAIN, "n_val": N_VAL, "n_test": int(len(test_idx)),
            "t_train_end": te, "checkpoint_seed": best[2], "runs": runs}

    os.makedirs(OUT, exist_ok=True)
    torch.save({"state_dict": best[1].state_dict(), "norm": best[3], "config": config},
               os.path.join(OUT, "model.pt"))
    np.savez(os.path.join(OUT, "meta.npz"), x=x.numpy(), t=t.numpy(),
             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    json.dump(config, open(os.path.join(OUT, "config.json"), "w"), indent=2)
    json.dump(info, open(os.path.join(OUT, "training_info.json"), "w"), indent=2)
    print("test in_dist=%.4f±%.4f | test extrap=%.4f±%.4f" %
          (info["test_in_dist"]["mean"], info["test_in_dist"]["std"],
           info["test_extrap"]["mean"], info["test_extrap"]["std"]))


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
