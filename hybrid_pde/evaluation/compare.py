import os
os.environ.setdefault("DDE_BACKEND", "pytorch")
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import deepxde as dde
from neuralop.models import FNO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data", "colehopf", "burgers_colehopf.pt")
R = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(R, "comparison.json")
dev = "cuda" if torch.cuda.is_available() else "cpu"

d = torch.load(DATA, weights_only=False)
u = d["u"].numpy()
ICs = d["ICs"].numpy()
x = d["x"].numpy()
t = d["t"].numpy()
nx, nt = len(x), len(t)
te = float(d["t_train_end"])
inm, exm = t <= te, t > te
i_te = int(np.where(inm)[0][-1])


def grid():
    Xg, Tg = np.meshgrid(x, t, indexing="xy")
    return np.stack([Xg.ravel(), Tg.ravel()], 1).astype(np.float32)


# ---- DeepONet (DeepXDE) ----
dmeta = np.load(os.path.join(R, "deeponet_relL2", "deeponet_meta.npz"))
sidx, Tmax, nff = dmeta["sidx"], float(dmeta["Tmax"]), int(dmeta["nff"])
FF = 2.0 ** np.arange(nff)


def don_feats(pts):
    xc, tc = pts[:, 0:1], pts[:, 1:2] / Tmax
    ang = np.pi * xc * FF[None, :]
    return np.concatenate([xc, tc, np.sin(ang), np.cos(ang)], 1).astype(np.float32)


don = dde.nn.DeepONetCartesianProd([len(sidx)] + [256] * 4 + [256],
                                   [2 + 2 * nff] + [256] * 4 + [256], "relu", "Glorot normal").to(dev)
don.load_state_dict(torch.load(os.path.join(R, "deeponet_relL2", "deeponet_net.pt"),
                               map_location=dev, weights_only=False))
don.eval()
don_trunk = torch.tensor(don_feats(grid()), device=dev)


def predict_don(idx):
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 100):
            br = torch.tensor(ICs[idx[s:s + 100]][:, sidx], dtype=torch.float32, device=dev)
            out.append(np.asarray(don((br, don_trunk)).cpu()))
    return np.concatenate(out, 0).reshape(len(idx), nt, nx)


# ---- FNO (neuralop) ----
fcfg = torch.load(os.path.join(R, "fno", "fno_config.pt"), weights_only=False)
fno = FNO(n_modes=(fcfg["n_modes"],), hidden_channels=fcfg["hidden_channels"],
          in_channels=3, out_channels=1).to(dev)
fno.load_state_dict(torch.load(os.path.join(R, "fno", "fno.pt"), map_location=dev, weights_only=False))
fno.eval()
xt, tt = torch.tensor(x, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)


def predict_fno(idx):
    out = []
    with torch.no_grad():
        for i in idx:
            ic = torch.tensor(ICs[i], dtype=torch.float32)
            ch_ic = ic.view(1, 1, nx).expand(nt, 1, nx)
            ch_t = tt.view(nt, 1, 1).expand(nt, 1, nx)
            ch_x = xt.view(1, 1, nx).expand(nt, 1, nx)
            inp = torch.cat([ch_ic, ch_t, ch_x], dim=1).to(dev)
            out.append(fno(inp)[:, 0].cpu().numpy())
    return np.stack(out)


# ---- PINN (DeepXDE FNN, one per IC) ----
def predict_pinn(idx):
    pts = torch.tensor(grid(), device=dev)
    out = []
    with torch.no_grad():
        for i in idx:
            net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal").to(dev)
            net.load_state_dict(torch.load(os.path.join(R, "pinn", f"pinn_ic{i}.pt"),
                                           map_location=dev, weights_only=False))
            net.eval()
            out.append(net(pts).cpu().numpy().reshape(nt, nx))
    return np.stack(out)


def block(pred, true, mask):
    p, r = pred[:, mask], true[:, mask]
    num = np.linalg.norm((p - r).reshape(len(p), -1), axis=1)
    den = np.linalg.norm(r.reshape(len(r), -1), axis=1) + 1e-12
    return num / den


def stats(pred, true):
    ind, ext = block(pred, true, inm), block(pred, true, exm)
    return {"in_dist_mean": float(ind.mean()), "in_dist_std": float(ind.std(ddof=1) if len(ind) > 1 else 0.0),
            "extrap_mean": float(ext.mean()), "extrap_std": float(ext.std(ddof=1) if len(ext) > 1 else 0.0)}


def persistence(true):
    pers = np.repeat(true[:, i_te:i_te + 1, :], nt, axis=1)
    return float(block(pers, true, exm).mean())


def plot_fields(true2d, preds, path, title):
    names = list(preds.keys())
    fig, ax = plt.subplots(2, 1 + len(names), figsize=(3.2 * (1 + len(names)), 6))
    ax[0, 0].pcolormesh(x, t, true2d, shading="auto", cmap="viridis")
    ax[0, 0].set_title("truth"); ax[0, 0].axhline(te, color="w", ls="--", lw=1)
    ax[1, 0].axis("off")
    for j, nm in enumerate(names, 1):
        p = preds[nm]
        ax[0, j].pcolormesh(x, t, p, shading="auto", cmap="viridis")
        ax[0, j].set_title(nm + " pred"); ax[0, j].axhline(te, color="w", ls="--", lw=1)
        im = ax[1, j].pcolormesh(x, t, np.abs(p - true2d), shading="auto", cmap="magma")
        ax[1, j].set_title(nm + " |error|"); ax[1, j].axhline(te, color="c", ls="--", lw=1)
        fig.colorbar(im, ax=ax[1, j])
    for a in ax.ravel():
        if a.has_data():
            a.set_xlabel("x"); a.set_ylabel("t")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


test_idx = np.arange(900, 1000)
pinn_idx = np.arange(10)

don_te, fno_te = predict_don(test_idx), predict_fno(test_idx)
don_pi, fno_pi, pinn_pi = predict_don(pinn_idx), predict_fno(pinn_idx), predict_pinn(pinn_idx)

report = {
    "metric": "per-IC relative L2 over the space-time block (norm over t,x); mean +/- std across ICs",
    "t_train_end": te,
    "test_ICs_900_999": {
        "DeepONet": stats(don_te, u[test_idx]),
        "FNO": stats(fno_te, u[test_idx]),
        "persistence_extrap": persistence(u[test_idx]),
    },
    "pinn_ICs_0_9_same_ICs_all_methods": {
        "DeepONet": stats(don_pi, u[pinn_idx]),
        "FNO": stats(fno_pi, u[pinn_idx]),
        "PINN": stats(pinn_pi, u[pinn_idx]),
        "persistence_extrap": persistence(u[pinn_idx]),
    },
}
json.dump(report, open(OUT_JSON, "w"), indent=2)

plot_fields(u[900], {"DeepONet": don_te[0], "FNO": fno_te[0]},
            os.path.join(R, "fields_test_ic900.png"), "Test IC 900 (unseen)")
plot_fields(u[0], {"DeepONet": don_pi[0], "FNO": fno_pi[0], "PINN": pinn_pi[0]},
            os.path.join(R, "fields_ic0.png"), "IC 0 (sin pi x)")

print("== test ICs 900-999 (operators) ==")
for m in ("DeepONet", "FNO"):
    s = report["test_ICs_900_999"][m]
    print("  %-9s in_dist=%.3f extrap=%.3f" % (m, s["in_dist_mean"], s["extrap_mean"]))
print("  persistence extrap=%.3f" % report["test_ICs_900_999"]["persistence_extrap"])
print("== ICs 0-9 (all three, same ICs) ==")
for m in ("DeepONet", "FNO", "PINN"):
    s = report["pinn_ICs_0_9_same_ICs_all_methods"][m]
    print("  %-9s in_dist=%.3f extrap=%.3f" % (m, s["in_dist_mean"], s["extrap_mean"]))
print("  persistence extrap=%.3f" % report["pinn_ICs_0_9_same_ICs_all_methods"]["persistence_extrap"])
print("saved", OUT_JSON, "and field PNGs in", R)
