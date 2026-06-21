import os
os.environ["DDE_BACKEND"] = "pytorch"
import glob, json
import numpy as np
import torch
import deepxde as dde

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(ROOT, "data", "colehopf", "burgers_colehopf.pt")
CKPT = os.path.join(ROOT, "results", "pinn")
OUT_JSON = os.path.join(CKPT, "extrapolation_analysis.json")
OUT_FIELDS = os.path.join(CKPT, "extrapolation_fields.npz")

dev = "cuda" if torch.cuda.is_available() else "cpu"

d = torch.load(DATA, weights_only=False)
u = d["u"].numpy()
x, t = d["x"].numpy(), d["t"].numpy()
nx, nt = len(x), len(t)
te = float(d["t_train_end"])

n_pinns = len(sorted(glob.glob(os.path.join(CKPT, "pinn_ic*.pt"))))
ic_idx = np.arange(n_pinns)

Xg, Tg = np.meshgrid(x, t, indexing="xy")
pts = torch.tensor(np.stack([Xg.ravel(), Tg.ravel()], 1), dtype=torch.float32, device=dev)


def load_pinn(i):
    net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal").to(dev)
    net.load_state_dict(torch.load(os.path.join(CKPT, f"pinn_ic{i}.pt"), map_location=dev, weights_only=False))
    net.eval()
    return net


preds = []
for i in ic_idx:
    net = load_pinn(i)
    with torch.no_grad():
        preds.append(net(pts).cpu().numpy().reshape(nt, nx))
pred = np.stack(preds)
true = u[ic_idx]

i_te = int(np.where(t <= te)[0][-1])
pers = np.repeat(true[:, i_te:i_te + 1, :], nt, axis=1)


def err_vs_t(p):
    num = np.linalg.norm(p - true, axis=2)
    den = np.linalg.norm(true, axis=2) + 1e-12
    return (num / den).mean(axis=0)


e_model, e_pers = err_vs_t(pred), err_vs_t(pers)
inm, exm = t <= te, t > te

report = {
    "t_train_end": te, "n_pinns": int(n_pinns), "ic_indices": ic_idx.tolist(),
    "time": t.tolist(),
    "error_vs_time_model": e_model.tolist(),
    "error_vs_time_persistence_extrap_baseline": e_pers.tolist(),
    "model_in_dist_mean": float(e_model[inm].mean()),
    "model_extrap_mean": float(e_model[exm].mean()),
    "model_extrap_final": float(e_model[-1]),
    "persistence_extrap_mean": float(e_pers[exm].mean()),
    "note": "one PINN per IC, trained on t<=t_train_end; persistence freezes the solution at t_train_end (naive t>1 baseline)",
}
os.makedirs(CKPT, exist_ok=True)
json.dump(report, open(OUT_JSON, "w"), indent=2)
np.savez(OUT_FIELDS, x=x, t=t, t_train_end=te,
         u_true=true[0], u_pred=pred[0], ic_index=int(ic_idx[0]))

print("model      : in_dist=%.3f  extrap=%.3f  extrap@T=%.3f" % (
    report["model_in_dist_mean"], report["model_extrap_mean"], report["model_extrap_final"]))
print("persistence: extrap=%.3f  (naive freeze-at-t=1 baseline)" % report["persistence_extrap_mean"])
print("beats persistence in extrap window:", report["model_extrap_mean"] < report["persistence_extrap_mean"])
print("saved", OUT_JSON, "and", OUT_FIELDS)
