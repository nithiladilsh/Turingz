import json, os
import numpy as np
import torch
from neuralop.models import FNO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(ROOT, "data", "colehopf", "burgers_colehopf.pt")
CKPT = os.path.join(ROOT, "results", "fno")
OUT_JSON = os.path.join(CKPT, "extrapolation_analysis.json")
OUT_FIELDS = os.path.join(CKPT, "extrapolation_fields.npz")

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = torch.load(os.path.join(CKPT, "fno_config.pt"), weights_only=False)
MODES, WIDTH, N_TRAIN = cfg["n_modes"], cfg["hidden_channels"], cfg["n_train"]

d = torch.load(DATA, weights_only=False)
u = d["u"].numpy()
ICs = d["ICs"].numpy()
x = d["x"].numpy()
t = d["t"].numpy()
nx, nt = len(x), len(t)
te = float(d["t_train_end"])
N = u.shape[0]
test_idx = np.arange(N_TRAIN, N)

model = FNO(n_modes=(MODES,), hidden_channels=WIDTH, in_channels=3, out_channels=1).to(dev)
model.load_state_dict(torch.load(os.path.join(CKPT, "fno.pt"), map_location=dev, weights_only=False))
model.eval()


def predict(ic):
    S = ic.shape[0]
    ict = torch.tensor(ic, dtype=torch.float32)
    tt = torch.tensor(t, dtype=torch.float32)
    xt = torch.tensor(x, dtype=torch.float32)
    ch_ic = ict.unsqueeze(1).expand(S, nt, nx)
    ch_t = tt.view(1, nt, 1).expand(S, nt, nx)
    ch_x = xt.view(1, 1, nx).expand(S, nt, nx)
    inp = torch.stack([ch_ic, ch_t, ch_x], dim=2).reshape(S * nt, 3, nx)
    outs = []
    with torch.no_grad():
        for s in range(0, inp.shape[0], 2048):
            outs.append(model(inp[s:s + 2048].to(dev)).cpu())
    return torch.cat(outs, 0).reshape(S, nt, nx).numpy()


pred = predict(ICs[test_idx])
true = u[test_idx]

i_te = int(np.where(t <= te)[0][-1])
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
os.makedirs(CKPT, exist_ok=True)
json.dump(report, open(OUT_JSON, "w"), indent=2)
np.savez(OUT_FIELDS, x=x, t=t, t_train_end=te,
         u_true=true[0], u_pred=pred[0], ic_index=int(test_idx[0]))

print("model      : in_dist=%.3f  extrap=%.3f  extrap@T=%.3f" % (
    report["model_in_dist_mean"], report["model_extrap_mean"], report["model_extrap_final"]))
print("persistence: extrap=%.3f  (naive freeze-at-t=1 baseline)" % report["persistence_extrap_mean"])
print("beats persistence in extrap window:", report["model_extrap_mean"] < report["persistence_extrap_mean"])
print("saved", OUT_JSON, "and", OUT_FIELDS)
