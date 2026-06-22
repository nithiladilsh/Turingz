import os, json, sys, traceback
os.environ.setdefault("DDE_BACKEND", "pytorch")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "colehopf", "burgers_colehopf.pt")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "eval")
os.makedirs(OUT, exist_ok=True)

d = torch.load(DATA, weights_only=False, map_location="cpu")
U = d["u"].numpy()
ICs, x, t = d["ICs"], d["x"], d["t"]
te = float(d["t_train_end"])
nx = U.shape[-1]
EVAL = np.arange(900, 910)
SEEN_OP = np.arange(10)

def fno_pred(idx):
    from neuralop.models import FNO
    torch.set_default_device("cpu")
    cfg = torch.load(os.path.join(RES, "fno", "fno_config.pt"), map_location="cpu", weights_only=False)
    m = FNO(n_modes=(cfg["n_modes"],), hidden_channels=cfg["hidden_channels"], in_channels=3, out_channels=1)
    m.load_state_dict(torch.load(os.path.join(RES, "fno", "fno.pt"), map_location="cpu", weights_only=False))
    m.eval()
    out = np.zeros((len(idx), len(t), nx), np.float32)
    with torch.no_grad():
        for r, s in enumerate(idx):
            inp = torch.stack([ICs[s].unsqueeze(0).repeat(len(t), 1),
                               t.view(-1, 1).repeat(1, nx),
                               x.view(1, -1).repeat(len(t), 1)], dim=1)
            out[r] = m(inp).squeeze(1).numpy()
    return out

def pinn_pred(idx):
    import deepxde as dde
    torch.set_default_device("cpu")
    X, T = np.meshgrid(x.numpy(), t.numpy())
    XT = torch.tensor(np.stack([X.ravel(), T.ravel()], 1), dtype=torch.float32)
    out = np.zeros((len(idx), len(t), nx), np.float32)
    for r, i in enumerate(idx):
        net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal")
        net.load_state_dict(torch.load(os.path.join(RES, "pinn", f"pinn_ic{i}.pt"), map_location="cpu"))
        net.eval()
        with torch.no_grad():
            out[r] = net(XT).numpy().reshape(len(t), nx)
    return out

def deeponet_pred(idx):
    from hybrid_pde.solvers.ml.deepOnet.deeponet import load_model
    s = load_model(os.path.join(RES, "deeponet"), x.numpy())
    return s.predict_grid(ICs.numpy()[idx], x.numpy(), t.numpy())

preds = {"u_true_eval": U[EVAL], "u_true_seen": U[SEEN_OP]}
jobs = [("PINN", pinn_pred, EVAL),
        ("FNO_eval", fno_pred, EVAL), ("FNO_seen", fno_pred, SEEN_OP),
        ("DeepONet_eval", deeponet_pred, EVAL), ("DeepONet_seen", deeponet_pred, SEEN_OP)]
for name, fn, idx in jobs:
    try:
        preds[name] = fn(idx)
        print(f"{name}: ok  shape {preds[name].shape}")
    except Exception:
        print(f"{name}: FAILED")
        traceback.print_exc()

np.savez(os.path.join(OUT, "predictions.npz"), x=x.numpy(), t=t.numpy(), te=te, **preds)
print(f"\nSaved predictions to {OUT}")
