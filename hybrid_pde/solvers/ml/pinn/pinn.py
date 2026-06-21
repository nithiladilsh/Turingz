import os
os.environ["DDE_BACKEND"] = "pytorch"
import numpy as np
import torch
import deepxde as dde

NU = 1.0 / (100 * np.pi)
T_TRAIN = 1.0
EVAL_IDS = list(range(900, 910))
ADAM_ITERS = 15000

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DATA = os.path.join(ROOT, "data", "colehopf", "burgers_colehopf.pt")
SAVE = os.path.join(ROOT, "results", "pinn")
os.makedirs(SAVE, exist_ok=True)

d = torch.load(DATA)
x_grid = d["x"].numpy()
ICs = d["ICs"].numpy()
np.save(os.path.join(SAVE, "train_ics.npy"), ICs[EVAL_IDS])

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, T_TRAIN)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def pde(x, u):
    u_t = dde.grad.jacobian(u, x, i=0, j=1)
    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    return u_t + u * u_x - NU * u_xx  # type: ignore[operator]

def make_ic_fn(ic):
    xe = np.concatenate([x_grid, [1.0]])
    ye = np.concatenate([ic, [ic[0]]])
    def f(X):
        return np.interp(X[:, 0], xe, ye)[:, None]
    return f

def train_one(i):
    ic = dde.icbc.IC(geomtime, make_ic_fn(ICs[i]), lambda _, on_initial: on_initial)
    bc = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary)
    data = dde.data.TimePDE(geomtime, pde, [ic, bc],
                            num_domain=2540, num_boundary=80, num_initial=160)
    net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal")  # type: ignore[attr-defined]
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.train(iterations=ADAM_ITERS, display_every=5000)
    model.compile("L-BFGS")
    model.train()
    torch.save(net.state_dict(), os.path.join(SAVE, f"pinn_ic{i}.pt"))

for i in EVAL_IDS:
    print(f"\n=== training PINN for IC {i} ===")
    train_one(i)

print(f"\nDone. {len(EVAL_IDS)} trained PINNs saved to {SAVE}")
