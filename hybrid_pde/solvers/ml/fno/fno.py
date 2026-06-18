import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from neuralop.models import FNO

N_TRAIN = 900
EPOCHS = 50
BATCH = 64
LR = 1e-3
MODES = 16
WIDTH = 64

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DATA = os.path.join(ROOT, "data", "colehopf", "burgers_colehopf.pt")
SAVE = os.path.join(ROOT, "results", "fno")
os.makedirs(SAVE, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

d = torch.load(DATA)
u = d["u"]
ICs = d["ICs"]
x = d["x"]
t = d["t"]
nx = u.shape[-1]
t_train_end = float(d["t_train_end"])
K = int((t <= t_train_end).sum())

def build(ic, sol):
    S = ic.shape[0]
    times = t[:K]
    ch_ic = ic.unsqueeze(1).expand(S, K, nx)
    ch_t = times.view(1, K, 1).expand(S, K, nx)
    ch_x = x.view(1, 1, nx).expand(S, K, nx)
    inp = torch.stack([ch_ic, ch_t, ch_x], dim=2).reshape(S * K, 3, nx)
    out = sol[:, :K, :].reshape(S * K, 1, nx)
    return inp, out

Xtr, Ytr = build(ICs[:N_TRAIN], u[:N_TRAIN])
loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=BATCH, shuffle=True)

model = FNO(n_modes=(MODES,), hidden_channels=WIDTH, in_channels=3, out_channels=1).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

def rel_l2(pred, target):
    return (torch.norm(pred - target, dim=-1) / torch.norm(target, dim=-1)).mean()

for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = rel_l2(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * xb.shape[0]
    sched.step()
    print(f"epoch {epoch + 1}/{EPOCHS}  train rel-L2 {total / len(Xtr):.4f}")

torch.save(model.state_dict(), os.path.join(SAVE, "fno.pt"))
torch.save({"n_modes": MODES, "hidden_channels": WIDTH, "in_channels": 3,
            "channels": "ic,t,x", "n_train": N_TRAIN, "t_train_end": t_train_end},
           os.path.join(SAVE, "fno_config.pt"))
print(f"\nDone. FNO operator saved to {SAVE}")
