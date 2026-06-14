import os
import numpy as np
import torch

nu = 1.0 / (100 * np.pi)
L, nx = 2.0, 512
T, nt = 2.0, 200
N_SAMPLES = 1000
OUT = "data/colehopf/burgers_colehopf.pt"
x = np.linspace(-1, 1, nx, endpoint=False)
dx = L / nx
t_start = 0.01
t = np.concatenate([[0.0], np.linspace(t_start, T, nt - 1)])
x_ext = np.concatenate([x - L, x, x + L])
diff = x[:, None] - x_ext

def random_ic(rng, n_modes=4):
    u = sum(rng.standard_normal() * np.sin(2*np.pi*m*x/L + rng.uniform(0, 2*np.pi))
            for m in range(1, n_modes + 1))
    return u / (np.abs(u).max() + 1e-12)

def solve_all(ICs):
    cumint = np.concatenate([np.zeros((len(ICs), 1)),
                             np.cumsum(0.5 * (ICs[:, :-1] + ICs[:, 1:]) * dx, axis=1)], axis=1)
    a = -cumint / (2 * nu)
    pe = np.tile(np.exp(a - a.max(axis=1, keepdims=True)), (1, 3))
    U = np.empty((len(ICs), nt, nx)); U[:, 0] = ICs
    for j in range(1, nt):
        K = np.exp(-diff**2 / (4 * nu * t[j]))
        U[:, j] = (pe @ (diff * K).T) / (pe @ K.T) / t[j]
    return U

rng = np.random.default_rng(42)
ICs = np.stack([np.sin(np.pi * x)] + [random_ic(rng) for _ in range(N_SAMPLES - 1)])
u = solve_all(ICs)
assert np.isfinite(u).all()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save({
    "u": torch.tensor(u, dtype=torch.float32),
    "ICs": torch.tensor(ICs, dtype=torch.float32),
    "x": torch.tensor(x, dtype=torch.float32),
    "t": torch.tensor(t, dtype=torch.float32),
    "nu": nu, "L": L, "x_start": -1.0, "x_end": 1.0,
    "T": T, "t_train_end": 1.0, "t_start": t_start, "nx": nx, "nt": nt, "N_samples": len(u),
}, OUT)
print("saved", OUT, tuple(u.shape))
