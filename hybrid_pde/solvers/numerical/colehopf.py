import numpy as np
import torch

nu = 1.0 / (100 * np.pi)
L, nx = 2.0, 512
T, nt = 2.0, 200
x = np.linspace(-1, 1, nx, endpoint=False)
dx = L / nx
t_start = 0.01
t = np.concatenate([[0.0], np.linspace(t_start, T, nt - 1)])
x_ext = np.concatenate([x - L, x, x + L])  
diff = x[:, None] - x_ext                    

def solve(u0):
    cumint = np.concatenate([[0.0], np.cumsum(0.5 * (u0[:-1] + u0[1:]) * dx)])
    phi0 = np.exp(-cumint / (2 * nu) - np.max(-cumint / (2 * nu)))   
    phi0_ext = np.tile(phi0, 3)
    U = np.empty((nt, nx)); U[0] = u0
    for j in range(1, nt):                  
        w = phi0_ext * np.exp(-diff**2 / (4 * nu * t[j]))            
        U[j] = (diff * w).sum(1) / w.sum(1) / t[j]
    return U

def random_ic(rng, n_modes=4):
    u = sum(rng.standard_normal() * np.sin(2*np.pi*m*x/L + rng.uniform(0, 2*np.pi))
            for m in range(1, n_modes + 1))
    return u / (np.abs(u).max() + 1e-12)

rng = np.random.default_rng(42)
ICs = np.stack([np.sin(np.pi * x)] + [random_ic(rng) for _ in range(7)])
u = np.stack([solve(ic) for ic in ICs])
assert np.isfinite(u).all()

torch.save({
    "u": torch.tensor(u, dtype=torch.float32),
    "ICs": torch.tensor(ICs, dtype=torch.float32),
    "x": torch.tensor(x, dtype=torch.float32),
    "t": torch.tensor(t, dtype=torch.float32),
    "nu": nu, "L": L, "x_start": -1.0, "x_end": 1.0,
    "T": T, "t_train_end": 1.0, "t_start": t_start, "nx": nx, "nt": nt, "N_samples": len(u),
}, "burgers_colehopf.pt")

X, Tg = np.meshgrid(x, t)
rows = np.vstack([np.column_stack([Tg.ravel(), X.ravel(), u[s].ravel()]) for s in range(len(u))])
np.savetxt("burgers_colehopf.csv", rows, delimiter=",", header="t,x,u", comments="", fmt="%.10f")
