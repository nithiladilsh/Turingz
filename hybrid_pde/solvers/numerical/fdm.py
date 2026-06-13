import numpy as np
import torch

nu = 1.0 / (100 * np.pi)
L, nx = 2.0, 512
T, nt = 2.0, 200
x = np.linspace(-1, 1, nx, endpoint=False)
dx = L / nx
t_start = 0.01
t = np.concatenate([[0.0], np.linspace(t_start, T, nt - 1)])

def step(u, dt):
    up, um = np.roll(u, -1), np.roll(u, 1)
    u_xx = (up - 2 * u + um) / dx**2
    u_x = np.where(u >= 0, (u - um) / dx, (up - u) / dx)
    return u + dt * (-u * u_x + nu * u_xx)

def solve(u0):
    dt = 0.4 * min(dx / (np.abs(u0).max() + 1e-9), dx**2 / (2 * nu))
    U = np.empty((nt, nx)); U[0] = u0
    u, tc = u0.copy(), 0.0
    for k in range(1, nt):
        while tc < t[k] - 1e-12:
            h = min(dt, t[k] - tc)
            u = step(u, h); tc += h
        U[k] = u
    return U

def random_ic(rng, n_modes=4):
    u = sum(rng.standard_normal() * np.sin(2*np.pi*m*x/L + rng.uniform(0, 2*np.pi))
            for m in range(1, n_modes + 1))
    return u / (np.abs(u).max() + 1e-12)

rng = np.random.default_rng(42)
ICs = np.stack([np.sin(np.pi * x)] + [random_ic(rng) for _ in range(7)])
u = np.stack([solve(ic) for ic in ICs])

torch.save({
    "u": torch.tensor(u, dtype=torch.float32),
    "ICs": torch.tensor(ICs, dtype=torch.float32),
    "x": torch.tensor(x, dtype=torch.float32),
    "t": torch.tensor(t, dtype=torch.float32),
    "nu": nu, "L": L, "x_start": -1.0, "x_end": 1.0,
    "T": T, "t_train_end": 1.0, "t_start": t_start, "nx": nx, "nt": nt, "N_samples": len(u),
}, "burgers_fdm.pt")

X, Tg = np.meshgrid(x, t)
rows = np.vstack([np.column_stack([Tg.ravel(), X.ravel(), u[s].ravel()]) for s in range(len(u))])
np.savetxt("burgers_fdm.csv", rows, delimiter=",", header="t,x,u", comments="", fmt="%.10f")