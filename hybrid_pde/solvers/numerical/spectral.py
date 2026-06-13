import numpy as np
import torch

nu = 1.0 / (100 * np.pi)
L, nx = 2.0, 512
T, nt = 2.0, 200
x = np.linspace(-1, 1, nx, endpoint=False)
dx = L / nx
t_start = 0.01
t = np.concatenate([[0.0], np.linspace(t_start, T, nt - 1)])
k = 2 * np.pi * np.arange(nx // 2 + 1) / L
mask = np.arange(nx // 2 + 1) <= nx // 3
dt_target = 1e-4

def rhs(uh):
    u = np.fft.irfft(uh * mask, n=nx)
    return -0.5j * k * np.fft.rfft(u * u)

def step(uh, E, E2, h):
    k1 = rhs(uh)
    k2 = rhs(E2 * uh + 0.5 * h * E2 * k1)
    k3 = rhs(E2 * uh + 0.5 * h * k2)
    k4 = rhs(E * uh + h * E2 * k3)
    uh = E * uh + h / 6 * (E * k1 + 2 * E2 * k2 + 2 * E2 * k3 + k4)
    uh[-1] = 0.0
    return uh

def solve(u0):
    uh = np.fft.rfft(u0)
    U = np.empty((nt, nx)); U[0] = u0
    for j in range(1, nt):
        h = t[j] - t[j - 1]
        m = max(1, round(h / dt_target))
        h /= m
        E, E2 = np.exp(-nu * k**2 * h), np.exp(-nu * k**2 * h * 0.5)
        for _ in range(m):
            uh = step(uh, E, E2, h)
        U[j] = np.fft.irfft(uh * mask, n=nx)
    return U

def random_ic(rng, n_modes=4):
    u = sum(rng.standard_normal() * np.sin(2*np.pi*m*x/L + rng.uniform(0, 2*np.pi))
            for m in range(1, n_modes + 1))
    return u / (np.abs(u).max() + 1e-12)

rng = np.random.default_rng(42)
ICs = np.stack([np.sin(np.pi * x)] + [random_ic(rng) for _ in range(7)])
u = np.stack([solve(ic) for ic in ICs])

U0 = u[0]
energy = 0.5 * dx * np.sum(U0**2, axis=1)
mass = dx * np.sum(U0, axis=1)
res = []
for ti in range(2, nt - 1):
    if t[ti] >= 0.3:
        break
    uh = np.fft.rfft(U0[ti])
    ux = np.fft.irfft(1j * k * uh, n=nx)
    uxx = np.fft.irfft(-k**2 * uh, n=nx)
    dudt = (U0[ti + 1] - U0[ti - 1]) / (t[ti + 1] - t[ti - 1])
    res.append(np.sqrt(np.mean((dudt + U0[ti] * ux - nu * uxx)**2)))
assert np.isfinite(u).all()
assert np.all(np.diff(energy[1:]) <= 1e-9)
print("finite=%s energy_dissipated=%.1f%% mass_drift=%.1e pde_residual=%.1e" % (
    np.isfinite(u).all(), 100*(1-energy[-1]/energy[0]),
    np.max(np.abs(mass-mass[0])), np.mean(res)))

torch.save({
    "u": torch.tensor(u, dtype=torch.float32),
    "ICs": torch.tensor(ICs, dtype=torch.float32),
    "x": torch.tensor(x, dtype=torch.float32),
    "t": torch.tensor(t, dtype=torch.float32),
    "nu": nu, "L": L, "x_start": -1.0, "x_end": 1.0,
    "T": T, "t_train_end": 1.0, "t_start": t_start, "nx": nx, "nt": nt, "N_samples": len(u),
}, "burgers_spectral.pt")

X, Tg = np.meshgrid(x, t)
rows = np.vstack([np.column_stack([Tg.ravel(), X.ravel(), u[s].ravel()]) for s in range(len(u))])
np.savetxt("burgers_spectral.csv", rows, delimiter=",", header="t,x,u", comments="", fmt="%.10f")
