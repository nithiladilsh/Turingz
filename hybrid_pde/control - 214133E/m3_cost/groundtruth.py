import numpy as np
from . import config as C

X = np.linspace(-1, 1, C.NX, endpoint=False)
DX = C.L / C.NX
T_GRID = np.concatenate([[0.0], np.linspace(C.T_START, C.T, C.NT - 1)])
TRAIN_MASK = T_GRID <= C.T_TRAIN_END
EXTRAP_MASK = T_GRID > C.T_TRAIN_END

_X_EXT = np.concatenate([X - C.L, X, X + C.L])
_DIFF = X[:, None] - _X_EXT


def _random_ic(rng, n_modes=4):
    u = sum(rng.standard_normal() * np.sin(2 * np.pi * m * X / C.L + rng.uniform(0, 2 * np.pi))
            for m in range(1, n_modes + 1))
    return u / (np.abs(u).max() + 1e-12)


def make_ics(n=C.N_SAMPLES, seed=C.SEED):
    rng = np.random.default_rng(seed)
    return np.stack([np.sin(np.pi * X)] + [_random_ic(rng) for _ in range(n - 1)])


def colehopf_solve(ic):
    cumint = np.concatenate([[0.0], np.cumsum(0.5 * (ic[:-1] + ic[1:]) * DX)])
    a = -cumint / (2 * C.NU)
    pe = np.tile(np.exp(a - a.max()), 3)
    U = np.empty((C.NT, C.NX)); U[0] = ic
    for j in range(1, C.NT):
        K = np.exp(-_DIFF ** 2 / (4 * C.NU * T_GRID[j]))
        U[j] = (pe @ (_DIFF * K).T) / (pe @ K.T) / T_GRID[j]
    return U


def _fdm_step(u, dt):
    up, um = np.roll(u, -1), np.roll(u, 1)
    u_xx = (up - 2 * u + um) / DX ** 2
    u_x = np.where(u >= 0, (u - um) / DX, (up - u) / DX)
    return u + dt * (-u * u_x + C.NU * u_xx)


def _fdm_dt(u):
    return 0.4 * min(DX / (np.abs(u).max() + 1e-9), DX ** 2 / (2 * C.NU))


def fdm_advance(u, t0, t1):
    u = u.copy(); tc = float(t0)
    while tc < t1 - 1e-12:
        h = min(_fdm_dt(u), t1 - tc)
        u = _fdm_step(u, h); tc += h
    return u


def fdm_solve(ic):
    U = np.empty((C.NT, C.NX)); U[0] = ic
    u = ic.copy()
    for kk in range(1, C.NT):
        u = fdm_advance(u, T_GRID[kk - 1], T_GRID[kk])
        U[kk] = u
    return U


_K = 2 * np.pi * np.arange(C.NX // 2 + 1) / C.L
_DEALIAS = np.arange(C.NX // 2 + 1) <= C.NX // 3
_DT_SPECTRAL = 1e-4


def _spec_rhs(uh):
    u = np.fft.irfft(uh * _DEALIAS, n=C.NX)
    return -0.5j * _K * np.fft.rfft(u * u)


def _spec_substep(uh, E, E2, h):
    k1 = _spec_rhs(uh)
    k2 = _spec_rhs(E2 * uh + 0.5 * h * E2 * k1)
    k3 = _spec_rhs(E2 * uh + 0.5 * h * k2)
    k4 = _spec_rhs(E * uh + h * E2 * k3)
    uh = E * uh + h / 6 * (E * k1 + 2 * E2 * k2 + 2 * E2 * k3 + k4)
    uh[-1] = 0.0
    return uh


def spectral_advance(u, t0, t1):
    uh = np.fft.rfft(u)
    span = t1 - t0
    m = max(1, round(span / _DT_SPECTRAL))
    h = span / m
    E, E2 = np.exp(-C.NU * _K ** 2 * h), np.exp(-C.NU * _K ** 2 * h * 0.5)
    for _ in range(m):
        uh = _spec_substep(uh, E, E2, h)
    return np.fft.irfft(uh * _DEALIAS, n=C.NX)


def spectral_solve(ic):
    U = np.empty((C.NT, C.NX)); U[0] = ic
    u = ic.copy()
    for j in range(1, C.NT):
        u = spectral_advance(u, T_GRID[j - 1], T_GRID[j])
        U[j] = u
    return U


NUMERICAL_SOLVERS = {
    "fdm": dict(advance=fdm_advance, solve=fdm_solve),
    "spectral": dict(advance=spectral_advance, solve=spectral_solve),
}


def rel_l2(pred, ref, mask=None):
    if mask is None:
        return np.linalg.norm(pred - ref, axis=-1) / (np.linalg.norm(ref, axis=-1) + 1e-12)
    num = np.linalg.norm((pred - ref)[mask])
    den = np.linalg.norm(ref[mask]) + 1e-12
    return float(num / den)
