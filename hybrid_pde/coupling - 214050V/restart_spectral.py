import numpy as np

NU = 1.0 / (100 * np.pi)                       
assert np.isclose(NU, 1.0 / (100.0 * np.pi))

L, NX = 2.0, 512
T, NT = 2.0, 200
X = np.linspace(-1, 1, NX, endpoint=False)
DX = L / NX
T_START = 0.01
TGRID = np.concatenate([[0.0], np.linspace(T_START, T, NT - 1)])
K = 2 * np.pi * np.arange(NX // 2 + 1) / L 
MASK = np.arange(NX // 2 + 1) <= NX // 3 
DT_TARGET = 1e-4

# Calculate the nonlinear change
def _rhs(uh):
    u = np.fft.irfft(uh * MASK, n=NX, axis=-1)
    return -0.5j * K * np.fft.rfft(u * u, axis=-1)

#time step
def _step(uh, E, E2, h):
    k1 = _rhs(uh)
    k2 = _rhs(E2 * uh + 0.5 * h * E2 * k1)
    k3 = _rhs(E2 * uh + 0.5 * h * k2)
    k4 = _rhs(E * uh + h * E2 * k3)
    uh = E * uh + h / 6 * (E * k1 + 2 * E2 * k2 + 2 * E2 * k3 + k4)
    uh[..., -1] = 0.0
    return uh


def solve_from(u0, i_start, nu=NU):
    assert np.isclose(nu, 1.0 / (100.0 * np.pi))
    u0 = np.asarray(u0, dtype=np.float64)
    batched = (u0.ndim == 2)
    U0 = u0 if batched else u0[None, :]
    B = U0.shape[0]
    out = np.empty((B, NT - i_start, NX))
    out[:, 0] = U0 # handoff state as the first state
    uh = np.fft.rfft(U0, axis=-1)
    for j in range(i_start + 1, NT):
        h = TGRID[j] - TGRID[j - 1]
        m = max(1, round(h / DT_TARGET))
        h = h / m
        E = np.exp(-nu * K**2 * h)
        E2 = np.exp(-nu * K**2 * h * 0.5)
        for _ in range(m):
            uh = _step(uh, E, E2, h)
        out[:, j - i_start] = np.fft.irfft(uh * MASK, n=NX, axis=-1) #store next physical state
    return out if batched else out[0]


def solve_full(u0, nu=NU):
    return solve_from(u0, 0, nu=nu)

#Find the index of the nearest time in array to a given time
def nearest_index(t_s):
    return int(np.argmin(np.abs(TGRID - t_s)))

#ML state cleaning
def lowpass(u, frac):
    u = np.asarray(u, dtype=np.float64)
    if frac >= 1.0:
        return u.copy()
    cut = int(frac * (NX // 3))
    uh = np.fft.rfft(u, axis=-1)
    uh[..., cut + 1:] = 0.0
    return np.fft.irfft(uh, n=NX, axis=-1)
