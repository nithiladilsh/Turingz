import numpy as np

NU = 1.0 / (100 * np.pi)

def _dx(u, dx):
    return (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2 * dx)

def _dxx(u, dx):
    return (np.roll(u, -1, axis=-1) - 2 * u + np.roll(u, 1, axis=-1)) / dx ** 2

def residual_field(u, x, t, nu=NU):
    dx = x[1] - x[0]
    u_t = np.gradient(u, t, axis=1)
    return u_t + u * _dx(u, dx) - nu * _dxx(u, dx)

def residual_signal(u, x, t, nu=NU):
    return np.sqrt((residual_field(u, x, t, nu) ** 2).mean(axis=-1))

def shock_coeff(u_true, x, t, nu=NU):
    dx = x[1] - x[0]
    r = np.abs(residual_field(u_true, x, t, nu))
    s = np.abs(_dxx(u_true, dx))
    return float((r * s).sum() / (s ** 2).sum())

def corrected_signal(u_pred, coeff, x, t, nu=NU):
    dx = x[1] - x[0]
    r = np.abs(residual_field(u_pred, x, t, nu))
    excess = np.maximum(0.0, r - coeff * np.abs(_dxx(u_pred, dx)))
    return np.sqrt((excess ** 2).mean(axis=-1))

def energy_signal(u, x):
    dx = x[1] - x[0]
    e = 0.5 * dx * (u ** 2).sum(axis=-1)
    de = np.diff(e, axis=1, prepend=e[:, :1])
    return np.cumsum(np.maximum(0.0, de), axis=1)

def roughness_signal(u, x):
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(u.shape[-1], d=dx)
    power = np.abs(np.fft.fft(u, axis=-1)) ** 2
    hi = np.abs(k) > 0.5 * np.abs(k).max()
    return power[..., hi].sum(-1) / (power.sum(-1) + 1e-12)
