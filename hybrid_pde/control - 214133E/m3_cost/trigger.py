import numpy as np
from . import groundtruth as G
from . import config as C


def _spatial_derivs(u):
    up, um = np.roll(u, -1, axis=-1), np.roll(u, 1, axis=-1)
    u_x = (up - um) / (2 * G.DX)
    u_xx = (up - 2 * u + um) / G.DX ** 2
    return u_x, u_xx


def physics_residual_signal(u_field, t=G.T_GRID):
    nt = u_field.shape[0]
    res = np.zeros(nt)
    u_x, u_xx = _spatial_derivs(u_field)
    for j in range(1, nt - 1):
        u_t = (u_field[j + 1] - u_field[j - 1]) / (t[j + 1] - t[j - 1])
        r = u_t + u_field[j] * u_x[j] - C.NU * u_xx[j]
        res[j] = np.sqrt(np.mean(r ** 2))
    res[0], res[-1] = res[1], res[-2]
    return res


def physics_residual_trust(u_field, scale=None):
    res = physics_residual_signal(u_field)
    if scale is None:
        in_win = res[G.TRAIN_MASK]
        scale = np.median(in_win[in_win > 0]) if np.any(in_win > 0) else (res.max() + 1e-12)
    return 1.0 / (1.0 + res / (scale + 1e-12)), res


def oracle_trust(u_field, u_true):
    err = G.rel_l2(u_field, u_true)
    return np.clip(1.0 - err, 0.0, 1.0), err


def synthetic_trust(t0=C.T_TRAIN_END, sharpness=8.0):
    return 1.0 / (1.0 + np.exp(sharpness * (G.T_GRID - t0)))


def reliable_horizon(trust, threshold, t=G.T_GRID):
    below = np.where(trust < threshold)[0]
    if len(below) == 0:
        return float(t[-1]), len(t) - 1
    j = int(below[0])
    return float(t[j]), j
