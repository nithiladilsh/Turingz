from typing import Dict
import numpy as np

SHOCK_T_LO = 0.25
SHOCK_T_HI = 0.75

def _dx_uniform(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(x[1] - x[0])


def d_dx_periodic(u: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dx)


def d2_dx2_periodic(u: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(u, -1, axis=1) - 2.0 * u + np.roll(u, 1, axis=1)) / (dx * dx)


def d_dt(u: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return np.gradient(u, t, axis=0)

def burgers_residual(u: np.ndarray, x: np.ndarray, t: np.ndarray,
                     nu: float) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    dx = _dx_uniform(x)
    u_t = d_dt(u, t)
    u_x = d_dx_periodic(u, dx)
    u_xx = d2_dx2_periodic(u, dx)
    return u_t + u * u_x - float(nu) * u_xx


def _agg(v: np.ndarray) -> Dict[str, float]:
    if v.size == 0:
        return {"mean": float("nan"), "max": float("nan")}
    return {"mean": float(np.mean(v)), "max": float(np.max(v))}


def _windowed(curve: np.ndarray, t: np.ndarray, t_train_end: float) -> Dict:
    pre = t < SHOCK_T_LO
    shock = (t >= SHOCK_T_LO) & (t <= SHOCK_T_HI)
    post = (t > SHOCK_T_HI) & (t <= t_train_end + 1e-12)
    extrap = t > t_train_end + 1e-12
    return {
        "pre_shock": _agg(curve[pre]),
        "shock": _agg(curve[shock]),
        "post_shock": _agg(curve[post]),
        "extrap": _agg(curve[extrap]),
    }

def energy(u: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = _dx_uniform(x)
    return 0.5 * dx * np.sum(np.asarray(u, dtype=np.float64) ** 2, axis=1)


def energy_drift(u: np.ndarray, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    e = energy(u, x)
    return np.gradient(e, np.asarray(t, dtype=np.float64))


def periodicity_gap(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    scale = np.maximum(np.max(np.abs(u), axis=1), 1e-12)
    return np.abs(u[:, -1] - u[:, 0]) / scale


def compute_signals(u: np.ndarray, x: np.ndarray, t: np.ndarray, nu: float,
                    t_train_end: float = 1.0, keep_curves: bool = True) -> Dict:
    u = np.asarray(u, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    r = burgers_residual(u, x, t, nu)
    res_rms = np.sqrt(np.mean(r ** 2, axis=1)) 
    drift = energy_drift(u, x, t)
    drift_pos = np.maximum(drift, 0.0)               
    seam = periodicity_gap(u)

    out = {
        "residual_rms": {
            **_agg(res_rms), "final": float(res_rms[-1]),
            "windowed": _windowed(res_rms, t, t_train_end),
        },
        "energy_drift_pos": {**_agg(drift_pos), "final": float(drift_pos[-1])},
        "periodicity_gap": {**_agg(seam), "final": float(seam[-1])},
        "windows": {"shock_t_lo": SHOCK_T_LO, "shock_t_hi": SHOCK_T_HI,
                    "t_train_end": float(t_train_end)},
        "method": "finite_difference (periodic central in x, np.gradient in t)",
    }
    if keep_curves:
        out["residual_rms"]["curve"] = res_rms.tolist()
        out["periodicity_gap"]["curve"] = seam.tolist()
        out["t"] = t.tolist()
    return out


def correlate_signal_with_error(signal_curve, error_curve) -> Dict[str, float]:
    s = np.asarray(signal_curve, dtype=np.float64)
    e = np.asarray(error_curve, dtype=np.float64)
    n = min(len(s), len(e))
    s, e = s[:n], e[:n]
    def _pearson(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / d)
    def _rank(a):
        order = np.argsort(np.argsort(a))
        return order.astype(np.float64)
    return {"pearson": _pearson(s, e), "spearman": _pearson(_rank(s), _rank(e))}


def residual_curve(u: np.ndarray, x: np.ndarray, t: np.ndarray, nu: float) -> np.ndarray:
    r = burgers_residual(u, x, t, nu)
    return np.sqrt(np.mean(r ** 2, axis=1))
