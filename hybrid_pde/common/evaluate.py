import numpy as np


def _per_ic(pred, ref, mask):
    num = np.linalg.norm((pred - ref)[:, mask], axis=(1, 2))
    den = np.linalg.norm(ref[:, mask], axis=(1, 2)) + 1e-12
    return num / den


def evaluate(solver, U, ICs, x, t, te, idx):
    tr, ex = t <= te, t > te
    idx = np.asarray(idx)
    if hasattr(solver, "predict_grid"):
        pred = solver.predict_grid(ICs[idx], x, t)
    else:
        pred = np.stack([solver.rollout(ICs[i], x, t) for i in idx])
    ref = U[idx]
    ind, exr = _per_ic(pred, ref, tr), _per_ic(pred, ref, ex)
    curves = np.linalg.norm(pred - ref, axis=2) / (np.linalg.norm(ref, axis=2) + 1e-12)
    n = len(idx)
    return {"n": int(n),
            "in_dist_mean": float(ind.mean()), "in_dist_std": float(ind.std(ddof=1) if n > 1 else 0.0),
            "in_dist_max": float(ind.max()),
            "extrap_mean": float(exr.mean()), "extrap_std": float(exr.std(ddof=1) if n > 1 else 0.0),
            "extrap_max": float(exr.max()),
            "per_ic_in_dist": ind.tolist(), "per_ic_extrap": exr.tolist(),
            "rel_l2_over_t": curves}
