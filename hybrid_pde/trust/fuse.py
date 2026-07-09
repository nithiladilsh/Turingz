import numpy as np

def _corr(a, b):
    a, b = a.ravel(), b.ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def fit_fusion(signals, err):
    F = np.stack(signals, -1)
    flat = F.reshape(-1, F.shape[-1])
    mean, std = flat.mean(0), flat.std(0) + 1e-8
    Z = (flat - mean) / std
    w = np.array([max(0.0, _corr(Z[:, i], err)) for i in range(F.shape[-1])])
    w = w / (w.sum() + 1e-9)
    return {"mean": mean, "std": std, "w": w}

def fuse(signals, params):
    F = np.stack(signals, -1)
    Z = (F - params["mean"]) / params["std"]
    return Z @ params["w"]
