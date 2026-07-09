import numpy as np

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def smooth(a, w):
    k = np.ones(w) / w
    return np.array([np.convolve(r, k, "same") for r in np.atleast_2d(a)])

def fit_calibration(fused_train, fail_train):
    s = fused_train.reshape(-1)
    mean, std = s.mean(), s.std() + 1e-8
    z = (s - mean) / std
    y = fail_train.reshape(-1)
    a, c = 0.0, 0.0
    for _ in range(3000):
        p = _sigmoid(a * z + c)
        g = p - y
        a -= 0.2 * (g * z).mean()
        c -= 0.2 * g.mean()
    return {"a": a, "c": c, "mean": mean, "std": std}

def trust_score(fused, cal):
    z = (fused - cal["mean"]) / cal["std"]
    return 1.0 - _sigmoid(cal["a"] * z + cal["c"])

def _first(mask, K):
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= K:
            return i - K + 1
    return len(mask) - 1

def reliable_horizon(trust, times, cutoff, K):
    return times[_first(trust < cutoff, K)]

def true_horizon(err, times, fail):
    return times[_first(err > fail, 1)]

def select_persistence(trust_train, times, err_train, fail, cutoff):
    true_h = np.array([true_horizon(e, times, fail) for e in err_train])
    best = None
    for K in range(4, 19):
        pred = np.array([reliable_horizon(tr, times, cutoff, K) for tr in trust_train])
        mae = np.abs(pred - true_h).mean()
        if best is None or mae < best[1] - 1e-9:
            best = (K, mae)
    return best[0]
