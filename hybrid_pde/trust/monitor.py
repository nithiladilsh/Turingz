import os
import numpy as np
from hybrid_pde.trust.signals import _dx, _dxx, NU, shock_coeff

WIN, CUT, START, FAIL = 5, 0.5, 2, 0.10

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def _hi_mask(nx, dx):
    k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    return np.abs(k) > 0.5 * np.abs(k).max()

def _frame(u, prev_u, ecum, eprev, dx, dt, coeff, hi):
    ut = (u - prev_u) / dt
    r = ut + u * _dx(u, dx) - NU * _dxx(u, dx)
    excess = np.maximum(0.0, np.abs(r) - coeff * np.abs(_dxx(u, dx)))
    res = np.sqrt((excess ** 2).mean())
    e = 0.5 * dx * (u ** 2).sum()
    ecum = ecum + max(0.0, e - eprev)
    p = np.abs(np.fft.fft(u)) ** 2
    rough = p[hi].sum() / (p.sum() + 1e-12)
    mom = u.sum() * dx
    return np.array([res, ecum, rough, mom]), ecum, e

def _raw_signals(u, dx, dt, coeff, hi):
    T = u.shape[0]
    F = np.zeros((T, 4))
    ecum, eprev = 0.0, 0.5 * dx * (u[0] ** 2).sum()
    for n in range(T):
        prev = u[n - 1] if n >= 1 else u[n]
        F[n], ecum, eprev = _frame(u[n], prev, ecum, eprev, dx, dt, coeff, hi)
    F[:, 3] = np.maximum.accumulate(np.abs(F[:, 3] - F[0, 3]))
    return F

def _trail(a, w):
    return np.array([a[max(0, i - w + 1):i + 1].mean() for i in range(len(a))])

def _first(mask, K):
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= K:
            return i - K + 1
    return len(mask) - 1

def fit_trust(preds, err, x, t, u_true_train):
    dx, dt = float(x[1] - x[0]), float(t[1] - t[0])
    hi = _hi_mask(len(x), dx)
    coeff = shock_coeff(u_true_train, x, t)
    F = np.array([_raw_signals(w, dx, dt, coeff, hi)[START:] for w in preds])
    e = err[:, START:]
    flat = F.reshape(-1, 4)
    mean, std = flat.mean(0), flat.std(0) + 1e-8
    Z = (flat - mean) / std
    ef = e.reshape(-1)
    w = np.array([max(0.0, np.corrcoef(Z[:, i], ef)[0, 1]) for i in range(4)])
    w = w / (w.sum() + 1e-9)
    fused = ((F - mean) / std) @ w
    fsm = np.array([_trail(f, WIN) for f in fused])
    smean, sstd = fsm.mean(), fsm.std() + 1e-8
    fail = (e > FAIL).astype(float)
    a, c = 0.0, 0.0
    zz = (fsm.reshape(-1) - smean) / sstd
    yy = fail.reshape(-1)
    for _ in range(3000):
        g = _sigmoid(a * zz + c) - yy
        a -= 0.2 * (g * zz).mean()
        c -= 0.2 * g.mean()
    te = t[START:]
    true_h = np.array([te[_first(row > FAIL, 1)] for row in e])
    trust = 1 - _sigmoid(a * (fsm - smean) / sstd + c)
    best = None
    for cut in np.arange(0.5, 0.91, 0.05):
        for K in range(4, 26):
            late, early = 0, 0.0
            for i in range(len(trust)):
                ph = te[_first(trust[i] < cut, K)]
                d = ph - true_h[i]
                if d > 1e-9:
                    late += 1
                else:
                    early += -d
            cost = 100 * late + early / len(trust)
            if best is None or cost < best[0] - 1e-9:
                best = (cost, float(cut), K)
    return dict(coeff=coeff, dx=dx, dt=dt, mean=mean, std=std, w=w,
               smean=smean, sstd=sstd, a=a, c=c, CUT=best[1], K=best[2])

def save_params(path, params):
    np.savez(path, **params)

def load_params(path):
    d = np.load(path)
    return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}

class TrustMonitor:
    def __init__(self, params):
        self.p = params
        self.hi = None
        self.reset()

    def reset(self):
        self.prev_u = None
        self.ecum = 0.0
        self.eprev = None
        self.fbuf = []
        self.M0 = None
        self.mdmax = 0.0
        self.run = 0
        self.n = 0
        self.failed = False

    def update(self, u, t=None):
        p = self.p
        u = np.asarray(u, float)
        if self.hi is None:
            self.hi = _hi_mask(len(u), p["dx"])
        if self.prev_u is None:
            self.prev_u = u
            self.eprev = 0.5 * p["dx"] * (u ** 2).sum()
        s, self.ecum, self.eprev = _frame(u, self.prev_u, self.ecum, self.eprev,
                                          p["dx"], p["dt"], float(p["coeff"]), self.hi)
        self.prev_u = u
        self.n += 1
        if self.M0 is None:
            self.M0 = s[3]
        self.mdmax = max(self.mdmax, abs(s[3] - self.M0))
        s = s.copy()
        s[3] = self.mdmax
        if self.n <= START:
            return {"trust": 1.0, "ok": True}
        fused = float(((s - p["mean"]) / p["std"]) @ p["w"])
        self.fbuf.append(fused)
        fsm = np.mean(self.fbuf[-WIN:])
        trust = float(1 - _sigmoid(p["a"] * (fsm - p["smean"]) / p["sstd"] + p["c"]))
        self.run = self.run + 1 if trust < float(p.get("CUT", CUT)) else 0
        if self.run >= int(p["K"]):
            self.failed = True
        return {"trust": trust, "ok": not self.failed}
