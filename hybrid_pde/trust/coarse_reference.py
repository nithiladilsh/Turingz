import numpy as np

NU = 1.0 / (100 * np.pi)

def _godunov_flux(uL, uR):
    fL = 0.5 * uL * uL
    fR = 0.5 * uR * uR
    s = 0.5 * (uL + uR)
    return np.where(uL > uR, np.where(s > 0, fL, fR),
                    np.where(uL > 0, fL, np.where(uR < 0, fR, 0.0)))

class CoarseReferenceMonitor:
    def __init__(self, ic, x, n=256, nu=NU, fail=0.10, K=3, cfl=0.4):
        self.x = np.asarray(x, float)
        self.nx = len(self.x)
        self.L = float(self.x[1] - self.x[0]) * self.nx
        self.n = n
        self.step = self.nx // n
        self.nu = nu
        self.fail = fail
        self.K = K
        self.cfl = cfl
        self.dxc = self.L / n
        self.xp = np.concatenate([self.x[::self.step], [self.x[0] + self.L]])
        self._ic0 = np.asarray(ic, float)[::self.step].copy()
        self.reset()

    def reset(self):
        self.uc = self._ic0.copy()
        self.tc = 0.0
        self.run = 0
        self.failed = False

    def _rhs(self, u):
        Fr = _godunov_flux(u, np.roll(u, -1))
        Fl = np.roll(Fr, 1)
        return -(Fr - Fl) / self.dxc + self.nu * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / self.dxc ** 2

    def _advance(self, t):
        while self.tc < t - 1e-12:
            u = self.uc
            dt = min(self.cfl * self.dxc / (np.abs(u).max() + 1e-9),
                     0.25 * self.dxc ** 2 / self.nu, t - self.tc)
            u1 = u + dt * self._rhs(u)
            self.uc = 0.5 * u + 0.5 * (u1 + dt * self._rhs(u1))
            self.tc += dt

    def _reference(self):
        return np.interp(self.x, self.xp, np.concatenate([self.uc, [self.uc[0]]]))

    def update(self, u, t):
        u = np.asarray(u, float)
        self._advance(float(t))
        ref = self._reference()
        est = float(np.linalg.norm(u - ref) / (np.linalg.norm(ref) + 1e-12))
        trust = float(np.clip(1.0 - est / (2.0 * self.fail), 0.0, 1.0))
        self.run = self.run + 1 if est > self.fail else 0
        if self.run >= self.K:
            self.failed = True
        return {"trust": trust, "ok": not self.failed, "est_error": est}
