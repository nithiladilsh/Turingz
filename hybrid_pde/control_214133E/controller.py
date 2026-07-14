from __future__ import annotations
from abc import ABC, abstractmethod
from .contracts import SwitchDecision


class Controller(ABC):
    @abstractmethod
    def configure(self, accuracy_target): ...

    @abstractmethod
    def decide(self, trust, flag, t, step) -> SwitchDecision: ...

    def reset(self):
        pass


class FixedIntervalController(Controller):
    def __init__(self, k=5, horizon=1):
        self.k = k
        self.horizon = horizon

    def configure(self, accuracy_target):
        pass

    def decide(self, trust, flag, t, step) -> SwitchDecision:
        return SwitchDecision(step % self.k == 0, self.horizon, "fixed")

    def reset(self):
        pass


class AdaptiveController(Controller):
    def __init__(self, theta_lo=0.4, theta_hi=0.6, horizon=1, model=None):
        self.theta_lo = theta_lo
        self.theta_hi = theta_hi
        self.base_horizon = horizon
        self.model = model
        self._horizon = horizon
        self._correcting = False

    def configure(self, accuracy_target):
        self._correcting = False
        self._horizon = self.base_horizon
        if self.model is not None:
            eff = self.model.budget_to_effort(accuracy_target)
            if eff == eff:
                self._horizon = max(1, int(round(eff)))

    def decide(self, trust, flag, t, step) -> SwitchDecision:
        if self._correcting:
            if trust > self.theta_hi:
                self._correcting = False
        else:
            if trust < self.theta_lo:
                self._correcting = True
        return SwitchDecision(self._correcting, self._horizon, "adaptive")

    def reset(self):
        self._correcting = False


def thresholds_for_target(target):
    lo = min(0.58, max(0.12, 0.62 - 1.4 * float(target)))
    return lo, min(0.9, lo + 0.12)
