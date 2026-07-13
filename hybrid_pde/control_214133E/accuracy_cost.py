from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Callable


@dataclass
class AccuracyCostPoint:
    effort: float
    error: float
    cost: float


@dataclass
class AccuracyCostModel:
    ml_step_s: float
    correction_step_s: float
    points: List[AccuracyCostPoint] = field(default_factory=list)

    def predict_cost(self, ml_steps, correction_steps):
        return ml_steps * self.ml_step_s + correction_steps * self.correction_step_s

    def sweep(self, run_fn: Callable[[float], "tuple[float, float]"], efforts):
        self.points = []
        for e in efforts:
            err, cost = run_fn(e)
            self.points.append(AccuracyCostPoint(float(e), float(err), float(cost)))
        self.points.sort(key=lambda p: p.effort)
        return self.points

    def budget_to_effort(self, target_error):
        if not self.points:
            return float("nan")
        for p in self.points:
            if p.error <= target_error:
                return p.effort
        return self.points[-1].effort

    def point_at_effort(self, effort):
        for p in self.points:
            if p.effort == effort:
                return p
        return None
