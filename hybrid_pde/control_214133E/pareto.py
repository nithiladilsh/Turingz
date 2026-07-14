from __future__ import annotations
from dataclasses import dataclass
from typing import List, Callable


@dataclass
class ParetoPoint:
    method: str
    cost: float
    error: float


def dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    return (a.cost <= b.cost and a.error <= b.error) and (a.cost < b.cost or a.error < b.error)


def pareto_front(points: List[ParetoPoint]) -> List[ParetoPoint]:
    return [p for p in points if not any(dominates(q, p) for q in points if q is not p)]


def build_frontier(run_point: Callable[[float], ParetoPoint], targets) -> List[ParetoPoint]:
    return [run_point(t) for t in targets]
