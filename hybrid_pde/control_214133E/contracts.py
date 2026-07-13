from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Optional, List
import numpy as np


@runtime_checkable
class Solver(Protocol):
    name: str
    def rollout(self, ic: np.ndarray, x: np.ndarray, t: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class TrustSignal(Protocol):
    def __call__(self, state: np.ndarray, t: float) -> "tuple[float, bool]": ...


@runtime_checkable
class Coupling(Protocol):
    def rollout(self, ic, x, t, ml: Solver, num: Solver, trigger) -> np.ndarray: ...
    def correct(self, state, x, t0: float, t1: float, num: Solver) -> np.ndarray: ...


@dataclass
class SwitchDecision:
    correct: bool
    horizon: int = 1
    reason: str = ""


@dataclass
class CostReport:
    ml_steps: int = 0
    correction_steps: int = 0
    wall_time_s: float = 0.0
    accuracy_target: float = float("nan")
    achieved_error: float = float("nan")
    met_target: bool = False


@dataclass
class HybridResult:
    u: np.ndarray
    cost: CostReport
    trust_curve: Optional[np.ndarray] = None
    switch_times: List[float] = field(default_factory=list)
