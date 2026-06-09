from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np


class AbstractSolver(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fit(self, dataset: Dict[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    def predict(self, ic: np.ndarray, x: np.ndarray, t: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...

    def rollout(self, ic: np.ndarray, x_grid: np.ndarray,
                t_grid: np.ndarray) -> np.ndarray:
        X, T = np.meshgrid(x_grid, t_grid, indexing="xy")
        u_flat = self.predict(ic, X.ravel(), T.ravel())
        return u_flat.reshape(len(t_grid), len(x_grid))

    def num_parameters(self) -> int:
        return 0

    def supported_samples(self, candidate_indices):
        """Which of the candidate IC indices this solver can predict.
        Operators handle all; a single-instance solver (e.g. PINN) overrides
        this to return only the IC it was trained on."""
        return list(candidate_indices)
