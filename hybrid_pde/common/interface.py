from abc import ABC, abstractmethod
import numpy as np


class AbstractSolver(ABC):
    name = "solver"

    @abstractmethod
    def fit(self, dataset):
        ...

    def predict(self, ic, x, t):
        raise NotImplementedError

    def rollout(self, ic, x, t):
        X, T = np.meshgrid(x, t, indexing="xy")
        return self.predict(ic, X.ravel(), T.ravel()).reshape(len(t), len(x))

    def num_parameters(self):
        return 0

    def supported_samples(self, idx):
        return list(idx)
