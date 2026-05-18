"""
================================================================================
ABSTRACT SOLVER INTERFACE
Team    : Turingz   (Module 3 - infrastructure)
File    : abstract_solver.py

All ML solvers (PINN, FNO, DeepONet) and hybrid solvers conform to this
interface so the same evaluation harness — reliability detector,
robustness module, cost meter, and switching-policy wrapper — can call
every solver the same way.
================================================================================
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np


class AbstractSolver(ABC):
    """Base class for every PDE solver in the FYP framework.

    Subclass contract
    -----------------
    A concrete solver must implement:
        - name              : short identifier used in logs / plots
        - fit(dataset)      : train on the reference dataset
        - predict(ic, x, t) : forward inference at scattered query points
        - save(path)        : persist weights + hyper-parameters to disk
        - load(path)        : restore the solver to a ready-to-predict state

    Concrete solvers may also override:
        - rollout(ic, x_grid, t_grid) : full-grid prediction.
          The default implementation just calls predict() with the flattened
          mesh, but operator solvers (FNO, DeepONet) can override it for
          a single batched forward pass.

    Why this interface
    ------------------
    Module 1 (reliability) calls .predict at every step to compute its
    four signals.  Module 2 (robustness) calls .rollout on out-of-
    distribution initial conditions.  Module 3 (cost) wraps fit/predict
    in the cost meter, and the switching-policy wrapper replaces the
    ML step with a Cole-Hopf step when the residual exceeds threshold.
    The unified interface keeps all three modules independent of any
    specific architecture.
    """

    # ─────────────────────────────────────────────────────────────────────
    # Required
    # ─────────────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier, e.g. 'DeepONet(m=128)'."""

    @abstractmethod
    def fit(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Train the solver on the reference dataset.

        Parameters
        ----------
        dataset : dict
            Loaded Cole-Hopf reference with keys
                'u'           : (N, nt, nx) ground-truth field
                'ICs'         : (N, nx)     initial conditions
                'x'           : (nx,)       spatial grid
                't'           : (nt,)       time grid
                'nu'          : float       viscosity
                't_train_end' : float       training / extrapolation split

        Returns
        -------
        dict
            Training info — final losses, wall time, per-IC errors, etc.
            Module 3's cost meter reads 'wall_time_s' from this dict.
        """

    @abstractmethod
    def predict(self, ic: np.ndarray, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Forward inference at scattered query points.

        Parameters
        ----------
        ic : (nx,)
            Initial condition on the full reference grid.  Operator
            solvers feed this through their branch network; function
            solvers (PINN) treat it as a selector for a per-IC model.
        x : (M,)
            Spatial query points.
        t : (M,)
            Temporal query points, same length as x.

        Returns
        -------
        (M,) ndarray of predicted u values.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist weights and hyper-parameters under `path`."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore a saved solver from `path` (in place)."""

    # ─────────────────────────────────────────────────────────────────────
    # Optional override
    # ─────────────────────────────────────────────────────────────────────

    def rollout(self, ic: np.ndarray, x_grid: np.ndarray,
                t_grid: np.ndarray) -> np.ndarray:
        """Predict u on a full (t × x) grid.

        Parameters
        ----------
        ic     : (nx,)        initial condition.
        x_grid : (nx_q,)      spatial output grid.
        t_grid : (nt_q,)      temporal output grid.

        Returns
        -------
        (nt_q, nx_q) ndarray.  Row index is t, column index is x —
        same layout as dataset['u'][i].
        """
        X, T = np.meshgrid(x_grid, t_grid, indexing="xy")
        u_flat = self.predict(ic, X.ravel(), T.ravel())
        return u_flat.reshape(len(t_grid), len(x_grid))