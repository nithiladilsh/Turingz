"""
================================================================================
SHARED EVALUATION RUNNER  —  one pipeline for every model
Team : Turingz   File : common/evaluation.py

Drives ANY solver that implements the AbstractSolver interface (PINN, FNO,
DeepONet) through exactly the same evaluation:

    for each evaluated initial condition:
        u_pred  = solver.rollout(ic, x_grid, t_grid)          # full horizon
        metrics = common.metrics.compute_metrics(u_pred, u_ref, t)   # accuracy
        signals = common.reliability_signals.compute_signals(u_pred, ...)  # no-ref

Because it only uses the shared interface, the shared metrics and the shared
(model-agnostic) reliability signals, the three models are graded by identical
code. This replaces the three separate per-model evaluation scripts AND ensures
the PDE-residual signal is computed the same way for all of them.
================================================================================
"""

import os
import json
import numpy as np

from . import canonical_split as cs
from . import metrics as M
from . import reliability_signals as RS


# ─────────────────────────────────────────────────────────────────────────────
# Reference dataset loader (numpy-only dict)
# ─────────────────────────────────────────────────────────────────────────────
def load_reference(dataset_path: str) -> dict:
    """Load the Cole-Hopf reference .pt as a numpy-only dict."""
    import torch  # imported lazily so this module loads without torch present
    blob = torch.load(dataset_path, map_location="cpu", weights_only=False)

    def npv(v):
        if hasattr(v, "detach"):
            return v.detach().cpu().numpy()
        if hasattr(v, "numpy"):
            return v.numpy()
        return np.asarray(v)

    return {
        "u": npv(blob["u"]).astype(np.float64),       # (N, nt, nx)
        "ICs": npv(blob["ICs"]).astype(np.float64),   # (N, nx)
        "x": npv(blob["x"]).astype(np.float64),       # (nx,)
        "t": npv(blob["t"]).astype(np.float64),       # (nt,)
        "nu": float(blob["nu"]),
        "t_train_end": float(blob.get("t_train_end", cs.T_TRAIN_END)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core: evaluate one fitted/loaded solver on a set of ICs
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_solver(solver, reference: dict, sample_indices=None,
                    keep_curves: bool = True, with_signals: bool = True) -> dict:
    """Evaluate a solver across initial conditions on the full-horizon grid.

    solver         : a loaded/fitted AbstractSolver (has .name, .rollout).
    reference      : dict from load_reference().
    sample_indices : which ICs to evaluate (default: the canonical EVAL_IDX).
    with_signals   : also compute model-agnostic reliability signals (PDE
                     residual etc.) from the predicted field, and how well the
                     residual tracks the true per-time error.
    Returns a uniform result dict: per-sample accuracy + reliability signals
    + cross-sample aggregate.
    """
    u = reference["u"]
    x = reference["x"]
    t = reference["t"]
    nu = reference["nu"]
    t_end = reference.get("t_train_end", cs.T_TRAIN_END)

    if sample_indices is None:
        sample_indices = cs.EVAL_IDX

    per_sample = {}
    for i in sample_indices:
        ic = u[i, 0, :]                          # IC on the full grid (t = 0)
        u_pred = solver.rollout(ic, x, t)        # (nt, nx), full horizon

        m = M.compute_metrics(u_pred, u[i], t, t_train_end=t_end)
        m["regime"] = cs.regime_of(i)

        if with_signals:
            sig = RS.compute_signals(u_pred, x, t, nu, t_train_end=t_end,
                                     keep_curves=keep_curves)
            # Reliability premise: does the no-reference residual track the
            # true per-time error? (We have the reference here, so we can check.)
            true_err_curve = m.get("per_time_rel_l2")
            res_curve = sig["residual_rms"].get("curve")
            if true_err_curve is not None and res_curve is not None:
                sig["residual_vs_error"] = RS.correlate_signal_with_error(
                    res_curve, true_err_curve)
            m["reliability_signals"] = sig

        if not keep_curves:
            m.pop("per_time_rel_l2", None)
            m.pop("t", None)
        per_sample[int(i)] = m

    return {
        "solver": getattr(solver, "name", str(type(solver).__name__)),
        "t_train_end": t_end,
        "sample_indices": list(int(i) for i in sample_indices),
        "samples": per_sample,
        "aggregate": M.aggregate_over_samples(per_sample),
        "split": cs.summary(),
    }


def write_results(results: dict, out_dir: str, model_key: str) -> str:
    """Write the evaluation dict to a uniform location/schema."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_key}_evaluation.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Solver loader — delegates to common.persistence (manifest-aware, one path)
# ─────────────────────────────────────────────────────────────────────────────
def load_solver(model_key, checkpoint: str):
    """Construct and restore a solver. If model_key is None, the type is read
    from the checkpoint directory's manifest.json via common.persistence."""
    from . import persistence
    if model_key is None:
        return persistence.load_any(checkpoint)
    return persistence.load_solver(model_key, checkpoint)


def evaluate_checkpoint(model_key, checkpoint: str, dataset_path: str,
                        sample_indices=None, out_dir: str = None,
                        with_signals: bool = True) -> dict:
    """End-to-end: load reference + solver, evaluate, optionally write JSON."""
    reference = load_reference(dataset_path)
    solver = load_solver(model_key, checkpoint)
    results = evaluate_solver(solver, reference, sample_indices=sample_indices,
                              with_signals=with_signals)
    if out_dir:
        results["_path"] = write_results(results, out_dir, model_key.lower())
    return results
