import os
import json
import numpy as np

from . import canonical_split as cs
from . import metrics as M
from . import reliability_signals as RS

def load_reference(dataset_path: str) -> dict:

    import torch
    blob = torch.load(dataset_path, map_location="cpu", weights_only=False)

    def npv(v):
        if hasattr(v, "detach"):
            return v.detach().cpu().numpy()
        if hasattr(v, "numpy"):
            return v.numpy()
        return np.asarray(v)

    return {
        "u": npv(blob["u"]).astype(np.float64),       
        "ICs": npv(blob["ICs"]).astype(np.float64),   
        "x": npv(blob["x"]).astype(np.float64),      
        "t": npv(blob["t"]).astype(np.float64),   
        "nu": float(blob["nu"]),
        "t_train_end": float(blob.get("t_train_end", cs.T_TRAIN_END)),
    }

def evaluate_solver(solver, reference: dict, sample_indices=None,
                    keep_curves: bool = True, with_signals: bool = True) -> dict:
    u = reference["u"]
    x = reference["x"]
    t = reference["t"]
    nu = reference["nu"]
    t_end = reference.get("t_train_end", cs.T_TRAIN_END)

    if sample_indices is None:
        sample_indices = cs.EVAL_IDX

    # A single-instance solver (PINN) can only predict its own trained IC;
    # operators handle all. This stops a per-IC model being graded on ICs
    # it never saw.
    if hasattr(solver, "supported_samples"):
        sample_indices = list(solver.supported_samples(sample_indices))

    per_sample = {}
    for i in sample_indices:
        ic = u[i, 0, :]                         
        u_pred = solver.rollout(ic, x, t)     

        m = M.compute_metrics(u_pred, u[i], t, t_train_end=t_end)
        m["regime"] = cs.regime_of(i)

        if with_signals:
            sig = RS.compute_signals(u_pred, x, t, nu, t_train_end=t_end,
                                     keep_curves=keep_curves)
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

def load_solver(model_key, checkpoint: str):
    from . import persistence
    if model_key is None:
        return persistence.load_any(checkpoint)
    return persistence.load_solver(model_key, checkpoint)


def evaluate_checkpoint(model_key, checkpoint: str, dataset_path: str,
                        sample_indices=None, out_dir: str = None,
                        with_signals: bool = True) -> dict:
    reference = load_reference(dataset_path)
    solver = load_solver(model_key, checkpoint)
    results = evaluate_solver(solver, reference, sample_indices=sample_indices,
                              with_signals=with_signals)
    if out_dir:
        results["_path"] = write_results(results, out_dir, model_key.lower())
    return results
