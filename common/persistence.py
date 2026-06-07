"""
================================================================================
UNIFORM PERSISTENCE  —  one save/load convention for every solver
Team : Turingz   File : common/persistence.py

Before this, the three models persisted differently: FNO wrote a single .pt
file while PINN and DeepONet wrote directories. That meant the evaluation
harness needed to know each model's quirk to reload it.

Convention now: every solver's save(path) writes a DIRECTORY containing the
model's native weights PLUS a uniform `manifest.json`. The manifest records
the model type and the checkpoint filename, so any solver can be reloaded
through ONE function — load_any(dir) — without being told which model it is.

    manifest.json
    {
      "schema": "turingz.solver.manifest/v1",
      "model_type": "pinn" | "fno" | "deeponet",
      "name": "<solver.name>",
      "checkpoint": "<weights filename inside this directory>",
      "framework": "<deepxde-pytorch | pytorch-neuralop>",
      "saved_at": "<iso8601>"
    }
================================================================================
"""

import os
import json
import datetime

MANIFEST_NAME = "manifest.json"
SCHEMA = "turingz.solver.manifest/v1"


# ─────────────────────────────────────────────────────────────────────────────
# Manifest read / write
# ─────────────────────────────────────────────────────────────────────────────
def write_manifest(out_dir: str, model_type: str, checkpoint: str,
                   name: str = None, framework: str = None,
                   extra: dict = None) -> str:
    """Write the uniform manifest.json into out_dir. Returns its path."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "schema": SCHEMA,
        "model_type": model_type,
        "name": name,
        "checkpoint": checkpoint,
        "framework": framework,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        manifest["extra"] = extra
    path = os.path.join(out_dir, MANIFEST_NAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def read_manifest(path: str) -> dict:
    """Read manifest.json from a directory (or return None if absent)."""
    mpath = path
    if os.path.isdir(path):
        mpath = os.path.join(path, MANIFEST_NAME)
    if not os.path.isfile(mpath):
        return None
    with open(mpath) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Loader dispatch — one entry point for all model types
# ─────────────────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def construct(model_type: str):
    """Build an empty solver of the requested type (lazy imports)."""
    import sys
    key = model_type.lower()
    if key == "pinn":
        from ml_models.pinn.model import BurgersPINN
        return BurgersPINN()
    if key == "fno":
        d = os.path.join(_REPO_ROOT, "ml_models", "fno")
        if d not in sys.path:
            sys.path.insert(0, d)
        from fno_solver import FNOSolver
        return FNOSolver()
    if key == "deeponet":
        d = os.path.join(_REPO_ROOT, "ml_models", "deeponet")
        if d not in sys.path:
            sys.path.insert(0, d)
        from deeponet_solver import DeepONetSolver
        return DeepONetSolver()
    raise ValueError(f"unknown model_type '{model_type}' (use pinn|fno|deeponet)")


def load_solver(model_type: str, checkpoint: str):
    """Construct the named solver and restore it from `checkpoint`."""
    s = construct(model_type)
    s.load(checkpoint)
    return s


def load_any(path: str):
    """Reload a solver from a directory WITHOUT being told its type — the
    model_type is read from the manifest. This is the single code path the
    evaluation/reliability/cost modules use."""
    manifest = read_manifest(path)
    if manifest is None:
        raise FileNotFoundError(
            f"No {MANIFEST_NAME} found in '{path}'. Re-save the model with the "
            "current code, or use load_solver(model_type, checkpoint) explicitly.")
    return load_solver(manifest["model_type"], path)
