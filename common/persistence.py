import os
import json
import datetime

MANIFEST_NAME = "manifest.json"
SCHEMA = "turingz.solver.manifest/v1"


def write_manifest(out_dir: str, model_type: str, checkpoint: str,
                   name: str = None, framework: str = None,
                   extra: dict = None) -> str:
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
    mpath = path
    if os.path.isdir(path):
        mpath = os.path.join(path, MANIFEST_NAME)
    if not os.path.isfile(mpath):
        return None
    with open(mpath) as f:
        return json.load(f)


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def construct(model_type: str):
    import sys
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    key = model_type.lower()
    if key == "pinn":
        from ml_models.pinn.model import BurgersPINN
        return BurgersPINN()
    if key == "fno":
        from ml_models.fno.fno_solver import FNOSolver
        return FNOSolver()
    if key == "deeponet":
        from ml_models.deeponet.deeponet_solver import DeepONetSolver
        return DeepONetSolver()
    raise ValueError(f"unknown model_type '{model_type}' (use pinn|fno|deeponet)")


def load_solver(model_type: str, checkpoint: str):
    s = construct(model_type)
    s.load(checkpoint)
    return s


def load_any(path: str):
    manifest = read_manifest(path)
    if manifest is None:
        raise FileNotFoundError(
            f"No {MANIFEST_NAME} found in '{path}'. Re-save the model with the "
            "current code, or use load_solver(model_type, checkpoint) explicitly.")
    return load_solver(manifest["model_type"], path)
