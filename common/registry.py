import os
import json

SCHEMA = "turingz.solver.manifest/v1"
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".ipynb_checkpoints"}


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def discover_models(root: str = ".") -> list:
    records = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        if "manifest.json" not in filenames:
            continue
        man = _read_json(os.path.join(dirpath, "manifest.json"))
        if not man or man.get("schema") != SCHEMA:
            continue
        rec = {
            "model_type": man.get("model_type"),
            "name": man.get("name"),
            "framework": man.get("framework"),
            "checkpoint": os.path.normpath(os.path.relpath(dirpath, root)),
            "sample": None,
            "regime": None,
        }
        if man.get("model_type") == "pinn":
            summ = _read_json(os.path.join(dirpath, "train_summary.json")) or {}
            meta = _read_json(os.path.join(dirpath, "metadata.json")) or {}
            rec["sample"] = summ.get("sample", meta.get("sample"))
            rec["regime"] = summ.get("regime")
        records.append(rec)

    # stable order: type, then sample number (PINN), then name
    records.sort(key=lambda r: (r["model_type"],
                                r["sample"] if r["sample"] is not None else -1,
                                r["name"] or ""))
    return records


def build_index(root: str = ".", out: str = "results/model_registry.json") -> list:
    recs = discover_models(root)
    out_path = os.path.join(root, out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"n_models": len(recs), "models": recs}, f, indent=2)
    return recs


def load_all(root: str = ".", model_type: str = None):
    from .persistence import load_any
    for rec in discover_models(root):
        if model_type and rec["model_type"] != model_type:
            continue
        yield rec, load_any(os.path.join(root, rec["checkpoint"]))


if __name__ == "__main__":
    recs = build_index()
    print(f"Discovered {len(recs)} models (written to results/model_registry.json):")
    for r in recs:
        tag = f"sample={r['sample']},{r['regime']}" if r["model_type"] == "pinn" else ""
        print(f"  {r['model_type']:<9} {r['name']:<26} {r['checkpoint']:<34} {tag}")
