import os
os.environ.setdefault("DDE_BACKEND", "pytorch")

import argparse
import json

import numpy as np
import torch

from .config import PINNConfig
from .dataset import ColeHopfDataset
from .model import BurgersPINN
from .metrics import compute_metrics


def evaluate(model_dir: str, data_path: str, sample: int = 0) -> dict:
    ds = ColeHopfDataset(data_path, sample=sample)
    pinn = BurgersPINN()
    pinn.load(model_dir)

    pred, t, x = pinn.predict_grid(train_only=False)     # (nt, nx) on dataset grid
    ref = ds.u_ref
    t_end = pinn.cfg.t_train_end or ds.t_train_end
    metrics = compute_metrics(pred, ref, t, t_train_end=t_end)

    torch.save(
        {"u_pred": torch.tensor(pred, dtype=torch.float32),
         "u_ref": torch.tensor(ref, dtype=torch.float32),
         "x": torch.tensor(x, dtype=torch.float32),
         "t": torch.tensor(t, dtype=torch.float32),
         "sample": sample, "nu": ds.nu, "t_train_end": t_end},
        os.path.join(model_dir, "prediction.pt"))

    with open(os.path.join(model_dir, "evaluation.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _print_report(m: dict):
    r = m["relative_l2"]; w = m["windowed_rel_l2"]
    print("\n--- PINN evaluation (relative L2) ---")
    print(f"  global   mean={r['mean']:.3e}  max={r['max']:.3e}  final={r['final']:.3e}")
    for k in ["pre_shock", "shock", "post_shock", "extrap"]:
        print(f"  {k:<10} mean={w[k]['mean']:.3e}  max={w[k]['max']:.3e}")
    print(f"  Linf     mean={m['linf']['mean']:.3e}  max={m['linf']['max']:.3e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model dir (contains metadata.json)")
    p.add_argument("--data", required=True)
    p.add_argument("--sample", type=int, default=0)
    args = p.parse_args()
    m = evaluate(args.model, args.data, sample=args.sample)
    _print_report(m)
    print(f"\n[saved] {os.path.join(args.model, 'evaluation.json')}")
    print(f"[saved] {os.path.join(args.model, 'prediction.pt')}")


if __name__ == "__main__":
    main()
main()
