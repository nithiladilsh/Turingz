import os
os.environ.setdefault("DDE_BACKEND", "pytorch")

import argparse
import json

from .config import PINNConfig
from .dataset import ColeHopfDataset
from .model import BurgersPINN


def main():
    p = argparse.ArgumentParser(description="Train PINN for periodic Burgers (Cole-Hopf reference).")
    p.add_argument("--data", required=True, help="path to burgers_1d_cole_hopf.pt")
    p.add_argument("--out", default="results/pinn", help="output directory")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--adam-iters", type=int, default=None)
    p.add_argument("--no-lbfgs", action="store_true")
    p.add_argument("--soft-periodic", action="store_true",
                   help="use explicit PeriodicBC loss instead of hard feature transform")
    p.add_argument("--anchors", action="store_true", help="data-informed PINN ablation")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed (default: canonical SEED from common.canonical_split)")
    p.add_argument("--float64", action="store_true")
    args = p.parse_args()

    from common.canonical_split import T_TRAIN_END, SEED, regime_of

    seed = args.seed if args.seed is not None else SEED
    cfg = PINNConfig(sample=args.sample, seed=seed, float64=args.float64)
    cfg.t_train_end = T_TRAIN_END         
    if args.adam_iters is not None:
        cfg.adam_iters = args.adam_iters
    if args.no_lbfgs:
        cfg.lbfgs = False
    if args.soft_periodic:
        cfg.hard_periodic = False
    if args.anchors:
        cfg.use_data_anchors = True

    regime = regime_of(args.sample)
    out_dir = os.path.join(args.out, f"sample{args.sample}")
    os.makedirs(out_dir, exist_ok=True)

    ds = ColeHopfDataset(args.data, sample=args.sample)
    print(f"[data] nu={ds.nu:.6e}  x in [{ds.x_start},{ds.x_end})  "
          f"nx={ds.nx} nt={ds.nt}  t_train_end={cfg.t_train_end}")
    print(f"[split] sample {args.sample} regime={regime}  (operators' in_dist/ood label)")
    print(f"[cfg ] hard_periodic={cfg.hard_periodic} anchors={cfg.use_data_anchors} "
          f"adam={cfg.adam_iters} lbfgs={cfg.lbfgs} seed={seed}")

    pinn = BurgersPINN(cfg)
    info = pinn.fit(ds, out_dir=out_dir)

    summary = {
        "final_loss_train": info["final_loss_train"],
        "final_loss_test": info["final_loss_test"],
        "loss_order": info["loss_order"],
        "wall_time_s": info["wall_time_s"],
        "name": info["name"],
        "sample": args.sample,
        "regime": regime,
        "t_train_end": cfg.t_train_end,
        "split_source": "common.canonical_split",
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {info['name']} ({regime}) trained in {info['wall_time_s']:.1f}s; "
          f"model + metadata saved to {out_dir}")


if __name__ == "__main__":
    main()
