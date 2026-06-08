"""
================================================================================
RUN COST  -  the Module 3 command-line tool
Team : Turingz   File : module3_deployment/run_cost.py

Mirrors common/evaluate.py in spirit: one CLI that works for every model,
loading it through the SHARED loader (common.persistence / common.evaluation)
so there is never a model-specific code path.

Two subcommands:

  profile  - measure ONE trained model and write its cost+accuracy JSON:
      python -m module3_deployment.run_cost profile \
          --model fno \
          --checkpoint ml_models/fno/checkpoints/fno_burgers.pt \
          --data data/colehopf/burgers_1d_cole_hopf.pt \
          --train-wall 312.5

  compare  - read every results/cost/*_cost.json, build the HEI trade-off
             table + recommendation, and render the charts:
      python -m module3_deployment.run_cost compare

Output schema mirrors the rest of the project: results/cost/<model>_cost.json,
plus results/cost/trade_off.json and PNG charts.
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import glob
import argparse
from typing import Dict, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common import evaluation as EV          # noqa: E402
from common import persistence as PERS       # noqa: E402

from . import cost_accuracy
from . import scalability
from . import cost_meter

DEFAULT_OUT = os.path.join("results", "cost")


# ─────────────────────────────────────────────────────────────────────────────
def _read_train_wall(train_wall: Optional[float], train_log: Optional[str]) -> float:
    """Resolve the one-training wall_time_s from a flag or a train-log JSON."""
    if train_wall is not None:
        return float(train_wall)
    if train_log and os.path.isfile(train_log):
        try:
            with open(train_log) as f:
                blob = json.load(f)
            for k in ("wall_time_s", "wall_time", "train_time_s"):
                if k in blob:
                    return float(blob[k])
            # nested fit info
            fit = blob.get("fit") or blob.get("train_summary") or {}
            if "wall_time_s" in fit:
                return float(fit["wall_time_s"])
        except Exception as e:
            print(f"  ! could not read train log {train_log}: {e}")
    return 0.0


def cmd_profile(args) -> None:
    print(f"[profile] loading reference: {args.data}")
    reference = EV.load_reference(args.data)

    if args.model:
        print(f"[profile] loading {args.model} from {args.checkpoint}")
        solver = EV.load_solver(args.model, args.checkpoint)
        model_type = args.model
    else:
        print(f"[profile] loading via manifest from {args.checkpoint}")
        solver = PERS.load_any(args.checkpoint)
        model_type = (PERS.read_manifest(args.checkpoint) or {}).get("model_type", "unknown")

    train_wall = _read_train_wall(args.train_wall, args.train_log)
    samples = [int(s) for s in args.samples.split(",")] if args.samples else None

    profile = cost_accuracy.profile_solver(
        solver, reference, model_type, train_wall_s=train_wall,
        eval_samples=samples, cost_sample=args.cost_sample,
        repeats=args.repeats, warmup=args.warmup)
    profile["device"] = cost_meter.device_info()

    if args.scalability:
        print("[profile] running grid-size scalability sweep ...")
        profile["scalability"] = scalability.grid_sweep(
            solver, reference, sample_index=args.cost_sample,
            repeats=max(3, args.repeats // 2))

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{model_type.lower()}_cost.json")
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"[profile] wrote {out_path}")
    _print_profile_summary(profile)


def cmd_compare(args) -> None:
    files = sorted(glob.glob(os.path.join(args.out, "*_cost.json")))
    if not files:
        print(f"[compare] no *_cost.json found in {args.out}. Run 'profile' first.")
        return
    profiles = []
    for fp in files:
        with open(fp) as f:
            profiles.append(json.load(f))
    print(f"[compare] {len(profiles)} model profile(s): "
          f"{', '.join(p['name'] for p in profiles)}")

    trade = cost_accuracy.build_trade_off(profiles, accuracy_key=args.accuracy_key)
    out_json = os.path.join(args.out, "trade_off.json")
    with open(out_json, "w") as f:
        json.dump(trade, f, indent=2)
    print(f"[compare] wrote {out_json}")

    print("\n=== Hybrid Efficiency Index (ranked) ===")
    for r in trade["ranked"]:
        print(f"  {r['name']:<18} HEI={r['HEI']:<10} "
              f"relL2={r['relative_l2']:<10} "
              f"lat={r.get('latency_s')}s  mem={r.get('memory_mb')}MB")
    print("\nRecommendation:", trade["recommendation"]["recommendation"])

    try:
        from . import plots
        png = plots.render_trade_off(trade, args.out)
        print(f"[compare] charts: {png}")
    except Exception as e:
        print(f"[compare] (charts skipped: {e})")


def _print_profile_summary(p: Dict) -> None:
    print("  ---------------------------------------------")
    print(f"  model        : {p['name']} ({p['model_type']})")
    print(f"  #parameters  : {p['n_parameters']:,}")
    print(f"  trainings    : {p['trainings_per_benchmark']} per full benchmark")
    print(f"  rel-L2 (glob): {p['relative_l2_global']}")
    print(f"  rel-L2(extr) : {p['relative_l2_extrap']}")
    print(f"  latency      : {p['latency_ms_median']} ms / rollout")
    print(f"  throughput   : {p['throughput_points_per_s']} points/s")
    print(f"  mem (delta)  : {p['memory_mb']} MB")
    print(f"  train wall   : {p['train_wall_s']} s (one run)")
    print("  ---------------------------------------------")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Module 3 cost/deployment tool")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile", help="measure one trained model")
    p.add_argument("--model", choices=["pinn", "fno", "deeponet"], default=None,
                   help="model type; omit to auto-detect via manifest")
    p.add_argument("--checkpoint", required=True, help="checkpoint dir or file")
    p.add_argument("--data", required=True, help="Cole-Hopf reference .pt")
    p.add_argument("--train-wall", type=float, default=None,
                   help="wall_time_s for one training run")
    p.add_argument("--train-log", default=None,
                   help="JSON train log to read wall_time_s from")
    p.add_argument("--samples", default=None, help="comma list of IC indices")
    p.add_argument("--cost-sample", type=int, default=0)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--scalability", action="store_true",
                   help="also run the grid-size complexity sweep")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.set_defaults(func=cmd_profile)

    c = sub.add_parser("compare", help="build HEI trade-off across all profiles")
    c.add_argument("--out", default=DEFAULT_OUT)
    c.add_argument("--accuracy-key", default="relative_l2_global",
                   choices=["relative_l2_global", "relative_l2_extrap",
                            "relative_l2_final"])
    c.set_defaults(func=cmd_compare)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
