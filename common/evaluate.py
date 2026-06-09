import argparse

from . import evaluation as E
from . import canonical_split as cs


def _print_report(results: dict):
    print(f"\n=== Shared evaluation: {results['solver']} ===")
    print(f"  t_train_end = {results['t_train_end']}   "
          f"samples = {results['sample_indices']}")
    print("  per-sample (regime | global relL2 | extrap relL2 | residual | resid-vs-error corr):")
    for i, m in results["samples"].items():
        g = m["relative_l2"]["mean"]
        ex = m["windowed_rel_l2"]["extrap"]["mean"]
        sig = m.get("reliability_signals", {})
        res = sig.get("residual_rms", {}).get("mean", float("nan"))
        corr = sig.get("residual_vs_error", {}).get("spearman", float("nan"))
        print(f"    sample {i:>2}  {m['regime']:<8} "
              f"global={g:.3e}  extrap={ex:.3e}  residual={res:.3e}  corr={corr:+.2f}")
    print("  aggregate (by regime):")
    for tag, a in results["aggregate"].items():
        print(f"    {tag:<8} n={a['n_samples']}  "
              f"global={a['global_rel_l2']['mean']:.3e}  "
              f"shock={a['shock_rel_l2']['mean']:.3e}  "
              f"extrap={a['extrap_rel_l2']['mean']:.3e}")


def main():
    p = argparse.ArgumentParser(description="Shared cross-model evaluation.")
    p.add_argument("--model", required=True, choices=["pinn", "fno", "deeponet"])
    p.add_argument("--checkpoint", required=True,
                   help="PINN/DeepONet: directory; FNO: .pt file")
    p.add_argument("--data", required=True, help="path to the Cole-Hopf .pt")
    p.add_argument("--samples", type=int, nargs="*", default=None,
                   help="IC indices to evaluate (default: canonical EVAL_IDX)")
    p.add_argument("--out", default="results/evaluation",
                   help="output directory for the uniform JSON")
    p.add_argument("--no-signals", action="store_true",
                   help="skip the model-agnostic reliability signals")
    args = p.parse_args()

    samples = args.samples if args.samples else cs.EVAL_IDX
    results = E.evaluate_checkpoint(
        args.model, args.checkpoint, args.data,
        sample_indices=samples, out_dir=args.out,
        with_signals=not args.no_signals,
    )
    _print_report(results)
    print(f"\n[saved] {results.get('_path')}")


if __name__ == "__main__":
    main()
