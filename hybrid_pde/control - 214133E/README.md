# Module 3 — Cost-Aware Adaptive Control & Deployment (214133E)

The **cost module** of the Turingz hybrid PDE solver. It is *not* a trained model;
it is a measured cost-aware control method plus a deployable runtime. See
[`DEFENSE.md`](DEFENSE.md) for the research gap, contribution, novelty, evidence,
and answers to likely examiner questions.

## What it contains (plan components #14–#19)

```
m3_cost/
  config.py        # benchmark grid, paths, splits (single source of truth)
  groundtruth.py   # torch-free exact ICs + Cole-Hopf + FDM/spectral solvers
  profiler.py      # #14 cost/latency/memory profiler (numerical + ML step costs)
  surrogate.py     # adapter: TorchSurrogate (host) | CachedSurrogate (no torch)
  trigger.py       # M1 stub: physics-residual / oracle / synthetic trust
  coupling.py      # M2 stub: switch / masked hybrid rollout + cost accounting
  controller.py    # #16 trust-gated accuracy-budget scheduler (+ fixed/oracle)
  accuracy_cost.py # #15 accuracy-cost model + #17 Pareto frontier
  runtime.py       # #18 one-knob deployable HybridRuntime
scripts/
  run_frontier.py    # measured cost-vs-accuracy frontier + plot
  run_controller.py  # #19 adaptive vs fixed ablation + plot
  demo.py            # headline live demo panel
tests/test_m3.py     # verification suite (12 invariants, torch-free)
```

## Reproduce in the sandbox (no torch needed)

Uses the real trained **DeepONet** field on test IC 900 and the pure-NumPy solvers.

```bash
cd "hybrid_pde/control - 214133E"
python3 tests/test_m3.py                 # 12/12 invariants pass
python3 scripts/run_frontier.py          # results/frontier_deeponet-cached.{json,png}
python3 scripts/run_controller.py        # results/controller_deeponet-cached.{json,png}
python3 scripts/demo.py                  # results/demo_deeponet-cached.png
```

## Run the official numbers on the training host (with torch)

Same code, real models, many ICs. The `--kind` flag selects the surrogate; the
adapter loads the trained weights and the profiler measures the real ML step cost.

```bash
# FNO is the recommended anchor (lowest expected handover error)
python3 scripts/run_frontier.py   --kind fno --ics 900 901 902 903 904
python3 scripts/run_controller.py --kind fno --ics 900 901 902 903 904
python3 scripts/demo.py           --kind fno --ic 900
```

Notes:
- **FNO** is fully wired (`surrogate.TorchSurrogate`). **DeepONet/PINN** host loaders
  are marked integration points — plug in the exact constructor used at train time;
  DeepONet accuracy on IC 900 already works via `CachedSurrogate`.
- Cost is reported in two units: **numerical-step-equivalents** (hardware-independent,
  use this in the report) and **wall-ms** (uses measured step costs, host-specific).
- At integration the synthetic/physics trust is replaced by **M1's calibrated trust**
  and the switch coupling by **M2's re-anchoring coupling**, through the fixed data
  contracts — no controller change required.

## Key measured results (sandbox, IC 900, DeepONet)
- one FDM step **0.64 ms**, one spectral step **15.6 ms** (≈24×), full spectral solve **3.29 s**.
- hybrid beats pure-ML by ~6× accuracy; reaches 2×-better-than-ML at **2.34× lower cost** than pure-numerical.
- adaptive controller saves up to **29% numerical steps** at loose targets and hits tight targets the fixed schedule misses.
- honest limitation: hybrid accuracy floor (0.102) is capped by surrogate handover error — quantified, and the reason FNO is worth running.
