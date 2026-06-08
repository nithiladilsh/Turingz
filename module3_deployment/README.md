# Module 3 — Deployment & Cost Analysis (214133E)

This package is the **deployment / computational-cost** workstream. It is built
**entirely on top of the shared framework** (`common/`) and the
`AbstractSolver` contract. It does **not** re-implement accuracy, the dataset
split, or model loading — so every number it produces stays consistent with the
reliability and robustness modules. **No file outside `module3_deployment/` was
changed.**

## What it measures (maps 1:1 to the interim slide)

| Interim metric | Where |
|---|---|
| Runtime (execution time) | `cost_meter.measure_inference` (median rollout latency) |
| Memory usage | `cost_meter` peak-memory sampler (CPU via psutil, GPU via pynvml) |
| Computational complexity | `scalability.grid_sweep` (latency vs grid size → fitted exponent `p`) |
| Throughput | `cost_meter` (`points_per_s`, `rollouts_per_s`) |
| Accuracy vs cost trade-off | `cost_accuracy` + `hei` (Hybrid Efficiency Index) |
| Cost–accuracy curves / comparison charts | `plots.render_trade_off` |
| Deployment recommendation | `hei.recommend` |

Plus the deployment engine from the README: `app.py` — a latency-aware FastAPI
**Engine Control Unit** that ingests Module 1's failure score, computes the HEI,
and routes *ML vs numerical correction* while throttling to avoid OOM.

## Files

```
cost_meter.py     runtime / peak memory / throughput / #params (wraps fit + rollout)
scalability.py    complexity vs grid size
cost_accuracy.py  joins shared accuracy (common.metrics) with cost
hei.py            Hybrid Efficiency Index + online RoutingPolicy (PINN per-IC accounting)
run_cost.py       CLI: profile one model, then compare across all
plots.py          trade-off + comparison PNGs
app.py            FastAPI latency-aware orchestrator (Engine Control Unit)
```

## How to run (do this in order tomorrow)

All commands from the **project root**.

### 1. Retrain under the canonical conventions (Handoff §6)
The old checkpoints have no `manifest.json`, so the shared loader can't read
them. Retrain so every model is loadable the same way:

```powershell
# PINN — one model per IC (all 8)
foreach ($i in 0..7) { python -m ml_models.pinn.train --data data/colehopf/burgers_1d_cole_hopf.pt --sample $i }
# FNO — single operator (trains on ICs 0–5, tests 6–7)
python ml_models/fno/train.py
# DeepONet — single operator
python ml_models/deeponet/train.py --data data/colehopf/burgers_1d_cole_hopf.pt --out ml_models/deeponet/checkpoints/m128
```

> If you're tight on time, the cost numbers don't need fully-converged weights
> (latency, memory, #params and throughput are set by the architecture). A
> shorter training run still gives a valid interim profile; just note it.

### 2. Profile each model (cost + accuracy, uniform JSON)

```powershell
python -m module3_deployment.run_cost profile --model fno      --checkpoint ml_models/fno/checkpoints/fno_burgers.pt --data data/colehopf/burgers_1d_cole_hopf.pt --train-log ml_models/fno/checkpoints/fno_train_log.json --scalability
python -m module3_deployment.run_cost profile --model deeponet  --checkpoint ml_models/deeponet/checkpoints/m128       --data data/colehopf/burgers_1d_cole_hopf.pt --scalability
python -m module3_deployment.run_cost profile --model pinn      --checkpoint results/pinn/sample0                      --data data/colehopf/burgers_1d_cole_hopf.pt --train-log results/pinn/sample0/train_summary.json --scalability
```

Each writes `results/cost/<model>_cost.json`. Pass `--train-wall <seconds>` if
you'd rather give the training time directly instead of `--train-log`.

### 3. Build the trade-off + charts + recommendation

```powershell
python -m module3_deployment.run_cost compare
```

Writes `results/cost/trade_off.json` and three PNGs:
`cost_accuracy_tradeoff.png`, `hei_ranking.png`, `performance_comparison.png`.

### 4. (Optional) Demo the deployment engine

```powershell
python -m module3_deployment.app                       # no-server self-demo
uvicorn module3_deployment.app:app --port 8000         # live API: /route, /simulate, /device
```

## The Hybrid Efficiency Index (HEI)

```
accuracy_score = 1 / (1 + relative_L2)
cost_index     = weighted, min-max-normalised blend of
                 (inference latency, peak memory, amortised training cost)
HEI            = accuracy_score / cost_index          # higher = more deployable
```

**Paradigm accounting (required by the compatibility review):** a PINN trains
**one model per IC**, so it is charged `len(EVAL_IDX)=8` trainings; FNO and
DeepONet train once and amortise. This is computed from
`common.canonical_split`, never hard-coded.

## Verified

The full pipeline (cost meter → scalability → accuracy+HEI → routing → plots)
was exercised end-to-end on a numpy-only dummy solver, confirming: parameter
counting, latency/memory/throughput, the complexity-exponent fit, the shared
accuracy join, the PINN 8× training accounting, the router switching to
numerical correction as the failure score rises, and PNG rendering. All
`module3_deployment/*.py` byte-compile.
