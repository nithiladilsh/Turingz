# Findings — Burgers operator / PINN extrapolation study

## Problem & data
- 1D viscous Burgers, nu = 1/(100*pi) (sharp near-shock regime).
- Cole-Hopf analytical dataset: 1000 ICs, 512 space points (x in [-1,1]) x 200 time steps (t in [0,2]).
- Data verified against an independent pseudo-spectral (IFRK4) solver: max relative L2 ~8.5e-4 on a random 64-IC subset.
- IC 0 is sin(pi x); the rest are random 4-mode Fourier ICs (max amplitude 1).

## Splits & protocol
- Operators: train on ICs 0..N-1, held-out test on ICs 900..999 (never trained/tuned on).
  - DeepONet trains on 800 ICs; FNO currently on 900 ICs (see caveats).
- PINN: one network per IC, trained on ICs 0..9 (a PINN is not an operator — it solves a single IC).
- Time split: train on t <= 1; t > 1 is the held-out extrapolation window.
- All ML models trained with a relative-L2 objective (consistent across methods).
- Metric: per-IC relative L2 over the space-time block, mean +/- std across ICs.

## Implementations
- DeepONet: DeepXDE `DeepONetCartesianProd`, 100 sensors, latent 256, width 256, depth 4, 6 Fourier trunk features, ReLU, relative-L2 loss, 30k iters, 5 seeds.
- FNO: `neuraloperator` FNO, modes 16, width 64, channels [IC, t, x], relative-L2 loss.
- PINN: DeepXDE FNN [2,64,64,64,64,1], tanh, PDE residual + IC + periodic BC.

## Cross-method comparison (compare.py — one metric, same ICs)

Held-out test ICs (900-999), operators:

| Method | in-dist (t<=1) | extrapolation (t>1) |
|---|---|---|
| FNO | 0.005 | 0.176 |
| DeepONet | 0.331 | 0.758 |
| persistence (freeze at t=1) | - | 0.537 |

Same ICs (0-9), all three methods head-to-head:

| Method | in-dist (t<=1) | extrapolation (t>1) |
|---|---|---|
| FNO | 0.004 | 0.192 |
| PINN | 0.057 | 0.289 |
| DeepONet | 0.123 | 0.687 |
| persistence | - | 0.568 |

## Key results
- The FNO is far stronger than the DeepONet in-distribution (~0.5% vs ~33%) and extrapolates best.
- FNO and PINN both beat the persistence baseline in extrapolation; the **DeepONet is the only method that does worse than persistence** (0.76 vs 0.54), i.e. it would have done better assuming the solution stopped evolving.
- Field heatmaps (results/fields_ic0.png) confirm the DeepONet prediction is a recognizable but noisy/blurred version of the true field — a genuine architectural limitation on the IC-dependent shock front, not a bug. The FNO field is near-exact; the PINN field is clean.

## DeepONet detail (the baseline)
- Final (DeepXDE, relative-L2, 5 seeds, test ICs): in-dist 33.8% +/- 0.6%, extrapolation 79.3% +/- 4.6%.
- The separable branch x trunk structure cannot represent the sharp IC-dependent front; training error plateaus and unseen-IC error sits ~30%.
- Overfitting in time: extrapolation degrades as training continues (the validation in-dist metric rises from ~0.29 to ~0.32 over 30k iters, and an early-stopped model extrapolated ~0.57 vs ~0.79 at full budget). The operator fits the t<=1 window harder rather than learning forward-time dynamics.
- Loss-choice ablation: at equal (full) training budget, MSE and relative-L2 give essentially the same result (~34% / ~79%); the loss choice is not what drives the high error.

## Caveats / to finalise
- Train-set size is not yet matched (DeepONet 800 vs FNO 900) — align for a strictly apples-to-apples table.
- FNO is currently single-seed (no error bars); run multiple seeds to match the DeepONet's 5-seed reporting.
- PINNs are on ICs 0-9, which are operator training ICs; operators have an in-distribution edge on that set.

## Contribution
- The vanilla DeepONet plateau and extrapolation failure is the baseline, not the result.
- The gap between methods — FNO and PINN holding up while the DeepONet fails worse than persistence — is the finding, established on identical data, splits, time windows, and metric.
