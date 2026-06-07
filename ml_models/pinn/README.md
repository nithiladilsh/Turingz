# PINN — Periodic Viscous Burgers (Team Turingz)

Physics-Informed Neural Network solver for

    u_t + u u_x = nu u_xx,   x in [-1, 1) periodic,   t in [0, T]

trained against the Cole-Hopf reference dataset. One trained network solves one
IC instance (single-instance solver), unlike FNO/DeepONet which learn an operator.

## Layout
```
ml_models/pinn/
  config.py      PINNConfig: all hyperparameters + domain/physics knobs
  dataset.py     ColeHopfDataset: grid, nu, IC, train/extrap split, anchors
  model.py       BurgersPINN: geometry, PDE residual, IC, periodicity, train/predict/save/load
  metrics.py     relative L2 / Linf, windowed (pre_shock / shock / post_shock / extrap)
  train.py       CLI training entry point
  evaluate.py    metrics vs Cole-Hopf, writes evaluation.json + prediction.pt
  visualize.py   heatmaps, snapshots, error-vs-time, energy
```

## Requirements
```
deepxde      # PINN framework
torch        # backend (DDE_BACKEND=pytorch is set in code)
numpy scipy matplotlib
```

## Train (focal sample 0 = sin(pi x))
```
python -m ml_models.pinn.train --data data/colehopf/burgers_1d_cole_hopf.pt --out results/pinn
```
Useful flags: `--sample N`, `--soft-periodic` (explicit PeriodicBC loss instead of the
hard sin/cos feature transform), `--anchors` (data-informed ablation), `--no-lbfgs`,
`--float64`, `--adam-iters K`.

## Evaluate + plot
```
python -m ml_models.pinn.evaluate  --model results/pinn/sample0 --data data/colehopf/burgers_1d_cole_hopf.pt
python -m ml_models.pinn.visualize --model results/pinn/sample0
```

## Interface for the reliability module
```python
from ml_models.pinn import BurgersPINN, ColeHopfDataset

ds   = ColeHopfDataset("data/colehopf/burgers_1d_cole_hopf.pt", sample=0)
pinn = BurgersPINN.load("results/pinn/sample0", ds)

u_pred, t, x = pinn.predict_grid()      # (nt, nx) on the dataset grid
u_pred       = pinn.predict(X)          # X: (N, 2) array of (x, t)
```
`predict_grid` returns the solution on the exact Cole-Hopf grid, so the reliability
module can diff PINN / FNO / DeepONet against the same reference identically.

## Key design notes
- **Periodic BC, not Dirichlet.** The Cole-Hopf solver is periodic; for sample 0 the
  shock forms at the seam x = +/-1. `hard_periodic=True` (default) encodes period-2
  periodicity via a sin/cos input transform, so periodicity holds exactly.
- **Physics-only by default.** Interior Cole-Hopf values are used for validation, not
  training. `use_data_anchors` enables a supervised ablation.
- **Extrapolation.** Training time domain is capped at `t_train_end` (1.0). Evaluation
  queries the full `[0, T]` grid; the `extrap` window (t > 1.0) is the headline metric,
  comparable across all three ML solvers (all trained on [0, 1] only).
- **Reproducible.** Seeded via `dde.config.set_random_seed`; config saved to metadata.json.
```
```
