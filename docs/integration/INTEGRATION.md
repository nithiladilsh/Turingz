# Module 3 — Integration Guide (Step 9)

Integration = replace the four stand-ins with the real modules by implementing
four hooks in `hybrid_pde/control_214133E/integrate.py`, then calling `main()`.
Because every seam is a fixed contract, nothing else in M3 changes.

## The four hooks to implement

Each returns an object matching a contract you already built against.

1. `load_ml_solver()` -> a solver with `rollout(ic, x, t) -> array (n_t, n_x)`.
   Wrap your FNO with the adapter:
   ```python
   from hybrid_pde.control_214133E.integrate import FunctionSolver
   def load_ml_solver():
       model = load_fno()                      # your existing loader
       def fno_rollout(ic, x, t):
           return fno_infer(model, ic, x, t)   # must return (len(t), len(x))
       return FunctionSolver("FNO", fno_rollout)
   ```

2. `load_numerical_solver()` -> same contract, wrapping the spectral solver:
   ```python
   def load_numerical_solver():
       return FunctionSolver("spectral", spectral_rollout)
   ```

3. `load_trust()` -> M1's trust as a callable `(state, t) -> (trust in [0,1], flag)`:
   ```python
   from hybrid_pde.control_214133E.trigger import RealTrust
   def load_trust():
       m1 = load_m1_estimator()                # your teammate's module
       return RealTrust(lambda state, t: m1.trust(state, t))
   ```

4. `load_coupling()` -> M2 with `correct(state, x, t0, t1, num)` and
   `rollout(ic, x, t, ml, num, trigger)`:
   ```python
   from hybrid_pde.control_214133E.coupling import RealCoupling
   def load_coupling():
       return RealCoupling(load_m2_coupling())
   ```

## Run it
```bash
python -c "from hybrid_pde.control_214133E.integrate import main; print(main())"
```
This loads the shared Cole-Hopf reference, runs the hybrid on the shared test
ICs (900-909), and returns the frontier rows with REAL numbers (cost, mean
error, std, hit-rate). Per-step costs come from the profiler automatically.

## Order of integration (restructured plan, Sec. 8)
1. M1 trust -> M2 coupling (trust-gated coupling).
2. That -> M3 runtime (cost-controlled runtime).
Swap one hook at a time and re-run, so any problem is easy to localise.

## Regenerate the real plots
Once `main()` runs, re-run the Step 7 (frontier) and Step 8 (robustness)
experiments with the real solvers to refresh `results/m3/step7_*` and
`step8_*` with real numbers and real error bars. The plotting code is
unchanged; only the solvers/trust/coupling differ.

## Checklist
- [ ] `load_ml_solver()` returns real FNO behind `rollout(ic,x,t)`
- [ ] `load_numerical_solver()` returns real spectral
- [ ] `load_trust()` returns M1 trust as `(state,t)->(trust,flag)`
- [ ] `load_coupling()` returns M2 with `correct(...)` and `rollout(...)`
- [ ] `integrate.main()` runs and returns real frontier rows
- [ ] OOD/unseen ICs taken from the shared test set (not authored by M3)
- [ ] Step 7 & Step 8 plots regenerated with real numbers

## Partial integration — do this NOW (no need to wait for teammates)

The FNO and spectral solvers already exist. So you can get REAL-solver numbers
today, keeping M1 (trust) and M2 (coupling) as honest stubs until they are ready.

Implement only the two solver hooks (`load_ml_solver`, `load_numerical_solver`),
then run:

```bash
python -c "from hybrid_pde.control_214133E.integrate import partial_main, save_report; save_report(partial_main(), 'results/m3/step9b_partial_integration', 'real_solver_frontier')"
```

This uses your REAL FNO + spectral on the shared test ICs (900-909), with a
time-based trust stub (switch near the training horizon) and the simple coupling
stub. It writes `real_solver_frontier.png` + `.json` under
`results/m3/step9b_partial_integration/`.

What you get: your first frontier with REAL solver costs and REAL errors. Expect
pure-numerical to have a small non-zero error and the hybrid to sit just above it
at lower cost - the realistic version of the headline plot.

When M1 and M2 are ready, implement the other two hooks and call `main()` for the
full integration.

## UPDATE — the two solver hooks are now pre-filled and (spectral) verified

`load_ml_solver()` (real FNO) and `load_numerical_solver()` (real spectral) are
already implemented in `integrate.py`:
- **spectral** is VERIFIED here, torch-free, against your Cole-Hopf reference:
  0.00036 relative L2 over the full trajectory (0.00006 in extrapolation) - it
  also works as the per-step corrector. Nothing for you to change.
- **FNO** is written (loads `results/fno/fno.pt` via neuralop) and needs your
  torch env to run. If your FNO loader differs, adjust `fno_rollout` / `load_ml_solver`.

So partial integration is now a single command (no editing needed if your FNO
checkpoint is at `results/fno/`):

```bash
python -c "from hybrid_pde.control_214133E.integrate import partial_main, save_report; save_report(partial_main(), 'results/m3/step9b_partial_integration', 'real_solver_frontier')"
```

If it errors on the FNO load, paste the error and we fix `load_ml_solver` together.
