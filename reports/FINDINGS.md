# Findings so far — Burgers operator/PINN study

## Solvers & data quality
- The spectral (IFRK4) solver had a hidden bug that made it only 2nd-order accurate; fixed, it is now properly 4th-order.
- Cole-Hopf (analytical) and spectral agree to ~8x10^-4 on a random 64-IC subset, so the training data is trustworthy (verified against an independent method).
- The original dataset had only 8 initial conditions (ICs) — far too few; it caused the operators to memorize rather than learn.
- Dataset scaled to 1000 ICs (Cole-Hopf), each on a 512-point space grid x 200 time steps (x in [-1, 1], t in [0, 2]).

## DeepONet
- With 8 ICs: ~4% error on training ICs but ~87% on unseen ICs — pure memorization.
- With 1000 ICs: generalization to unseen ICs improved from ~87% to ~28%.

### Final DeepONet results (1000 ICs, 5 seeds, mean +/- std)
- Train, in-distribution (seen ICs, t <= 1): 14.0% +/- 0.4%
- Validation, in-distribution (unseen ICs, t <= 1): 27.4% +/- 0.4%
- Test, in-distribution (unseen ICs, t <= 1): 28.1% +/- 0.3%
- Test, extrapolation (unseen ICs, t > 1): 68.2% +/- 6.2%
- The tight std across seeds shows the ~14% training ceiling and the 28% -> 68% in-distribution-to-extrapolation cliff are robust, not noise.

### Extrapolation analysis (held-out test ICs, t > 1)
- Error grows with time past t = 1, reaching ~99% (near-total) at the final time t = 2.
- DeepONet extrapolation error (~70%) is WORSE than a naive persistence baseline (~53%) that just freezes the solution at t = 1.
- Interpretation: in the extrapolation window the operator has not learned the forward-time dynamics at all — it would have done better assuming the solution stopped evolving. Strong, clean evidence of extrapolation failure.

- The model then hit a ceiling: training error will not go below ~14%. It underfits.
- Standard fixes tried (bigger network, spatial Fourier features, space-time Fourier features) did NOT break the ~14% plateau.
- Conclusion: this is a known limitation of vanilla DeepONet on shock-forming problems — not a bug. The sharp, IC-dependent shock front is what its separable structure cannot represent well.
- Extrapolation in time (t > 1) fails badly (~60-100% error) — expected, and characterizing this failure is a goal of the study.
- Sensor count (64 to 256) makes no meaningful difference — reported as "insensitive."
- ReLU is used, not tanh: tanh fits marginally better in-distribution but destroys extrapolation, which is the metric of interest.

## Data splits
- IC split: 800 train / 100 validation / 100 test (ICs 0-799 / 800-899 / 900-999). Test ICs are never used in training or tuning.
- Time split: train only on t <= 1 (in-distribution window, ~100 steps); t > 1 is held out as the extrapolation window.

## DeepONet training setup
- Trains on the 800 training ICs, restricted to t <= 1.
- Mini-batched: 64 random ICs and 8,192 random space-time points per step.
- Evaluated on train/val/test ICs in both time windows.

## Sensor sweep setup
- Trains on the same 800 training ICs (t <= 1).
- Selects sensor count using the 100 validation ICs (in-distribution error); test ICs stay untouched.

## Cross-method comparison (the actual contribution)
- Vanilla DeepONet plateauing and failing to extrapolate is the baseline.
- FNO is designed for shock/transport problems and is expected to do better; the gap between methods is the finding.
