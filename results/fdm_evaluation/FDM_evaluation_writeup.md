# Why FDM is not used as the main reference solver

**Module:** Finite-Difference Method (FDM) — Sandeepa (214039V), Team Turingz
**Question this answers:** We built three numerical solvers (Cole-Hopf, spectral, FDM). Cole-Hopf is used as the main reference for generating the dataset. This document gives the reasons and evidence for *why FDM was not chosen* for that role.

The argument is built in two parts. **Part A judges FDM entirely on its own** — no other method is used. **Part B confirms Part A** by comparing against the other two solvers.

---

## Short version

FDM is a *valid and stable* solver, but it is *not accurate enough* to be the source of truth for our dataset. On a realistic input, at our 512-point grid, FDM does not even agree with a finer FDM run (it is still ~12% off), and the method adds a large amount of "fake" smoothing that blurs the sharp part of the wave — the exact feature our research is about. We therefore keep FDM only as a cheap, independent baseline, and use Cole-Hopf (confirmed by the spectral method) as the reference.

---

## Part A — FDM judged on its own (no other method used)

### Test 1 — Does FDM agree with itself?

The simplest test of any numerical method: if you make the grid finer, a *good* method gives almost the same answer. If the answer keeps changing a lot, the method has not "settled" and cannot be trusted. We compare FDM at each grid size to FDM at the next finer grid size. **No other solver is involved.**

| Grid compared to next finer | Smooth input (easy) | Realistic bumpy input |
| :--- | :--- | :--- |
| 256 → 512 | 1.45% | 16.86% |
| **512 → 1024 (our grid)** | **0.91%** | **11.90%** |
| 1024 → 2048 | 0.53% | 7.56% |

Two findings:

1. On a **realistic input, FDM at our 512 grid is still ~12% away** from the same method run on a finer grid. It has not converged to its own answer.
2. The differences shrink **slowly** — the measured convergence order is about **0.5–0.6** (where 1.0 already counts as slow). This means even doubling the grid only cuts the error by roughly a third, so reaching high accuracy would need an enormous, expensive grid.

### Test 2 — How much "fake" smoothing does FDM add?

The simple upwind rule FDM uses has a well-known side effect: it quietly behaves like *extra* viscosity (extra smoothing) of size `|u| · dx / 2`. This is called **numerical diffusion**. We can compare this fake smoothing directly to the *real* viscosity in the problem (`ν = 1/(100π) ≈ 0.00318`). This is purely a property of the method.

| Grid points | Fake smoothing as % of the real physics |
| :--- | :--- |
| 256 | 123% |
| **512 (our grid)** | **61%** |
| 1024 | 31% |
| 2048 | 15% |

At our 512 grid, FDM adds fake smoothing equal to **61% of the real physical effect**. In other words, the simulation behaves as if the viscosity were much larger than it actually is, which rounds off the sharp wave front. To push this fake smoothing below 10% of the real value, we would need about **3,100 grid points — six times more than we use**, and far more time steps.

### Test 3 — Does FDM obey the physics?

To be fair to FDM, we also check what it does *right*. The equation requires total "mass" to stay constant and energy to only ever decrease. On the 512 grid, FDM keeps mass constant (drift ≈ 1e-16, machine precision) and energy never wrongly increases. **So FDM is a stable, well-behaved solver — it is simply not accurate.** This is exactly why it stays useful as a baseline.

---

## Part B — Cross-check (confirming Part A)

Part A already shows FDM is inaccurate. To confirm the *size* of that error on our real dataset, we compare the three already-generated datasets. All three use the **same 8 initial conditions and the same grid** (verified identical), so any difference is purely the method.

Averaged over all 8 cases and all time steps, difference from Cole-Hopf (the exact method):

| Method | Difference from Cole-Hopf |
| :--- | :--- |
| Spectral | **0.01%** |
| FDM | **16.3%** |

This is the decisive confirmation. Cole-Hopf and the spectral method are built in completely different ways, yet they agree to **0.01%** — so that answer is trustworthy. FDM is the **odd one out, disagreeing about 1,500 times more**. FDM is also fine on the easy smooth case (~2%) but poor on the realistic bumpy cases (12%–39%), which matches Part A exactly.

---

## Conclusion

| Reason | Evidence |
| :--- | :--- |
| FDM does not converge at our resolution | 12% self-difference at 512 on realistic inputs; slow order ~0.5 (Test 1) |
| FDM blurs the sharp wave (the feature we study) | adds fake smoothing = 61% of the real physics; needs ~3,100 points to fix (Test 2) |
| FDM error grows with time | difference from Cole-Hopf rises toward 20%+ by t=2 (Part B) |
| FDM is still a valid baseline | stable, conserves mass, energy only decreases (Test 3) |

**Decision:** Cole-Hopf is the main reference (confirmed by spectral). FDM is retained as a cheap, independent baseline / sanity check, not as the source of truth.

*Figures: `fdm_standalone_evaluation.png` (Part A), `fdm_crosscheck.png` (Part B). All numbers: `fdm_evaluation_values.json`. Reproduce with `experiments/evaluate_fdm.py`.*

*Note on "exact": the reference's correctness rests on two independently-built methods (Cole-Hopf and spectral) agreeing to 0.01%. This is the standard, accepted way to establish a trusted reference when no closed-form answer is available for every case.*
