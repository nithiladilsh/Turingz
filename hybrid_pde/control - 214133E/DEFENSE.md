# Module 3 — Cost-Aware Adaptive Control & Deployment
## Research defense: gap, contribution, novelty, and evidence

**Author:** Mendis B.N.D. (214133E) · Team Turingz (Group 74) · University of Moratuwa
**Module:** M3 of *Hybrid Machine Learning and Numerical Methods for PDE Extrapolation*
**Benchmark PDE:** 1D viscous Burgers, ν = 1/(100π), domain x∈[−1,1], t∈[0,2], shock-forming.
**Status of numbers below:** every figure is *measured*, produced by the scripts in
this folder. Sandbox results use the real trained DeepONet field on test IC 900 and
the pure-NumPy numerical solvers; the multi-model (FNO/DeepONet/PINN), multi-IC
numbers are produced by the same scripts on the training host (see `README.md`).

---

## 0. The criticism this module answers

> *"You only have boilerplate code and no contribution — you copied a code and trained an ML model."*

That criticism is fair about **Phase 1** (training surrogates is engineering, not
research). Module 3 is deliberately **not** a trained model. It is a *method*: a
measurement instrument, a control law, and a deployable runtime that together
answer a question nobody in this project had answered — **"how much numerical
computation must a hybrid solver spend, and where, to hit a target accuracy at
minimum cost, and does the resulting hybrid actually beat both pure-ML and
pure-numerical?"** Nothing here is copied; the controller, the cost model, the
profiler and the runtime are new artifacts with their own evaluation.

---

## 1. Research gap (M3-specific)

A hybrid that always corrects with numerics is as slow as a pure numerical solver,
which defeats the purpose. To be worthwhile, numerical effort must be spent **only
where it pays off**. Before this module there was:

1. **no controller** that, given a target accuracy, decides *how much* numerical
   computation to spend and *where* on a learned-operator rollout;
2. **no measured cost–accuracy frontier** for such a hybrid on a real
   shock-forming PDE — only the folk claim that "ML is fast, numerics is accurate";
3. **no deployable artifact** turning that trade-off into a single tunable knob.

The gap is therefore *control under uncertainty without ground truth*: at deployment
there is no true answer to tell us where the ML model failed, so the amount and
placement of numerical work must be decided from a trust signal that is informative
but imperfect.

## 2. Objective

Build (i) a **cost/latency/memory profiler** for the hybrid's unit operations,
(ii) a **measured accuracy–cost model** and **Pareto frontier**, (iii) an
**accuracy-budget adaptive controller** driven by the trust signal, and (iv) a
**deployable runtime** with one accuracy knob — and prove with real measurements
whether the hybrid dominates both extremes.

## 3. Contribution (stated narrowly and honestly)

> **An accuracy-budget adaptive scheduler for hybrid ML–numerical PDE solvers,
> driven by a trust signal, together with the first measured cost-vs-accuracy
> frontier for this hybrid on a shock-forming PDE, packaged as a deployable
> runtime.** This converts "computational-cost analysis" from a *claim* into a
> *method* and a *product*.

Six concrete, new artifacts (none of which existed in the project before):

| # | Artifact | File | What is new |
|---|----------|------|-------------|
| 14 | Cost/latency/memory profiler | `m3_cost/profiler.py` | rigorous, solver-agnostic unit-cost measurement (warmup, repeats, median+IQR, peak memory) |
| 15 | Accuracy–cost model | `m3_cost/accuracy_cost.py` | maps numerical-correction budget → achieved accuracy and cost |
| 16 | Adaptive controller | `m3_cost/controller.py` | trust-gated, accuracy-budget switch/corrector scheduler + calibration with no test leakage |
| 17 | Pareto-frontier construction | `m3_cost/accuracy_cost.py` | non-dominated hybrid frontier vs pure-ML and pure-numerical |
| 18 | Deployment runtime | `m3_cost/runtime.py` | one-knob live tool returning solution + measured cost + accuracy |
| 19 | Robustness / ablation | `scripts/run_controller.py` | adaptive vs fixed schedule, target hit-rate, unseen ICs |

## 4. Novelty point (precise)

The novel object is the **trust-gated accuracy-budget scheduler**: a control law
that places the *minimum* numerical work needed to meet a requested error, using a
calibrated trust threshold to locate the ML model's reliable horizon, and selecting
the cheapest corrector (finite-difference vs spectral) that meets the target. The
**second novelty** is empirical: the **first measured cost–accuracy frontier** for
an ML–numerical hybrid on the viscous Burgers shock problem, including the
quantified finding that the hybrid's accuracy floor is set by the surrogate's
hand-over error — not by the numerical solver. Both are method/finding, not model.

## 5. How it is proven correct (real evidence)

All numbers below are reproducible via the scripts in this folder.

### 5.1 The cost backbone is real and measured
Unit step costs on the benchmark grid (512 points, median of repeated timings):

| operation | median cost |
|-----------|-------------|
| one FDM step | **0.64 ms** |
| one spectral step | **15.6 ms** (≈ 24× an FDM step) |
| full pure-numerical (spectral) solve | **3.29 s** (199 steps) |
| spectral corrector sub-rollout | **≈ 17 ms / step** (linear in length) |

These establish the trade-off space: the corrector is expensive, so spending fewer
numerical steps is the entire point.

### 5.2 Ground truth is verified, not assumed
The Cole-Hopf reference regenerated by M3 matches the stored project ground truth to
**2.4×10⁻⁸ relative L2** — so every accuracy number here is measured against a
verified true solution, on the held-out **test** IC 900 (never used in training).

### 5.3 The measured Pareto frontier (real, IC 900, DeepONet surrogate)

| policy | cost (numerical steps) | extrapolation rel-L2 |
|--------|------------------------|----------------------|
| pure ML | 0 | **0.590** |
| hybrid (beat-ML-by-2× point) | 85 | ≤ 0.295 |
| hybrid (switch @ t=1) | 99 | 0.188 |
| hybrid (accuracy floor) | 149 | **0.102** |
| pure numerical (spectral) | 199 | 4.6×10⁻⁵ |

**Headline measured result:** the hybrid reaches a solution **2× more accurate than
pure-ML using 85 numerical steps — a 2.34× cost saving versus the full pure-numerical
solve.** The frontier strictly dominates pure-ML everywhere.

### 5.4 The honest, scientifically interesting finding
With DeepONet, the hybrid's accuracy **floor is 0.102** and it does **not** reach
pure-numerical accuracy. We traced this to a specific mechanism: re-seeding the
numerical solver from the ML state inherits the ML hand-over error (DeepONet carries
~24% in-distribution error on IC 900), and the numerical solver then faithfully
propagates that wrong state. This is a *quantified* result, not a hand-wave, and it
makes two falsifiable predictions the host run tests directly:
(a) a more accurate surrogate (FNO) lowers the floor and may achieve full dominance;
(b) a re-anchoring corrector (M2) that pulls the trajectory back, rather than a
single re-seed, can break the floor.

### 5.5 The controller is adaptive, not a fixed recipe (ablation, IC 900)

| accuracy target | fixed (switch @ t=1) | adaptive (trust-gated) | outcome |
|-----------------|----------------------|------------------------|---------|
| 0.50 | 99 steps, hit | **70 steps, hit** | adaptive saves 29 steps (29%) |
| 0.40 | 99 steps, hit | **79 steps, hit** | adaptive saves 20 steps |
| 0.15 | 99 steps, **miss** | 189 steps, **hit** | adaptive reaches target fixed cannot |

The adaptive controller spends *less* when the target is loose and *more* (and
succeeds) when the target is tight — the defining behaviour of a real controller.

### 5.6 The deployable runtime (the panel's "working demonstration")
`scripts/demo.py` exposes one knob. As the requested accuracy tightens 0.50 → 0.15,
the runtime moves along the measured frontier (70 → 189 numerical steps; 1.3 s →
3.6 s wall-time) and meets every target. The demo figure shows the hybrid recovering
the true shock front at t = 2 that pure-ML completely misses, with error held near
0.1 while pure-ML diverges to ~0.5.

## 6. Positioning vs related work (honest)

Hybrid ML–numerical solving and error-indicator switching both exist in the
literature. This module does **not** claim to invent hybrid solving. Its contribution
is **incremental but concrete and demonstrated**: a *cost-aware* controller that
schedules numerical effort to an accuracy budget, plus the *measured* frontier and a
*deployable* runtime on a real shock problem — the cost/deployment angle is the
under-explored part. The framing is stated narrowly so it is defensible: a clearly
specified, properly evaluated, runnable method-module, which is the correct bar for a
final-year project.

## 7. Anticipated examiner questions (and answers)

- **"Isn't this just timing code?"** No. Timing is component #14. The contribution is
  the *control law* (#16) that uses those costs plus a trust signal to decide
  numerical effort, and the *frontier* (#17) proving the trade-off — both evaluated.
- **"Where is the novelty if hybrids exist?"** In the cost-aware *scheduling* and the
  *measured* frontier + runtime for this problem; we position against the closest
  related work rather than over-claiming (Section 6).
- **"Your hybrid doesn't beat pure-numerical on accuracy."** Correct, and we
  *quantify why* (handover-error floor, §5.4). That is a finding, not a failure, and
  it yields two testable predictions. It also beats pure-ML by 6× and is 2.3× cheaper
  than pure-numerical at a 2×-better-than-ML accuracy.
- **"Does it generalise?"** The controller calibrates on train ICs with no test
  leakage and is evaluated on unseen test ICs; the host script reports target
  hit-rate across many ICs and the three surrogates (#19).
- **"Is the trust signal real?"** M3 develops against a parameterised trigger and a
  real reference-free physics-residual signal; we *observed and documented* that the
  naive residual is contaminated at the shock (it fires immediately) — exactly the
  problem M1 is built to fix. At integration, M1's calibrated trust replaces the stub
  through a fixed data contract, with zero change to the controller.

## 8. Honest limitations

1. Hybrid accuracy is floored by surrogate handover error (quantified, §5.4).
2. The deployable trust signal's quality depends on M1; the oracle ablation gives the
   achievable ceiling.
3. Sandbox wall-times are CPU; the host regenerates official (and GPU) timings with
   identical code. The **numerical-step-equivalent** cost metric is hardware-independent.

## 9. One-line statement for the supervisor

> *M3 is not a trained model. It is a measured cost-aware control method and a
> deployable runtime that, on a real shock-forming PDE, turns the ML/numerical
> trade-off into a single tunable knob — beating pure-ML by ~6× accuracy and
> pure-numerical by ~2.3× cost at matched-to-ML accuracy — with every number
> reproducible and the one failure mode (the handover floor) quantified rather than
> hidden.*
