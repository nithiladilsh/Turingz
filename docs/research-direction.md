# Research Direction — Turingz Hybrid PDE Solver

**Team Turingz (Group 74), University of Moratuwa** · Benchmark: 1D viscous Burgers equation

## The problem
Time-dependent PDEs can be solved two ways, each flawed. Classical **numerical solvers** are accurate but slow. **ML solvers** (PINN, FNO, DeepONet) are fast but only reliable inside their training window — pushed beyond it (temporal extrapolation) they fail *silently*, producing smooth but wrong answers. We want fast **and** trustworthy long-horizon prediction, which neither family gives alone.

## What already exists
- ML PDE surrogates are established: [PINN (Raissi et al., 2019)](https://doi.org/10.1016/j.jcp.2018.10.045), [DeepONet (Lu et al., 2021)](https://www.nature.com/articles/s42256-021-00302-5), [FNO (Li et al., 2020)](https://arxiv.org/abs/2010.08895).
- Their **extrapolation failure** is well documented — error compounds beyond the training horizon ([TI-DeepONet, 2025](https://arxiv.org/abs/2505.17341)).
- **Hybrid ML–numerical solvers** and error-triggered switching already exist: coupling a learned operator to a numerical solver ([NO–FE coupling / HINTS, 2025](https://arxiv.org/abs/2504.11383)), and residual-gated correction during rollout ([ANCHOR, 2025](https://arxiv.org/abs/2512.19643), the closest prior work).

So the *idea* of a hybrid solver is not new. The building blocks exist.

## Our approach and how it differs
We build **one trust-gated, cost-aware hybrid solver** on a known PDE, and our contribution is a *specific combination* that prior work does not deliver together: a **calibrated, ground-truth-free trust signal**, a **minimal-cost coupling**, and a **cost-budgeted adaptive controller with a measured cost/accuracy frontier, deployed as a runnable tool**. We do not claim to invent hybrid solving — we claim a new control-and-deployment method, demonstrated end-to-end, built on existing solver foundations.

The distinction from prior work (e.g. ANCHOR) is that we don't just *switch when error looks high* — we decide *how much* numerical effort to spend to hit a chosen accuracy target at minimum cost, prove the trade-off with real measurements, and ship it with a single accuracy knob.

## How the three members combine
The work splits into three independent modules joined by fixed interfaces, forming one system:

- **M1 — Trust (Sandeepa).** A reference-free, calibrated estimator that watches the running ML rollout and outputs a trust score + failure flag, without ground truth. *Answers: when is the ML no longer reliable?*
- **M2 — Coupling (Dharmapala).** A minimal-cost re-anchoring mechanism that injects numerical correction with a continuous (jump-free) hand-over. *Answers: how do we correct without breaking the trajectory?*
- **M3 — Control & Deployment (Mendis).** A cost-aware adaptive controller that, driven by the trust signal, spends the least numerical effort needed to meet an accuracy target, with a measured cost/accuracy frontier and a deployable runtime. *Answers: how much to correct, and how to ship it?*

**Alignment:** M1's trust signal feeds M2's coupling, which M3 controls and deploys — `trust → couple → control`. Each member owns a distinct novel mechanism; together they make a single demonstrable solver: fast ML where trustworthy, numerical correction only where needed, tuned by one accuracy knob.

---
*Note: hybrid solving and error-indicator switching exist in the literature; our contribution is the three specific, evaluated mechanisms combined into one working, tunable system. Confirm final novelty wording with the supervisor.*
