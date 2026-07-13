# Master Study Guide — FYP Viva Preparation (Module 3, Turingz)

**Owner:** Mendis B.N.D. (214133E) · **Module:** M3 — Cost-Aware Adaptive Control & Deployment
**Purpose:** A single living document to take me from "know very little" to confidently defending the whole project in the viva. Updated every session — never recreated.

> How to use: read the current day's section, then answer the mock-viva questions at the end of it *out loud in my own words*. If I can't, that topic isn't done yet.

---

## Progress Tracker

| Day | Topic | Status | Confidence (1–10) |
|----|-------|--------|-------------------|
| 1 | Big picture: motivation, the hybrid idea, the 3 modules, my scope | In progress | — |
| 2 | Shared foundation: Burgers PDE, Cole–Hopf ground truth, the 6 solvers, data, metric | Pending | — |
| 3 | My module I: the problem M3 solves; config, contracts, groundtruth, stubs, profiler | Pending | — |
| 4 | My module II: accuracy–cost model + adaptive controller (line-by-line) | Pending | — |
| 5 | My module III: runtime, Pareto frontier, robustness (line-by-line) | Pending | — |
| 6 | Results & plots: every figure in results/m3, real vs stand-in, honest framing | Pending | — |
| 7 | ANCHOR deep dive: method, figures, tables, experiments, metrics | Pending | — |
| 8 | Comparison vs ANCHOR: novelty, contribution, industry impact, limits, future work | Pending | — |
| 9 | Integration, demo, end-to-end system + full mock viva | Pending | — |
| 10 | Hard examiner mock viva + weak-area revision + export MD/Word/PDF | Pending | — |

**Weak areas (updated as we go):** _none logged yet_
**Remaining work:** all 10 days.

---

## Day-by-Day Study Plan

**Day 1 — Big picture (today).** Project motivation; the two solver families and their weaknesses; the hybrid idea; the three modules and how they fit; exactly what *my* scope (M3) is. Mock viva: "Explain your project in 2 minutes," "What is your scope?"

**Day 2 — Shared foundation.** The Burgers equation (what it is, why it's a good test); Cole–Hopf as exact ground truth; the three ML surrogates (FNO/PINN/DeepONet) at a concept level; the three numerical solvers; how the data is stored; the shared relative-L2 metric. Files: `data/`, `solvers/`, `common/`. Mock viva on dataset & why-this-benchmark.

**Day 3 — My module, part I.** The gap M3 fills; then code review of `config.py`, `contracts.py`, `groundtruth.py`, the `trigger.py`/`coupling.py` stubs, and `profiler.py`. Mock viva on each.

**Day 4 — My module, part II.** `accuracy_cost.py` and `controller.py` — line by line, including hysteresis and the accuracy-budget map. Mock viva.

**Day 5 — My module, part III.** `runtime.py`, `pareto.py`, `robustness.py`, `integrate.py` — line by line. Mock viva.

**Day 6 — Results & plots.** Walk every figure in `results/m3/` (steps 2–10 + real frontier): what it shows, good/bad, the ideal viva answer, follow-ups. The honest "trade-off, not free lunch" story.

**Day 7 — ANCHOR deep dive.** Read arXiv:2512.19643v2 fully; method, pipeline, algorithm, every figure/table/experiment/metric/dataset/result/conclusion.

**Day 8 — Comparison vs ANCHOR.** Similarities, differences, advantages, disadvantages, our novelty & contribution, industrial relevance, limitations, future work.

**Day 9 — Integration + demo + end-to-end + mock viva.** How the three modules join; the demo; a full mock viva across the whole project.

**Day 10 — Hard mock viva + export.** Difficult examiner questions; revise weak areas; export this guide to Word and PDF.

---

## Day 1 — Big Picture

### 1. Big picture
Many real-world things (heat spreading, air/water flowing, a shock wave forming) are described by **PDEs** — rules that say how a quantity changes over space and time. "Solving" a PDE means: start from a shape at time zero and march it forward in time to see what it becomes.

Our project builds **one hybrid solver** that predicts far into the future, fast *and* accurately, by combining two kinds of solver that each have one fatal weakness.

### 2. Why this exists / 3. The problem it solves
- **Numerical solvers** (classical maths methods) are accurate and trustworthy, but **slow** — they take many tiny time steps.
- **Machine-learning (ML) solvers** are **fast** once trained, but only reliable inside the time window they were trained on. Pushed *beyond* that window ("extrapolation"), they drift and can **fail silently** — giving a smooth, believable, but wrong answer.

Neither alone gives *fast + trustworthy long-horizon* prediction. That's the gap.

### 4. Why previous approaches were insufficient
Using ML alone → fast but wrong in the future. Using numerics alone → right but too slow. The obvious "just switch to numerics when ML fails" is hard because **at deployment there is no ground truth** — nothing tells you *when* the ML has gone wrong, *how* to fix it cleanly, or *how much* numerical effort to spend. Those three unknowns are exactly the three team modules.

### 5. How this solution works
One pipeline: run the cheap ML model while it can be trusted; when it's about to go wrong, bring in numerical computation — but only as much as needed.
- **Module 1 (Sandeepa) — Trust:** watches the ML and raises a flag when it's no longer reliable ("when to switch").
- **Module 2 (Dharmapala) — Coupling:** injects/corrects with the numerical solver, keeping the answer smooth and accurate ("how to correct").
- **Module 3 (me) — Control + Deploy:** decides *how much* numerical effort to spend to hit a target accuracy at minimum cost, and packages it as a runnable one-knob tool ("how much, and ship it").

### 6. How it connects to the rest of the project
My module sits at the top: a user gives my runtime a starting shape and an accuracy target. My runtime runs the ML model, reads M1's trust each step, and when trust drops, asks M2's coupling to correct — while I keep total cost minimal for the chosen accuracy. The output is one accurate long-horizon solution, produced far faster than pure numerics.

**My scope in one line:** the *cost-aware adaptive controller* + the *deployable runtime* + the *measured cost-vs-accuracy frontier*. I do **not** build the trust detector (M1) or the coupling mechanism (M2) — I consume them through fixed interfaces.

### Day 1 mock-viva questions (answer out loud)
1. In 2 minutes, explain what your project does and why it's needed.
2. What exactly is *your* scope, and what is *not* your scope?
3. Why is this a *research* problem and not just an engineering one?
4. Why can't you just always use the numerical solver?

---

## Viva Question Bank (growing)
- Explain your project. / Explain your scope. / What did you personally implement?
- Why is this a research problem? (Ans idea: no ground truth at deployment → must decide when/how/how-much to correct blindly.)
- Why not just use numerics always? (Ans idea: accurate but too slow for long horizons — defeats the purpose.)

_(more added each day)_

---

## Confidence Log
| Date | Topic | Self-rating (1–10) | Note |
|------|-------|--------------------|------|
| — | — | — | — |
