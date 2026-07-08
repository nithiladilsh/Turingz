# Module 2 — Viva Guide (Plain-Language Progress Document)

**Student:** Dharmapala R.D. (214050V) · **Module:** Coupling (ML → Numerical handoff)
**Purpose of this file:** a running, simple-English record of everything we have done, why,
what the results mean, and how to explain it in the viva. It is updated at the end of every phase.

> **How to read this:** each phase has four parts — *What we did*, *Why*, *What we found*,
> *How to say it in the viva*. Technical words are explained the first time they appear.
> A change log at the bottom tracks every update.

---

## 0. The one-paragraph summary (say this if asked "what is your module?")

> "The team has two kinds of solver for a fluid-flow equation. One is a machine-learning
> model (fast, but it becomes wrong when asked to predict far into the future). The other is
> a numerical solver (slow, but always accurate). My module is the **bridge** between them:
> I run the fast model early, then **hand its answer over** to the accurate solver to finish
> the job. I then measure **when** this handover helps and when it doesn't. My first real
> experiment shows the handover cuts the future-prediction error from **14% down to about 1%**
> when done early."

---

## 1. The big picture in simple words

**The equation.** We study the 1-D **viscous Burgers equation**. Think of it as a simple
model of a wave in a slightly sticky fluid: the wave moves, steepens, and the "stickiness"
(called **viscosity**) smooths it out. We want to know the shape of the wave `u(x, t)` at
every position `x` and every time `t`.

**Two ways to solve it.**
- **Numerical solvers** (Cole–Hopf, pseudo-spectral, finite-difference): they follow the
  real maths step by step. Very accurate, but slow.
- **Machine-learning (ML) solvers** (FNO, DeepONet, PINN): they *learn* the answer from lots
  of examples. Very fast, but they were only *trained* on times up to `t = 1`. When asked
  about `t > 1` (called **extrapolation** — predicting beyond what they were taught), they
  drift away from the truth.

**The trade-off we exploit.** Fast-but-wrong vs slow-but-right. A **hybrid** uses the fast
one while it can be trusted and the accurate one only when needed.

**Key numbers (our own measurements, file `results/eval/reliability_summary.json`):**
- The FNO model has **0.57%** error inside its training window (`t ≤ 1`).
- It jumps to **14.1%** error when extrapolating (`t > 1`).
- It stays trustworthy until about **t = 1.46** ("reliable horizon").

---

## 2. What exactly is *my* contribution (the coupling module)

The other two teammates handle **when** to switch (a "trust" signal) and **how much** compute
to spend (cost control). **I build and study the actual handover** between the ML model and
the numerical solver.

My research question, in plain words:

> **"When I take the ML model's guess at some moment and let the accurate solver continue from
> it, does the final answer get better — and at what point is the ML guess already too wrong
> for this to help?"**

The moment where it stops helping is what I call the **handoff viability boundary**. Finding
and measuring that boundary is the heart of my contribution.

---

## 3. Small glossary (for quick viva recall)

| Word | Simple meaning |
|---|---|
| Viscosity (ν) | How "sticky"/smoothing the fluid is. Ours is ν = 1/(100π) ≈ 0.0032 (only slightly sticky). |
| Extrapolation | Predicting beyond the times the model was trained on (here, t > 1). |
| Rollout | The full predicted wave over all times. |
| Handoff / handover | Taking the ML model's wave at a chosen time and giving it to the numerical solver as a fresh starting point. |
| Switch time (t_s) | The moment we hand over from ML to numerical. |
| Reference / ground truth | The "correct" answer. We use the Cole–Hopf solver for this. |
| Relative L2 error | A single number for "how wrong" a prediction is (0 = perfect). |
| Held-out ICs | Starting waves the ML model never saw in training — a fair test. |

---

## 4. The journey (decisions we made, in order)

We refined the plan several times before writing code. The important decisions:

1. **Use the pseudo-spectral solver as the "continuer", and keep Cole–Hopf as the independent
   referee.** Reason: the pseudo-spectral solver naturally continues from any wave, and using a
   *separate* solver as the referee avoids the accusation that we "cheated" by inserting the
   true answer.
2. **Test on many starting waves, not one.** A result on a single example is an anecdote; a
   result across 10–20 waves is evidence.
3. **Add a "perfect-handoff" measurement** (restart from the *true* wave, not the ML wave) as a
   yardstick, so we can tell whether any error came from the ML guess or from the solver itself.
4. **Judge success by a rule fixed in advance** (at least a 10% improvement), so we don't fool
   ourselves after seeing results.

---

## 5. Phase 1 — The Audit (Day 1)

**What we did.** Before writing any hybrid code, we read the team's real code and files to
write down a exact "contract": grid size, viscosity, time steps, which model is real, and
whether the numerical solver can restart. (File: `reports/module2/baseline_audit.md`.)

**Why.** If you connect two systems that secretly disagree on a detail (like viscosity, or
whether numbers are "normalised"), the hybrid can look correct but be silently wrong.

**What we found — three important things:**

1. **The FNO is NOT a step-by-step model.** This surprised us. We had assumed the FNO predicts
   one small time step at a time (`u(now) → u(next)`), with errors piling up step by step. The
   real trained model instead takes `(starting wave, a time t, position x)` and **jumps directly**
   to the answer at that time, in one shot. In plain words: *it is a direct "time machine", not
   a step-by-step walker.*
   - **Why this matters:** the story "errors build up each step" is wrong for our model. The
     correct story is: *the model simply gets less accurate the further past t = 1 you ask it.*
   - It also means an old helper file (`fno_rollout.py`) that does step-by-step prediction does
     **not** match the real model and must not be used.

2. **The numerical solver could not restart.** The team's pseudo-spectral solver only ran from
   the very beginning (`t = 0`). To hand over in the middle, we needed to teach it to start from
   an arbitrary wave. (We built this — see Phase 2.)

3. **The training dataset was not saved on disk**, but a results file
   (`results/eval/predictions.npz`) already stored the true answers and the FNO's answers for
   **10 unseen test waves**. This let us run real experiments immediately, without re-training
   anything.

**How to say it in the viva:**
> "In my Day-1 audit I discovered our FNO is a direct space-time operator, not an
> autoregressive one — so I corrected the whole framing of the problem before writing code.
> I also found the numerical solver couldn't restart mid-run, which became my first engineering
> task."

---

## 6. Phase 2 — The Restart Gate + First Hybrid Result (Day 2)

### 6.1 The code we wrote (explained simply)

**File 1: `restart_spectral.py` — the "continuer".**
The team solver only starts from the beginning. This file re-creates the *exact same* numerical
recipe (same maths, same settings) but lets it **start from any wave at any time** and run to the
end. One important detail it handles: the solver takes many tiny internal steps for stability,
but only *records* the answer at the same time points as our dataset — so all three methods can
be compared fairly on the same clock.

*Key function:* `solve_from(u0, i_start)` — "start from this wave `u0` at time-index `i_start`
and give me the rest of the trajectory."

**File 2: `day2_gate_and_sweep.py` — the experiment.**
For each test wave and each candidate switch time, it builds **three** trajectories:
- **Pure FNO** — the fast model alone (our baseline to beat).
- **Hybrid (H)** — run FNO up to the switch, then hand its wave to the numerical continuer.
- **Upper Bound (UB)** — hand over the *true* wave instead of the FNO's wave. This is not a real
  method; it is a **yardstick** that tells us how much error the continuer adds on its own.

It then measures how wrong each one is compared to the Cole–Hopf truth.

### 6.2 The "gate" — a safety check that had to pass first

**What:** before trusting any hybrid result, we checked: *if we restart the numerical solver from
the perfectly correct wave, does it reproduce the correct future?* 

**Result:** yes — the error was about **0.000005** (essentially zero). **The gate passed.**

**Why this matters:** it proves our restart machinery (indexing, timing, the maths) is correct.
So any error we see later is genuinely the ML model's fault, not a bug in our plumbing.

### 6.3 The first real result (10 unseen test waves, real viscosity)

The table below is our first genuine evidence. "Tail error" = how wrong the method is, on
average, over the future part of the prediction (0 = perfect). "Benefit" = how much the hybrid
improves on the pure FNO.

| Switch time t_s | FNO's wave error at handover | Pure-FNO future error | **Hybrid future error** | Benefit | Waves improved |
|----:|----:|----:|----:|----:|:--:|
| 1.0 | 1.4% | 14.1% | **1.1%** | 92% | 10/10 |
| 1.2 | 3.2% | 16.9% | **2.1%** | 86% | 10/10 |
| 1.4 | 7.7% | 20.9% | **7.5%** | 62% | 10/10 |
| 1.6 | 16% | 25.4% | **16.3%** | 34% | 10/10 |
| 1.8 | 25% | 30.1% | **25.0%** | 16% | 9/10 |

### 6.4 What this table means (three conclusions)

1. **The hybrid works, and strongly when done early.** Switching at t = 1.0 cuts the future
   error from **14.1% to 1.1%** — better on *all 10* unseen waves. This is the headline result.

2. **Earlier is better; there is a viability boundary.** As we wait longer to switch (t_s going
   1.0 → 1.8), the benefit steadily falls (92% → 16%). Reason: the longer we let the FNO run, the
   more wrong its wave already is at handover (see column 2: 1.4% → 25%), so the numerical solver
   inherits a worse starting point.

3. **The deepest finding: the solver stops the bleeding, but can't heal old wounds.** Look closely:
   the **hybrid's future error is almost equal to the FNO's error at the moment of handover**
   (e.g. at t_s = 1.4: handover error 7.7%, hybrid error 7.5%). And the "perfect-handoff" yardstick
   (UB) was near zero. Put together, this means: *the numerical continuer adds almost no new error.
   It prevents the FNO's error from growing any further — but it cannot remove the error that was
   already baked into the FNO's wave at the switch.* This is exactly why switching **early**
   (while the FNO wave is still good) matters so much.

**How to say it in the viva:**
> "My first result shows the hybrid reduces extrapolation error from 14% to about 1% for an early
> switch, on every held-out wave. I also show *why*: using a true-state yardstick, I prove the
> numerical continuation adds essentially no error of its own — it freezes further growth but
> can't recover error already present in the ML state. That gives a measurable 'viability
> boundary': the hybrid helps most while the ML wave is still accurate."

### 6.5 Honest limitations (be ready for these)
- We used a faithful **copy** of the team's numerical recipe (the coding environment didn't have
  the team's exact library loaded). Next phase: confirm the copy matches the original exactly.
- Only **10** test waves so far; we will extend to 10–20 for the final numbers.
- We used the raw ML wave with **no cleaning/filtering**, and it was already stable — so cleaning
  may not even be needed (we will check).

---

## 7. The novelty — explained through the code (VERY IMPORTANT for viva)

**What is *not* novel:** connecting an ML model to a numerical solver is a known idea (papers like
ANCHOR and PDE-Refiner exist). We do not claim to invent hybrid solving.

**What *is* novel here, and where it lives in the code:**

1. **A working, solver-agnostic restart bridge — `restart_spectral.py`, function `solve_from`.**
   The team's solver could only start from the beginning. My `solve_from(u0, i_start)` turns it
   into something that can **continue from any ML-produced wave at any moment**. That single
   capability is what makes a hybrid possible at all. It is written to accept *any* wave, so the
   same bridge works for FNO, DeepONet or PINN — that is the "solver-agnostic" claim.

2. **A fair, self-checking measurement design — `day2_gate_and_sweep.py`.** The novelty is not
   just "it works" but "we can *prove* when and why it works". Two code ideas do this:
   - The **UB (true-state) yardstick**: by restarting from the true wave as well as the ML wave,
     the code cleanly separates "error the ML model brought in" from "error the solver added".
     Very few student projects include this control — it is what makes the conclusion trustworthy.
   - The **switch-time sweep with a fixed success rule**: the code measures benefit across many
     switch times and many held-out waves, so the "viability boundary" is a measured curve, not a
     lucky single example.

**One-line novelty statement for the viva:**
> "My novelty is not hybrid solving itself — it is a **validated, solver-agnostic handoff** plus a
> **measurement that quantifies exactly when it helps**. In code: `solve_from` makes the handoff
> possible, and the true-state yardstick in the sweep proves the numerical part adds no error, so
> the remaining error is purely inherited from the ML model. That turns 'it seems to work' into
> 'here is the boundary where it works, with proof.'"

---

## 8. Graphs / figures

**Status: not generated yet.** So far our evidence is the numbers in Section 6.3
(saved in `day2_sweep_results.json`). The next phase produces the figures below, and this section
will be filled in with a plain-language explanation of each one:

- **Figure 1 — Error over time (three lines):** pure FNO vs hybrid vs the numerical reference.
  *(Will show FNO drifting up after t=1 while the hybrid stays low.)*
- **Figure 2 — Switch time vs benefit:** the "viability boundary" curve.
- **Figure 3 — Handover-wave error vs benefit:** shows benefit shrinking as the handed-over wave
  gets worse.
- **Figure 4 — Hybrid vs the true-state yardstick (UB):** shows the gap equals the inherited ML error.
- **Figure 5 — Accuracy vs compute used:** the cost/accuracy trade-off.

*(This section auto-updates when the figures are made.)*

---

## 9. What's next (roadmap)

- Reword the written report to drop "autoregressive" and use the direct-map framing.
- Install dependencies + regenerate the dataset → verify the restart copy matches the team solver,
  extend to 10–20 waves, and test the same handoff on DeepONet.
- Generate the five figures and fill in Section 8.
- Write automated tests and the final results tables.

---

## 10. Change log (every update to this document)

| Date | Phase | What was added |
|---|---|---|
| 2026-07-08 | Day 1 (Audit) + Day 2 (Restart gate & first sweep) | Initial document: big picture, glossary, audit findings (direct-map FNO discovery), restart gate PASS, first 10-IC hybrid result table + conclusions, novelty-through-code section. Figures pending. |
