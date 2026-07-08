# Module 2 — Baseline Audit + Restart Gate

**Author:** Dharmapala R.D. (214050V) · **Branch:** `Ruwandie-PhaseTwo` · **ν = 1/(100π) ≈ 0.0031831**

This is the data contract every coupling experiment depends on. It was produced by
reading the actual repo files, not the older proposal/interim documents.

---

## 1. HEADLINE FINDING — the FNO is NOT autoregressive

The production FNO (`hybrid_pde/solvers/ml/fno/fno.py`, checkpoint `results/fno/fno.pt`)
is trained with **`in_channels=3`, channels `[ic, t, x]`**. It maps

> (initial condition, query time *t*, grid *x*)  →  u(x, t)

in a **single forward pass**. It is a **direct space-time operator**, confirmed by
`common/interface.py` where `rollout()` builds a full `(x, t)` meshgrid and calls
`predict(ic, X, T)` once. There is **no step-by-step rollout and no autoregressive
error accumulation** for this checkpoint.

The autoregressive `rollout()` in the personal `fno_rollout.py` (`u = model(u)`) belongs
to a *different, older* FNO variant and must not be used with this checkpoint.

### Consequences for the plan (must be reworded)
- The mechanism is **not** "viscosity halts autoregressive compounding." It is:
  the FNO's **direct extrapolation error grows with query time past the training
  horizon (t>1); the spectral solver replaces further extrapolation with true Burgers
  dynamics from the handed-over state.**
- `u_FNO(x, t_s)` is one clean forward query — the handoff state is a direct prediction,
  not an error-accumulated rollout endpoint.
- **Return-to-FNO is moot** (you can query the FNO at any time directly) → drop it as a
  stretch goal.
- Cost model: FNO query cost is ~constant per queried time (batchable), independent of
  t_s. Hybrid saving vs full-numerical = the spectral integration over [0, t_s] avoided.

---

## 1b. Scope and research question

The switch time is an **external input** to this module (a teammate's trust signal will later
supply it); my module studies **what happens at a supplied switch time**, not when to trigger
it. The handoff is one-way (no return to the FNO). **Research question:** *How does an
externally supplied switch time, and the FNO state quality at that moment, affect
FNO-to-numerical continuation performance?*

## 2. Data contract (verified)

| Item | Value | Source |
|---|---|---|
| Viscosity ν | 1/(100π) ≈ 0.0031831 | `colehopf.py`, `spectral.py` |
| Domain | x = linspace(−1, 1, 512, endpoint=False), L=2 | both solvers |
| Time grid | t = [0, 0.01, …, 2.0], nt=200, **non-uniform first step** (0→0.01, then ~0.01005) | `colehopf.py` L12 |
| Training horizon | t_train_end = 1.0 (K = #{t ≤ 1}) | `fno.py` L27–28 |
| IC split | train 0–799, val 800–899, **test 900–999** | `common/split.py` |
| FNO | neuralop FNO, modes 16, width 64, in=3 (ic,t,x), out=1 | `fno.py`, `fno_config.pt` |
| Reference | Cole–Hopf, analytic heat-kernel (no time-stepping) | `colehopf.py` |
| Continuation solver | pseudo-spectral IFRK4, 2/3 mask, dt_target 1e-4 | `spectral.py` |

### Reliability numbers (use these, not the proposal's)
FNO in-window **0.57%**, extrapolation **14.1%**, reliable horizon **1.457**
(`results/eval/reliability_summary.json`). DeepONet worst (61% extrap), PINN 35.8%.

---

## 3. Two blockers found, both resolved

1. **Spectral solver does not restart.** `spectral.solve(u0)` only runs t=0→T on the
   fixed grid. → Wrote `coupling - 214050V/restart_spectral.py`: a faithful numpy port
   of the exact team scheme (same k, mask, integrating-factor RK4 step, dt_target,
   Nyquist zeroing) exposed as `solve_from(u0, i_start)`, batched over ICs.
   *VERIFIED (Phase 1): `solve_from(u0,0)` reproduces the team `spectral.solve(u0)` **bit-for-bit** (rel diff 0.0e+00) via `verify_restart.py`, which runs the team's real `solve()` code. The wrapper IS the team solver.*
2. **Dataset not on disk** (`data/colehopf/` is empty; gitignored). But
   `results/eval/predictions.npz` contains Cole–Hopf truth (`u_true_eval`) and FNO
   predictions (`FNO_eval`) for **10 held-out ICs**, full (200,512) — enough to run the
   gate and a first hybrid sweep offline, no torch / no regen.

---

## 4. Restart Gate — PASS

Restarting the spectral wrapper from the **true Cole–Hopf state** at t_s and continuing
to T reproduces the Cole–Hopf tail to **time-integrated rel-L2 ≈ 3.6e-7 … 5.7e-6** across
all switch times and 10 ICs. The restart machinery (indexing, local-time, output
sampling, real ν) is therefore correct. Any error in the real hybrid below is the FNO
state's fault, not the plumbing.

Sanity cross-check: pure-FNO tail error at t_s=1.0 = 0.1409, matching the independent
`reliability_summary.json` extrapolation value 0.14127 → pipeline is consistent.

---

## 5. Preliminary hybrid result (10 held-out ICs, real data)

Tail = time-integrated rel-L2 over [t_s, 2]. B_int = mean benefit vs pure FNO.

| t_s | idx | FNO handoff-state err e_s | UB tail (gate) | pure-FNO tail | **hybrid tail** | B_int | ICs improved |
|----:|----:|--------------------------:|---------------:|--------------:|----------------:|------:|:---:|
| 1.0 | 100 | 0.0143 | 5.7e-6 | 0.1409 | **0.0109** | 0.92 | 10/10 |
| 1.2 | 119 | 0.0323 | 2.8e-6 | 0.1690 | **0.0214** | 0.86 | 10/10 |
| 1.4 | 139 | 0.0765 | 1.4e-6 | 0.2087 | **0.0745** | 0.62 | 10/10 |
| 1.6 | 159 | 0.1618 | 6.9e-7 | 0.2540 | **0.1628** | 0.34 | 10/10 |
| 1.8 | 179 | 0.2528 | 3.6e-7 | 0.3010 | **0.2496** | 0.16 | 9/10 |

### What this shows (the contribution, quantified)
1. **The hybrid works, strongly, for early switches.** At t_s=1.0 it cuts extrapolation
   error from 14.1% to **1.1%** — a 92% reduction — on all 10 unseen ICs.
2. **The handoff viability region is early.** Benefit falls monotonically as t_s rises,
   because the FNO state handed over is already more degraded (e_s: 1.4% → 25%).
3. **The clean scientific result:** UB ≈ 0 everywhere, and **hybrid tail ≈ e_s** at each
   t_s (e.g. t_s=1.4: e_s=7.65%, hybrid=7.45%). So the spectral continuation adds almost
   no error — the hybrid's accuracy is **bounded by the FNO state quality at handoff**.
   Numerical continuation **halts further error growth but does not recover error already
   in the state.** That is Result category B/C from the plan, measured.

### Caveats
- Restart wrapper is now **verified bit-for-bit identical** to the team solver (Phase 1, `verify_restart.py`).
- n=10 held-out ICs; regenerate to 10–20 for the final table.
- Raw handoff, no filtering — and it is stable at the real low ν (no NaNs), so filtering
  may be unnecessary (confirm via ablation).

---

## 6. Immediate next steps
- Reword the research identity to drop "autoregressive"; keep the direct-map framing above.
- Regenerate the dataset (`scripts/generate_dataset.py`) + install deps to (a) verify the
  wrapper against the imported team solver, (b) extend to 10–20 ICs, (c) add DeepONet.
- Figures built in `results/module2/figures/` (see the viva guide, Section 8).
