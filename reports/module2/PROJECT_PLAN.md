# Module 2 - Project Plan (Phase-Based)

**Student:** Dharmapala R.D. (214050V) | **Module:** Coupling (ML-to-numerical handoff)
**Scope:** Given an externally supplied switch time, hand the FNO's predicted state to a
pseudo-spectral numerical continuation and quantify when this improves long-horizon accuracy.

## How to read the time allocation
The remaining window is about **two working weeks** alongside exams, so allocations are
**effort estimates with built-in buffer**, not fixed deadlines. Effort is given as a share of
total remaining work and a rough number of focused working days. Phases on the **critical path**
must be done in order; **optional** phases are dropped first if time runs short (see cut rule).

## Phase overview

| Phase | Objective | Key deliverables | Effort | Est. days | Priority | Status |
|---|---|---|---:|---:|---|---|
| 0 | Foundation, audit & restart gate | Data contract, restart wrapper, gate PASS, preliminary sweep, 5 figures | 20% | ~2 | Critical | **Done** |
| 1 | Solver equivalence & full restart validation | Wrapper == team solver (**done: bit-identical**); regenerated-dataset restart on 10-20 ICs (needs torch) | 10% | ~1 | Critical | **In progress** |
| 2 | Core handoff experiment (viability region) | Switch-time sweep on 10-20 ICs; benefit metrics (rel-L2 + spectral distance); a-priori viability rule; final figures | 25% | ~2.5 | Critical | Planned |
| 3 | State-preparation study | Raw vs filtered handoff ablation; decide if cleaning is needed | 10% | ~1 | Should | Planned |
| 4 | Generalisation | One OOD case; transfer to DeepONet (same pipeline) | 15% | ~1.5 | Optional | Planned |
| 5 | Validation & reproducibility | Automated tests; experiment records; equivalence assertions | 10% | ~1 | Critical | Planned |
| 6 | Write-up & viva prep | Methodology + results + limitations; viva guide finalised; optional dashboard | 20% | ~2 | Critical | Planned |

**Rough calendar:** Week 1 = Phases 1-2 (+ start 3). Week 2 = Phases 3-6, with the last day held as buffer.

## Phase detail

### Phase 0 - Foundation, audit & restart gate  (DONE)
Audited the real repo, wrote the data contract, discovered the FNO is a direct space-time
operator, built the restart wrapper, passed the true-state restart gate (~1e-6), and produced a
preliminary hybrid sweep + five figures. Deliverables: `baseline_audit.md`, `restart_spectral.py`,
`make_figures.py`, `results/module2/figures/`.

### Phase 1 - Solver equivalence & full restart validation
**Objective:** prove the restart wrapper is the team solver, and validate restart on a
regenerated dataset. **Tasks:** install deps + regenerate the Cole-Hopf dataset; assert
`solve_from(u0, 0)` reproduces `spectral.solve(u0)` to ~1e-12; re-run the true-state restart gate
on several ICs. **Exit criteria:** equivalence assertion passes (**DONE — bit-identical, `verify_restart.py`**); gate passes on a regenerated set beyond the cached 10 (needs torch on your machine to run FNO on more test ICs).

### Phase 2 - Core handoff experiment (viability region)  [CRITICAL PATH]
**Objective:** the defensible result. **Tasks:** switch-time sweep over 10-20 held-out ICs;
report rel-L2 and spectral-distance benefit with mean +/- spread; freeze the viability rule
(benefit >= 10% AND absolute error bound) before viewing results; finalise the five figures.
**Exit criteria:** clean viability-region result with error bars across ICs.

### Phase 3 - State-preparation study
**Objective:** is handoff cleaning needed? **Tasks:** raw vs light/moderate spectral-filter
ablation on validation ICs; freeze one setting before the final test. **Exit criteria:** a stated,
justified handoff setting (current evidence: raw is already stable, so cleaning may be unnecessary).

### Phase 4 - Generalisation  [OPTIONAL / DROPPABLE]
**Objective:** show the coupling is not FNO-specific. **Tasks:** one OOD initial condition; apply
the same pipeline to DeepONet without changing the coupling logic. **Exit criteria:** at least one
generalisation result, or a documented reason it does not transfer.

### Phase 5 - Validation & reproducibility
**Objective:** make the result trustworthy. **Tasks:** automated tests (restart consistency,
finite output, time-grid alignment, no ground-truth leakage, determinism); per-experiment records
(git commit, dataset version, switch time, metrics). **Exit criteria:** tests pass; every figure
is reproducible from a recorded config.

### Phase 6 - Write-up & viva preparation
**Objective:** communicate the work. **Tasks:** methodology, results, honest limitations, and
individual-contribution sections; finalise the viva guide; optional two-mode dashboard only if
core results are complete. **Exit criteria:** a defensible written module section + rehearsed
viva story. The final day is buffer.

## Mandatory vs optional
- **Mandatory:** Phases 1, 2, 5, 6 (equivalence, viability region, tests, write-up).
- **Should:** Phase 3 (filtering ablation).
- **Optional:** Phase 4 (OOD + DeepONet), dashboard.

## Cut rule
If the Phase 2 sweep is not clean by the end of Week 1, **drop Phase 4 and the dashboard** and use
the remaining time for correctness, more ICs, tests, figures, and the write-up. The core viability
result matters more than extra features.

## Change log
| Date | Update |
|---|---|
| 2026-07-08 | Initial phase-based plan created (replaces the earlier day-by-day plan). Phase 0 complete. |
