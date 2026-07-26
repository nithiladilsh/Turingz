# Final Report Skeleton + Evidence Map - Team Turingz (Group 74)

Rules for every chapter:
- "N.1 Introduction" first, Summary last. Figures/tables captioned (Figure N.M: ...) and cited in text BEFORE appearing.
- Every reference cited; Mendeley + IEEE. All numbers from docs/report/RESULTS_PACK.md - never from memory.
- Bulk code/screenshots -> Appendices, cited from chapters.

Chapters: 1 Introduction [done, 2 fixes pending] | 2 Literature Review | 3 Technology Adapted |
4 Approach & Design | 5 Implementation | 6 Evaluation & Discussion | 7 Conclusion | Refs | Appendices A-D

---------------------------------------------------------------
## Chapter 2 - Literature Review (per-module blocks; shared background written once)
2.1 Introduction (recap Ch1's three problems, roadmap sections; NO citations)
2.2 Background [shared, salvage interim, written ONCE]
    2.2.1 Classical numerical methods - FD, spectral, Cole-Hopf. Refs [18]-[29]
    2.2.2 ML surrogates - PINN, DeepONet, FNO. Refs [1]-[6]
2.3 Module 1 review - Reliability & trust [Sandeepa]
    2.3.1 Introduction | 2.3.2 Review of others' work | 2.3.3 Challenges | 2.3.4 Research gap
    Refs: [7] Wang, [8] Krishnapriyan, [9] Fesser, [10] Zhu, [12] Lu, [13] PDEBench;
          [14] Psaros, [15] Gopakumar, [16] deep ensembles, [17] conformal
2.4 Module 2 review - Hybrid coupling [Dharmapala]
    2.4.1 Introduction | 2.4.2 Review of others' work | 2.4.3 Challenges | 2.4.4 Research gap
    Refs: [30] Kochkov, [32] HINTS (Zhang/Nature MI), [11] PDE-Refiner,
          [31] ANCHOR (arXiv:2512.19643 - closest work, longest treatment), [33] Wu
2.5 Module 3 review - Cost-aware control [Mendis]
    2.5.1 Introduction | 2.5.2 Review of others' work | 2.5.3 Challenges | 2.5.4 Research gap
    Refs: [34] Hespanha, [35] Liberzon, [36] Astrom, [37] Mayne MPC, [38] Peherstorfer,
          [39] Zilberstein, [40] Dean & Boddy, [41] Miettinen / [42] Deb
    Gap: no controller takes a TARGET ACCURACY and schedules numerical effort to meet it at
    minimum cost; no measured cost/accuracy frontier for a deployed hybrid.
2.6 Comparison of approaches (required table)
2.7 Consolidated research gap - ties the three per-module gaps together
2.8 Summary
EVIDENCE:
- TABLE 2.1 (in 2.6 or 2.7): approach | detects failure how | corrects how |
  target accuracy requestable? | cost measured? Prior rows No/No; this project Yes/Yes.
- Each module block (2.3-2.5) uses the same four sub-parts (matches interim; each member's gap visible).
- Cite ONLY the 42 consolidated refs; insert via Mendeley (numbering by first appearance).
  ANCHOR = [31], HINTS = [32] (published Nature MI version).

## Chapter 3 - Technology Adapted (short; justify, don't tutorialize)
3.1 Introduction | 3.6 Summary (required)
3.2 Benchmark: 1D viscous Burgers, nu = 1/(100*pi) - shock formation + exact Cole-Hopf reference
3.3 Numerical solvers: Cole-Hopf primary (no time-stepping drift), spectral verifier+corrector, FDM baseline
3.4 ML architectures: PINN per-instance vs FNO/DeepONet amortized (plant the regime distinction here)
3.5 Libraries: PyTorch, DeepXDE, neuraloperator, NumPy/SciPy; FastAPI+React demo
EVIDENCE:
- FIGURE 3.1: shock formation illustration -> use results/eval/3_snapshots.png (or regenerate cleaner).
- TABLE 3.1: component inventory (solver/model | role | library | key parameter).

## Chapter 4 - Approach and Design (NO result numbers here)
4.1 Introduction | 4.7 Summary (required)
4.2 System concept: one pipeline, three modules, fixed contracts, stub-first development
4.3 M1 design - trust signals, fusion, calibration, coarse-reference check [M1]
4.4 M2 design - hand-off operator, corrector, switching mechanics, return-to-ML, safeguards [M2]
4.5 M3 design - profiler; accuracy-cost model + thresholds_for_target (the budget map);
    controller state machine (one-way latch shipped, hysteresis tested-not-shipped, forward-ref Ch6);
    deployment runtime run(IC, target) -> solution + cost report [M3]
4.6 Integration design: stubs swapped at integration; no module depends on another's internals
EVIDENCE:
- FIGURE 4.1 (required by guidelines): top-level architecture. Redraw clean:
  IC -> ML surrogate -> [M1 Trust] -> [M2 Coupling] -> [M3 Control] -> solution + cost report.
  Base exists at docs/architecture/image.png - redraw, don't paste as-is.
- TABLE 4.1: the three data contracts (from restructured plan Table 4).
- FIGURE 4.2: M3 controller decision flow (base: docs/architecture/module3-architecture.png).
- PSEUDOCODE only in Ch4 (3-6 lines for decide()); real code belongs in Ch5.
- M1/M2 add one design figure each (M1: signal-fusion diagram, trust_module_mechanism.png as base;
  M2: hand-off sequence diagram).

## Chapter 5 - Implementation
5.1 Introduction | 5.7 Summary (required)
5.2 Shared foundation (brief) | 5.3 M1 impl | 5.4 M2 impl | 5.5 M3 impl | 5.6 Integration
EVIDENCE (M3 = 5.5):
- LISTING 5.x: AdaptiveController.decide() - the real ~10 lines.
- LISTING 5.x: thresholds_for_target - both lines INCLUDING the 0.58 clip (saturation shown, discussed 6.5.4).
- Runtime step loop as short pseudocode; integrate.py timed-frontier protocol described in prose
  (warm-up, per-IC timing, mean/std over 10 ICs).
- One paragraph each: _ablation_switching.py, _pinn_regime.py, _run_full.py with their one-line
  reproduce commands (python -m hybrid_pde.control_214133E._run_full etc.).
- FIGURE 5.x: ONE demo screenshot here (Cost control - Three-way race, light mode); all other
  demo screenshots -> Appendix C.
- 5.6: quote the conformance check result (M2 contract checker 10/10) - one line, no screenshot needed.

## Chapter 6 - Evaluation and Discussion (numbers ONLY from RESULTS_PACK.md)
6.1 Introduction (restate the evaluation questions = plan Table 5) | 6.8 Summary (required)
6.2 Setup: machine spec, repeats, 10 test ICs, 200 steps, rel-L2 metric; state ONCE the
    timing-vs-determinism caveat (timings session-dependent; errors/hit-rates/corrections deterministic).
    - TABLE 6.1: experimental setup summary.
6.3 M1 results [M1]
    - FIGURE: results/m3/step11_coarse_detector/coarse_vs_residual.png (detector 0.999 vs residual 0.683)
      + M1's own calibration/ROC figures. Attribution note: M3 prototyped concept, M1 productionised.
6.4 M2 results [M2]
    - FIGURES from results/module2/figures/: fig1_error_over_time, handoff_viability_gate,
      restart_safety_boundary (owner picks 3-4; each needs caption + in-text citation).
6.5 M3 results [YOU]
    6.5.1 Full-system frontier: TABLE 6.x = Pack T1 (6 targets, cost/error/ratios/hit-rate).
          FIGURE 6.x: frontier plot - REGENERATE from timed_cost_result_m2.json; the existing
          step9d/timed_pareto.png PREDATES the rerun. Do not reuse without regenerating.
    6.5.2 Switching ablation: TABLE 6.x = condensed T2 (5 policies at targets 0.3/0.05/0.01);
          full table -> Appendix D. FIGURE 6.x (generate): hit-rate vs target per policy, or
          error-range bar showing hardcoded FLAT vs adaptive RESPONDS.
    6.5.3 Operating regime: TABLE 6.x = T3 (PINN) + T5 (DeepONet, rel-to-own-numerical).
          FIGURE 6.x (generate): FNO vs DeepONet frontiers normalised by each session's own
          numerical baseline, numerical = 1.0 line.
    6.5.4 Limitations as findings: 3.00% floor; saturation below target 0.029; hysteresis
          unnecessary (latch shipped); wall-clock noise -> corrections are the controlled variable.
6.6 Robustness/OOD [shared]: in-dist vs OOD; deployment profile figures
    (results/deployment/2_cost_accuracy.png, 4_amortization.png) if used, verify against Pack T4.
6.7 Discussion (required): vs ANCHOR - proactive budget-driven vs reactive fixed threshold;
    revisit Table 2.1 with a "this project (measured)" row.

## Chapter 8 - Conclusion and Further Work   [FINALIZED]
Rule: NO new numbers anywhere in this chapter - every conclusion points back to the Ch7 section that
proved it. Intro first, Summary last (guideline). Individual contributions live in Appendix A, so
contributions are folded into 8.2 rather than given a standalone section (DeepThinkers pattern).

8.1 Introduction - restate the problem the project set out to solve + roadmap of the chapter (1 para).
8.2 Conclusions against objectives - walk O1-O5 in turn; state the conclusion and cite where it was
    demonstrated (Ch7 sections). Contribution per objective folded in here. NO new results.
    [ACTION: paste exact O1-O5 wording from Chapter 1 so conclusions map one-to-one.]
8.3 Limitations - project level, concise: one PDE (1D viscous Burgers), single-machine timing,
    3% accuracy floor. (Distinct from the finer 7.5.5 findings; this is the project-level summary.)
8.4 Further work - lower-floor trust monitor; learned controller (declared optional stretch, not a
    promise); more PDEs and 2D; broader OOD testing.
8.5 Summary (required) - short closing.

NOTE: header chapter list above still shows old 7-chapter numbering (pre Ch4/Ch5 split); working
report is 8 chapters with Ch7 = Evaluation & Discussion, Ch8 = Conclusion.

## Appendices
A: Individual Contribution x3 (one page each; yours = six plan components -> what exists)
B: Code listings (controller.py complete; integrate.py + _ablation_switching.py excerpts; M1/M2 equivalents)
C: Demo screenshots (all pages: Overview, solvers, Reliability, Robustness, Cost analysis,
   Trust, Coupling, Cost control x3 tabs, Hybrid engine)
D: Full result tables (complete T2 ablation; T3 PINN regime; per-IC frontier data)

---------------------------------------------------------------
## FIGURES THAT MUST NOT BE USED (stand-in era / superseded)
- step5_controller/controller.png        - synthetic hysteresis demo; cut from the story
- step7_pareto/pareto.png                - stand-in Pareto, superseded by timed frontier
- step8_robustness/adaptive_vs_fixed.png - the 0.0-vs-8.6 stand-in; superseded by T2
- step9_integration/ + step9b_*/ dry-run PNGs - stand-in era
- step10_demo/demo_preview.png           - superseded by real app screenshots
- step9d PNGs (timed_pareto, coarse_frontier) - REAL but predate the rerun; regenerate before use

## REPORT-READY FIGURES (already generated - docs/report/figures/)
Built from the current JSONs by docs/report/build_figures.py; re-run that script after any experiment re-run.
- fig_6_5_1_frontier.png   - full-system frontier, error bars, both baselines (Ch6.5.1)
- fig_6_5_2_ablation.png   - error + hit-rate vs target, five policies (Ch6.5.2)
- fig_6_5_3_regime.png     - FNO vs DeepONet frontiers, normalised, numerical = 1.0 line (Ch6.5.3)
- fig_6_5_3b_pinn.png      - controlled PINN vs standalone 35.8% line + hit-rate (Ch6.5.3)
Distinct markers/linestyles per series, so they survive grayscale printing.
