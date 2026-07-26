# Team Turingz - Verified Results Pack (report source of truth)

Every number below is read directly from a results file in the repository at build time.
Regenerate any of them with the command shown. Do NOT retype numbers into the report from
memory - copy them from here, and if an experiment is re-run, rebuild this file first.
Build: python docs/report/build_results_pack.py

## T1. Full-system timed frontier (FNO + M1 coarse trust + M2 coupling + M3 controller)
Source: results/m3/step9d_coarse_integration/timed_cost_result_m2.json
Reproduce: python -m hybrid_pde.control_214133E._run_full

Pure ML baseline:        0.174 s at 7.77% error
Pure numerical baseline: 1.715 s at 0.0107% error

| target | cost (s) | error | vs numerical | vs pure ML | hit rate |
|---|---|---|---|---|---|
| 0.3 | 0.499 | 5.31% | 3.43x cheaper | 1.46x more accurate | 100% |
| 0.2 | 0.644 | 4.77% | 2.66x cheaper | 1.63x more accurate | 100% |
| 0.1 | 0.784 | 4.00% | 2.19x cheaper | 1.94x more accurate | 100% |
| 0.05 | 0.849 | 3.59% | 2.02x cheaper | 2.16x more accurate | 100% |
| 0.02 | 0.985 | 3.00% | 1.74x cheaper | 2.59x more accurate | 10% |
| 0.01 | 0.988 | 3.00% | 1.74x cheaper | 2.59x more accurate | 10% |

HEADLINE: 0.50-0.99 s at 3.0-5.3% error, vs 1.71 s numerical and 7.8% pure ML => 1.7-3.4x cheaper, up to 2.6x more accurate.
CAVEAT: timing is machine/session dependent; error, hit-rate and correction counts are deterministic.
All cost figures in one report table must come from the SAME run (this one).

## T2. Switching ablation (five policies, six targets, 10 ICs, 200 steps)
Source: results/m3/step11_switching_ablation/switching_ablation.json
Reproduce: python -m hybrid_pde.control_214133E._ablation_switching

| target | policy | error | corrections | switch events | hit rate |
|---|---|---|---|---|---|
| 0.3 | latch | 5.31% | 31.1 | 0.9 | 100% |
| 0.3 | deadband | 5.93% | 30.9 | 1.1 | 100% |
| 0.3 | naive | 6.19% | 30.2 | 1.8 | 100% |
| 0.3 | hardcoded | 4.45% | 43.0 | 1.0 | 100% |
| 0.3 | fixed_matched | 7.73% | 34.0 | - | 100% |
| 0.2 | latch | 4.77% | 39.0 | 1.0 | 100% |
| 0.2 | deadband | 5.48% | 38.8 | 1.2 | 100% |
| 0.2 | naive | 5.36% | 38.1 | 1.9 | 100% |
| 0.2 | hardcoded | 4.45% | 43.0 | 1.0 | 100% |
| 0.2 | fixed_matched | 7.73% | 40.0 | - | 100% |
| 0.1 | latch | 4.00% | 48.2 | 1.0 | 100% |
| 0.1 | deadband | 4.76% | 48.0 | 1.2 | 100% |
| 0.1 | naive | 4.91% | 47.2 | 2.0 | 100% |
| 0.1 | hardcoded | 4.45% | 43.0 | 1.0 | 100% |
| 0.1 | fixed_matched | 7.72% | 50.0 | - | 80% |
| 0.05 | latch | 3.59% | 52.3 | 1.0 | 100% |
| 0.05 | deadband | 4.51% | 52.0 | 1.3 | 80% |
| 0.05 | naive | 4.56% | 51.4 | 1.9 | 80% |
| 0.05 | hardcoded | 4.45% | 43.0 | 1.0 | 70% |
| 0.05 | fixed_matched | 7.72% | 50.0 | - | 20% |
| 0.02 | latch | 3.00% | 66.8 | 1.0 | 10% |
| 0.02 | deadband | 4.36% | 56.5 | 1.4 | 0% |
| 0.02 | naive | 4.42% | 54.1 | 2.1 | 0% |
| 0.02 | hardcoded | 4.45% | 43.0 | 1.0 | 0% |
| 0.02 | fixed_matched | 7.70% | 67.0 | - | 0% |
| 0.01 | latch | 3.00% | 66.8 | 1.0 | 10% |
| 0.01 | deadband | 4.36% | 56.5 | 1.4 | 0% |
| 0.01 | naive | 4.42% | 54.1 | 2.1 | 0% |
| 0.01 | hardcoded | 4.45% | 43.0 | 1.0 | 0% |
| 0.01 | fixed_matched | 7.70% | 67.0 | - | 0% |

- latch: error spans 3.00%-5.31% (range 2.31pp) -> RESPONDS to target
- deadband: error spans 4.36%-5.93% (range 1.57pp) -> RESPONDS to target
- naive: error spans 4.42%-6.19% (range 1.77pp) -> RESPONDS to target
- hardcoded: error spans 4.45%-4.45% (range 0.00pp) -> FLAT (ignores the target)

KEY CLAIMS THIS SUPPORTS:
1. vs fixed-interval schedule at matched cost: adaptive 3.00-5.31% vs fixed 7.70-7.73% (dominates).
2. vs hand-tuned fixed threshold (theta=0.4): same frontier, but hardcoded is FLAT (0.0445 at every
   target) while the adaptive controller responds (0.0531 -> 0.0300). The contribution is controllability.
3. Hysteresis deadband: measured unnecessary on the real trust signal (naive re-engages ~2x only);
   the shipped controller is a one-way latch. Do NOT claim deadband as novelty.
4. Knob saturates below target ~0.029 (theta_lo clips at 0.58): targets 0.02 and 0.01 identical.

## T3. PINN through the controller (10 pre-trained ICs, training cost EXCLUDED)
Source: results/m3/step12_pinn_regime/pinn_controller.json
Reproduce: python demo_app/backend/_pinn_regime.py

| target | error | corr fraction | hit rate |
|---|---|---|---|
| 0.3 | 18.24% | 33% | 80% |
| 0.2 | 14.51% | 36% | 60% |
| 0.1 | 10.42% | 39% | 50% |
| 0.05 | 8.55% | 41% | 40% |
| 0.02 | 8.03% | 45% | 40% |
| 0.01 | 8.03% | 45% | 30% |

Knob responds: True (error 8.03%-18.24%), best hit-rate 80%.
CLAIM: PINN is excluded by DEPLOYMENT cost (2114 s retrain per problem, ~950x numerical),
NOT by controllability - the controller works on it and cuts standalone extrapolation error ~4x.

## T4. Per-model deployment profile (one machine, 30 repeats)
Source: results/deployment/cost_summary.json

| model | deploy cost | error in-window | error beyond | params | disk | peak mem | scaling exp |
|---|---|---|---|---|---|---|---|
| FNO | 0.444 s | 0.57% | 14.13% | 198,465 | 1.40 MB | 1197 MB | 1.00 |
| DeepONet | 0.729 s | 29.31% | 61.11% | 546,817 | 2.19 MB | 1309 MB | 1.04 |
| PINN | 35 min | 1.47% | 35.80% | 12,737 | 0.06 MB | 1061 MB | 1.02 |
| FDM | 0.086 s | 8.83% | 12.97% | 0 | 0.00 MB | 847 MB | 0.11 |
| Spectral | 1.750 s | 0.01% | 0.00% | 0 | 0.00 MB | 847 MB | 0.02 |
| ColeHopf | 2.228 s | 0.00% | 0.00% | 0 | 0.00 MB | 859 MB | 1.01 |

PINN training split: 1344 s Adam + 771 s L-BFGS = 2114 s per problem.

## T5. DeepONet hybrid frontier (operating-regime evidence)
Source: results/m3/step9d_coarse_integration/timed_cost_result_deeponet.json
Reproduce: python -m hybrid_pde.control_214133E._run_full DeepONet

Baselines that session: ML 0.120 s at 39.0%; numerical 1.684 s.

| target | cost (s) | rel. to numerical | error | hit rate |
|---|---|---|---|---|
| 0.3 | 2.114 | 1.26x | 22.7% | 80% |
| 0.2 | 2.620 | 1.56x | 22.2% | 60% |
| 0.1 | 2.744 | 1.63x | 23.0% | 0% |
| 0.05 | 4.594 | 2.73x | 23.0% | 0% |
| 0.02 | 4.635 | 2.75x | 23.0% | 0% |
| 0.01 | 5.273 | 3.13x | 23.0% | 0% |

CLAIM: with DeepONet the hybrid costs 1.26-3.13x the numerical solver at ~23-39% error - strictly
dominated. Precondition violated: accuracy in-window (29.3% standalone). NOTE: normalise each frontier
by its OWN session's numerical baseline; never mix cost numbers across sessions.

## T6. Coarse-reference drift detector (M1 integration evidence)
Source: results/m3/step11_coarse_detector/verification.json
- Coarse-detector correlation with true error: 0.999
- Physics-residual correlation with true error: 0.683
- True failure onset detected at t = 1.1457286432160805
NOTE: production detector is M1's contribution; M3 prototyped the concept (status: 'concept verified,
stand-ins'). Attribute accordingly in the report.

## T7. Out-of-distribution frontier (real FNO + M1 coarse + M2 coupling + M3 controller)
Source: results/m3/ood_frontier/m3_ood_frontier.json  (experiments/m3_ood_frontier.py)
Grid nx=512, nt=200. OOD set = sin(5..8 pi x) + 3 Gaussian bumps (7 waves); exact Cole-Hopf references.
Self-check 1: analytic reference vs committed dataset (IC 900) rel diff 3.8e-08.
Self-check 2: in-distribution frontier reproduces the committed deterministic headline exactly
(5.31->3.00% error, 100% hit down to 0.05, 3.00% floor); timing session-dependent, not compared.

Baselines on the OOD set: pure ML 36.64% error; pure numerical 0.042% error.
(In-distribution same session: pure ML 7.77%; pure numerical 0.011%.)

| target | in-dist error | in-dist hit | OOD error | OOD hit | OOD corrections (of 200) |
|---|---|---|---|---|---|
| 0.30 | 5.31% | 100% | 12.51% (+/-3.29) | 100% | 145.7 |
| 0.20 | 4.77% | 100% | 11.10% (+/-2.62) | 100% | 147.7 |
| 0.10 | 4.00% | 100% | 10.92% (+/-2.62) | 14% | 172.4 |
| 0.05 | 3.59% | 100% | 10.58% (+/-2.75) | 0% | 173.1 |
| 0.02 | 3.00% | 10% | 10.11% (+/-2.95) | 0% | 173.6 |
| 0.01 | 3.00% | 10% | 10.11% (+/-2.95) | 0% | 173.6 |

CLAIM: on unseen inputs the surrogate degrades ~4.7x (7.77% -> 36.64%) while the numerical solver is
unaffected (0.042%). The controller detects the drift and corrects far more aggressively - corrections
rise from 31-67 to 146-174 of 200 steps - so the hybrid CAPS error at 10-12.5%, about 3x better than
pure ML OOD, at a cost that rises toward numerical (1.07-1.25 s vs ~1.42 s). This is graceful
degradation / fail-safe, quantified. HONEST LIMIT: the error floor rises from 3.00% (in-dist) to ~10%
(OOD) and tight targets (<=0.1) become unreachable (hit-rate 0-14%); only relaxed budgets (0.2, 0.3)
are still honoured OOD. Mechanism: the monitor's pre-flag window runs on a badly-wrong ML seed OOD, so
the same monitor-set floor of 7.5.5 is amplified. Cross-session cost pairing forbidden as elsewhere.

## Numbers that must NOT be claimed (superseded or unsupported)
- 'adaptive 0.0% vs fixed 8.6%' - superseded synthetic result; use T2 (3.00-5.31% vs 7.70%).
- 'hysteresis deadband prevents chatter in the system' - T2 shows it is unnecessary on the real signal.
- 'nearly numerical-level accuracy' - floor is 3.00%, numerical is 0.011%. Say 'usable accuracy at a
  fraction of the cost' with the actual numbers.
- Any cost pairing that mixes sessions (e.g. 2.54 s numerical with this run's hybrid costs).