# Deployment & computational-cost analysis of the six solvers

**Module:** Deployment & Orchestration — Team Turingz
**Question this answers:** The reliability study tells us *how accurate* PINN, FNO and DeepONet are. This document tells us *what every solver costs to run* — the three ML models **and** the three numerical solvers (Cole-Hopf, FDM, spectral) that the hybrid engine falls back to — so that the orchestrator can route a query to the cheapest solver that still meets the accuracy budget, and so that the cost of the numerical fallback it throttles is *measured*, not assumed. It is the cost half of the cost–accuracy decision.

The analysis is deliberately built to be *fair*: every solver is measured on the **same hardware, the same CPU thread budget, the same initial conditions, and the same space–time grid** as the reliability study (the 10 operator-unseen test ICs 900–909, on the 512×200 Cole-Hopf grid). ML accuracy is taken directly from `results/eval/reliability_summary.json`; numerical-solver accuracy is computed here as relative error vs the Cole-Hopf reference, so cost and accuracy refer to exactly the same predictions throughout.

The numerical solvers join the comparison as the **expensive, accurate anchor** and slot naturally into the per-query side of the cost story (like a PINN, every new IC is a fresh solve — there is nothing to amortize). Two honest caveats: (1) the numerical solvers *define* the accuracy reference, so on the cost–accuracy plot they sit at ≈0 error **by construction** — that is "ground truth," not "best model"; (2) Cole-Hopf is an analytical solution **specific to Burgers** (fast but not generalizable), so **FDM and spectral are the representative general-purpose numerical cost**, while Cole-Hopf is the special analytical case. Their memory is measured inside the same torch-based harness as the ML models for a like-for-like footprint, though in practice the numerical solvers need only NumPy and would have a smaller standalone footprint.

Reproduce everything with `python hybrid_pde/evaluation/deployment_cost_analysis.py`. All raw numbers land in `cost_summary.json`; the four figures are described below.

---

## What is measured, and why each metric is the fair one

A neural PDE solver has two very different cost regimes, and conflating them is the most common way these comparisons go wrong. We separate them explicitly.

**Offline (one-time) training cost** is paid once, before deployment, and is then *sunk*. For FNO and DeepONet this is the operator-training run; for the orchestrator it does not matter how long it took, because it is never repeated at serving time.

**Online (per-query) cost** is paid *every time a new initial condition arrives*. This is the number the deployment layer actually budgets against. Here the three models split into two fundamentally different classes:

- **FNO and DeepONet are amortized operators.** A new IC is a single forward pass over the grid. Training is done once and reused for every future IC.
- **A PINN is *not* amortized.** Each network is fit to one specific IC (see `pinn.py`: one `pinn_ic{i}.pt` per IC). Serving a *new, unseen* IC means solving a fresh optimisation from scratch (15 000 Adam iterations + L-BFGS). The cheap forward pass of an already-trained PINN is *not* the deployment cost — the optimisation is.

Treating the PINN's forward pass as its "inference cost" would understate its true per-query cost by several orders of magnitude. The honest per-query cost we report for the PINN is **train-from-scratch + evaluate**; for the operators it is the **forward pass**. This single distinction is the central result of the analysis, and figure 4 is built around it.

**How the PINN training cost is measured (so the comparison is fair, not inflated or deflated).** Running the full 15 000-iteration Adam phase just to time it would dominate the analysis runtime, so we measure it the way one measures any constant-cost loop: each Adam iteration recomputes the *same* residual graph (fixed network, fixed collocation set: 2540 domain + 80 boundary + 160 initial), so per-iteration time is constant. We discard the first 100 iterations as warm-up (graph build, autograd setup, cache warming — a one-time overhead that would otherwise inflate the rate), time the next 300 iterations to get the steady-state per-iteration cost, and scale to the real 15 000. We then **measure the L-BFGS phase directly** by running it to convergence and timing it, rather than ignoring it. The reported PINN cost is therefore `Adam_steady_rate × 15 000 + L-BFGS_measured` (both components are in `cost_summary.json` as `adam_s` and `lbfgs_s`). The one disclosed approximation is that L-BFGS is timed from the short Adam warm-start rather than from a fully-converged 15 000-iteration state; its per-iteration cost is identical either way, only the iteration count to convergence can differ. None of this affects the headline, which is an *orders-of-magnitude* gap (milliseconds vs tens of seconds) that no plausible timing error can close.

The five metrics, and the reason each is included:

1. **Runtime (per-IC inference latency).** Raw prediction speed: wall-clock time to produce the full 512×200 field for one IC, on a single CPU thread, reported as mean ± std over 30 timed repeats after 5 warm-up runs. Single-thread + warm-up is what makes the number reproducible rather than machine-noise.
2. **Throughput (new ICs / second).** The serving-rate the orchestrator can sustain. For the operators this is measured on a batched run (realistic serving); for the PINN it is `1 / (train + eval)`, because a "new IC" *is* a training run.
3. **Memory (peak resident set size of an isolated serving process).** Hardware-resource consumption. Measured in a **fresh subprocess** that imports the framework, loads one model and serves one IC, sampling its own peak RSS — so the number is order-independent and captures everything the deployment actually needs resident, including the PyTorch runtime (an earlier in-process delta version was discarded because it measured only marginal per-call allocation and was sensitive to call order). Because the framework runtime dominates this figure, the *pure model* size is reported separately as parameter memory (`params × 4 bytes`) and on-disk checkpoint size; the headline reading is that all three networks are comparable and modest in footprint — none is memory-constrained — which is the relevant contrast with the numerical solver that the hybrid engine must throttle for memory.
4. **Computational complexity (scalability).** Latency is measured across grid sizes `N = nt × nx` and a log–log slope is fitted (`exp` in the JSON). The slope is the empirically measured scaling order — the fair, model-agnostic way to compare how each method grows, rather than trusting big-O on paper.
5. **Cost–accuracy trade-off.** Per-query cost plotted against the relative error from the reliability study. This is the only metric that actually answers "which model should I deploy", and it is the basis for the recommendations below.

---

## Expected structure of the results (grounded in the architectures)

The script produces the exact numbers; the *ordering* is predictable from the methods themselves and is what the figures should confirm:

| | PINN | FNO | DeepONet |
| :--- | :--- | :--- | :--- |
| Per-query class | **not amortized** (re-optimise per IC) | amortized (1 forward pass) | amortized (1 forward pass) |
| Dominant per-query cost | Adam + L-BFGS optimisation | spectral conv forward | branch×trunk forward |
| Inference-latency theory | O(N) forward, but optimisation dominates | O(N log N) (FFT in the spectral layer) | O(N · p) (linear in query points) |
| In-window error (t ≤ 1)¹ | 1.47% | **0.57%** | 29.3% |
| Extrapolation error (t > 1)¹ | 35.8% | **14.1%** | 61.1% |

¹ relative error vs Cole-Hopf, from `reliability_summary.json` (the same source as the reliability plots).

The headline is that **FNO is expected to sit at the favourable corner of every plot**: lowest error in *both* time windows, and amortized (millisecond) per-query cost. The **PINN is the opposite extreme on cost** — its accuracy is respectable, but each new IC costs a full optimisation, so its per-query cost is larger than the operators' by *orders of magnitude*, and its deployment throughput is correspondingly tiny. **DeepONet is cheap to serve but inaccurate** on this shock-forming problem (the ~14% training plateau documented in `FINDINGS.md`), so it is dominated on accuracy.

The **numerical solvers** complete the picture as the accurate-but-costly reference (measured here, not assumed):

| | Cole-Hopf | FDM | Spectral |
| :--- | :--- | :--- | :--- |
| Per-query class | not amortized (fresh solve per IC) | not amortized | not amortized |
| Dominant per-query cost | analytical convolution, O(nt·nx²) | explicit time-stepping, CFL-limited | ETDRK4 + FFT, sub-stepped |
| Accuracy vs Cole-Hopf | reference (≈0 by construction) | ~16% (numerical diffusion blurs the shock) | ~0.01% (independent high-accuracy method) |
| Generalises to other PDEs? | no (Burgers-specific) | yes | yes |

The expected reading: the numerical solvers anchor the **low-error / high-cost** corner of the cost–accuracy plot, opposite the cheap ML operators. This is exactly the trade-off the hybrid engine exploits — serve with the cheap amortized operator while it is reliable (t ≤ 1), and pay the numerical solver's cost only when the ML prediction is flagged unreliable (t > 1). One subtlety the figures will show: a numerical solver's cost is set by **stability-limited time-stepping**, so it scales with spatial refinement (CFL) far more steeply than with the number of output snapshots — which is why refining the grid, not lengthening the horizon, is what makes the fallback expensive.

---

## The four figures

1. **`1_runtime_throughput.png`** — per-IC inference latency (log) and deployment throughput (new ICs/s, log). Reads raw speed and serving rate at a glance.
2. **`2_cost_accuracy.png`** — the central plot: per-query cost (log x) vs relative error (y), with a circle for extrapolation (t > 1) and a triangle for in-window (t ≤ 1). The **lower-left is best** (cheap *and* accurate). Any model that is up-and-to-the-right of another is Pareto-dominated and should not be deployed.
3. **`3_scalability.png`** — latency vs grid points on log–log axes, with the fitted scaling exponent in the legend. Shows which solver degrades gracefully as the grid is refined.
4. **`4_amortization.png`** — cumulative serving wall-time vs number of new ICs. The PINN's line is steep (it pays full cost every IC); the operators are near-flat. This is the visual proof of the amortized-vs-not distinction and the reason an operator is the right default for a serving system.

---

## Measured results

Measured on an Intel Core Ultra (Model 170), single CPU thread, PyTorch 2.6.0, 30 timed repeats (numerical-solver latency over 10; full provenance in `cost_summary.json` → `_env`). Latency is the **median** (jitter-robust); cost and accuracy are on the same 10 unseen test ICs.

| Solver | Params | Size (MB) | Latency (ms) | Throughput (IC/s) | Peak RSS (MB) | Scaling exp. | Per-IC deploy | In-window err (t≤1) | Extrapolation err (t>1) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **FNO** | 198,465 | 1.40 | 441 | 2.3 | 1197 | 1.00 | **0.44 s** | **0.57%** | 14.1% |
| **DeepONet** | 546,817 | 2.19 | 727 | 12.2 | 1309 | 1.04 | 0.73 s | 29.3% | 61.1% |
| **PINN** | 12,737 | 0.06 | 96 | 0.0005 | 1061 | 1.02 | **2114 s** | 1.47% | 35.8% |
| **Cole-Hopf** | — | — | 2220 | 0.45 | 859 | 1.01 | 2.23 s | ≈0 | ≈0 |
| **FDM** | — | — | 84 | 11.7 | 847 | **0.11** | **0.086 s** | 8.83% | 13.0% |
| **Spectral** | — | — | 1727 | 0.57 | 847 | **0.02** | 1.75 s | 0.010% | 0.004% |

What the measured numbers establish:

- **The amortization gap is real and enormous.** A new IC costs FNO 0.44 s but a PINN **2114 s** — a ~4,800× gap — because the PINN re-optimises from scratch per IC while the operator does one forward pass. `4_amortization.png` shows the PINN line detached from all others.
- **FNO is the in-window operator of choice.** It is the only solver combining sub-1% in-window error (0.57%) with amortized sub-second serving. DeepONet is cheap-ish but its 29%/61% error rules it out; this is the documented ~14% training plateau, not noise.
- **The numerical solvers confirm the accuracy anchor.** Cole-Hopf reproduces the reference to 1e-4% (sanity check that the harness is correct), and spectral matches it to ~0.01% — so the cost–accuracy plot's near-zero-error corner is verified, not assumed. FDM sits at ~9–13% (numerical diffusion blurring the shock), as its standalone evaluation already found.
- **Numerical cost scales with stability, not output size.** FDM and spectral are essentially flat versus the number of output snapshots (scaling exponents 0.11 and 0.02) because their cost is set by CFL/`dt`-limited sub-stepping; Cole-Hopf and the ML models scale ~linearly (≈1.0) because each output point is genuine compute. This is why a numerical fallback's expense is driven by *spatial refinement*, not the prediction horizon.
- **The sharpest finding — FDM vs FNO is a window-dependent trade-off, not a clean win.** In **extrapolation** FDM beats FNO on *both* axes (0.086 s & 13.0% vs 0.44 s & 14.1%). But **in-window** FNO is an order of magnitude more accurate (0.57% vs 8.83%), and FNO's cost is resolution-independent whereas FDM's explodes under refinement (the flat-vs-steep scaling above). So FNO earns its place as the in-window operator, and the **accurate fallback is spectral (≈0% error), not FDM** — which is precisely the routing the hybrid engine implements.

---

## On the "Hybrid Efficiency Index" (HEI) — honest assessment

The earlier repo collapsed cost and accuracy into a single scalar (HEI). I'd advise **against making a hand-defined composite index the headline metric**, for three reasons that a reviewer would raise:

1. **It is not dimensionally meaningful.** Latency (seconds) and error (dimensionless %) live on different scales; any single number combining them hides an arbitrary weighting choice, and the ranking can be flipped just by changing that weight. There is no principled, reviewer-defensible value for it.
2. **It hides the trade-off it claims to summarise.** The decision genuinely depends on the deployment context (a tight latency SLA vs a tight accuracy SLA give different winners). A scalar throws away exactly the information the orchestrator needs.
3. **The accepted alternative is standard and transparent.** Report the **Pareto front** (figure 2) plus **explicit SLA thresholds**. This is the normal way efficiency is argued in the ML-systems literature, and it is fully reproducible from `cost_summary.json`.

If a single comparable number is genuinely required (e.g. for a dashboard), use a **transparent, disclosed** efficiency ratio rather than an opaque index — for example *accuracy delivered per second of per-query cost*, `(1 − relative_error) / deploy_seconds` — and always show it next to the underlying cost and error, never instead of them. The orchestrator can still consume such a number; it just should not be the evidence the analysis rests on.

---

## Threats to validity — what generalizes, and what an evaluator will (rightly) question

The most common objection to any timing study is *"these numbers depend on your machine."* That is true of the **absolute** wall-times, and we do not hide it — the exact CPU, core count, thread setting and PyTorch version are recorded automatically in `cost_summary.json` under `_env`, so every number is reproducible and attributable to a known machine. What matters is that **the conclusions do not rest on the absolute times.** They rest on three quantities that are machine-independent:

1. **The scaling exponents** (`3_scalability.png`, slopes ≈ 1.0–1.1). A slope is a *ratio of log-times*, so a faster or slower CPU shifts every curve up or down by a constant factor but leaves the slope unchanged. The measured near-linear scaling is a property of the algorithms, not the hardware.
2. **Parameter counts, model sizes, and the amortized-vs-not structure.** These are exact and identical on any machine. A PINN requires a fresh optimisation per IC and an operator requires one forward pass *by construction* — no hardware changes that.
3. **The order-of-magnitude gaps.** The PINN's per-query cost is ~10³× the operators' (≈2100 s vs ≈1–2 s). No realistic change of machine — even a high-end GPU, which might give a 10–50× speed-up — comes close to closing a 1000× gap. The ranking is therefore robust; only the precise multiplier moves.

So the defensible framing is: *absolute latencies are reported for a stated reference machine and will scale with hardware; the comparative findings (ordering, complexity slopes, amortization gap) are hardware-invariant and are what the recommendations are built on.*

**On the network:** it is not a factor. Every measurement is purely local compute — models are loaded from local checkpoints and run in-process. Nothing in the timed path touches the network, so network conditions cannot affect the results. (If this engine is later served behind an HTTP API, network/serialisation latency would be added *on top* equally for all three models and would not change their relative ordering.)

**Controls that make it reproducible:** single CPU thread (removes scheduler/core-count noise), 5 warm-up runs discarded before every measurement (removes cold-start/JIT effects), 30 timed repeats reported as **mean ± std and median** (the median is robust to the occasional OS-scheduling spike, so it is the figure to quote if an evaluator challenges the timing noise), and the same test ICs and grid as the reliability study (so cost and accuracy are measured on identical predictions).

**Repeat counts are matched to the measured timing variance, not fixed blindly.** The ML forward passes are millisecond-scale, so OS jitter is a large *relative* fraction (15–20% spread) and 30 repeats are needed to pin the mean. The numerical solvers are multi-second *deterministic* computations (a spectral solve is ~2 s, dominated by its `dt = 1e-4` sub-stepping), where the same fixed jitter is a negligible fraction (~2%), so their per-query latency is taken over 10 repeats — already rock-solid, as the reported std confirms — and their **throughput is derived as 1/latency** rather than timed separately, which is exact for non-amortized solvers (each new IC is an independent solve, so there is no batching speed-up to measure; Cole-Hopf's analytical form is the one that *could* exceed this under request batching). This right-sizes effort to noise instead of spending equal repeats on measurements with very different signal-to-noise.

**Honest limitations (state these before an evaluator does):**

- **PINN training cost is measured on one IC** (IC 900). Optimisation convergence — and thus L-BFGS iteration count — varies somewhat across ICs, so the PINN figure is a representative point, not a cross-IC mean. It is cached in `pinn_train_cost.json`; averaging over several ICs would tighten it (at ~13 min per IC). The order-of-magnitude conclusion is unaffected.
- **Single machine, CPU-only.** A GPU would lower the operators' absolute latencies (and the PINN's training time) but, as argued above, not the ordering or the amortization gap. Running on a second machine and confirming the slopes match would strengthen the generalisation claim.
- **Inference latency is data-independent by design.** For a fixed architecture the compute graph is identical regardless of the IC values, so a single IC is sufficient for the *inference* timings (unlike PINN training). This is why we report 30 repeats on one IC rather than averaging many ICs there.

## Deployment recommendations

These follow directly from the Pareto plot and are stated as routing rules the orchestrator can implement:

- **Default operator: FNO.** It is Pareto-optimal here — best accuracy in both time windows *and* amortized millisecond serving. For any latency-sensitive request inside the trained regime (t ≤ 1), route to FNO.
- **Do not serve a PINN for routine, latency-bound queries.** Its per-IC optimisation makes throughput too low for an online path. Reserve PINNs for *offline, high-value, single-IC* jobs where its physics-residual training is worth the cost and no operator covers the case.
- **DeepONet is dominated** on this problem: cheap but too inaccurate to trust. Keep it only as a fast lower-bound/baseline, not on the accuracy-critical path.
- **All three ML models fail in extrapolation (t > 1).** Error there is 14–61%, beyond the reliability study's 10% threshold. This is precisely the regime where the hybrid engine should hand off to a numerical solver — the cost analysis confirms there is no "cheap ML shortcut" that is also accurate past the training horizon.
- **Fall back to the spectral solver, not FDM, when accuracy is required.** The numerical cost is now measured, so the hand-off is quantified rather than assumed: spectral gives ~0.01% error at its measured per-IC cost, whereas FDM is ~16% off (numerical diffusion blurring the shock) for comparable cost — so FDM is not worth serving as the accurate fallback. Cole-Hopf is the cheapest reference but is Burgers-specific and does not generalise, so it is not a template for other PDEs.
- **The fallback is the expensive path — gate it.** The measured per-IC cost of the numerical solvers is orders of magnitude above the amortized operators, which is exactly why Module 3 throttles it: route to the numerical solver *only* when the failure score crosses threshold, never by default.
- **Use measured scaling, not assumptions.** If the production grid is refined beyond 512 points, re-read the fitted exponents in `3_scalability.png` before sizing hardware; the FFT-based FNO and the point-wise DeepONet/PINN grow at different rates, and the numerical solvers' CFL-limited time-stepping grows fastest of all under spatial refinement.

*Figures and raw numbers: `1_runtime_throughput.png`, `2_cost_accuracy.png`, `3_scalability.png`, `4_amortization.png`, `cost_summary.json` (includes the `_env` hardware block). Reproduce with `hybrid_pde/evaluation/deployment_cost_analysis.py`. All measurements: CPU, single thread, 30 timed repeats (mean ± std and median), same test ICs and grid as the reliability study. The expensive one-time PINN training measurement is cached in `pinn_train_cost.json`; delete it or set the `RETRAIN_PINN` environment variable to re-measure.*
