# Module 2 — Viva Defense Pack
**Dharmapala R.D. (214050V) · Coupling: the verified ML→numerical handoff**

## The headline (one figure, one sentence)

Figure: `results/module2/figures/restart_safety_boundary.png`

> **Approximate ML→numerical couplings that pass at benchmark viscosity are latently unsafe as gradients sharpen; a bit-exact restart of the production scheme is required for reliability under regime shift — proven (rel diff 0.0), measured (safety boundary Re_cell ≈ 3), and predicted by a reference-free diagnostic.**

## The three-layer contribution

1. **Mechanism** — the verified restartable re-anchor: hard one-way switch, continuous by construction (state jump = 0 to machine precision), restart proven bit-for-bit identical to the production pseudo-spectral solver (`verify_restart.py`, rel diff 0.0e+00).
2. **Law** — hybrid accuracy is bounded by handoff-state quality; the numerical continuation adds ~1e−6 (oracle decomposition). Holds across FNO/PINN/DeepONet (92%/86%/56% benefit) and across switch times (viability boundary t_s ≈ 1.47 ≈ FNO reliable horizon 1.457).
3. **Necessity** — restart fidelity is not cosmetic: a restart that drops only the 2/3 de-aliasing step is indistinguishable at operating viscosity yet fails the 1% target 0/10 and destabilises (energy ×49.6) as the shock sharpens, while the verified re-anchor meets the target 30/30.

## Attack → control (each answer is an experiment already run)

| Likely challenge | Answer | Evidence |
|---|---|---|
| "Why not a blended/overlap hand-over?" | The hard switch is already continuous (jump = 0, measured); the residual *drops* across the handoff (0.10→0.03), so there is no transient to remove. The oracle shows all hybrid error is inherited state error (~1e−6 from continuation); a blend re-weights the erroneous ML state and can only be equal or worse — tested, and it was. | §13.2 diagnostic; oracle control §13; decision log |
| "Isn't this just de-aliasing — Orszag 1971?" | Correct that the *mechanism* of failure is aliasing — and that is the point: the from-t0 control shows the failure is scheme fidelity, not the restart index. The contribution is not inventing de-aliasing; it is proving the restart preserves the exact production scheme and measuring when approximate fidelity becomes unsafe. | from-t0 control (careless also fails from t=0 at 1/(1600π)) |
| "Isn't the low-ν result just under-resolution?" | Measured, not assumed: grid-refinement control (N=1024/2048) shows the 512 grid converged to ≤0.5% through 1/(200π), 1.2% at 1/(400π), 3–8% spatial error at 800π–1600π. Under-resolution degrades *both* restarts smoothly and equally; only the non-de-aliased one **destabilises** on the same grid while the verified one stays at its scheme's 1e−12 floor. Instability is a fidelity artefact, not a resolution artefact — and the low-ν claims are explicitly scoped as fixed-grid restart-fidelity results. | `restart_safety_boundary.json` → `refinement` |
| "Does fidelity matter at operating conditions?" | No — and that is the danger. The careless restart passes every check at ν = 1/(100π) (identical tails, 10/10 targets). It is a latent defect: invisible at the benchmark, catastrophic beyond Re_cell ≈ 3–6. That is why bit-for-bit verification is a contribution and not plumbing. | stress test §15; conformance at operating ν |
| "Which safety step does the work?" | Split ablation: removing only Nyquist zeroing ≡ verified everywhere; removing only the 2/3 mask ≡ fully careless everywhere. The de-alias mask is decisive; Nyquist zeroing is hygiene. Mechanism: aliasing of the quadratic term feeds spurious energy into the resolved band as the spectrum fills. | ext JSON `multi_ic` |
| "One IC?" | Multi-IC sweep, n = 10 random Fourier ICs per viscosity, clean handoff states: verified 30/30 meet 1%; careless 10/10 → 1/10 → 0/10 (+1 blow-up). Blow-up is IC-dependent (sharpest shock worst); degradation is universal. | ext JSON `multi_ic` |
| "Descriptive, not predictive?" | The failure maps onto the cell Reynolds number at handoff (Re_cell = max\|u\|Δx/ν): the approximate restart crosses 1% at Re_cell ≈ 3.2 and 5% at ≈ 5.6 (threshold-sensitivity reported; the 1% crossing sits near the edge of grid convergence, the 5% crossing and instability are robustly beyond it). A **reference-free diagnostic** — the high-k energy fraction of the handoff state, computable from the ML state alone — predicts approximate-restart failure monotonically over four decades (unsafe beyond ≈ 3e−3). | `restart_safety_boundary.json`, both figure panels |
| "You made the baseline artificially dumb" | The careless baseline is the literature's default treatment — "set the solver's initial condition to the network output." At operating viscosity it is indistinguishable from the verified restart, i.e. it is precisely the version a careful-looking implementation could ship unnoticed. | §3.1, §15 |
| "What's new vs ANCHOR?" | Different question: ANCHOR triggers correction (reference-free residual, autoregressive operators, breadth). This module is the transition layer itself — verified equivalence, oracle attribution, viability boundary, restart-fidelity safety boundary and diagnostic, on one deeply-controlled PDE. Smaller, verified, deeply evaluated — stated as such. | §20 (report) |
| "Where's the cost claim?" | Measured: at matched ~1% accuracy the hybrid (t_s = 1.0) spends 50% of the numerical steps of a full re-solve; the sweep quantifies the whole accuracy–cost curve (0.50→1.0%, 0.41→2.3%, 0.31→7.2%, 0.21→14.9%, 0.11→23.6%). | §13.1 |

## Team-level link (say this at integration questions)

The safety boundary defines the **envelope in which the coupling is trustworthy**; Module 3's cost-aware controller operates inside it, and the reference-free high-k diagnostic is cheap enough for the runtime to evaluate at every candidate switch. M1 answers *when the ML fails*; my boundary answers *when the correction itself is safe*; M3 decides *what it may cost*. One system, three scoped questions.

Known open tuning item (theirs, with my numbers): the current trust trigger fires at t ≈ 0.2 (90% numerical work); the viability window says aim for t ≈ 1.3–1.47.

## Two honest sentences to volunteer before being asked

- "At the operating viscosity the restart details look optional — I reported that first, and it is exactly why the defect is dangerous."
- "At the lowest viscosity the 512-grid solution itself carries ~8% spatial error — so those claims are scoped to fixed-grid restart fidelity, and the instability contrast is the part no resolution argument can explain."
