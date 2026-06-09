# Module 2 — Concepts + Line-by-Line Code Explanations

This is the "understand it from scratch" companion. Part A answers the concept questions
(exact solution, mental picture, Cole-Hopf derivation, trapezoidal rule). Part B walks the
actual code line by line so you can answer "what does this line do?" in the room.

---

# PART A — THE CONCEPTS

## A1. "The known way to get the exact answer" — what is it?

Most nonlinear PDEs have **no formula** for their solution; you can only approximate them
numerically. The viscous Burgers equation is special: it is one of the very few nonlinear
PDEs that **does** have an exact, closed-form solution — and that formula is obtained
*through the Cole-Hopf transform itself.*

So the "known way to get the answer" **is** the Cole-Hopf method. The chain is:

```
Burgers (nonlinear, no formula)
   --Cole-Hopf transform-->  Heat equation (linear, HAS a formula: the Gaussian kernel)
   --solve exactly-->        φ(x,t)
   --transform back-->       u(x,t)   <-- exact solution of Burgers
```

That is why Burgers is the "lab rat": it is hard enough to be interesting (it forms shocks),
yet we can still write down the true answer to check our ML models against. With most
equations you would have no trustworthy "truth" to compare to.

The only reason we say *semi*-analytical (not fully analytical) is that the Gaussian-kernel
formula contains an integral, and a computer evaluates that one integral numerically
(trapezoidal rule). Everything else is exact algebra.

## A2. The mental picture (shock formation), concretely

Start with `u(x,0) = sin(πx)` on `x ∈ [-1, 1)`. So at `t=0` you have one smooth sine hump:
positive on the left half, negative on the right half.

Now read the equation `u_t = -u·u_x + ν·u_xx` as "what makes `u` change in time":

- **`-u·u_x` (advection, the nonlinear part).** This term means **each point of the wave
  moves horizontally at a speed equal to its own height `u`.** Tall positive parts move
  *right fast*; negative parts move *left*. Because the speed depends on the height, the
  fast-moving crest **catches up to** the slower part ahead of it. The profile leans
  forward and **steepens**, like an ocean wave about to break. For `sin(πx)` the steepening
  concentrates near `x=0`, and the slope there becomes nearly vertical around
  `t ≈ 1/π ≈ 0.32`. That near-vertical region is the **shock**.

- **`+ν·u_xx` (diffusion, the viscous part).** `u_xx` is curvature; multiplying by the small
  `ν` adds a gentle **smoothing** that is strongest exactly where the curve bends most — i.e.
  at the forming cliff. It prevents the slope from becoming truly infinite. So instead of a
  perfectly vertical cliff (which would be a true discontinuity), you get a **steep but smooth
  ramp** whose width is roughly `√ν` — very thin because `ν` is tiny.

Picture summary: *a smooth sine hump leans forward → steepens into a thin near-vertical
ramp at x≈0 by t≈0.32 → viscosity keeps that ramp smooth rather than a true cliff → then
the whole thing slowly decays as energy dissipates.* Predicting that thin moving ramp
correctly far into the future (t up to 2) is what trips up the ML models.

Why this is the hard case for FNO specifically: a thin sharp ramp is built from **many
high-frequency** Fourier modes. FNO keeps only the lowest modes, so it tends to **blur the
ramp** — and that blurring shows up in the frequency spectrum, which is what your spectral
metric measures.

## A3. Derivation of the Cole-Hopf transform (whiteboard-ready)

**Goal:** show that `u = -2ν·φ_x/φ` turns Burgers into the heat equation `φ_t = ν·φ_xx`.

**Step 1 — introduce a potential.** Burgers is `u_t + u·u_x = ν·u_xx`. Note
`u·u_x = (½u²)_x`, so
```
u_t + (½u²)_x = ν·u_xx.
```
Define a potential `ψ` (psi) by `u = ψ_x` (i.e. `u` is the slope of `ψ`). Substitute and
integrate once in `x` (the integration "constant" can be absorbed into `ψ`):
```
ψ_t + ½(ψ_x)² = ν·ψ_xx.            (this is the "potential Burgers" equation)
```

**Step 2 — the Cole-Hopf substitution.** Let
```
ψ = -2ν·ln(φ)        ⇔        φ = exp(-ψ / 2ν).
```
Differentiate (chain rule):
```
ψ_t  = -2ν · φ_t/φ
ψ_x  = -2ν · φ_x/φ                 →  u = ψ_x = -2ν·φ_x/φ   ✓ (the transform you quote)
ψ_xx = -2ν · ( φ_xx/φ − (φ_x/φ)² )
```

**Step 3 — plug into the potential equation.** Substitute the three expressions into
`ψ_t + ½(ψ_x)² = ν·ψ_xx`:
```
-2ν·φ_t/φ  +  ½·(4ν²)(φ_x/φ)²   =   ν·( -2ν·φ_xx/φ + 2ν·(φ_x/φ)² )
-2ν·φ_t/φ  +  2ν²(φ_x/φ)²        =   -2ν²·φ_xx/φ + 2ν²(φ_x/φ)²
```
The `+2ν²(φ_x/φ)²` term appears on **both sides and cancels.** Left with:
```
-2ν·φ_t/φ = -2ν²·φ_xx/φ
```
Divide both sides by `-2ν/φ`:
```
φ_t = ν·φ_xx        ✓  the linear heat equation.
```
That cancellation of the nonlinear term is the whole magic of Cole-Hopf.

**Step 4 — transform the initial condition.** Since `u = ψ_x`, `ψ(x,0) = ∫_{x0}^{x} u0(s) ds`.
Then from `φ = exp(-ψ/2ν)`:
```
φ(x,0) = exp( -1/(2ν) · ∫_{x0}^{x} u0(s) ds ).
```

**Step 5 — solve the heat equation exactly (Gaussian kernel).** The heat equation's known
solution is convolution of the initial data with a spreading Gaussian:
```
φ(x,t) = 1/√(4πνt) · ∫ φ(y,0) · exp( -(x-y)² / (4νt) ) dy.
```

**Step 6 — transform back.** Differentiate `φ` in `x` (only the kernel depends on `x`):
```
φ_x(x,t) = 1/√(4πνt) · ∫ φ(y,0) · [ -(x-y)/(2νt) ] · exp(-(x-y)²/4νt) dy,
```
then
```
u(x,t) = -2ν · φ_x / φ.
```
That is the exact Burgers solution. The code computes `φ` and `φ_x` (Step 5 & 6) and divides.

## A4. Where the trapezoidal rule comes in (and why it's the *only* approximation)

Two integrals appear above and neither can be done symbolically for an arbitrary starting
shape, so the computer approximates each by the **trapezoidal rule**.

**The trapezoidal rule** approximates the area under a curve by chopping the x-axis into
strips and treating each strip as a trapezoid:
```
∫ f(x) dx  ≈  Σ_i  (x_{i+1} − x_i) · (f_i + f_{i+1}) / 2.
```
You only know `f` at grid points, so you connect consecutive points with straight lines and
sum the little trapezoids. More grid points (`nx=512`) → more accurate.

It is used in **two** places:

1. **The IC integral** `∫ u0 ds` in `compute_phi0` (done "by hand" as a cumulative trapezoid:
   `0.5*(u[:-1]+u[1:])*dx`, then `cumsum` to get a running total at every x).
2. **The Gaussian-kernel integral** `∫ φ0(y)·kernel dy` in `solve_heat_batch`, done with
   `scipy.integrate.trapezoid`.

So when you say "semi-analytical," **the trapezoidal rule is the 'semi' part** — the single
numerical approximation inside an otherwise exact method. That is the precise, defensible
answer to "where is it numerical?"

---

# PART B — LINE-BY-LINE CODE

## B1. `numerical_solvers/colehopf/colehopf.py` — the generator (your core)

### `Config` (lines 12–37) — the shared constants
```python
x_start=-1.0, x_end=1.0, L=2.0, nx=512   # domain x in [-1,1), 512 grid points, length L=2
T=2.0, nt_out=200, t_start=0.01          # time 0..2, 200 output times, first nonzero t=0.01
nu = 1.0/(100.0*np.pi)                    # viscosity ~0.00318 (small -> sharp shock)
N_samples=8, n_modes=4, ic_seed=42        # 8 trajectories; random ICs use 4 modes, seed 42
t_train_end=1.0                           # models train on t<=1, extrapolate to t=2
```
*Talking point:* this single class is the team's "single source of truth" — every other
solver and the ML models read these same constants so all data lives on the same grid.

### `build_grid` (lines 39–44)
```python
x  = np.linspace(cfg.x_start, cfg.x_end, cfg.nx, endpoint=False)  # periodic grid: excludes +1
dx = cfg.L / cfg.nx                                               # spacing between points
t  = np.concatenate([[0.0], np.linspace(cfg.t_start, cfg.T, cfg.nt_out-1)])  # t=0 then 0.01..2
```
- `endpoint=False` is **the periodicity choice**: `x=+1` is the same physical point as `x=-1`
  on a ring, so we don't store it twice.
- The time grid is `0.0` glued in front of `[0.01 ... 2.0]`. **This is the t=0 singularity
  fix** — we never ask the formula to evaluate at exactly `t=0` (which would divide by zero in
  `1/√(4νt)`); the `t=0` row is filled with the exact IC later.

### `ic_sinpi` (lines 46–47)
`return np.sin(np.pi * x)` — the canonical starting shape, sample 0.

### `ic_random_fourier` (lines 50–61) — diverse but reproducible ICs
```python
for m in range(1, n_modes+1):
    amp   = rng.standard_normal()        # random height for mode m (Gaussian)
    phase = rng.uniform(0, 2*pi)         # random shift for mode m
    u += amp * np.sin(2*pi*m*x/L + phase)
return u / (np.max(np.abs(u)) + 1e-12)   # normalise so peak amplitude = 1
```
- Builds a random smooth wave by adding `n_modes` sine waves with random heights/phases.
- `rng` is a **seeded** generator → same seed gives the same "random" ICs every run
  (reproducible). The `+1e-12` avoids divide-by-zero if `u` is all zeros.

### `make_ic` (lines 64–69)
Sample 0 → `sin(πx)`; samples 1–7 → a random Fourier IC. Gives one canonical case plus
seven varied-but-reproducible cases.

### `compute_phi0` (lines 72–77) — the transformed initial condition φ(x,0)
```python
dx       = x[1] - x[0]
trap     = 0.5 * (u_ic[:-1] + u_ic[1:]) * dx     # trapezoid areas between consecutive points
cumint   = np.concatenate([[0.0], np.cumsum(trap)])  # running integral ∫ u0 from x0 to x
exponent = -cumint / (2.0 * nu)                  # the exponent  -1/(2ν) ∫u0
return np.exp(exponent - np.max(exponent))       # exp, with the max-subtraction stability trick
```
- Lines 74–75 are **the trapezoidal rule for `∫u0`**, accumulated so we get the integral up to
  every grid point at once.
- Line 76 is `φ(x,0)`'s exponent from the derivation (Step 4).
- Line 77 is **the overflow fix**: `exp(big number)` overflows to `inf`. Subtracting the max
  exponent shifts everything so the largest value is `exp(0)=1`. This is allowed because the
  constant factor `exp(-max)` cancels later in `u = -2ν·φ_x/φ` (it multiplies both `φ_x` and
  `φ`, so the ratio is unchanged). *This line impresses examiners — know why it's valid.*

### `solve_heat_batch` (lines 80–102) — the Gaussian-kernel solve (the numerical heart)
```python
x_ext    = np.concatenate([x - L, x, x + L])     # three copies of the domain (left, centre, right)
phi0_ext = np.tile(phi0, 3)                       # φ0 copied to match the three images
```
- **This is the periodicity fix.** On a ring, the Gaussian smear at a point must collect
  contributions that wrap around the seam. Laying three domain copies side by side lets the
  kernel "see" neighbours across the wrap without special-casing the edges.

```python
diff  = x[:, None] - x_ext[None, :]   # matrix of (x - y) for every output x vs every source y
diff2 = diff ** 2                     # (x - y)^2, precomputed once (doesn't depend on t)
```
- `x[:,None] - x_ext[None,:]` uses broadcasting to build the full `(nx, 3nx)` table of
  pairwise distances in one shot (fast, vectorised).

```python
for ti, tt in enumerate(t_array):
    denom    = 4.0 * nu * tt              # the 4νt inside the Gaussian
    kernel   = np.exp(-diff2 / denom)     # exp(-(x-y)^2 / 4νt) — the heat kernel
    d_kernel = (-2.0 * diff / denom) * kernel  # ∂/∂x of the kernel = -(x-y)/(2νt) * kernel
    norm     = np.sqrt(np.pi * denom)     # the 1/√(4πνt) normaliser (here √(π·4νt))
    phi_all[ti]     = trapezoid(phi0_ext * kernel,   x_ext, axis=1) / norm   # φ(x,t)
    dphi_dx_all[ti] = trapezoid(phi0_ext * d_kernel, x_ext, axis=1) / norm   # φ_x(x,t)
```
- **The loop is over time, but each `tt` is computed independently from `phi0`** — there is no
  "previous step" feeding the next. *This is the "each output time computed directly → no
  time-stepping drift" property, in code.*
- `trapezoid(..., x_ext, axis=1)` is **the Gaussian-kernel integral by the trapezoidal rule**
  (Step 5/6 of the derivation).
- `kernel` → `φ`; `d_kernel` → `φ_x`. Two integrals, one for the value, one for its derivative.

### `solve_burgers` (lines 105–119) — assemble u
```python
phi0 = compute_phi0(x, u_ic, nu)
if np.any(phi0 == 0.0): raise ValueError(...)     # guard: if φ0 underflowed, stop loudly
U[0] = u_ic                                        # t=0 row = exact IC (avoids the singularity)
phi, dphi_dx = solve_heat_batch(x, t_array[1:], phi0, nu, L)   # solve for t>0
U[1:] = -2.0 * nu * dphi_dx / phi                  # inverse transform u = -2ν φ_x/φ
```
- Line 118 is **the inverse Cole-Hopf transform** turning `φ` back into the Burgers solution.

### `validate_solution` (lines 154–206) — the four quality checks
- **[1/4] PDE residual (lines 163–179):** uses FFT-based spatial derivatives
  (`np.fft.rfft` → multiply by `ik` for `u_x`, by `-k²` for `u_xx`) and a centred time
  difference, then checks `du_dt + u·u_x − ν·u_xx ≈ 0`. If the field solves the equation, the
  residual is ~0. *Why FFT derivatives? They're spectrally accurate on a periodic grid, so the
  check itself isn't introducing error.*
- **[2/4] Energy (lines 181–185):** `E = ½·dx·Σu²` must be non-increasing (`np.diff(energy)<=0`).
  Physically viscous Burgers can only lose energy. Also reports % dissipated.
- **[3/4] Mass (lines 187–190):** `mass = dx·Σu`; the drift from its initial value must be ~0
  (conservation on a periodic domain).
- **[4/4] Deterministic re-run (lines 192–196):** regenerate sample 0 from scratch and require
  `max|U − U_re| < 1e-12`. Proves reproducibility.

### `save_dataset` (lines 253–311) — the two output files
- Lines 265–289: `torch.save(...)` writes the **`.pt`** with raw `u`, a normalised copy, the
  ICs, grids, all metadata, and the validation dict.
- Lines 293–304: builds `(t, x, u)` rows and `np.savetxt(...)` writes the **`.csv`**.
*Talking point:* `.pt` for the ML code, `.csv` for human/other-tool inspection — same data, two
formats.

---

## B2. `ml_models/fno/fno_solver.py` — the FNO (your model)

### `FNOConfig` (lines 63–90) — architecture + training knobs
```python
n_modes_t=16, n_modes_x=16     # number of Fourier modes kept in time and space (truncation)
hidden_channels=32, n_layers=4 # 4 Fourier layers, 32-wide latent
in_channels=3, out_channels=1  # input: (IC, x-coord, t-coord); output: u
epochs=500, batch_size=4, lr=1e-3, weight_decay=1e-4, scheduler="cosine"
```
- `n_modes_*` = **mode truncation** (the FNO keeps only these lowest modes). This is the knob
  the mode-sweep varies.

### `_build_input_tensor` (lines 112–141) — the 3 input channels
```python
ic_ch = ic_batch[:,None,None,:].expand(B,1,nt,nx)   # channel 0: the IC tiled across all times
x_ch  = x_grid[None,None,None,:].expand(B,1,nt,nx)   # channel 1: the x coordinate
t_ch  = t_rel[None,None,:,None].expand(B,1,nt,nx)    # channel 2: the relative time coordinate
return torch.cat([ic_ch, x_ch, t_ch], dim=1)         # stack into (B, 3, nt, nx)
```
- The network is told, at every (t,x) cell: "here is the starting shape, here is where you are
  in space, here is when you are." From that it predicts `u(x,t)`.

### `fit` (lines 216–342) — training
- Lines 224–234: load `u, x, t`, sanity-check shapes.
- Lines 236–248: optionally restrict to `train_idx`; slice the **training-time block** `t<=t_train_end`.
- Lines 261–266: compute normalisation stats; **targets are normalised**, inputs kept physical.
- Lines 287, 303: loss is `LpLoss(d=2, p=2)` = **relative L2 over the 2-D (t,x) field**.
- Lines 292–322: standard training loop — forward, `loss.backward()`, `opt.step()`, cosine LR.
*One-liner:* "It learns the operator `IC → whole solution block` by minimising relative L2."

### `rollout` (lines 372–431) — long-time extrapolation
```python
T_blk = t_rel[-1]                              # length of one training block (=t_train_end)
n_blocks = ceil(t_max / T_blk)                 # how many blocks to cover t up to 2
for b in range(n_blocks):
    xin = _build_input_tensor(ic_t, x_native, t_rel)   # build input from current IC
    pred = self.model(xin)                              # predict one block
    u_blk = pred*std + mean                             # un-normalise back to physical u
    ...
    ic_t = u_blk[-1:, :]                                # LAST slice becomes next block's IC
```
- **This is the block-wise autoregressive rollout.** Line `ic_t = u_blk[-1:]` is the key:
  the model **eats its own last output** to keep predicting past where it was trained. It's
  also *why* errors compound across blocks.
- `_regrid` / `_bilinear_interp` (lines 540–608) resample onto any requested grid, **periodic
  in x** (the `_wrap_x` calls).

### `save` / `load` (lines 442–502)
Save the weights + config + grid + normalisation stats, and write a `manifest.json` so the
robustness runner can reload by directory. `load` rebuilds the model and restores everything.

---

## B3. The robustness framework

### `common/ood_spec.py` — the three OOD cases
```python
_ic_high_freq_sin(x) = sin(3πx)               # OOD-1: frequency shift
_ic_gaussian_bump(x) = exp(-x²/(2*0.2²))      # OOD-2: shape shift
_ic_sin_pi(x)        = sin(πx)  with nu'=1/(50π)  # OOD-3: parameter (viscosity) shift
```
- `cole_hopf_reference(case, x, t)` (lines 151–161) calls your **`solve_burgers`** with the
  case's `nu` to produce the exact ground truth for each OOD case. Only OOD-3 passes a
  different `nu`, so only its reference changes.
- `native_grid()` returns the **same grid** the FNO trained on, so comparisons are pointwise.

### `common/metrics.py` — relative L2
```python
def _per_time_rel_l2(pred, ref):
    return np.linalg.norm(pred-ref, axis=1) / (np.linalg.norm(ref, axis=1) + 1e-12)
```
- For each time row: size of the error divided by size of the truth. `axis=1` = across space,
  giving one number per time step. 0 = perfect, ~1 = totally wrong.

### `common/spectral_metrics.py` — the spectral distance + early warning
```python
def amplitude_spectrum(field):
    return np.abs(np.fft.rfft(field, axis=-1))     # |FFT| along space = "frequency fingerprint"

def mode_weights(n, mode):
    low_freq -> 1/(1+k),  uniform -> 1,  high_freq -> k   # per-mode weighting choices

def spectral_distance_per_t(pred, ref, weight_mode):
    sp_p = amplitude_spectrum(pred); sp_r = amplitude_spectrum(ref)
    w = mode_weights(sp_p.shape[1], weight_mode)
    return np.sum(w * np.abs(sp_p - sp_r), axis=1)  # D(t) = Σ_k w_k·| |U_pred,k| - |U_ref,k| |
```
- This is the **weighted Fourier-amplitude distance** (the WWF-inspired metric). It compares
  *how much of each frequency* the prediction has vs the truth, weighted by mode.

```python
def first_crossing_time(series, t, baseline_window=(0,0.25), factor=2.0):
    baseline = mean of series over t in [0,0.25]      # the "calm" reference level
    threshold = factor * baseline                     # 2x the calm level
    return first t where series > threshold           # when it departs from baseline

def early_warning_lead(rel_l2, spec_dist, t):
    return (time rel_l2 crosses) - (time spec_dist crosses)   # positive => spectral warned first
```
- **Early warning logic:** each signal gets a baseline over the trusted early window, then we
  time when it exceeds 2× that baseline. If the spectral signal crosses *before* the L2 error,
  the lead is positive — frequency drift warned us early.

### `module2_robustness/robustness_eval.py` — the runner
- `load_fno_solver` (74–84): the **only** FNO-specific part — loads the checkpoint, returns the
  model and its native grid.
- `evaluate_case` (90–122): for one OOD case → build IC, **`u_pred = rollout(ic,x,t)`**, build
  `u_ref = cole_hopf_reference(...)`, compute the rel-L2 series and the D(t) series, and the
  early-warning lead **under all three weightings** (so the choice is empirical).
- `run_robustness` (125–136): loop over the three cases.
- `write_csv / write_json / plot_results` (142–201): the table, the time-series, the 2×3 plot.
- Grid guard (235–240): refuse to run if the FNO grid ≠ the reference grid (a pointwise
  comparison would otherwise be meaningless).
- Model-agnostic: the core takes any `rollout_fn(ic, x, t) -> u(t,x)`, so PINN/DeepONet plug in
  by swapping only `load_fno_solver`.

---

# PART C — PRESENTATION SCRIPT CHECK (verified against the code)

Every Module-2 claim in your script is **accurate**. Specifics confirmed:

| Script line | Verdict | Code evidence |
|---|---|---|
| "turns the hard nonlinear equation into a simpler one solved exactly" | ✅ | Cole-Hopf → heat eq (`compute_phi0`+`solve_heat_batch`) |
| "computes each moment directly, never builds up drift" | ✅ | `solve_heat_batch` loops over t, each from `phi0` independently |
| "handles the sharp shock cleanly without ripples" | ✅ defensible | real-space kernel, no Gibbs ringing (spectral methods do ring) |
| "checked it four ways: physics, energy, mass, repeatability" | ✅ | `validate_solution` = PDE residual / energy / mass / re-run |
| "agree to within about three-millionths" | ✅ | cross-verification ≈ 3×10⁻⁶ |
| "FNO thinks in frequencies, keeps a handful, reassembles" | ✅ | `n_modes=(16,16)` truncation in neuralop FNO |
| "eight frequencies worked best; too many predicts worse" | ✅ | mode sweep 4/8/16 → 8 best, 16 extrapolates worse |
| "feeds its own output back in to predict beyond training" | ✅ | `rollout`: `ic_t = u_blk[-1:]` |
| "three unfamiliar inputs: wigglier wave / single bump / thicker fluid" | ✅ | `ood_spec.py`: sin(3πx) / Gaussian / ν'=1/(50π) |
| "measured: how far it drifts + whether frequency mix goes wrong" | ✅ | rel-L2 + spectral distance |
| "model breaks down on unfamiliar inputs" | ✅ | final rel-L2 ≈ 1.33 / 0.97 / 0.49 |
| "early-warning built and running; doesn't fire early yet" | ✅ honest | leads = −0.83 / nan / 0 |

**Numbers you can safely read out:** mode sweep = 4 / 8 / 16 (8 best); OOD final errors =
**1.33** (high-freq), **0.97** (bump), **0.49** (thicker fluid).

**One refinement on the early-warning "why".** Your script says the lead isn't positive because
it "baselines against the unfamiliar run's own start, where the model is already wrong." That is
a reasonable and defensible explanation. A slightly more bulletproof version (because the data
shows the *physical* error crosses very early, at t≈0.28 on OOD-1) is:

> "The lead isn't positive yet for two linked reasons: on these deliberately out-of-family
> inputs the physical error rises almost immediately, so there's little precursor window to
> warn within; and the baseline is taken on the OOD run's own early window, where the model is
> already somewhat off, which inflates the threshold the spectral signal must beat. Fixing the
> baseline — e.g. referencing an in-distribution calm level — is the immediate next step."

Either version is fine; the second pre-empts a follow-up about why L2 itself crosses so early.

**Small wording suggestions (optional):**
- Slide 16: when you say "eight worked best," add "from a short sweep, so it's a relative
  indicator" if anyone presses — keeps you honest about the 30-epoch runs.
- Slide 15: "without the ripples some methods produce" — if asked which methods, say "spectral
  methods can show Gibbs oscillations near a sharp shock; the real-space Cole-Hopf kernel
  doesn't." Only volunteer this if asked.
