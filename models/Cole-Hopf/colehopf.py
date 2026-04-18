# """
# ================================================================================
# 1D VISCOUS BURGERS' EQUATION DATASET GENERATOR
# ================================================================================
# Method: Cole-Hopf Transformation (near-exact analytical solution)
# Team:   Turingz — Dataset Generator (Cole-Hopf Method)

# PDE:  ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
#       x ∈ [0, 2π], periodic BCs, t ∈ [0, T]

# The Cole-Hopf transform  u = -2ν · φ_x / φ  reduces Burgers' to the
# heat equation  ∂φ/∂t = ν ∂²φ/∂x²  which is solved exactly via
# convolution with the Gaussian kernel — no time-stepping, no CFL limit.

# Aligned with teammates:
#   - Domain      : x ∈ [0, 2π]         (matches spectral solver)
#   - Viscosity   : ν = 1/(100π)         (matches spectral and FDM)
#   - IC (sample 0): sin(x)              (canonical validation case)
#   - Output      : .pt (PyTorch) + .csv (same format as spectral solver)
#   - Array shape : (N_samples, N_t, N_x)

# Quadrature note
# ---------------
# The Gaussian kernel has width √(4νt). When √(4νt) < dx the trapezoid
# rule is under-resolved. The minimum safe time is:

#     t_min ≈ dx² / ν = (2π/256)² / (1/100π) ≈ 0.19

# Therefore t[0] is set to exactly 0 (the IC, which is trivially exact)
# and the remaining nt_out-1 snapshots span [t_start, T] where
# t_start = 0.2 sits safely above the under-resolved zone.
# This matches what the spectral solver produces and avoids silent
# accuracy degradation near t = 0.

# Speed optimisation
# ------------------
# For each trajectory, x_ext, phi0_ext, and the squared-distance matrix
# diff² = (x[:,None] - x_ext[None,:])² are computed once.
# All nt_out-1 time steps reuse these arrays, eliminating 199 redundant
# allocations per sample.

# References:
#   - Cole (1951), Hopf (1950) — exact transformation
#   - Raissi et al. (2019)     — PINN baseline
#   - Li et al. (2021)         — FNO benchmark
# ================================================================================
# """

# import numpy as np
# import scipy.integrate as si
# import os
# import time
# import torch
# import matplotlib
# matplotlib.use("Agg")          # headless — works on any server
# import matplotlib.pyplot as plt


# # ─────────────────────────────────────────────────────────────
# # CONFIGURATION  — change values only here
# # ─────────────────────────────────────────────────────────────
# class Config:
#     # Domain — MUST match teammates
#     L           = 2.0 * np.pi          # spatial period  [0, 2π]
#     nx          = 256                   # spatial resolution
#     T           = 2.0                   # total simulation time
#     nt_out      = 200                   # number of output time snapshots
#                                         # (includes t=0; reliable data from t_start)

#     # Quadrature safety: first nt_out-1 interior snapshots start here.
#     # t_min ≈ dx²/ν ≈ 0.19 for nx=256, ν=1/(100π).
#     # t_start = 0.2 sits safely above this threshold.
#     t_start     = 0.2

#     # Physics — MUST match teammates
#     nu          = 1.0 / (100.0 * np.pi)   # ν = 1/(100π) ≈ 0.00318

#     # Dataset
#     # ── Dataset size guide ──────────────────────────────────────
#     # Each sample contributes nt_out × nx = 200 × 256 = 51,200 rows to the CSV.
#     #   N_samples =  4  →  204,800 rows  ≈ 200k
#     #   N_samples =  8  →  409,600 rows  ≈ 400k   ← DEFAULT (≈50s on laptop)
#     #   N_samples = 12  →  614,400 rows  ≈ 600k
#     # ────────────────────────────────────────────────────────────
#     N_samples   = 8                     # number of trajectories
#                                         # sample 0 is always sin(x) for validation
#     ic_type     = "random_fourier"      # IC for samples 1+
#                                         # "sinusoidal" | "multi_mode" | "random_fourier"
#     ic_seed     = 42
#     n_modes     = 4                     # Fourier modes for random_fourier IC

#     # Train / extrapolation split
#     t_train_end = 1.0

#     # Output
#     output_dir  = "./data"
#     pt_filename = "burgers_1d_cole_hopf.pt"
#     csv_filename = "burgers_1d_cole_hopf.csv"
#     plot_filename = "burgers_sample0_diagnostics.png"
#     validate    = True


# # ─────────────────────────────────────────────────────────────
# # GRID
# # ─────────────────────────────────────────────────────────────
# def build_grid(cfg: Config):
#     """
#     Uniform periodic grid on [0, L) and time array.

#     t[0]  = 0         (exact IC, no quadrature)
#     t[1:] = linspace(t_start, T, nt_out-1)   (kernel integral, safely resolved)
#     """
#     x  = np.linspace(0, cfg.L, cfg.nx, endpoint=False)   # endpoint=False → periodic
#     dx = cfg.L / cfg.nx
#     t  = np.concatenate(
#         [[0.0], np.linspace(cfg.t_start, cfg.T, cfg.nt_out - 1)]
#     )
#     return x, dx, t


# # ─────────────────────────────────────────────────────────────
# # INITIAL CONDITIONS
# # ─────────────────────────────────────────────────────────────
# def ic_sinusoidal(x):
#     """u(x,0) = sin(x) — canonical IC, matches spectral solver."""
#     return np.sin(x)


# def ic_multi_mode(x):
#     """u(x,0) = sin(x) + 0.5sin(2x) + 0.25sin(3x) — matches spectral solver."""
#     return np.sin(x) + 0.5 * np.sin(2 * x) + 0.25 * np.sin(3 * x)


# def ic_random_fourier(x, n_modes=4, rng=None):
#     """
#     Bandlimited random field — modes 1 to n_modes.
#     Normalized to max|u| = 1. Matches spectral solver convention.
#     """
#     if rng is None:
#         rng = np.random.default_rng()
#     u = np.zeros_like(x)
#     for m in range(1, n_modes + 1):
#         amp   = rng.standard_normal()
#         phase = rng.uniform(0, 2 * np.pi)
#         u    += amp * np.sin(m * x + phase)
#     return u / (np.max(np.abs(u)) + 1e-12)


# def get_ic(x, ic_type, rng=None, n_modes=4):
#     if ic_type == "sinusoidal":
#         return ic_sinusoidal(x)
#     elif ic_type == "multi_mode":
#         return ic_multi_mode(x)
#     elif ic_type == "random_fourier":
#         return ic_random_fourier(x, n_modes=n_modes, rng=rng)
#     else:
#         raise ValueError(f"Unknown ic_type: {ic_type!r}")


# # ─────────────────────────────────────────────────────────────
# # COLE-HOPF CORE
# # ─────────────────────────────────────────────────────────────
# def compute_phi0(x, u_ic, nu):
#     """
#     Compute the initial heat-equation potential φ(x,0) from u(x,0).

#     The Cole-Hopf transformation gives:
#         φ(x,0) = exp( -1/(2ν) · ∫₀ˣ u(s,0) ds )

#     Cumulative integral computed via trapezoidal rule.

#     Note: for u₀ = sin(x), the integral from 0 to x is 1-cos(x),
#     so φ₀ = exp(-(1-cos(x))/(2ν)). With ν ≈ 0.003 the exponent reaches
#     ~-314 at x=π, giving φ₀ ≈ 0 there. This extreme dynamic range is
#     handled safely by the Gaussian kernel convolution below.
#     """
#     dx = x[1] - x[0]
#     cumint = np.zeros_like(x)
#     for i in range(1, len(x)):
#         cumint[i] = cumint[i-1] + 0.5 * (u_ic[i-1] + u_ic[i]) * dx

#     exponent = -cumint / (2.0 * nu)

#     # Shift exponent so its maximum is exactly 0.
#     # This keeps phi safely away from float64 underflow at low viscosities.
#     # Mathematically safe: the constant factor exp(max_exp) cancels exactly
#     # in the Cole-Hopf ratio u = -2nu * phi_x / phi.
#     exponent_shifted = exponent - np.max(exponent)
#     return np.exp(exponent_shifted)


# def solve_heat_equation_batch(x, t_array, phi0, nu):
#     """
#     Solve the heat equation exactly for ALL time steps in one pass.

#     Optimisation vs the original single-t version
#     -----------------------------------------------
#     x_ext, phi0_ext, and the squared-distance matrix diff² are each
#     allocated ONCE and reused across all nt time steps.  The inner loop
#     only recomputes the Gaussian width (a scalar) and applies it to the
#     pre-built diff² array — eliminating 199 redundant matrix allocations
#     per sample.

#     Parameters
#     ----------
#     x       : (nx,)     spatial grid [0, 2π)
#     t_array : (nt,)     time points — must all be > 0 (t=0 is handled by
#                         the caller as the exact IC)
#     phi0    : (nx,)     initial heat potential
#     nu      : float     viscosity

#     Returns
#     -------
#     phi_all     : (nt, nx)
#     dphi_dx_all : (nt, nx)
#     """
#     L        = x[-1] + (x[1] - x[0])        # full period = 2π
#     x_ext    = np.concatenate([x - L, x, x + L])   # (3·nx,)
#     phi0_ext = np.tile(phi0, 3)                      # (3·nx,)

#     # Compute distance matrix ONCE — reused for every t
#     diff  = x[:, None] - x_ext[None, :]   # (nx, 3·nx)
#     diff2 = diff ** 2                      # (nx, 3·nx)  ← precomputed

#     nt  = len(t_array)
#     phi_all     = np.empty((nt, len(x)))
#     dphi_dx_all = np.empty((nt, len(x)))

#     for ti, tt in enumerate(t_array):
#         denom    = 4.0 * nu * tt                        # scalar
#         kernel   = np.exp(-diff2 / denom)               # (nx, 3·nx)
#         d_kernel = (-2.0 * diff / denom) * kernel       # (nx, 3·nx)

#         phi_t     = si.trapezoid(phi0_ext * kernel,   x_ext, axis=1)
#         dphi_dx_t = si.trapezoid(phi0_ext * d_kernel, x_ext, axis=1)

#         norm = np.sqrt(np.pi * denom)
#         phi_all[ti]     = phi_t     / norm
#         dphi_dx_all[ti] = dphi_dx_t / norm

#     return phi_all, dphi_dx_all


# def solve_burgers_cole_hopf(x, t_array, u_ic, nu):
#     """
#     Full Cole-Hopf pipeline for one trajectory.

#     u(x,t) = -2ν · φ_x(x,t) / φ(x,t)

#     t_array[0] must be 0; the IC is returned exactly for that slot.
#     All t > 0 slots are filled via the batched Gaussian-kernel solver.

#     Parameters
#     ----------
#     x       : (nx,)    spatial grid
#     t_array : (nt,)    time points — t_array[0] == 0 required
#     u_ic    : (nx,)    initial condition
#     nu      : float    viscosity

#     Returns
#     -------
#     U : (nt, nx)
#     """
#     phi0 = compute_phi0(x, u_ic, nu)
#     if np.any(phi0 == 0.0):
#         raise ValueError(
#             f"phi0 underflowed to 0.0 for nu={nu:.6f}. "
#             f"The IC cumulative integral range exceeds float64 limits. "
#             f"Try increasing nu or reducing IC amplitude."
#         )

#     U    = np.empty((len(t_array), len(x)))
#     U[0] = u_ic                          # t=0: exact, no quadrature needed

#     # Solve for all t > 0 in a single batched call
#     t_pos = t_array[1:]                  # (nt-1,) — all positive times
#     phi_all, dphi_dx_all = solve_heat_equation_batch(x, t_pos, phi0, nu)
#     U[1:] = -2.0 * nu * dphi_dx_all / phi_all

#     return U


# # ─────────────────────────────────────────────────────────────
# # DATASET GENERATION
# # ─────────────────────────────────────────────────────────────
# def generate_dataset(cfg: Config):
#     """
#     Generate N_samples trajectories.

#     Sample 0 is always sin(x) so all three teammates have an identical
#     reference trajectory for cross-validation.
#     """
#     x, dx, t = build_grid(cfg)
#     rng       = np.random.default_rng(cfg.ic_seed)

#     U   = np.zeros((cfg.N_samples, cfg.nt_out, cfg.nx), dtype=np.float64)
#     ICs = np.zeros((cfg.N_samples, cfg.nx),              dtype=np.float64)

#     print(f"\n  Generating {cfg.N_samples} trajectories...")
#     print(f"  Grid   : nx={cfg.nx}, nt={cfg.nt_out}, dx={dx:.5f}")
#     print(f"  Physics: ν={cfg.nu:.6f}, L={cfg.L:.4f}, T={cfg.T}")
#     print(f"  Time   : t[0]=0 (exact IC) | t[1]={cfg.t_start} | t[-1]={cfg.T}")
#     print(f"  Split  : train t∈[0,{cfg.t_train_end}] | "
#           f"extrap t∈[{cfg.t_train_end},{cfg.T}]")
#     print("  " + "─" * 55)

#     total_t0 = time.perf_counter()

#     for i in range(cfg.N_samples):
#         t0 = time.perf_counter()

#         # Sample 0 → sin(x) always (cross-validation anchor)
#         u_ic = ic_sinusoidal(x) if i == 0 \
#                else get_ic(x, cfg.ic_type, rng=rng, n_modes=cfg.n_modes)

#         ICs[i] = u_ic
#         U[i]   = solve_burgers_cole_hopf(x, t, u_ic, cfg.nu)

#         # Progress: print every 50 samples (plus first/last)
#         if i == 0 or (i + 1) % 50 == 0 or i == cfg.N_samples - 1:
#             print(f"  Sample {i+1:>4}/{cfg.N_samples} | "
#                   f"IC max={np.max(np.abs(u_ic)):.3f} | "
#                   f"u(T) max={np.max(np.abs(U[i,-1,:])):.4f} | "
#                   f"{time.perf_counter()-t0:.2f}s")

#     elapsed = time.perf_counter() - total_t0
#     print(f"\n  Total: {elapsed:.1f}s  ({elapsed/cfg.N_samples:.2f}s/sample)")
#     return {"U": U, "ICs": ICs, "x": x, "t": t}


# # ─────────────────────────────────────────────────────────────
# # VALIDATION — mirrors the spectral solver's four checks
# # ─────────────────────────────────────────────────────────────
# def validate_solution(dataset, cfg: Config):
#     """
#     Four validation checks (same as spectral solver for comparability):
#       1. PDE residual   — spectral derivatives
#       2. Energy dissipation — L2 norm must decrease
#       3. Mass conservation  — mean must stay constant (periodic BCs)
#       4. Self-consistency   — sample 0 re-run to confirm reproducibility

#     Note: residual check skips t[0]=0 and t[1] (first kernel-computed
#     point) because the centred finite difference needs two neighbours;
#     the check begins at index 2 to avoid the t_start boundary.
#     """
#     U = dataset["U"]
#     x = dataset["x"]
#     t = dataset["t"]
#     k = np.fft.rfftfreq(cfg.nx, d=1.0 / cfg.nx)
#     dx = cfg.L / cfg.nx

#     print("\n─── Validation ─────────────────────────────────────────────────")

#     # 1. PDE Residual (sample 0, interior time steps — skip first two)
#     print("  [1/4] PDE residual check  (sample 0)...")
#     residuals = []
#     u_s = U[0]
#     for ti in range(2, len(t) - 1):
#         u       = u_s[ti]
#         dt_c    = t[ti+1] - t[ti-1]
#         u_hat   = np.fft.rfft(u)
#         du_dx   = np.fft.irfft(1j * k * u_hat, n=cfg.nx)
#         d2u_dx2 = np.fft.irfft(-k**2 * u_hat,  n=cfg.nx)
#         du_dt   = (u_s[ti+1] - u_s[ti-1]) / dt_c
#         res     = du_dt + u * du_dx - cfg.nu * d2u_dx2
#         residuals.append(np.sqrt(np.mean(res**2)))

#     mean_res = float(np.mean(residuals))
#     max_res  = float(np.max(residuals))
#     print(f"     Mean RMS residual: {mean_res:.3e}")
#     print(f"     Max  RMS residual: {max_res:.3e}  "
#           f"{'✓ EXCELLENT' if max_res < 1e-3 else '⚠ check resolution'}")

#     # 2. Energy dissipation (sample 0, skip t=0 → t_start gap)
#     print("  [2/4] Energy dissipation check...")
#     energy      = 0.5 * dx * np.sum(U[0]**2, axis=1)
#     # Gap between t[0]=0 and t[1]=t_start means energy *must* drop there;
#     # check monotonicity only over the uniformly-spaced t[1:] portion.
#     energy_mono = bool(np.all(np.diff(energy[1:]) <= 1e-8))
#     dE          = float(100 * (1 - energy[-1] / energy[0]))
#     print(f"     Monotone decreasing (t≥{cfg.t_start}): "
#           f"{energy_mono}  {'✓' if energy_mono else '✗ VIOLATION'}")
#     print(f"     Energy dissipated  : {dE:.1f}%")

#     # 3. Mass conservation (sample 0)
#     print("  [3/4] Mass conservation check...")
#     mass       = dx * np.sum(U[0], axis=1)
#     mass_drift = float(np.max(np.abs(mass - mass[0])))
#     print(f"     Max mass drift: {mass_drift:.3e}  "
#           f"{'✓' if mass_drift < 1e-8 else '⚠ small drift acceptable'}")

#     # 4. Self-consistency — re-solve sample 0 and compare
#     print("  [4/4] Self-consistency check  (re-run sample 0)...")
#     x_c, _, t_c = build_grid(cfg)
#     U_rerun = solve_burgers_cole_hopf(x_c, t_c, ic_sinusoidal(x_c), cfg.nu)
#     max_diff = float(np.max(np.abs(U[0] - U_rerun)))
#     print(f"     Max diff vs re-run: {max_diff:.3e}  "
#           f"{'✓ DETERMINISTIC' if max_diff < 1e-12 else '⚠ check RNG'}")

#     print("─────────────────────────────────────────────────────────────────\n")

#     return {
#         "pde_residual_rms_mean" : mean_res,
#         "pde_residual_rms_max"  : max_res,
#         "energy_monotone"       : energy_mono,
#         "energy_dissipated_pct" : dE,
#         "mass_drift"            : mass_drift,
#         "self_consistency_diff" : max_diff,
#     }


# # ─────────────────────────────────────────────────────────────
# # PLOT — diagnostic figure for sample 0
# # ─────────────────────────────────────────────────────────────
# def plot_sample(dataset, val_results, cfg: Config):
#     """
#     Four-panel diagnostic figure for sample 0 (sin(x) IC).

#       Panel A  Space-time heatmap of u(x,t)
#       Panel B  Selected u(x,t) snapshots (t=0, 0.5, 1.0, 1.5, 2.0)
#       Panel C  Energy vs time (log scale) — should decay monotonically
#       Panel D  PDE RMS residual vs time  — should stay below ~1e-3

#     The train/extrap split is indicated by a vertical dashed line.
#     """
#     U  = dataset["U"][0]      # (nt, nx)
#     x  = dataset["x"]
#     t  = dataset["t"]
#     dx = cfg.L / cfg.nx
#     k  = np.fft.rfftfreq(cfg.nx, d=1.0 / cfg.nx)

#     # ── energy ────────────────────────────────────────────────
#     energy = 0.5 * dx * np.sum(U**2, axis=1)

#     # ── PDE residuals (t[2:-1] only — needs centred difference) ──
#     res_t   = []
#     res_rms = []
#     for ti in range(2, len(t) - 1):
#         u       = U[ti]
#         dt_c    = t[ti+1] - t[ti-1]
#         u_hat   = np.fft.rfft(u)
#         du_dx   = np.fft.irfft(1j * k * u_hat, n=cfg.nx)
#         d2u_dx2 = np.fft.irfft(-k**2 * u_hat,  n=cfg.nx)
#         du_dt   = (U[ti+1] - U[ti-1]) / dt_c
#         res     = du_dt + u * du_dx - cfg.nu * d2u_dx2
#         res_t.append(t[ti])
#         res_rms.append(np.sqrt(np.mean(res**2)))
#     res_t   = np.array(res_t)
#     res_rms = np.array(res_rms)

#     # ── figure ────────────────────────────────────────────────
#     fig, axes = plt.subplots(2, 2, figsize=(13, 9))
#     fig.suptitle("Burgers' Equation — Cole-Hopf  |  Sample 0: sin(x)",
#                  fontsize=14, fontweight="bold")

#     cmap = "RdBu_r"

#     # Panel A — heatmap
#     ax = axes[0, 0]
#     T_grid, X_grid = np.meshgrid(t, x, indexing="ij")
#     im = ax.pcolormesh(X_grid, T_grid, U, cmap=cmap, shading="auto",
#                        vmin=-1, vmax=1)
#     fig.colorbar(im, ax=ax, label="u(x,t)")
#     ax.axhline(cfg.t_train_end, color="k", ls="--", lw=1.2,
#                label=f"train end t={cfg.t_train_end}")
#     ax.axhline(cfg.t_start,     color="grey", ls=":", lw=1.0,
#                label=f"t_start={cfg.t_start}")
#     ax.set_xlabel("x"); ax.set_ylabel("t")
#     ax.set_title("A  Space-time heatmap")
#     ax.legend(fontsize=8, loc="upper right")

#     # Panel B — snapshots
#     ax = axes[0, 1]
#     snap_times = [0.0, 0.5, 1.0, 1.5, 2.0]
#     colors     = plt.cm.viridis(np.linspace(0, 1, len(snap_times)))
#     for tt, col in zip(snap_times, colors):
#         idx = int(np.argmin(np.abs(t - tt)))
#         ax.plot(x, U[idx], color=col, lw=1.6,
#                 label=f"t = {t[idx]:.2f}")
#     ax.axvline(np.pi, color="grey", ls=":", lw=0.8)
#     ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
#     ax.set_title("B  Snapshots")
#     ax.legend(fontsize=8)

#     # Panel C — energy
#     ax = axes[1, 0]
#     ax.semilogy(t, energy, lw=1.8, color="steelblue")
#     ax.axvline(cfg.t_train_end, color="k",    ls="--", lw=1.2, label="train end")
#     ax.axvline(cfg.t_start,     color="grey", ls=":",  lw=1.0, label=f"t_start={cfg.t_start}")
#     ax.set_xlabel("t"); ax.set_ylabel("E(t) = ½∫u² dx")
#     ax.set_title("C  Energy dissipation")
#     ax.legend(fontsize=8)

#     # Panel D — PDE residual
#     ax = axes[1, 1]
#     ax.semilogy(res_t, res_rms, lw=1.5, color="tomato")
#     ax.axhline(1e-3, color="green", ls="--", lw=1.0, label="1e-3 threshold")
#     ax.axvline(cfg.t_train_end, color="k", ls="--", lw=1.2, label="train end")
#     ax.set_xlabel("t"); ax.set_ylabel("RMS PDE residual")
#     ax.set_title("D  PDE residual over time")
#     ax.legend(fontsize=8)

#     # ── annotation box ────────────────────────────────────────
#     stats = (
#         f"ν = {cfg.nu:.5f}   nx = {cfg.nx}   nt = {cfg.nt_out}\n"
#         f"PDE res max = {val_results.get('pde_residual_rms_max', float('nan')):.2e}\n"
#         f"Energy dissipated = {val_results.get('energy_dissipated_pct', float('nan')):.1f}%\n"
#         f"N_samples = {cfg.N_samples}"
#     )
#     fig.text(0.5, 0.01, stats, ha="center", va="bottom", fontsize=9,
#              bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="grey"))

#     plt.tight_layout(rect=[0, 0.05, 1, 1])
#     os.makedirs(cfg.output_dir, exist_ok=True)
#     plot_path = os.path.join(cfg.output_dir, cfg.plot_filename)
#     fig.savefig(plot_path, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     print(f"  → Saved plot : {plot_path}")
#     return plot_path


# # ─────────────────────────────────────────────────────────────
# # SAVE — same format as spectral solver (.pt + .csv)
# # ─────────────────────────────────────────────────────────────
# def save_dataset(dataset, val_results, cfg: Config):
#     """
#     .pt  — PyTorch file, shape U: (N_samples, N_t, N_x)
#     .csv — flat (sample_id, t, x, u) rows for ALL samples

#     Total CSV rows = N_samples × nt_out × nx
#     e.g. 8 × 200 × 256 = 409,600  ≈ 400k data points

#     Identical key names to spectral solver output.
#     t[0] = 0 (exact IC); t[1:] start from t_start (kernel-computed).
#     """
#     U   = dataset["U"]
#     ICs = dataset["ICs"]
#     x   = dataset["x"]
#     t   = dataset["t"]

#     u_mean = float(U.mean())
#     u_std  = float(U.std())
#     u_min  = float(U.min())
#     u_max  = float(U.max())
#     u_norm = (U - u_mean) / (u_std + 1e-12)

#     os.makedirs(cfg.output_dir, exist_ok=True)
#     pt_path  = os.path.join(cfg.output_dir, cfg.pt_filename)
#     csv_path = os.path.join(cfg.output_dir, cfg.csv_filename)

#     torch.save({
#         # Core data
#         "u"             : torch.tensor(U,      dtype=torch.float32),
#         "u_normalized"  : torch.tensor(u_norm, dtype=torch.float32),
#         "ICs"           : torch.tensor(ICs,    dtype=torch.float32),
#         "x"             : torch.tensor(x,      dtype=torch.float32),
#         "t"             : torch.tensor(t,      dtype=torch.float32),
#         # Physics
#         "nu"            : cfg.nu,
#         "L"             : cfg.L,
#         "T"             : cfg.T,
#         "t_start"       : cfg.t_start,
#         "t_train_end"   : cfg.t_train_end,
#         # Grid
#         "nx"            : cfg.nx,
#         "nt"            : cfg.nt_out,
#         "N_samples"     : cfg.N_samples,
#         "dx"            : cfg.L / cfg.nx,
#         "dt_output"     : (cfg.T - cfg.t_start) / (cfg.nt_out - 2),
#         # Normalisation
#         "u_mean"        : u_mean,
#         "u_std"         : u_std,
#         "u_min"         : u_min,
#         "u_max"         : u_max,
#         # Provenance
#         "method"        : "cole_hopf_gaussian_kernel_batched",
#         "validation"    : val_results,
#         "description"   : (
#             "1D viscous Burgers dataset via Cole-Hopf transformation. "
#             "Heat equation solved exactly with Gaussian kernel convolution. "
#             "Periodic BCs handled by domain tiling. "
#             "t[0]=0 is the exact IC; reliable quadrature from t_start=0.2 onward. "
#             f"diff² precomputed once per trajectory; {cfg.N_samples} samples total. "
#             "Sample 0 = sin(x) for cross-validation with spectral solver."
#         ),
#     }, pt_path)

#     print(f"  → Saved .pt  : {pt_path}")
#     print(f"     Shape: U = {U.shape}  (N_samples, N_t, N_x)")
#     print(f"     t[0]={t[0]:.1f}  t[1]={t[1]:.3f}  t[-1]={t[-1]:.1f}")

#     # CSV — ALL samples, columns: sample_id, t, x, u
#     # Total rows = N_samples × nt_out × nx
#     X_grid, T_grid = np.meshgrid(x, t)          # each (nt, nx)
#     rows = []
#     for s in range(cfg.N_samples):
#         sid = np.full(X_grid.size, s, dtype=np.int32)
#         rows.append(np.column_stack((
#             sid,
#             T_grid.ravel(),
#             X_grid.ravel(),
#             U[s].ravel(),
#         )))
#     all_rows = np.vstack(rows)
#     total_rows = all_rows.shape[0]
#     np.savetxt(csv_path, all_rows, delimiter=",",
#                header="sample_id,t,x,u", comments="", fmt=["%d", "%.8f", "%.8f", "%.8f"])
#     print(f"  → Saved .csv : {csv_path}")
#     print(f"     Rows: {total_rows:,}  ({cfg.N_samples} samples × {cfg.nt_out} steps × {cfg.nx} points)")


# # ─────────────────────────────────────────────────────────────
# # BENCHMARK — quick speed comparison (optional, ~10 samples)
# # ─────────────────────────────────────────────────────────────
# def run_benchmark(cfg: Config, n_bench: int = 5):
#     """
#     Time the batched solver on n_bench random samples and extrapolate
#     to the full N_samples budget.  Prints a one-line summary.
#     """
#     x, _, t = build_grid(cfg)
#     rng = np.random.default_rng(0)
#     times = []
#     for _ in range(n_bench):
#         u_ic = ic_random_fourier(x, rng=rng)
#         t0   = time.perf_counter()
#         solve_burgers_cole_hopf(x, t, u_ic, cfg.nu)
#         times.append(time.perf_counter() - t0)
#     mean_t = float(np.mean(times))
#     print(f"\n  Benchmark ({n_bench} samples): "
#           f"{mean_t:.3f}s/sample  →  "
#           f"estimated total for {cfg.N_samples} samples: "
#           f"{mean_t * cfg.N_samples / 60:.1f} min")
#     return mean_t


# # ─────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────
# def main():
#     print("=" * 70)
#     print("  1D VISCOUS BURGERS DATASET GENERATOR — COLE-HOPF METHOD")
#     print("  Team Turingz")
#     print("=" * 70)

#     cfg = Config()
#     print(f"\nConfiguration:")
#     print(f"  nx={cfg.nx} | nt_out={cfg.nt_out} | N_samples={cfg.N_samples}")
#     print(f"  ν={cfg.nu:.6f} | L={cfg.L:.4f} | T={cfg.T}")
#     print(f"  t[0]=0 (exact IC) | t[1]={cfg.t_start} | t[-1]={cfg.T}")
#     print(f"  IC type (samples 1+): {cfg.ic_type} | seed: {cfg.ic_seed}")
#     print(f"  t_train_end: {cfg.t_train_end}")

#     # 0. Quick benchmark before the full run
#     print("\n[0/4] Benchmarking speed...")
#     run_benchmark(cfg, n_bench=5)

#     # 1. Generate
#     print("\n[1/4] Generating dataset...")
#     dataset = generate_dataset(cfg)

#     # 2. Validate
#     val_results = {}
#     if cfg.validate:
#         print("[2/4] Validating...")
#         val_results = validate_solution(dataset, cfg)

#     # 3. Plot
#     print("[3/4] Plotting diagnostics (sample 0)...")
#     plot_sample(dataset, val_results, cfg)

#     # 4. Save
#     print("[4/4] Saving...")
#     save_dataset(dataset, val_results, cfg)

#     # Summary
#     print("\n" + "=" * 70)
#     if val_results:
#         ok = (val_results.get("pde_residual_rms_max", 1.0) < 1e-3
#               and val_results.get("energy_monotone", False))
#         print(f"  Quality : {'✓ RESEARCH GRADE' if ok else '⚠ REVIEW WARNINGS ABOVE'}")
#     print(f"\n  Load with:")
#     print(f"    data = torch.load('{cfg.output_dir}/{cfg.pt_filename}')")
#     print(f"    U    = data['u']   # shape ({cfg.N_samples}, {cfg.nt_out}, {cfg.nx})")
#     print(f"    x    = data['x']   # shape ({cfg.nx},)")
#     print(f"    t    = data['t']   # shape ({cfg.nt_out},)")
#     print(f"    # t[0]=0 (IC), t[1]={cfg.t_start} (first kernel snapshot)")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()

"""
================================================================================
1D VISCOUS BURGERS' EQUATION DATASET GENERATOR
================================================================================
Method: Cole-Hopf Transformation (exact analytical solution)
Team:   Turingz

PDE:  ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
      x ∈ [-1, 1), periodic BCs, t ∈ [0, T]

CSV output format (matches your existing data):
    t, x, u
    all samples concatenated — no sample_id column

Dataset sizing:
    Each sample  = nt_out × nx rows
    3 samples    = 3 × 200 × 512  = 307,200 rows  (~300k)
    4 samples    = 4 × 200 × 512  = 409,600 rows  (~400k)  ← default
    8 samples    = 8 × 200 × 512  = 819,200 rows  (~800k)
================================================================================
"""

import numpy as np
import scipy.integrate as si
import os
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
class Config:
    # Spatial domain  x ∈ [-1, 1)  — matches your CSV format exactly
    x_start     = -1.0
    x_end       =  1.0
    L           =  2.0              # period = x_end - x_start
    nx          = 512               # dx = 2/512 = 0.00390625  (matches your data)

    # Time
    T           = 2.0               # total time
    nt_out      = 200               # output snapshots (includes t=0)

    # Quadrature safety:
    #   t_min ≈ dx²/ν = (2/512)² / (1/100π) ≈ 0.0048
    #   t_start = 0.01 sits safely above this.
    t_start     = 0.01

    # Physics
    nu          = 1.0 / (100.0 * np.pi)   # ν ≈ 0.00318 — same as teammates

    # Dataset size
    # ── Row counts ──────────────────────────────────────────
    #   N_samples = 3  →  307,200  (~300k)
    #   N_samples = 4  →  409,600  (~400k)  ← default
    #   N_samples = 8  →  819,200  (~800k)
    # ────────────────────────────────────────────────────────
    N_samples   = 4
    ic_type     = "random_fourier"  # for samples 1+
    ic_seed     = 42
    n_modes     = 4

    # Train / extrapolation split
    t_train_end = 1.0

    # Output
    output_dir   = "./data"
    pt_filename  = "burgers_1d_cole_hopf.pt"
    csv_filename = "burgers_1d_cole_hopf.csv"
    plot_filename = "burgers_diagnostics.png"
    validate     = True


# ─────────────────────────────────────────────────────────────
# GRID
# ─────────────────────────────────────────────────────────────
def build_grid(cfg: Config):
    """
    x ∈ [-1, 1)  with nx=512  →  dx = 0.00390625  (matches your CSV)
    t[0] = 0     (exact IC, no quadrature)
    t[1:] = linspace(t_start, T, nt_out-1)
    """
    x  = np.linspace(cfg.x_start, cfg.x_end, cfg.nx, endpoint=False)
    dx = cfg.L / cfg.nx
    t  = np.concatenate([[0.0],
                          np.linspace(cfg.t_start, cfg.T, cfg.nt_out - 1)])
    return x, dx, t


# ─────────────────────────────────────────────────────────────
# INITIAL CONDITIONS
# ─────────────────────────────────────────────────────────────
def ic_sinpi(x):
    """u(x,0) = sin(πx)  — matches your existing CSV data exactly."""
    return np.sin(np.pi * x)


def ic_random_fourier(x, L, n_modes=4, rng=None):
    """Bandlimited random IC; modes 1..n_modes; normalised to max|u|=1."""
    if rng is None:
        rng = np.random.default_rng()
    u = np.zeros_like(x)
    for m in range(1, n_modes + 1):
        amp   = rng.standard_normal()
        phase = rng.uniform(0, 2 * np.pi)
        u    += amp * np.sin(2 * np.pi * m * x / L + phase)
    return u / (np.max(np.abs(u)) + 1e-12)


def get_ic(x, cfg: Config, rng=None):
    if cfg.ic_type == "random_fourier":
        return ic_random_fourier(x, cfg.L, n_modes=cfg.n_modes, rng=rng)
    raise ValueError(f"Unknown ic_type: {cfg.ic_type!r}")


# ─────────────────────────────────────────────────────────────
# COLE-HOPF CORE
# ─────────────────────────────────────────────────────────────
def compute_phi0(x, u_ic, nu):
    """
    phi(x,0) = exp( -1/(2nu) * integral_{x[0]}^x u(s,0) ds )

    Exponent-shifted so max=0 to prevent float64 underflow.
    The constant shift cancels exactly in u = -2nu * phi_x / phi.
    """
    dx = x[1] - x[0]
    cumint = np.zeros_like(x)
    for i in range(1, len(x)):
        cumint[i] = cumint[i-1] + 0.5 * (u_ic[i-1] + u_ic[i]) * dx
    exponent = -cumint / (2.0 * nu)
    return np.exp(exponent - np.max(exponent))


def solve_heat_batch(x, t_array, phi0, nu, L):
    """
    Exact Gaussian-kernel solution to the heat equation for all t in t_array.

    phi(x,t)   = (1/sqrt(4*pi*nu*t)) * integral phi0(xi) * exp(-(x-xi)^2/(4*nu*t)) dxi
    dphi_dx(t) = derivative under the integral sign

    Periodic BCs: domain tiled three times [-L, 0), [0, L), [L, 2L).
    diff^2 precomputed once and reused for all time steps (main speed gain).

    Returns
    -------
    phi_all     : (len(t_array), nx)
    dphi_dx_all : (len(t_array), nx)
    """
    x_ext    = np.concatenate([x - L, x, x + L])
    phi0_ext = np.tile(phi0, 3)

    diff  = x[:, None] - x_ext[None, :]    # (nx, 3nx)  allocated once
    diff2 = diff ** 2                       # (nx, 3nx)  allocated once

    nt = len(t_array)
    phi_all     = np.empty((nt, len(x)))
    dphi_dx_all = np.empty((nt, len(x)))

    for ti, tt in enumerate(t_array):
        denom    = 4.0 * nu * tt
        kernel   = np.exp(-diff2 / denom)
        d_kernel = (-2.0 * diff / denom) * kernel
        norm     = np.sqrt(np.pi * denom)

        phi_all[ti]     = si.trapezoid(phi0_ext * kernel,   x_ext, axis=1) / norm
        dphi_dx_all[ti] = si.trapezoid(phi0_ext * d_kernel, x_ext, axis=1) / norm

    return phi_all, dphi_dx_all


def solve_burgers(x, t_array, u_ic, nu, L):
    """
    Full Cole-Hopf pipeline for one trajectory.
    u(x,t) = -2nu * phi_x / phi
    t_array[0] must be 0; that slot receives u_ic directly.
    """
    phi0 = compute_phi0(x, u_ic, nu)
    if np.any(phi0 == 0.0):
        raise ValueError(f"phi0 underflowed (nu={nu:.6f}). Increase nu or reduce IC amplitude.")

    U    = np.empty((len(t_array), len(x)))
    U[0] = u_ic
    phi_all, dphi_dx_all = solve_heat_batch(x, t_array[1:], phi0, nu, L)
    U[1:] = -2.0 * nu * dphi_dx_all / phi_all
    return U


# ─────────────────────────────────────────────────────────────
# DATASET GENERATION
# ─────────────────────────────────────────────────────────────
def generate_dataset(cfg: Config):
    x, dx, t = build_grid(cfg)
    rng = np.random.default_rng(cfg.ic_seed)

    total_rows = cfg.N_samples * cfg.nt_out * cfg.nx
    print(f"\n  Generating {cfg.N_samples} trajectories...")
    print(f"  Domain : x ∈ [{cfg.x_start}, {cfg.x_end})  "
          f"nx={cfg.nx}  dx={dx:.8f}")
    print(f"  Time   : t[0]=0.0  t[1]={cfg.t_start}  t[-1]={cfg.T}  "
          f"nt={cfg.nt_out}")
    print(f"  Physics: nu={cfg.nu:.6f}  L={cfg.L}")
    print(f"  Target : {total_rows:,} CSV rows  "
          f"({cfg.N_samples}×{cfg.nt_out}×{cfg.nx})")
    print("  " + "─" * 55)

    U   = np.zeros((cfg.N_samples, cfg.nt_out, cfg.nx), dtype=np.float64)
    ICs = np.zeros((cfg.N_samples, cfg.nx),              dtype=np.float64)

    total_t0 = time.perf_counter()
    for i in range(cfg.N_samples):
        t0 = time.perf_counter()
        # Sample 0 → sin(πx) always to match your existing CSV exactly
        u_ic   = ic_sinpi(x) if i == 0 else get_ic(x, cfg, rng=rng)
        ICs[i] = u_ic
        U[i]   = solve_burgers(x, t, u_ic, cfg.nu, cfg.L)
        print(f"  Sample {i+1}/{cfg.N_samples} | "
              f"IC max={np.max(np.abs(u_ic)):.3f} | "
              f"u(T) max={np.max(np.abs(U[i,-1,:])):.4f} | "
              f"{time.perf_counter()-t0:.1f}s")

    elapsed = time.perf_counter() - total_t0
    print(f"\n  Total: {elapsed:.1f}s  ({elapsed/cfg.N_samples:.1f}s/sample)")
    return {"U": U, "ICs": ICs, "x": x, "t": t}


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────
def validate_solution(dataset, cfg: Config):
    U  = dataset["U"]
    x  = dataset["x"]
    t  = dataset["t"]
    dx = cfg.L / cfg.nx

    print("\n─── Validation (sample 0) ───────────────────────────────────────")

    # 1. PDE residual (spectral derivatives; skip t[0] and t[1] boundary)
    kk = np.fft.rfftfreq(cfg.nx, d=cfg.L / cfg.nx) * 2 * np.pi
    res_vals = []
    u_s = U[0]
    for ti in range(2, len(t) - 1):
        u       = u_s[ti]
        dt_c    = t[ti+1] - t[ti-1]
        u_hat   = np.fft.rfft(u)
        du_dx   = np.fft.irfft(1j * kk * u_hat, n=cfg.nx)
        d2u_dx2 = np.fft.irfft(-kk**2 * u_hat,  n=cfg.nx)
        du_dt   = (u_s[ti+1] - u_s[ti-1]) / dt_c
        res     = du_dt + u * du_dx - cfg.nu * d2u_dx2
        res_vals.append(np.sqrt(np.mean(res**2)))

    mean_res = float(np.mean(res_vals))
    max_res  = float(np.max(res_vals))
    print(f"  [1/4] PDE residual — Mean: {mean_res:.3e}  Max: {max_res:.3e}  "
          f"{'✓' if max_res < 1e-3 else '⚠ check resolution'}")

    # 2. Energy dissipation
    energy      = 0.5 * dx * np.sum(U[0]**2, axis=1)
    energy_mono = bool(np.all(np.diff(energy[1:]) <= 1e-8))
    dE          = float(100 * (1 - energy[-1] / energy[0]))
    print(f"  [2/4] Energy — monotone: {energy_mono}  "
          f"dissipated: {dE:.1f}%  {'✓' if energy_mono else '✗'}")

    # 3. Mass conservation
    mass_drift = float(np.max(np.abs(dx * np.sum(U[0], axis=1) -
                                     dx * np.sum(U[0][0]))))
    print(f"  [3/4] Mass drift: {mass_drift:.3e}  "
          f"{'✓' if mass_drift < 1e-8 else '⚠'}")

    # 4. Self-consistency
    x_c, _, t_c = build_grid(cfg)
    U_re  = solve_burgers(x_c, t_c, ic_sinpi(x_c), cfg.nu, cfg.L)
    rdiff = float(np.max(np.abs(U[0] - U_re)))
    print(f"  [4/4] Re-run diff: {rdiff:.3e}  "
          f"{'✓ DETERMINISTIC' if rdiff < 1e-12 else '⚠'}")
    print("─────────────────────────────────────────────────────────────────\n")

    return dict(pde_residual_rms_mean=mean_res, pde_residual_rms_max=max_res,
                energy_monotone=energy_mono, energy_dissipated_pct=dE,
                mass_drift=mass_drift, self_consistency_diff=rdiff)


# ─────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────
def plot_sample(dataset, val_results, cfg: Config):
    U  = dataset["U"][0]
    x  = dataset["x"]
    t  = dataset["t"]
    dx = cfg.L / cfg.nx

    energy = 0.5 * dx * np.sum(U**2, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Burgers — Cole-Hopf  |  Sample 0: sin(πx)  |  x ∈ [-1,1)  "
        f"|  N={cfg.N_samples} samples  ({cfg.N_samples*cfg.nt_out*cfg.nx:,} rows)",
        fontsize=11, fontweight="bold")

    im = axes[0].pcolormesh(x, t, U, cmap="RdBu_r", shading="auto",
                             vmin=-1, vmax=1)
    fig.colorbar(im, ax=axes[0], label="u")
    axes[0].axhline(cfg.t_train_end, color="k", ls="--", lw=1, label="train end")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    axes[0].set_title("Space-time heatmap"); axes[0].legend(fontsize=8)

    for tt, col in zip([0, 0.5, 1.0, 1.5, 2.0],
                        plt.cm.viridis(np.linspace(0, 1, 5))):
        idx = int(np.argmin(np.abs(t - tt)))
        axes[1].plot(x, U[idx], color=col, lw=1.5, label=f"t={t[idx]:.2f}")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("u")
    axes[1].set_title("Snapshots"); axes[1].legend(fontsize=7)

    axes[2].semilogy(t, energy, lw=1.8, color="steelblue")
    axes[2].axvline(cfg.t_train_end, color="k", ls="--", lw=1)
    axes[2].set_xlabel("t"); axes[2].set_ylabel("E(t)")
    axes[2].set_title("Energy dissipation")

    plt.tight_layout()
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, cfg.plot_filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved plot : {path}")


# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
def save_dataset(dataset, val_results, cfg: Config):
    """
    .pt  — PyTorch tensor file (all samples)

    .csv — columns: t, x, u   (ALL samples concatenated)
           Format matches your existing CSV exactly:
             - header row:  t,x,u
             - 10 decimal places
             - no sample_id column
             - x ∈ [-1, 1)  starting at -1.0000000000
    """
    U   = dataset["U"]
    ICs = dataset["ICs"]
    x   = dataset["x"]
    t   = dataset["t"]

    u_mean = float(U.mean()); u_std = float(U.std())
    u_min  = float(U.min());  u_max = float(U.max())
    u_norm = (U - u_mean) / (u_std + 1e-12)

    os.makedirs(cfg.output_dir, exist_ok=True)
    pt_path  = os.path.join(cfg.output_dir, cfg.pt_filename)
    csv_path = os.path.join(cfg.output_dir, cfg.csv_filename)

    # ── PyTorch file ─────────────────────────────────────────
    torch.save({
        "u"           : torch.tensor(U,      dtype=torch.float32),
        "u_normalized": torch.tensor(u_norm, dtype=torch.float32),
        "ICs"         : torch.tensor(ICs,    dtype=torch.float32),
        "x"           : torch.tensor(x,      dtype=torch.float32),
        "t"           : torch.tensor(t,      dtype=torch.float32),
        "nu"          : cfg.nu,     "L"         : cfg.L,
        "x_start"     : cfg.x_start,"x_end"     : cfg.x_end,
        "T"           : cfg.T,      "t_start"   : cfg.t_start,
        "t_train_end" : cfg.t_train_end,
        "nx"          : cfg.nx,     "nt"        : cfg.nt_out,
        "N_samples"   : cfg.N_samples,
        "dx"          : cfg.L / cfg.nx,
        "u_mean"      : u_mean,     "u_std"     : u_std,
        "u_min"       : u_min,      "u_max"     : u_max,
        "method"      : "cole_hopf_gaussian_kernel",
        "validation"  : val_results,
    }, pt_path)
    print(f"  → Saved .pt  : {pt_path}")
    print(f"     Shape: U = {U.shape}  (N_samples, N_t, N_x)")

    # ── CSV — all samples, format: t, x, u ──────────────────
    # Build the (nt × nx) t and x grids once, then loop over samples.
    X_grid, T_grid = np.meshgrid(x, t)      # both (nt_out, nx)
    t_col = T_grid.ravel()                   # (nt_out * nx,)
    x_col = X_grid.ravel()                   # (nt_out * nx,)

    chunks = []
    for s in range(cfg.N_samples):
        chunks.append(np.column_stack((t_col, x_col, U[s].ravel())))
    all_rows   = np.vstack(chunks)
    total_rows = all_rows.shape[0]

    np.savetxt(csv_path, all_rows,
               delimiter=",",
               header="t,x,u",
               comments="",
               fmt="%.10f")     # 10 d.p. matches your sample data

    print(f"  → Saved .csv : {csv_path}")
    print(f"     Rows : {total_rows:,}  "
          f"({cfg.N_samples} samples × {cfg.nt_out} steps × {cfg.nx} points)")
    print(f"     First row preview:  "
          f"t={all_rows[0,0]:.1f}  x={all_rows[0,1]:.10f}  "
          f"u={all_rows[0,2]:.10f}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  BURGERS DATASET — COLE-HOPF  |  x ∈ [-1, 1)  |  Team Turingz")
    print("=" * 70)

    cfg  = Config()
    rows = cfg.N_samples * cfg.nt_out * cfg.nx
    print(f"\n  N_samples={cfg.N_samples}  nt={cfg.nt_out}  nx={cfg.nx}")
    print(f"  → {rows:,} CSV rows  (t, x, u)")

    dataset     = generate_dataset(cfg)
    val_results = validate_solution(dataset, cfg) if cfg.validate else {}
    plot_sample(dataset, val_results, cfg)
    save_dataset(dataset, val_results, cfg)

    ok = (val_results.get("pde_residual_rms_max", 1.0) < 1e-3
          and val_results.get("energy_monotone", False))
    print("\n" + "=" * 70)
    print(f"  Quality : {'✓ RESEARCH GRADE' if ok else '⚠  REVIEW WARNINGS'}")
    print(f"\n  CSV:   {cfg.output_dir}/{cfg.csv_filename}  ({rows:,} rows)")
    print(f"  .pt :  {cfg.output_dir}/{cfg.pt_filename}")
    print("=" * 70)


if __name__ == "__main__":
    main()