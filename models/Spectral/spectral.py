"""
================================================================================
HIGH-FIDELITY 1D VISCOUS BURGERS' EQUATION DATASET GENERATOR
================================================================================
Method: Pseudo-Spectral (Fourier) with Integrating Factor + RK4
Verification: Cole-Hopf analytical transform (exact for specific ICs)
Standard: Research-grade, suitable for ML/PINN/Neural Operator benchmarking

PDE:  ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
      x ∈ [0, 2π], periodic BCs, t ∈ [0, T]

Author: Generated for ML-PDE Solver Research Project
References:
  - Canuto et al., "Spectral Methods in Fluid Dynamics" (1988)
  - Cole (1951), Hopf (1950) - exact transformation
  - Trefethen, "Spectral Methods in MATLAB" (2000)
================================================================================
"""

import numpy as np
import torch
import os
import time
import warnings
warnings.filterwarnings("error")  # Turn warnings into errors for clean research code

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  (change only here — never scatter magic numbers)
# ─────────────────────────────────────────────────────────────
class Config:
    # Domain
    L       = 2.0 * np.pi      # Spatial period
    nx      = 1024              # Spatial resolution (power of 2 for FFT efficiency)
    T       = 2.0               # Total simulation time
    nt_out  = 200               # Number of output snapshots saved
    nt_rk4  = 20000             # Internal RK4 steps (for accuracy; CFL << 1 guaranteed)

    # Physics
    nu      = 1.0 / (100.0 * np.pi)   # Viscosity: ν = 0.01/π  (Re ~ 100π)
                                        # This is the standard benchmark value used in
                                        # Li et al. (FNO, 2021) and related literature.

    # Initial Condition
    ic_type = "sinusoidal"      # Options: "sinusoidal" | "multi_mode" | "random_fourier"
    ic_seed = 42                # RNG seed for reproducibility

    # Numerics
    dealias = True              # 2/3 dealiasing rule (Orszag 1971) to suppress aliasing errors

    # Output
    output_dir  = "models/Spectral/data"
    pt_filename = "burgers_spectral_dataset.pt"
    csv_filename = "burgers_spectral_dataset.csv"
    validate    = True          # Run PDE residual validation after generation


# ─────────────────────────────────────────────────────────────
# GRID SETUP
# ─────────────────────────────────────────────────────────────
def build_grid(cfg: Config):
    """Build spatial grid and wavenumber arrays."""
    x  = np.linspace(0, cfg.L, cfg.nx, endpoint=False)   # endpoint=False → periodic
    dx = cfg.L / cfg.nx                                    # uniform spacing
    # Wavenumbers for rfft (only non-negative, saves memory & is correct for real signals)
    k  = np.fft.rfftfreq(cfg.nx, d=1.0 / cfg.nx)          # integer wavenumbers: 0,1,...,nx//2
    return x, dx, k


# ─────────────────────────────────────────────────────────────
# INITIAL CONDITIONS
# ─────────────────────────────────────────────────────────────
def initial_condition(x: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Return the initial velocity field u(x, 0).

    sinusoidal:     u₀ = sin(x)   — canonical IC; Cole-Hopf exact solution exists
    multi_mode:     u₀ = sin(x) + 0.5·sin(2x) + 0.25·sin(3x)
    random_fourier: bandlimited random field (seed-reproducible)
    """
    rng = np.random.default_rng(cfg.ic_seed)

    if cfg.ic_type == "sinusoidal":
        return np.sin(x)

    elif cfg.ic_type == "multi_mode":
        return np.sin(x) + 0.5 * np.sin(2 * x) + 0.25 * np.sin(3 * x)

    elif cfg.ic_type == "random_fourier":
        # Bandlimited Gaussian random field: only modes 1–8 excited
        u0 = np.zeros_like(x)
        for m in range(1, 9):
            amp   = rng.standard_normal()
            phase = rng.uniform(0, 2 * np.pi)
            u0   += amp * np.sin(m * x + phase)
        return u0 / (np.max(np.abs(u0)) + 1e-12)   # normalise to [-1, 1]

    else:
        raise ValueError(f"Unknown ic_type: {cfg.ic_type!r}")


# ─────────────────────────────────────────────────────────────
# DEALIASING MASK (Orszag 2/3 Rule)
# ─────────────────────────────────────────────────────────────
def dealias_mask(k: np.ndarray, nx: int) -> np.ndarray:
    """
    Zero-out modes above k_max = nx//3 to prevent aliasing in the quadratic
    nonlinearity u·∂u/∂x.  This is the standard 2/3 rule (Orszag 1971).
    Returns a boolean mask of shape (len(k),).
    """
    k_max = nx // 3
    return np.abs(k) <= k_max


# ─────────────────────────────────────────────────────────────
# PSEUDO-SPECTRAL RHS
# ─────────────────────────────────────────────────────────────
def compute_rhs_hat(u_hat: np.ndarray, k: np.ndarray,
                    nu: float, mask: np.ndarray) -> np.ndarray:
    """
    Compute the RHS of the Burgers equation in Fourier space.

    PDE in spectral space (before integrating factor):
        d(û_k)/dt = -ik · F[u²/2]_k  −  ν k² û_k
                    ╰─────nonlinear─╯   ╰─diffusion─╯

    We use the conservative form  u·∂u/∂x = ∂(u²/2)/∂x  which is
    spectrally exact and conserves momentum.

    Parameters
    ----------
    u_hat : complex array, Fourier coefficients of u (rfft convention)
    k     : integer wavenumber array
    nu    : kinematic viscosity
    mask  : dealiasing mask (True = keep)
    """
    # Apply dealiasing before computing nonlinear term
    u_hat_d = u_hat * mask

    # Physical-space velocity (dealiased)
    u = np.fft.irfft(u_hat_d, n=len(k) * 2 - 2 + (1 if len(k) % 2 == 0 else 0))
    # Note: irfft with explicit n to match nx
    nx = u_hat_d.shape[0] * 2 - (0 if (u_hat_d.shape[0] * 2 - 1) % 2 == 0 else -1)

    # Conservative nonlinear term: ∂(u²/2)/∂x  ↔  ik · F[u²/2]
    u2_hat = np.fft.rfft(u * u)
    nonlinear_hat = 1j * k * u2_hat * 0.5     # ∂(u²/2)/∂x in spectral space

    # Linear diffusion term: −ν k² û
    diffusion_hat = -nu * k**2 * u_hat

    return -nonlinear_hat + diffusion_hat


# ─────────────────────────────────────────────────────────────
# INTEGRATING FACTOR + RK4 SOLVER
# ─────────────────────────────────────────────────────────────
def solve_burgers_spectral(cfg: Config) -> tuple:
    """
    Solve 1D viscous Burgers' equation using the pseudo-spectral method
    with integrating factor technique + classic RK4 time-stepping.

    The integrating factor  v_hat = û · exp(ν k² t)  removes the stiff
    linear diffusion term from the RK4 loop, allowing a much larger stable
    time step compared to solving the full PDE naïvely.

    Returns
    -------
    x       : (nx,) spatial grid
    t_save  : (nt_out,) output times
    u_all   : (nt_out, nx) velocity snapshots
    meta    : dict of solver metadata for provenance
    """
    x, dx, k = build_grid(cfg)
    nx = cfg.nx

    # Time arrays
    dt_internal = cfg.T / cfg.nt_rk4                  # internal RK4 step
    t_save      = np.linspace(0, cfg.T, cfg.nt_out)   # output checkpoints
    save_every  = cfg.nt_rk4 // (cfg.nt_out - 1)      # steps between saves
    # Recompute to ensure exact alignment
    save_steps  = np.round(np.linspace(0, cfg.nt_rk4, cfg.nt_out)).astype(int)

    # Dealiasing
    mask = dealias_mask(k, nx) if cfg.dealias else np.ones(len(k), dtype=bool)

    # CFL safety check (for advection: CFL = |u|_max · dt / dx ≪ 1)
    # With RK4 + spectral, stability is set by the advective CFL.
    # Rule of thumb: CFL < 0.5 is safe.
    u0       = initial_condition(x, cfg)
    cfl_init = np.max(np.abs(u0)) * dt_internal / dx
    if cfl_init > 0.8:
        raise RuntimeError(
            f"CFL number {cfl_init:.3f} > 0.8! Reduce dt or increase nx.\n"
            f"Suggested nt_rk4 >= {int(cfg.nt_rk4 * cfl_init / 0.4) + 1}"
        )
    print(f"  CFL number (initial): {cfl_init:.4f}  ✓")

    # Diffusion stability: ν k²_max dt ≪ 1
    diff_number = cfg.nu * (nx // 2)**2 * dt_internal
    print(f"  Diffusion number:     {diff_number:.4f}  ✓  (integrating factor removes stiffness)")

    # ── Integrating factor pre-computation ───────────────────
    # IF at a single step: exp(-ν k² dt)
    # This exactly integrates the linear part; nonlinear residual handled by RK4
    ek  = np.exp(-cfg.nu * k**2 * dt_internal)         # (len(k),) — real
    ek2 = np.exp(-cfg.nu * k**2 * dt_internal * 0.5)   # for RK4 half-step

    # ── Initial state in spectral space ──────────────────────
    u_hat = np.fft.rfft(u0)

    # ── Storage ──────────────────────────────────────────────
    u_all  = np.zeros((cfg.nt_out, nx), dtype=np.float64)
    u_all[0] = u0
    save_idx = 1   # next snapshot index to fill

    print(f"\n  Running RK4 integration: {cfg.nt_rk4} steps, dt = {dt_internal:.2e}")
    t_start = time.perf_counter()

    # ── RK4 loop with integrating factor ─────────────────────
    for step in range(1, cfg.nt_rk4 + 1):

        # Classic RK4 adapted for the integrating factor formulation
        # Reference: Cox & Matthews (2002) "Exponential Time Differencing for Stiff Systems"
        # Here we use the simpler integrating-factor RK4 (IFRK4) variant:
        #
        #   k1 = N(u_hat)
        #   k2 = N(ek2·u_hat + dt/2·ek2·k1)
        #   k3 = N(ek2·u_hat + dt/2·k2)          ← k3 uses ek2 too
        #   k4 = N(ek·u_hat  + dt·ek·k3)          ← full step
        #   u_hat_new = ek·u_hat + dt/6·(ek·k1 + 2·ek2·k2 + 2·ek2·k3 + k4)

        def N(uh):
            """Nonlinear part of RHS only (diffusion is in integrating factor)."""
            u_hat_d = uh * mask
            u_phys  = np.fft.irfft(u_hat_d, n=nx)
            u2_hat  = np.fft.rfft(u_phys * u_phys)
            return -0.5j * k * u2_hat   # -(ik/2)·F[u²]  →  -∂(u²/2)/∂x

        k1 = N(u_hat)
        k2 = N(ek2 * u_hat + 0.5 * dt_internal * ek2 * k1)
        k3 = N(ek2 * u_hat + 0.5 * dt_internal * k2)
        k4 = N(ek  * u_hat + dt_internal * ek * k3)

        u_hat = (ek * u_hat
                 + dt_internal / 6.0 * (ek * k1
                                        + 2.0 * ek2 * k2
                                        + 2.0 * ek2 * k3
                                        + k4))

        # Zero Nyquist mode for even nx (prevents aliasing at highest mode)
        u_hat[-1] = 0.0

        # Save snapshot if at a checkpoint
        if save_idx < cfg.nt_out and step == save_steps[save_idx]:
            u_all[save_idx] = np.fft.irfft(u_hat * mask, n=nx)
            save_idx += 1

    elapsed = time.perf_counter() - t_start
    print(f"  Integration complete in {elapsed:.2f}s")

    meta = {
        "method"        : "pseudo_spectral_ifrk4",
        "nx"            : cfg.nx,
        "nt_internal"   : cfg.nt_rk4,
        "nt_output"     : cfg.nt_out,
        "dt_internal"   : dt_internal,
        "nu"            : cfg.nu,
        "L"             : cfg.L,
        "T"             : cfg.T,
        "ic_type"       : cfg.ic_type,
        "ic_seed"       : cfg.ic_seed,
        "dealias"       : cfg.dealias,
        "cfl_initial"   : float(cfl_init),
        "elapsed_s"     : elapsed,
    }
    return x, t_save, u_all, meta


# ─────────────────────────────────────────────────────────────
# COLE-HOPF ANALYTICAL SOLUTION (for sinusoidal IC only)
# ─────────────────────────────────────────────────────────────
def cole_hopf_solution(x: np.ndarray, t: np.ndarray, nu: float,
                       n_terms: int = 200) -> np.ndarray:
    """
    Exact Cole-Hopf solution for u(x,0) = sin(x) on [0, 2π] periodic domain.

    The Cole-Hopf transform  φ = exp(-∫u dx / (2ν))  reduces Burgers to
    the heat equation:  ∂φ/∂t = ν ∂²φ/∂x²

    For u₀ = sin(x), the initial φ field is:
        φ₀(x) = exp( cos(x) / (2ν) )    [with integration constant chosen for 2π-periodicity]

    This is expanded in a Fourier series:
        φ₀(x) = Σ_n  a_n · cos(nx)
    where a_n = (1/π) · I_n(1/(2ν)) · exp(-1/(2ν))  [Modified Bessel functions]
    (The sin terms vanish by symmetry since φ₀ is even.)

    The heat equation propagates each mode independently:
        φ(x, t) = Σ_n  a_n · cos(nx) · exp(-ν n² t)

    Finally: u = -2ν · ∂φ/∂x / φ

    Parameters
    ----------
    n_terms : number of Fourier terms (converges rapidly; 100 is sufficient for ν > 0.01)
    """
    from scipy.special import iv as bessel_I   # Modified Bessel function of first kind

    alpha = 1.0 / (2.0 * nu)
    # Bessel-function Fourier coefficients
    # a_0 = I_0(alpha) · exp(-alpha)  [from the normalization of the cosine series]
    # a_n = 2 · I_n(alpha) · exp(-alpha)  for n >= 1
    coeffs = np.array([bessel_I(n, alpha) * np.exp(-alpha) for n in range(n_terms + 1)])

    u_exact = np.zeros((len(t), len(x)), dtype=np.float64)

    for ti, tt in enumerate(t):
        if tt == 0.0:
            u_exact[ti] = np.sin(x)
            continue

        # φ(x, t) = Σ_n  c_n · cos(nx) · exp(-ν n² t)
        # ∂φ/∂x   = -Σ_n n · c_n · sin(nx) · exp(-ν n² t)
        exp_factors = np.exp(-nu * np.arange(n_terms + 1)**2 * tt)

        phi     = coeffs[0] * exp_factors[0]
        dphi_dx = 0.0

        for n in range(1, n_terms + 1):
            cn           = 2.0 * coeffs[n]   # cosine series factor
            decay        = exp_factors[n]
            phi          = phi     + cn * decay * np.cos(n * x)
            dphi_dx      = dphi_dx - cn * decay * n * np.sin(n * x)

        # Guard against near-zero φ (shouldn't happen for ν > 0, but be safe)
        phi = np.where(np.abs(phi) < 1e-300, 1e-300 * np.sign(phi + 1e-300), phi)

        u_exact[ti] = -2.0 * nu * dphi_dx / phi

    return u_exact


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────
def validate_solution(x: np.ndarray, t: np.ndarray,
                      u_num: np.ndarray, cfg: Config) -> dict:
    """
    Validate the numerical solution via:

    1. PDE Residual  — spectral evaluation of ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x²
    2. Energy dissipation — ∫u² dx should be monotonically decreasing
    3. Cole-Hopf comparison (only for sinusoidal IC)
    4. Mass/momentum conservation — ∫u dx should be constant (for periodic BCs)
    """
    print("\n─── Validation ─────────────────────────────────────────────────")
    k  = np.fft.rfftfreq(cfg.nx, d=1.0 / cfg.nx)
    nx = cfg.nx
    results = {}

    # 1. PDE Residual (interior time steps only — avoid endpoint derivative issues)
    print("  [1/4] PDE residual check...")
    residuals = []
    for ti in range(1, len(t) - 1):
        u    = u_num[ti]
        dt   = t[ti+1] - t[ti-1]    # central difference in time

        u_hat    = np.fft.rfft(u)
        du_dx    = np.fft.irfft(1j * k * u_hat, n=nx)
        d2u_dx2  = np.fft.irfft(-k**2 * u_hat,  n=nx)
        du_dt    = (u_num[ti+1] - u_num[ti-1]) / dt   # 2nd-order central

        res = du_dt + u * du_dx - cfg.nu * d2u_dx2
        residuals.append(np.sqrt(np.mean(res**2)))

    mean_res = np.mean(residuals)
    max_res  = np.max(residuals)
    results["pde_residual_rms_mean"] = float(mean_res)
    results["pde_residual_rms_max"]  = float(max_res)
    print(f"     Mean RMS PDE residual: {mean_res:.3e}")
    print(f"     Max  RMS PDE residual: {max_res:.3e}  "
          f"{'✓ EXCELLENT' if max_res < 1e-3 else '⚠ CHECK RESOLUTION'}")

    # 2. Energy (L2 norm) should be non-increasing
    print("  [2/4] Energy dissipation check...")
    dx          = cfg.L / nx
    energy      = np.array([0.5 * dx * np.sum(u_num[ti]**2) for ti in range(len(t))])
    energy_mono = np.all(np.diff(energy) <= 1e-8)   # allow tiny numerical noise
    results["energy_monotone"]       = bool(energy_mono)
    results["energy_initial"]        = float(energy[0])
    results["energy_final"]          = float(energy[-1])
    results["energy_dissipated_pct"] = float(100 * (1 - energy[-1] / energy[0]))
    print(f"     Energy monotone decreasing: {energy_mono}  "
          f"{'✓' if energy_mono else '✗ VIOLATION!'}")
    print(f"     Energy dissipated: {results['energy_dissipated_pct']:.1f}%")

    # 3. Mass conservation: ∫u dx = const (for periodic BCs, mean is preserved)
    print("  [3/4] Mass conservation check...")
    mass = np.array([dx * np.sum(u_num[ti]) for ti in range(len(t))])
    mass_drift = np.max(np.abs(mass - mass[0]))
    results["mass_conservation_max_drift"] = float(mass_drift)
    print(f"     Max mass drift: {mass_drift:.3e}  "
          f"{'✓' if mass_drift < 1e-10 else '⚠ small drift is acceptable'}")

    # 4. Cole-Hopf comparison (only for sinusoidal IC)
    if cfg.ic_type == "sinusoidal":
        print("  [4/4] Cole-Hopf exact solution comparison...")
        u_exact = cole_hopf_solution(x, t, cfg.nu, n_terms=150)
        l2_errs = []
        for ti in range(len(t)):
            norm_exact = np.sqrt(np.mean(u_exact[ti]**2))
            if norm_exact < 1e-12:
                continue
            l2_err = np.sqrt(np.mean((u_num[ti] - u_exact[ti])**2)) / norm_exact
            l2_errs.append(l2_err)
        mean_l2 = float(np.mean(l2_errs))
        max_l2  = float(np.max(l2_errs))
        results["cole_hopf_l2_mean"] = mean_l2
        results["cole_hopf_l2_max"]  = max_l2
        print(f"     Mean relative L2 error vs Cole-Hopf: {mean_l2:.3e}")
        print(f"     Max  relative L2 error vs Cole-Hopf: {max_l2:.3e}  "
              f"{'✓ EXCELLENT' if max_l2 < 1e-4 else '⚠ consider finer grid'}")
    else:
        print("  [4/4] Cole-Hopf comparison skipped (not sinusoidal IC)")
        results["cole_hopf_l2_mean"] = None
        results["cole_hopf_l2_max"]  = None

    print("─────────────────────────────────────────────────────────────────\n")
    return results


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  HIGH-FIDELITY 1D VISCOUS BURGERS DATASET GENERATOR")
    print("  Method: Pseudo-Spectral + Integrating Factor + RK4")
    print("=" * 70)

    cfg = Config()
    print(f"\nConfiguration:")
    print(f"  nx = {cfg.nx}  |  nt_internal = {cfg.nt_rk4}  |  nt_out = {cfg.nt_out}")
    print(f"  ν  = {cfg.nu:.6f}  |  L = {cfg.L:.4f}  |  T = {cfg.T}")
    print(f"  IC = {cfg.ic_type}  |  dealiasing = {cfg.dealias}")

    # ── Solve ────────────────────────────────────────────────
    print("\n[1/4] Solving Burgers equation...")
    x, t, u, meta = solve_burgers_spectral(cfg)

    # ── Validate ─────────────────────────────────────────────
    val_results = {}
    if cfg.validate:
        print("[2/4] Validating solution...")
        val_results = validate_solution(x, t, u, cfg)

    # ── Statistics (for ML normalization reference) ───────────
    print("[3/4] Computing statistics...")
    u_mean = float(u.mean())
    u_std  = float(u.std())
    u_min  = float(u.min())
    u_max  = float(u.max())
    print(f"  u ∈ [{u_min:.4f}, {u_max:.4f}]  |  mean = {u_mean:.4f}  |  std = {u_std:.4f}")

    # ML-normalised version (z-score)  — stored alongside raw for flexibility
    u_norm = (u - u_mean) / (u_std + 1e-12)

    # ── Save ─────────────────────────────────────────────────
    print("[4/4] Saving dataset...")
    os.makedirs(cfg.output_dir, exist_ok=True)
    pt_path  = os.path.join(cfg.output_dir, cfg.pt_filename)
    csv_path = os.path.join(cfg.output_dir, cfg.csv_filename)

    torch.save({
        # ── Core data ──
        "u"             : torch.tensor(u,      dtype=torch.float32),
        "u_normalized"  : torch.tensor(u_norm, dtype=torch.float32),
        "x"             : torch.tensor(x,      dtype=torch.float32),
        "t"             : torch.tensor(t,      dtype=torch.float32),

        # ── Physics ──
        "nu"            : cfg.nu,
        "L"             : cfg.L,
        "T"             : cfg.T,

        # ── Grid ──
        "nx"            : cfg.nx,
        "nt"            : cfg.nt_out,
        "dx"            : cfg.L / cfg.nx,
        "dt_output"     : cfg.T / (cfg.nt_out - 1),

        # ── Normalisation ──
        "u_mean"        : u_mean,
        "u_std"         : u_std,
        "u_min"         : u_min,
        "u_max"         : u_max,

        # ── Provenance ──
        "solver_meta"   : meta,
        "validation"    : val_results,
        "description"   : (
            "High-fidelity 1D viscous Burgers dataset. "
            "Solved with pseudo-spectral IFRK4 method. "
            "Suitable for ML/PINN/Neural-Operator training and benchmarking."
        ),
    }, pt_path)
    print(f"  → Saved .pt  : {pt_path}")
    print(f"     Shape: u = {u.shape}")

    # CSV (t, x, u) for interoperability
    X_grid, T_grid = np.meshgrid(x, t)
    data_flat = np.column_stack((
        T_grid.ravel(),
        X_grid.ravel(),
        u.ravel()
    ))
    header = "t,x,u"
    np.savetxt(csv_path, data_flat, delimiter=",", header=header, comments="", fmt="%.8f")
    print(f"  → Saved .csv : {csv_path}")

    print("\n" + "=" * 70)
    print("  DATASET GENERATION COMPLETE")
    if val_results:
        ok = (val_results.get("pde_residual_rms_max", 1.0) < 1e-3
              and val_results.get("energy_monotone", False))
        print(f"  Overall quality: {'✓ RESEARCH GRADE' if ok else '⚠ REVIEW WARNINGS'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
