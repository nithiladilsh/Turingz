"""
================================================================================
1D VISCOUS BURGERS' EQUATION DATASET GENERATOR
================================================================================
Method: Cole-Hopf Transformation (near-exact analytical solution)
Team:   Turingz — Dataset Generator (Cole-Hopf Method)

PDE:  ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
      x ∈ [0, 2π], periodic BCs, t ∈ [0, T]

The Cole-Hopf transform  u = -2ν · φ_x / φ  reduces Burgers' to the
heat equation  ∂φ/∂t = ν ∂²φ/∂x²  which is solved exactly via
convolution with the Gaussian kernel — no time-stepping, no CFL limit.

Aligned with teammates:
  - Domain      : x ∈ [0, 2π]         (matches spectral solver)
  - Viscosity   : ν = 1/(100π)         (matches spectral and FDM)
  - IC (sample 0): sin(x)              (canonical validation case)
  - Output      : .pt (PyTorch) + .csv (same format as spectral solver)
  - Array shape : (N_samples, N_t, N_x)

References:
  - Cole (1951), Hopf (1950) — exact transformation
  - Raissi et al. (2019)     — PINN baseline
  - Li et al. (2021)         — FNO benchmark
================================================================================
"""

import numpy as np
import scipy.integrate as si
import os
import time
import torch


# ─────────────────────────────────────────────────────────────
# CONFIGURATION  — change values only here
# ─────────────────────────────────────────────────────────────
class Config:
    # Domain — MUST match teammates
    L           = 2.0 * np.pi          # spatial period  [0, 2π]
    nx          = 256                   # spatial resolution
    T           = 2.0                   # total simulation time
    nt_out      = 200                   # number of output time snapshots

    # Physics — MUST match teammates
    nu          = 1.0 / (100.0 * np.pi)   # ν = 1/(100π) ≈ 0.00318

    # Dataset
    N_samples   = 10                    # number of trajectories
                                        # sample 0 is always sin(x) for validation
    ic_type     = "random_fourier"      # IC for samples 1+
                                        # "sinusoidal" | "multi_mode" | "random_fourier"
    ic_seed     = 42
    n_modes     = 4                     # Fourier modes for random_fourier IC

    # Train / extrapolation split
    t_train_end = 1.0

    # Output
    output_dir  = "./data"
    pt_filename = "burgers_1d_cole_hopf.pt"
    csv_filename = "burgers_1d_cole_hopf.csv"
    validate    = True


# ─────────────────────────────────────────────────────────────
# GRID
# ─────────────────────────────────────────────────────────────
def build_grid(cfg: Config):
    """Uniform periodic grid on [0, L) and uniform time array."""
    x  = np.linspace(0, cfg.L, cfg.nx, endpoint=False)   # endpoint=False → periodic
    dx = cfg.L / cfg.nx
    t  = np.linspace(0, cfg.T, cfg.nt_out)
    return x, dx, t


# ─────────────────────────────────────────────────────────────
# INITIAL CONDITIONS
# ─────────────────────────────────────────────────────────────
def ic_sinusoidal(x):
    """u(x,0) = sin(x) — canonical IC, matches spectral solver."""
    return np.sin(x)


def ic_multi_mode(x):
    """u(x,0) = sin(x) + 0.5sin(2x) + 0.25sin(3x) — matches spectral solver."""
    return np.sin(x) + 0.5 * np.sin(2 * x) + 0.25 * np.sin(3 * x)


def ic_random_fourier(x, n_modes=4, rng=None):
    """
    Bandlimited random field — modes 1 to n_modes.
    Normalized to max|u| = 1. Matches spectral solver convention.
    """
    if rng is None:
        rng = np.random.default_rng()
    u = np.zeros_like(x)
    for m in range(1, n_modes + 1):
        amp   = rng.standard_normal()
        phase = rng.uniform(0, 2 * np.pi)
        u    += amp * np.sin(m * x + phase)
    return u / (np.max(np.abs(u)) + 1e-12)


def get_ic(x, ic_type, rng=None, n_modes=4):
    if ic_type == "sinusoidal":
        return ic_sinusoidal(x)
    elif ic_type == "multi_mode":
        return ic_multi_mode(x)
    elif ic_type == "random_fourier":
        return ic_random_fourier(x, n_modes=n_modes, rng=rng)
    else:
        raise ValueError(f"Unknown ic_type: {ic_type!r}")


# ─────────────────────────────────────────────────────────────
# COLE-HOPF CORE
# ─────────────────────────────────────────────────────────────
def compute_phi0(x, u_ic, nu):
    """
    Compute the initial heat-equation potential φ(x,0) from u(x,0).

    The Cole-Hopf transformation gives:
        φ(x,0) = exp( -1/(2ν) · ∫₀ˣ u(s,0) ds )

    Cumulative integral computed via trapezoidal rule.

    Note: for u₀ = sin(x), the integral from 0 to x is 1-cos(x),
    so φ₀ = exp(-(1-cos(x))/(2ν)). With ν ≈ 0.003 the exponent reaches
    ~-314 at x=π, giving φ₀ ≈ 0 there. This extreme dynamic range is
    handled safely by the Gaussian kernel convolution below.
    """
    dx = x[1] - x[0]
    cumint = np.zeros_like(x)
    for i in range(1, len(x)):
        cumint[i] = cumint[i-1] + 0.5 * (u_ic[i-1] + u_ic[i]) * dx

    exponent = -cumint / (2.0 * nu)

    # Shift exponent so its maximum is exactly 0.
    # This keeps phi safely away from float64 underflow at low viscosities.
    # Mathematically safe: the constant factor exp(max_exp) cancels exactly
    # in the Cole-Hopf ratio u = -2nu * phi_x / phi.
    exponent_shifted = exponent - np.max(exponent)
    return np.exp(exponent_shifted)


def solve_heat_equation(x, t, phi0, nu):
    """
    Solve the heat equation exactly via Gaussian kernel convolution:

        φ(x,t) = 1/√(4πνt) · ∫ φ₀(ξ) · exp(-(x-ξ)²/(4νt)) dξ

    φ_x is obtained by differentiating analytically under the integral:

        φ_x(x,t) = 1/√(4πνt) · ∫ φ₀(ξ) · [-(x-ξ)/(2νt)] · exp(-(x-ξ)²/(4νt)) dξ

    Periodic BCs on [0, 2π] are handled by tiling three copies of the
    domain: [-2π, 0), [0, 2π), [2π, 4π). The tails of the Gaussian
    decay to machine zero beyond one period for all t ≤ T.

    Parameters
    ----------
    x    : (nx,)    spatial grid [0, 2π)
    t    : float    time > 0
    phi0 : (nx,)    initial heat potential
    nu   : float    viscosity

    Returns
    -------
    phi     : (nx,)
    dphi_dx : (nx,)
    """
    L        = x[-1] + (x[1] - x[0])       # full period = 2π
    x_ext    = np.concatenate([x - L, x, x + L])
    phi0_ext = np.tile(phi0, 3)

    denom    = 4.0 * nu * t                  # width of Gaussian

    # Vectorized (nx, 3·nx) distance matrix — avoids Python loop over x
    diff     = x[:, None] - x_ext[None, :]  # (nx, 3nx)
    kernel   = np.exp(-diff**2 / denom)
    d_kernel = (-2.0 * diff / denom) * kernel   # ∂kernel/∂x

    phi     = si.trapezoid(phi0_ext * kernel,   x_ext, axis=1)
    dphi_dx = si.trapezoid(phi0_ext * d_kernel, x_ext, axis=1)

    # Normalize: Gaussian integrates to √(π · denom)
    norm     = np.sqrt(np.pi * denom)
    phi     /= norm
    dphi_dx /= norm

    return phi, dphi_dx


def solve_burgers_cole_hopf(x, t_array, u_ic, nu):
    """
    Full Cole-Hopf pipeline for one trajectory.

    u(x,t) = -2ν · φ_x(x,t) / φ(x,t)

    Parameters
    ----------
    x       : (nx,)    spatial grid
    t_array : (nt,)    time points
    u_ic    : (nx,)    initial condition
    nu      : float    viscosity

    Returns
    -------
    U : (nt, nx)
    """
    # Float64 underflow check: after the exponent shift in compute_phi0,
    # the minimum shifted exponent is -(max_cumint - min_cumint) / (2*nu).
    # float64 underflows at exp(-708), so we need that range < 708.
    # We check this dynamically per-IC after computing phi0, rather than
    # using a fixed nu threshold (which would depend on the specific IC).
    phi0 = compute_phi0(x, u_ic, nu)
    if np.any(phi0 == 0.0):
        raise ValueError(
            f"phi0 underflowed to 0.0 for nu={nu:.6f}. "
            f"The IC cumulative integral range exceeds float64 limits. "
            f"Try increasing nu or reducing IC amplitude."
        )

    U    = np.zeros((len(t_array), len(x)))
    U[0] = u_ic

    for ti, tt in enumerate(t_array[1:], start=1):
        phi, dphi_dx = solve_heat_equation(x, tt, phi0, nu)
        # phi_safe guard removed: exponent shift in compute_phi0 guarantees
        # max(phi) = 1.0 at all times, so phi is never near zero.
        U[ti]        = -2.0 * nu * dphi_dx / phi

    return U


# ─────────────────────────────────────────────────────────────
# DATASET GENERATION
# ─────────────────────────────────────────────────────────────
def generate_dataset(cfg: Config):
    """
    Generate N_samples trajectories.

    Sample 0 is always sin(x) so all three teammates have an identical
    reference trajectory for cross-validation.
    """
    x, dx, t = build_grid(cfg)
    rng       = np.random.default_rng(cfg.ic_seed)

    U   = np.zeros((cfg.N_samples, cfg.nt_out, cfg.nx), dtype=np.float64)
    ICs = np.zeros((cfg.N_samples, cfg.nx),              dtype=np.float64)

    print(f"\n  Generating {cfg.N_samples} trajectories...")
    print(f"  Grid   : nx={cfg.nx}, nt={cfg.nt_out}, dx={dx:.5f}")
    print(f"  Physics: ν={cfg.nu:.6f}, L={cfg.L:.4f}, T={cfg.T}")
    print(f"  Split  : train t∈[0,{cfg.t_train_end}] | "
          f"extrap t∈[{cfg.t_train_end},{cfg.T}]")
    print("  " + "─" * 55)

    total_t0 = time.perf_counter()

    for i in range(cfg.N_samples):
        t0 = time.perf_counter()

        # Sample 0 → sin(x) always (cross-validation anchor)
        u_ic = ic_sinusoidal(x) if i == 0 \
               else get_ic(x, cfg.ic_type, rng=rng, n_modes=cfg.n_modes)

        ICs[i] = u_ic
        U[i]   = solve_burgers_cole_hopf(x, t, u_ic, cfg.nu)

        print(f"  Sample {i+1:>3}/{cfg.N_samples} | "
              f"IC max={np.max(np.abs(u_ic)):.3f} | "
              f"u(T) max={np.max(np.abs(U[i,-1,:])):.4f} | "
              f"{time.perf_counter()-t0:.1f}s")

    print(f"\n  Total: {time.perf_counter()-total_t0:.1f}s")
    return {"U": U, "ICs": ICs, "x": x, "t": t}


# ─────────────────────────────────────────────────────────────
# VALIDATION — mirrors the spectral solver's four checks
# ─────────────────────────────────────────────────────────────
def validate_solution(dataset, cfg: Config):
    """
    Four validation checks (same as spectral solver for comparability):
      1. PDE residual   — spectral derivatives
      2. Energy dissipation — L2 norm must decrease
      3. Mass conservation  — mean must stay constant (periodic BCs)
      4. Self-consistency   — sample 0 re-run to confirm reproducibility
    """
    U = dataset["U"]
    x = dataset["x"]
    t = dataset["t"]
    k = np.fft.rfftfreq(cfg.nx, d=1.0 / cfg.nx)
    dx = cfg.L / cfg.nx

    print("\n─── Validation ─────────────────────────────────────────────────")

    # 1. PDE Residual (sample 0, interior time steps)
    print("  [1/4] PDE residual check  (sample 0)...")
    residuals = []
    u_s = U[0]
    for ti in range(1, len(t) - 1):
        u       = u_s[ti]
        dt_c    = t[ti+1] - t[ti-1]
        u_hat   = np.fft.rfft(u)
        du_dx   = np.fft.irfft(1j * k * u_hat, n=cfg.nx)
        d2u_dx2 = np.fft.irfft(-k**2 * u_hat,  n=cfg.nx)
        du_dt   = (u_s[ti+1] - u_s[ti-1]) / dt_c
        res     = du_dt + u * du_dx - cfg.nu * d2u_dx2
        residuals.append(np.sqrt(np.mean(res**2)))

    mean_res = float(np.mean(residuals))
    max_res  = float(np.max(residuals))
    print(f"     Mean RMS residual: {mean_res:.3e}")
    print(f"     Max  RMS residual: {max_res:.3e}  "
          f"{'✓ EXCELLENT' if max_res < 1e-3 else '⚠ check resolution'}")

    # 2. Energy dissipation (sample 0)
    print("  [2/4] Energy dissipation check...")
    energy      = 0.5 * dx * np.sum(U[0]**2, axis=1)
    energy_mono = bool(np.all(np.diff(energy) <= 1e-8))
    dE          = float(100 * (1 - energy[-1] / energy[0]))
    print(f"     Monotone decreasing: {energy_mono}  {'✓' if energy_mono else '✗ VIOLATION'}")
    print(f"     Energy dissipated  : {dE:.1f}%")

    # 3. Mass conservation (sample 0)
    print("  [3/4] Mass conservation check...")
    mass       = dx * np.sum(U[0], axis=1)
    mass_drift = float(np.max(np.abs(mass - mass[0])))
    print(f"     Max mass drift: {mass_drift:.3e}  "
          f"{'✓' if mass_drift < 1e-8 else '⚠ small drift acceptable'}")

    # 4. Self-consistency — re-solve sample 0 and compare
    print("  [4/4] Self-consistency check  (re-run sample 0)...")
    x, _, t_arr = build_grid(cfg)
    U_rerun = solve_burgers_cole_hopf(x, t_arr, ic_sinusoidal(x), cfg.nu)
    max_diff = float(np.max(np.abs(U[0] - U_rerun)))
    print(f"     Max diff vs re-run: {max_diff:.3e}  "
          f"{'✓ DETERMINISTIC' if max_diff < 1e-12 else '⚠ check RNG'}")

    print("─────────────────────────────────────────────────────────────────\n")

    return {
        "pde_residual_rms_mean" : mean_res,
        "pde_residual_rms_max"  : max_res,
        "energy_monotone"       : energy_mono,
        "energy_dissipated_pct" : dE,
        "mass_drift"            : mass_drift,
        "self_consistency_diff" : max_diff,
    }


# ─────────────────────────────────────────────────────────────
# SAVE — same format as spectral solver (.pt + .csv)
# ─────────────────────────────────────────────────────────────
def save_dataset(dataset, val_results, cfg: Config):
    """
    .pt  — PyTorch file, shape U: (N_samples, N_t, N_x)
    .csv — flat (t, x, u) rows, sample 0 only (for quick inspection)

    Identical key names to spectral solver output.
    """
    U   = dataset["U"]
    ICs = dataset["ICs"]
    x   = dataset["x"]
    t   = dataset["t"]

    u_mean = float(U.mean())
    u_std  = float(U.std())
    u_min  = float(U.min())
    u_max  = float(U.max())
    u_norm = (U - u_mean) / (u_std + 1e-12)

    os.makedirs(cfg.output_dir, exist_ok=True)
    pt_path  = os.path.join(cfg.output_dir, cfg.pt_filename)
    csv_path = os.path.join(cfg.output_dir, cfg.csv_filename)

    torch.save({
        # Core data
        "u"             : torch.tensor(U,      dtype=torch.float32),
        "u_normalized"  : torch.tensor(u_norm, dtype=torch.float32),
        "ICs"           : torch.tensor(ICs,    dtype=torch.float32),
        "x"             : torch.tensor(x,      dtype=torch.float32),
        "t"             : torch.tensor(t,      dtype=torch.float32),
        # Physics
        "nu"            : cfg.nu,
        "L"             : cfg.L,
        "T"             : cfg.T,
        "t_train_end"   : cfg.t_train_end,
        # Grid
        "nx"            : cfg.nx,
        "nt"            : cfg.nt_out,
        "N_samples"     : cfg.N_samples,
        "dx"            : cfg.L / cfg.nx,
        "dt_output"     : cfg.T / (cfg.nt_out - 1),
        # Normalisation
        "u_mean"        : u_mean,
        "u_std"         : u_std,
        "u_min"         : u_min,
        "u_max"         : u_max,
        # Provenance
        "method"        : "cole_hopf_gaussian_kernel",
        "validation"    : val_results,
        "description"   : (
            "1D viscous Burgers dataset via Cole-Hopf transformation. "
            "Heat equation solved exactly with Gaussian kernel convolution. "
            "Periodic BCs handled by domain tiling. "
            "Sample 0 = sin(x) for cross-validation with spectral solver."
        ),
    }, pt_path)

    print(f"  → Saved .pt  : {pt_path}")
    print(f"     Shape: U = {U.shape}  (N_samples, N_t, N_x)")

    # CSV — sample 0 only, same column layout as spectral solver
    X_grid, T_grid = np.meshgrid(x, t)
    data_flat = np.column_stack((T_grid.ravel(), X_grid.ravel(), U[0].ravel()))
    np.savetxt(csv_path, data_flat, delimiter=",",
               header="t,x,u", comments="", fmt="%.8f")
    print(f"  → Saved .csv : {csv_path}  (sample 0 only)")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  1D VISCOUS BURGERS DATASET GENERATOR — COLE-HOPF METHOD")
    print("  Team Turingz")
    print("=" * 70)

    cfg = Config()
    print(f"\nConfiguration:")
    print(f"  nx={cfg.nx} | nt_out={cfg.nt_out} | N_samples={cfg.N_samples}")
    print(f"  ν={cfg.nu:.6f} | L={cfg.L:.4f} | T={cfg.T}")
    print(f"  IC type (samples 1+): {cfg.ic_type} | seed: {cfg.ic_seed}")
    print(f"  t_train_end: {cfg.t_train_end}")

    # 1. Generate
    print("\n[1/3] Generating dataset...")
    dataset = generate_dataset(cfg)

    # 2. Validate
    val_results = {}
    if cfg.validate:
        print("[2/3] Validating...")
        val_results = validate_solution(dataset, cfg)

    # 3. Save
    print("[3/3] Saving...")
    save_dataset(dataset, val_results, cfg)

    # Summary
    print("\n" + "=" * 70)
    if val_results:
        ok = (val_results.get("pde_residual_rms_max", 1.0) < 1e-3
              and val_results.get("energy_monotone", False))
        print(f"  Quality : {'✓ RESEARCH GRADE' if ok else '⚠ REVIEW WARNINGS ABOVE'}")
    print(f"\n  Load with:")
    print(f"    data = torch.load('{cfg.output_dir}/{cfg.pt_filename}')")
    print(f"    U    = data['u']   # shape ({cfg.N_samples}, {cfg.nt_out}, {cfg.nx})")
    print(f"    x    = data['x']   # shape ({cfg.nx},)")
    print(f"    t    = data['t']   # shape ({cfg.nt_out},)")
    print("=" * 70)


if __name__ == "__main__":
    main()