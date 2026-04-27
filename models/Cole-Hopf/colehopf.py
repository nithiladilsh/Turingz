"""
================================================================================
1D VISCOUS BURGERS' EQUATION — DATASET GENERATOR
Method  : Cole-Hopf Transformation (exact semi-analytical solution)
Team    : Turingz

PDE     : ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
Domain  : x ∈ [-1, 1), periodic BCs, t ∈ [0, T]

CSV output  : columns  t, x, u   (all samples concatenated, no sample_id)
Dataset size: N_samples × nt_out × nx  rows
              e.g. 4 × 200 × 512 = 409,600 rows  (default)
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import time

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
from scipy.integrate import trapezoid
import torch
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must precede pyplot
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    # Spatial domain  x ∈ [-1, 1)
    x_start: float = -1.0
    x_end:   float =  1.0
    L:       float =  2.0          # period  = x_end - x_start
    nx:      int   = 512           # dx = 2/512 ≈ 0.00391

    # Time
    T:       float = 2.0           # total integration time
    nt_out:  int   = 200           # output snapshots  (includes t = 0)
    # t_start > 0 avoids quadrature singularity at t = 0.
    # Safety check: t_start ≫ dx²/ν ≈ (2/512)² / (1/100π) ≈ 0.0048
    t_start: float = 0.01

    # Physics
    nu: float = 1.0 / (100.0 * np.pi)   # ν ≈ 0.00318

    # Dataset
    N_samples: int = 4             # 4 × 200 × 512 = 409,600 CSV rows
    n_modes:   int = 4             # Fourier modes for random ICs
    ic_seed:   int = 42            # reproducibility seed

    # Train / extrapolation split (used only for plots and metadata)
    t_train_end: float = 1.0

    # Output paths
    output_dir:    str = "./data"
    pt_filename:   str = "burgers_1d_cole_hopf.pt"
    csv_filename:  str = "burgers_1d_cole_hopf.csv"
    plot_filename: str = "burgers_diagnostics.png"

    # Run built-in validation after generation
    validate: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# GRID
# ─────────────────────────────────────────────────────────────────────────────
def build_grid(cfg: Config):
    """
    Return (x, dx, t).

    x : uniform grid on [-1, 1)  with nx points (endpoint excluded).
    t : t[0] = 0 (exact IC snapshot); t[1:] = linspace(t_start, T, nt_out-1).
    """
    x  = np.linspace(cfg.x_start, cfg.x_end, cfg.nx, endpoint=False)
    dx = cfg.L / cfg.nx
    t  = np.concatenate([[0.0],
                          np.linspace(cfg.t_start, cfg.T, cfg.nt_out - 1)])
    return x, dx, t


# ─────────────────────────────────────────────────────────────────────────────
# INITIAL CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────
def ic_sinpi(x: np.ndarray) -> np.ndarray:
    """Canonical IC: u(x, 0) = sin(πx).  Used for sample 0."""
    return np.sin(np.pi * x)


def ic_random_fourier(x: np.ndarray, L: float,
                      n_modes: int = 4,
                      rng: np.random.Generator = None) -> np.ndarray:
    """
    Band-limited random IC using Fourier modes 1 … n_modes.
    Amplitude is normalised so max|u| = 1.
    """
    if rng is None:
        rng = np.random.default_rng()
    u = np.zeros_like(x)
    for m in range(1, n_modes + 1):
        amp   = rng.standard_normal()
        phase = rng.uniform(0.0, 2.0 * np.pi)
        u    += amp * np.sin(2.0 * np.pi * m * x / L + phase)
    return u / (np.max(np.abs(u)) + 1e-12)


def make_ic(x: np.ndarray, cfg: Config, sample_idx: int,
            rng: np.random.Generator) -> np.ndarray:
    """
    Return the initial condition for a given sample index.
    Sample 0 always uses the canonical sin(πx) IC so the first trajectory
    matches existing CSV files.  Subsequent samples use random Fourier ICs.
    """
    if sample_idx == 0:
        return ic_sinpi(x)
    return ic_random_fourier(x, cfg.L, n_modes=cfg.n_modes, rng=rng)


# ─────────────────────────────────────────────────────────────────────────────
# COLE-HOPF CORE
# ─────────────────────────────────────────────────────────────────────────────
def compute_phi0(x: np.ndarray, u_ic: np.ndarray, nu: float) -> np.ndarray:
    """
    Compute φ(x, 0) = exp(−1/(2ν) ∫_{x[0]}^x u(s, 0) ds).

    The cumulative integral is evaluated with the trapezoidal rule.
    The exponent is shifted by its maximum before exponentiation so that
    φ_max = 1; the constant shift cancels exactly in  u = −2ν φ_x / φ.
    """
    dx     = x[1] - x[0]
    cumint = np.zeros_like(x)
    for i in range(1, len(x)):
        cumint[i] = cumint[i - 1] + 0.5 * (u_ic[i - 1] + u_ic[i]) * dx

    exponent = -cumint / (2.0 * nu)
    return np.exp(exponent - np.max(exponent))   # max-shifted for stability


def solve_heat_batch(x: np.ndarray, t_array: np.ndarray,
                     phi0: np.ndarray, nu: float, L: float):
    """
    Exact Gaussian-kernel solution to the periodic heat equation at all
    times in t_array.

        φ(x, t) = (1/√(4πνt)) ∫ φ₀(ξ) exp(−(x−ξ)²/(4νt)) dξ

    Periodicity is enforced by tiling the domain three times.
    The squared-distance matrix diff² is allocated once and reused for
    every time step.

    Returns
    -------
    phi_all     : (len(t_array), nx)
    dphi_dx_all : (len(t_array), nx)
    """
    x_ext    = np.concatenate([x - L, x, x + L])
    phi0_ext = np.tile(phi0, 3)

    diff  = x[:, None] - x_ext[None, :]   # (nx, 3*nx) — allocated once
    diff2 = diff ** 2

    nt          = len(t_array)
    phi_all     = np.empty((nt, len(x)))
    dphi_dx_all = np.empty((nt, len(x)))

    for ti, tt in enumerate(t_array):
        denom    = 4.0 * nu * tt
        kernel   = np.exp(-diff2 / denom)
        d_kernel = (-2.0 * diff / denom) * kernel
        norm     = np.sqrt(np.pi * denom)

        phi_all[ti]     = trapezoid(phi0_ext * kernel,   x_ext, axis=1) / norm
        dphi_dx_all[ti] = trapezoid(phi0_ext * d_kernel, x_ext, axis=1) / norm

    return phi_all, dphi_dx_all


def solve_burgers(x: np.ndarray, t_array: np.ndarray,
                  u_ic: np.ndarray, nu: float, L: float) -> np.ndarray:
    """
    Full Cole-Hopf pipeline for one trajectory.

    u(x, t) = −2ν φ_x(x, t) / φ(x, t)

    t_array[0] must equal 0; that snapshot is filled with u_ic directly.
    """
    phi0 = compute_phi0(x, u_ic, nu)
    if np.any(phi0 == 0.0):
        raise ValueError(
            f"phi0 underflowed (nu={nu:.6f}).  "
            "Increase nu or reduce the IC amplitude."
        )

    U    = np.empty((len(t_array), len(x)))
    U[0] = u_ic

    phi, dphi_dx = solve_heat_batch(x, t_array[1:], phi0, nu, L)
    U[1:] = -2.0 * nu * dphi_dx / phi
    return U


# ─────────────────────────────────────────────────────────────────────────────
# DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_dataset(cfg: Config) -> dict:
    """Generate N_samples trajectories and return them as a dict."""
    x, dx, t = build_grid(cfg)
    rng       = np.random.default_rng(cfg.ic_seed)

    print(f"  Domain  : x ∈ [{cfg.x_start}, {cfg.x_end})  "
          f"nx={cfg.nx}  dx={dx:.8f}")
    print(f"  Time    : t[0]=0  t[1]={cfg.t_start}  t[-1]={cfg.T}  "
          f"nt={cfg.nt_out}")
    print(f"  Physics : ν={cfg.nu:.6f}   L={cfg.L}")
    print(f"  Samples : {cfg.N_samples}  →  "
          f"{cfg.N_samples * cfg.nt_out * cfg.nx:,} CSV rows\n"
          f"  {'─' * 54}")

    U   = np.empty((cfg.N_samples, cfg.nt_out, cfg.nx), dtype=np.float64)
    ICs = np.empty((cfg.N_samples, cfg.nx),              dtype=np.float64)

    t0_total = time.perf_counter()
    for i in range(cfg.N_samples):
        t0      = time.perf_counter()
        u_ic    = make_ic(x, cfg, sample_idx=i, rng=rng)
        ICs[i]  = u_ic
        U[i]    = solve_burgers(x, t, u_ic, cfg.nu, cfg.L)
        elapsed = time.perf_counter() - t0
        print(f"  Sample {i + 1}/{cfg.N_samples} | "
              f"IC max={np.max(np.abs(u_ic)):.3f} | "
              f"u(T) max={np.max(np.abs(U[i, -1, :])):.4f} | "
              f"{elapsed:.1f}s")

    total = time.perf_counter() - t0_total
    print(f"\n  Total: {total:.1f}s  ({total / cfg.N_samples:.1f}s / sample)")
    return {"U": U, "ICs": ICs, "x": x, "t": t}


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def validate_solution(dataset: dict, cfg: Config) -> dict:
    """
    Four independent checks on sample 0.

    1. PDE residual  — spectral-derivative residual on interior time steps.
    2. Energy dissipation — ½∫u² dx must decrease monotonically.
    3. Mass conservation  — ∫u dx must be constant (periodic + zero-mean IC).
    4. Self-consistency   — re-run produces bit-for-bit identical output.
    """
    U  = dataset["U"]
    x  = dataset["x"]
    t  = dataset["t"]
    dx = cfg.L / cfg.nx

    print("\n─── Validation (sample 0) ───────────────────────────────────────")

    # 1. PDE residual (spectral derivatives; skip t[0], t[1], t[-1])
    kk     = np.fft.rfftfreq(cfg.nx, d=cfg.L / cfg.nx) * 2.0 * np.pi
    u_s    = U[0]
    res_list = []
    for ti in range(2, len(t) - 1):
        u        = u_s[ti]
        dt_c     = t[ti + 1] - t[ti - 1]
        u_hat    = np.fft.rfft(u)
        du_dx    = np.fft.irfft(1j * kk * u_hat, n=cfg.nx)
        d2u_dx2  = np.fft.irfft(-kk ** 2 * u_hat, n=cfg.nx)
        du_dt    = (u_s[ti + 1] - u_s[ti - 1]) / dt_c
        residual = du_dt + u * du_dx - cfg.nu * d2u_dx2
        res_list.append(np.sqrt(np.mean(residual ** 2)))

    mean_res = float(np.mean(res_list))
    max_res  = float(np.max(res_list))
    print(f"  [1/4] PDE residual — mean: {mean_res:.3e}  max: {max_res:.3e}  "
          f"{'✓' if max_res < 1e-3 else '⚠  check resolution'}")

    # 2. Energy dissipation
    energy       = 0.5 * dx * np.sum(u_s ** 2, axis=1)
    energy_mono  = bool(np.all(np.diff(energy[1:]) <= 1e-8))
    dE_pct       = float(100.0 * (1.0 - energy[-1] / energy[0]))
    print(f"  [2/4] Energy — monotone: {energy_mono}  "
          f"dissipated: {dE_pct:.1f}%  {'✓' if energy_mono else '✗'}")

    # 3. Mass conservation
    mass         = dx * np.sum(u_s, axis=1)
    mass_drift   = float(np.max(np.abs(mass - mass[0])))
    print(f"  [3/4] Mass drift: {mass_drift:.3e}  "
          f"{'✓' if mass_drift < 1e-8 else '⚠'}")

    # 4. Self-consistency (deterministic re-run)
    x_re, _, t_re = build_grid(cfg)
    U_re   = solve_burgers(x_re, t_re, ic_sinpi(x_re), cfg.nu, cfg.L)
    rdiff  = float(np.max(np.abs(U[0] - U_re)))
    print(f"  [4/4] Re-run diff: {rdiff:.3e}  "
          f"{'✓ DETERMINISTIC' if rdiff < 1e-12 else '⚠'}")
    print("─────────────────────────────────────────────────────────────────\n")

    return dict(
        pde_residual_rms_mean = mean_res,
        pde_residual_rms_max  = max_res,
        energy_monotone       = energy_mono,
        energy_dissipated_pct = dE_pct,
        mass_drift            = mass_drift,
        self_consistency_diff = rdiff,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_sample(dataset: dict, cfg: Config) -> None:
    """Save a three-panel diagnostic figure for sample 0."""
    U      = dataset["U"][0]
    x      = dataset["x"]
    t      = dataset["t"]
    dx     = cfg.L / cfg.nx
    energy = 0.5 * dx * np.sum(U ** 2, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Burgers — Cole-Hopf  |  sample 0: sin(πx)  |  "
        f"x ∈ [-1, 1)  |  N={cfg.N_samples} samples  "
        f"({cfg.N_samples * cfg.nt_out * cfg.nx:,} rows)",
        fontsize=11, fontweight="bold",
    )

    # Panel 1: space-time heatmap
    im = axes[0].pcolormesh(x, t, U, cmap="RdBu_r",
                             shading="auto", vmin=-1, vmax=1)
    fig.colorbar(im, ax=axes[0], label="u")
    axes[0].axhline(cfg.t_train_end, color="k", ls="--", lw=1,
                    label="train / extrap split")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    axes[0].set_title("Space-time heatmap")
    axes[0].legend(fontsize=8)

    # Panel 2: snapshots at selected times
    snap_times = [0.0, 0.5, 1.0, 1.5, 2.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(snap_times)))
    for tt, col in zip(snap_times, colors):
        idx = int(np.argmin(np.abs(t - tt)))
        axes[1].plot(x, U[idx], color=col, lw=1.5, label=f"t={t[idx]:.2f}")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("u")
    axes[1].set_title("Snapshots")
    axes[1].legend(fontsize=7)

    # Panel 3: energy dissipation
    axes[2].semilogy(t, energy, lw=1.8, color="steelblue")
    axes[2].axvline(cfg.t_train_end, color="k", ls="--", lw=1)
    axes[2].set_xlabel("t"); axes[2].set_ylabel("E(t)")
    axes[2].set_title("Energy dissipation")

    plt.tight_layout()
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, cfg.plot_filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → plot  : {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
def save_dataset(dataset: dict, val_results: dict, cfg: Config) -> None:
    """
    Write two output files:

    .pt   — PyTorch tensor file containing all arrays plus metadata.
    .csv  — Columns  t, x, u  (all samples concatenated, 10 d.p.).
            Format matches existing team CSV files exactly:
              - header row: t,x,u
              - x ∈ [-1, 1) starting at -1.0000000000
              - no sample_id column
    """
    U   = dataset["U"]
    ICs = dataset["ICs"]
    x   = dataset["x"]
    t   = dataset["t"]

    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Normalisation stats (stored in .pt only) ──────────────────────────
    u_mean = float(U.mean()); u_std = float(U.std())
    u_min  = float(U.min());  u_max = float(U.max())
    u_norm = (U - u_mean) / (u_std + 1e-12)

    # ── PyTorch file ──────────────────────────────────────────────────────
    pt_path = os.path.join(cfg.output_dir, cfg.pt_filename)
    torch.save({
        "u"           : torch.tensor(U,      dtype=torch.float32),
        "u_normalized": torch.tensor(u_norm, dtype=torch.float32),
        "ICs"         : torch.tensor(ICs,    dtype=torch.float32),
        "x"           : torch.tensor(x,      dtype=torch.float32),
        "t"           : torch.tensor(t,      dtype=torch.float32),
        "nu"          : cfg.nu,
        "L"           : cfg.L,
        "x_start"     : cfg.x_start,
        "x_end"       : cfg.x_end,
        "T"           : cfg.T,
        "t_start"     : cfg.t_start,
        "t_train_end" : cfg.t_train_end,
        "nx"          : cfg.nx,
        "nt"          : cfg.nt_out,
        "N_samples"   : cfg.N_samples,
        "dx"          : cfg.L / cfg.nx,
        "u_mean"      : u_mean,
        "u_std"       : u_std,
        "u_min"       : u_min,
        "u_max"       : u_max,
        "method"      : "cole_hopf_gaussian_kernel",
        "validation"  : val_results,
    }, pt_path)
    print(f"  → .pt   : {pt_path}")
    print(f"     shape: U = {U.shape}  (N_samples, N_t, N_x)")

    # ── CSV — all samples concatenated ───────────────────────────────────
    X_grid, T_grid = np.meshgrid(x, t)       # both (nt_out, nx)
    t_col = T_grid.ravel()
    x_col = X_grid.ravel()

    all_rows = np.vstack([
        np.column_stack((t_col, x_col, U[s].ravel()))
        for s in range(cfg.N_samples)
    ])

    csv_path = os.path.join(cfg.output_dir, cfg.csv_filename)
    np.savetxt(csv_path, all_rows,
               delimiter=",", header="t,x,u", comments="", fmt="%.10f")

    total_rows = all_rows.shape[0]
    print(f"  → .csv  : {csv_path}")
    print(f"     rows : {total_rows:,}  "
          f"({cfg.N_samples} × {cfg.nt_out} × {cfg.nx})")
    print(f"     first row:  t={all_rows[0, 0]:.1f}  "
          f"x={all_rows[0, 1]:.10f}  u={all_rows[0, 2]:.10f}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("  BURGERS DATASET — COLE-HOPF  |  x ∈ [-1, 1)  |  Team Turingz")
    print("=" * 70)

    cfg = Config()
    print(f"\n  N_samples={cfg.N_samples}  nt={cfg.nt_out}  nx={cfg.nx}  "
          f"→  {cfg.N_samples * cfg.nt_out * cfg.nx:,} CSV rows\n")

    dataset     = generate_dataset(cfg)
    val_results = validate_solution(dataset, cfg) if cfg.validate else {}
    plot_sample(dataset, cfg)
    save_dataset(dataset, val_results, cfg)

    passed = (
        val_results.get("pde_residual_rms_max", 1.0) < 1e-3
        and val_results.get("energy_monotone", False)
    )
    print("\n" + "=" * 70)
    print(f"  Quality : {'✓ RESEARCH GRADE' if passed else '⚠  REVIEW WARNINGS'}")
    print(f"\n  CSV  : {cfg.output_dir}/{cfg.csv_filename}")
    print(f"  .pt  : {cfg.output_dir}/{cfg.pt_filename}")
    print("=" * 70)


if __name__ == "__main__":
    main()