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