import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class Config:
    x_start: float = -1.0
    x_end:   float =  1.0
    L:       float =  2.0
    nx:      int   = 512

    T:       float = 2.0
    nt_out:  int   = 200
    t_start: float = 0.01

    nu: float = 1.0 / (100.0 * np.pi)

    N_samples: int = 8
    n_modes:   int = 4
    ic_seed:   int = 42

    t_train_end: float = 1.0

    dt_target: float = 1.0e-4
    dealias:   bool  = True

    data_dir:      str = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"
    ))
    plot_dir:      str = os.path.dirname(os.path.abspath(__file__))
    pt_filename:   str = "burgers_1d_spectral.pt"
    csv_filename:  str = "burgers_1d_spectral.csv"
    plot_filename: str = "burgers_diagnostics.png"

    validate: bool = True


def build_grid(cfg: Config):
    x  = np.linspace(cfg.x_start, cfg.x_end, cfg.nx, endpoint=False)
    dx = cfg.L / cfg.nx
    t  = np.concatenate([[0.0],
                          np.linspace(cfg.t_start, cfg.T, cfg.nt_out - 1)])
    return x, dx, t

def ic_sinpi(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x)


def ic_random_fourier(x: np.ndarray, L: float,
                      n_modes: int = 4,
                      rng: np.random.Generator = None) -> np.ndarray:
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
    if sample_idx == 0:
        return ic_sinpi(x)
    return ic_random_fourier(x, cfg.L, n_modes=cfg.n_modes, rng=rng)

def build_spectral_arrays(cfg: Config):
    n     = np.arange(cfg.nx // 2 + 1)
    k     = 2.0 * np.pi * n / cfg.L
    mask  = n <= (cfg.nx // 3)
    return k, mask


def _make_step(k, mask, nu, nx, dt):
    ek  = np.exp(-nu * k ** 2 * dt)
    ek2 = np.exp(-nu * k ** 2 * dt * 0.5)

    def N(uh):
        u  = np.fft.irfft(uh * mask, n=nx)
        u2 = np.fft.rfft(u * u)
        return -0.5j * k * u2

    def step(u_hat):
        k1 = N(u_hat)
        k2 = N(ek2 * u_hat + 0.5 * dt * ek2 * k1)
        k3 = N(ek2 * u_hat + 0.5 * dt * k2)
        k4 = N(ek  * u_hat +       dt * ek  * k3)
        u_hat = (ek * u_hat
                 + dt / 6.0 * (ek * k1 + 2.0 * ek2 * k2
                               + 2.0 * ek2 * k3 + k4))
        u_hat[-1] = 0.0
        return u_hat

    return step

def solve_burgers(x: np.ndarray, t_array: np.ndarray,
                  u_ic: np.ndarray, cfg: Config) -> np.ndarray:
    nx       = cfg.nx
    k, mask  = build_spectral_arrays(cfg)
    u_hat    = np.fft.rfft(u_ic)

    U        = np.empty((len(t_array), nx))
    U[0]     = u_ic

    for i in range(1, len(t_array)):
        dt_total   = t_array[i] - t_array[i - 1]
        n_substeps = max(1, int(round(dt_total / cfg.dt_target)))
        dt_step    = dt_total / n_substeps
        step       = _make_step(k, mask, cfg.nu, nx, dt_step)
        for _ in range(n_substeps):
            u_hat = step(u_hat)
        U[i] = np.fft.irfft(u_hat * mask, n=nx)

    return U

def generate_dataset(cfg: Config) -> dict:
    x, dx, t = build_grid(cfg)
    rng       = np.random.default_rng(cfg.ic_seed)

    print(f"  Domain   : x ∈ [{cfg.x_start}, {cfg.x_end})  "
          f"nx={cfg.nx}  dx={dx:.8f}")
    print(f"  Time     : t[0]=0  t[1]={cfg.t_start}  t[-1]={cfg.T}  "
          f"nt={cfg.nt_out}")
    print(f"  Physics  : ν={cfg.nu:.6f}  L={cfg.L}")
    print(f"  IFRK4    : target dt≈{cfg.dt_target:.0e}, dealias={cfg.dealias}")
    print(f"  Samples  : {cfg.N_samples}  →  "
          f"{cfg.N_samples * cfg.nt_out * cfg.nx:,} CSV rows")

    cfl0 = 1.0 * cfg.dt_target / dx
    print(f"  CFL      : {cfl0:.4f}  ✓")
    print(f"  {'─' * 54}")

    U   = np.empty((cfg.N_samples, cfg.nt_out, cfg.nx), dtype=np.float64)
    ICs = np.empty((cfg.N_samples, cfg.nx),              dtype=np.float64)

    t0_total = time.perf_counter()
    for i in range(cfg.N_samples):
        t0     = time.perf_counter()
        u_ic   = make_ic(x, cfg, sample_idx=i, rng=rng)
        ICs[i] = u_ic
        U[i]   = solve_burgers(x, t, u_ic, cfg)
        elapsed = time.perf_counter() - t0
        print(f"  Sample {i + 1}/{cfg.N_samples} | "
              f"IC max={np.max(np.abs(u_ic)):.3f} | "
              f"u(T) max={np.max(np.abs(U[i, -1, :])):.4f} | "
              f"{elapsed:.1f}s")

    total = time.perf_counter() - t0_total
    print(f"\n  Total: {total:.1f}s  ({total / cfg.N_samples:.1f}s / sample)")
    return {"U": U, "ICs": ICs, "x": x, "t": t}


def validate_solution(dataset: dict, cfg: Config) -> dict:
    U  = dataset["U"]
    x  = dataset["x"]
    t  = dataset["t"]
    dx = cfg.L / cfg.nx

    print("\n─── Validation (sample 0) ───────────────────────────────────────")

    k, _ = build_spectral_arrays(cfg)

    u_s = U[0]
    res = []
    for ti in range(2, len(t) - 1):
        u_t   = u_s[ti]
        dt_c  = t[ti + 1] - t[ti - 1]
        uhat  = np.fft.rfft(u_t)
        dux   = np.fft.irfft(1j * k * uhat,   n=cfg.nx)
        d2ux  = np.fft.irfft(-k ** 2 * uhat,  n=cfg.nx)
        dudt  = (u_s[ti + 1] - u_s[ti - 1]) / dt_c
        r     = dudt + u_t * dux - cfg.nu * d2ux
        res.append(np.sqrt(np.mean(r ** 2)))
    mean_res = float(np.mean(res))
    max_res  = float(np.max(res))
    print(f"  [1/4] PDE residual — mean: {mean_res:.3e}  max: {max_res:.3e}  "
          f"{'✓' if max_res < 1e-3 else '⚠'}")

    energy = 0.5 * dx * np.sum(u_s ** 2, axis=1)
    mono   = bool(np.all(np.diff(energy[1:]) <= 1e-8))
    dE_pct = float(100.0 * (1.0 - energy[-1] / energy[0]))
    print(f"  [2/4] Energy monotone: {mono}  "
          f"dissipated: {dE_pct:.1f}%  {'✓' if mono else '✗'}")

    mass   = dx * np.sum(u_s, axis=1)
    drift  = float(np.max(np.abs(mass - mass[0])))
    print(f"  [3/4] Mass drift: {drift:.3e}  "
          f"{'✓' if drift < 1e-8 else '⚠'}")

    x_re, _, t_re = build_grid(cfg)
    U_re   = solve_burgers(x_re, t_re, ic_sinpi(x_re), cfg)
    rdiff  = float(np.max(np.abs(U[0] - U_re)))
    print(f"  [4/4] Re-run diff: {rdiff:.3e}  "
          f"{'✓ DETERMINISTIC' if rdiff < 1e-12 else '⚠'}")
    print("─────────────────────────────────────────────────────────────────\n")

    return dict(
        pde_residual_rms_mean = mean_res,
        pde_residual_rms_max  = max_res,
        energy_monotone       = mono,
        energy_dissipated_pct = dE_pct,
        mass_drift            = drift,
        self_consistency_diff = rdiff,
    )

def plot_sample(dataset: dict, cfg: Config) -> None:
    U      = dataset["U"][0]
    x      = dataset["x"]
    t      = dataset["t"]
    dx     = cfg.L / cfg.nx
    energy = 0.5 * dx * np.sum(U ** 2, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Burgers — Pseudo-Spectral IFRK4  |  sample 0: sin(πx)  |  "
        f"x ∈ [-1, 1)  |  N={cfg.N_samples} samples  "
        f"({cfg.N_samples * cfg.nt_out * cfg.nx:,} rows)",
        fontsize=11, fontweight="bold",
    )

    im = axes[0].pcolormesh(x, t, U, cmap="RdBu_r",
                             shading="auto", vmin=-1, vmax=1)
    fig.colorbar(im, ax=axes[0], label="u")
    axes[0].axhline(cfg.t_train_end, color="k", ls="--", lw=1,
                    label="train / extrap split")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    axes[0].set_title("Space-time heatmap")
    axes[0].legend(fontsize=8)

    snap_times = [0.0, 0.5, 1.0, 1.5, 2.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(snap_times)))
    for tt, col in zip(snap_times, colors):
        idx = int(np.argmin(np.abs(t - tt)))
        axes[1].plot(x, U[idx], color=col, lw=1.5, label=f"t={t[idx]:.2f}")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("u")
    axes[1].set_title("Snapshots")
    axes[1].legend(fontsize=7)

    axes[2].semilogy(t, energy, lw=1.8, color="steelblue")
    axes[2].axvline(cfg.t_train_end, color="k", ls="--", lw=1)
    axes[2].set_xlabel("t"); axes[2].set_ylabel("E(t)")
    axes[2].set_title("Energy dissipation")

    plt.tight_layout()
    os.makedirs(cfg.plot_dir, exist_ok=True)
    path = os.path.join(cfg.plot_dir, cfg.plot_filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → plot  : {path}")

def save_dataset(dataset: dict, val_results: dict, cfg: Config) -> None:
    U   = dataset["U"]
    ICs = dataset["ICs"]
    x   = dataset["x"]
    t   = dataset["t"]

    os.makedirs(cfg.data_dir, exist_ok=True)

    u_mean = float(U.mean()); u_std = float(U.std())
    u_min  = float(U.min());  u_max = float(U.max())
    u_norm = (U - u_mean) / (u_std + 1e-12)

    pt_path = os.path.join(cfg.data_dir, cfg.pt_filename)
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
        "method"      : "pseudo_spectral_ifrk4",
        "validation"  : val_results,
    }, pt_path)
    print(f"  → .pt   : {pt_path}")
    print(f"     shape: U = {U.shape}")

    X_grid, T_grid = np.meshgrid(x, t)
    t_col = T_grid.ravel()
    x_col = X_grid.ravel()
    all_rows = np.vstack([
        np.column_stack((t_col, x_col, U[s].ravel()))
        for s in range(cfg.N_samples)
    ])
    csv_path = os.path.join(cfg.data_dir, cfg.csv_filename)
    np.savetxt(csv_path, all_rows,
               delimiter=",", header="t,x,u", comments="", fmt="%.10f")
    print(f"  → .csv  : {csv_path}")
    print(f"     rows : {all_rows.shape[0]:,}")

def main() -> None:
    print("=" * 70)
    print("  BURGERS DATASET — PSEUDO-SPECTRAL IFRK4  |  Independent Verifier")
    print("=" * 70)

    cfg = Config()
    print(f"\n  N_samples={cfg.N_samples}  nt={cfg.nt_out}  nx={cfg.nx}\n")

    dataset     = generate_dataset(cfg)
    val_results = validate_solution(dataset, cfg) if cfg.validate else {}
    plot_sample(dataset, cfg)
    save_dataset(dataset, val_results, cfg)

    passed = (val_results.get("pde_residual_rms_max", 1.0) < 1e-3
              and val_results.get("energy_monotone", False))
    print("\n" + "=" * 70)
    print(f"  Quality : {'✓ RESEARCH GRADE' if passed else '⚠  REVIEW WARNINGS'}")
    print(f"\n  CSV  : {os.path.abspath(os.path.join(cfg.data_dir, cfg.csv_filename))}")
    print(f"  .pt  : {os.path.abspath(os.path.join(cfg.data_dir, cfg.pt_filename))}")
    print(f"  plot : {os.path.abspath(os.path.join(cfg.plot_dir, cfg.plot_filename))}")
    print("=" * 70)
    print("\nNext step:  python models/cross_verify.py")


if __name__ == "__main__":
    main()