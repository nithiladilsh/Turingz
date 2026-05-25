import os
import time
import warnings

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

    nu:      float = 1.0 / (100.0 * np.pi)
    cfl:     float = 0.4

    t_train_end: float = 1.0

    data_dir: str = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "fdm"
    ))
    pt_filename:   str = "burgers_fdm_dataset.pt"
    csv_filename:  str = "burgers_fdm_dataset.csv"
    plot_filename: str = "burgers_fdm_diagnostics.png"

    validate: bool = True


def build_grid(cfg: Config):
    x  = np.linspace(cfg.x_start, cfg.x_end, cfg.nx, endpoint=False)
    dx = cfg.L / cfg.nx
    t  = np.concatenate([[0.0],
                          np.linspace(cfg.t_start, cfg.T, cfg.nt_out - 1)])
    return x, dx, t


def initial_condition(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x)


def solve(x: np.ndarray, t_out: np.ndarray, u0: np.ndarray,
          cfg: Config) -> np.ndarray:
    nx  = len(x)
    dx  = x[1] - x[0]
    nu  = cfg.nu

    dt_adv  = cfg.cfl * dx
    dt_diff = cfg.cfl * dx ** 2 / nu
    dt_fine = min(dt_adv, dt_diff)

    U    = np.empty((len(t_out), nx))
    U[0] = u0.copy()
    u    = u0.copy()

    for i in range(1, len(t_out)):
        dt_total   = t_out[i] - t_out[i - 1]
        n_substeps = max(1, int(np.ceil(dt_total / dt_fine)))
        dt_step    = dt_total / n_substeps

        for _ in range(n_substeps):
            u_left  = np.roll(u,  1)
            u_right = np.roll(u, -1)

            advection = np.where(
                u >= 0,
                u * (u - u_left)  / dx,
                u * (u_right - u) / dx,
            )
            diffusion = nu * (u_right - 2.0 * u + u_left) / dx ** 2

            u = u - dt_step * advection + dt_step * diffusion

            if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 1e4:
                warnings.warn(
                    f"INSTABILITY at t ≈ {t_out[i - 1] + _ * dt_step:.4f}  "
                    f"max|u| = {np.max(np.abs(u)):.2e}  "
                    "Try reducing cfl or increasing nx.",
                    RuntimeWarning,
                )
                U[i:] = np.nan
                return U

        U[i] = u.copy()

    return U


def validate_solution(U: np.ndarray, x: np.ndarray,
                      t: np.ndarray, cfg: Config) -> dict:
    dx = cfg.L / cfg.nx
    print("\n─── Validation ──────────────────────────────────────────────────")

    max_u  = float(np.nanmax(np.abs(U)))
    stable = bool(np.all(np.isfinite(U)) and max_u < 1e4)
    print(f"  [1/3] Stability  : {'✓  max|u| = ' + f'{max_u:.4f}' if stable else '✗  UNSTABLE'}")

    mass       = dx * np.sum(U, axis=1)
    mass_drift = float(np.max(np.abs(mass - mass[0])))
    print(f"  [2/3] Mass drift : {mass_drift:.2e}  "
          f"{'✓' if mass_drift < 1e-3 else '⚠  upwind numerical diffusion expected'}")

    energy      = 0.5 * dx * np.sum(U ** 2, axis=1)
    energy_mono = bool(np.all(np.diff(energy) <= 1e-6))
    dE_pct      = float(100.0 * (energy[0] - energy[-1]) / energy[0])
    print(f"  [3/3] Energy     : monotone={energy_mono}  "
          f"dissipated={dE_pct:.1f}%  {'✓' if energy_mono else '⚠'}")

    print("─────────────────────────────────────────────────────────────────\n")
    return {
        "stable":                stable,
        "max_u":                 max_u,
        "mass_drift":            mass_drift,
        "energy_monotone":       energy_mono,
        "energy_dissipated_pct": dE_pct,
    }


def plot_solution(U: np.ndarray, x: np.ndarray, t: np.ndarray,
                  cfg: Config) -> None:
    dx     = cfg.L / cfg.nx
    energy = 0.5 * dx * np.sum(U ** 2, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        "Burgers — FDM  |  sin(πx)  |  Upwind + Central Diffusion",
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
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(snap_times)))
    for tt, col in zip(snap_times, colors):
        idx = int(np.argmin(np.abs(t - tt)))
        axes[1].plot(x, U[idx], color=col, lw=1.5, label=f"t={t[idx]:.2f}")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("u")
    axes[1].set_title("Snapshots")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(t, energy, lw=1.8, color="steelblue")
    axes[2].axvline(cfg.t_train_end, color="k", ls="--", lw=1)
    axes[2].set_xlabel("t"); axes[2].set_ylabel("E(t)")
    axes[2].set_title("Energy dissipation")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(cfg.data_dir, exist_ok=True)
    path = os.path.join(cfg.data_dir, cfg.plot_filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → plot  : {path}")


def save_dataset(U: np.ndarray, x: np.ndarray, t: np.ndarray,
                 val_results: dict, cfg: Config) -> None:
    os.makedirs(cfg.data_dir, exist_ok=True)

    u_mean = float(U.mean()); u_std = float(U.std())
    u_min  = float(U.min());  u_max = float(U.max())
    u_norm = (U - u_mean) / (u_std + 1e-12)

    U_pt      = U[np.newaxis]
    u_norm_pt = u_norm[np.newaxis]

    pt_path = os.path.join(cfg.data_dir, cfg.pt_filename)
    torch.save({
        "u"           : torch.tensor(U_pt,      dtype=torch.float32),
        "u_normalized": torch.tensor(u_norm_pt, dtype=torch.float32),
        "x"           : torch.tensor(x,         dtype=torch.float32),
        "t"           : torch.tensor(t,         dtype=torch.float32),
        "nu"          : cfg.nu,
        "L"           : cfg.L,
        "x_start"     : cfg.x_start,
        "x_end"       : cfg.x_end,
        "T"           : cfg.T,
        "t_start"     : cfg.t_start,
        "t_train_end" : cfg.t_train_end,
        "nx"          : cfg.nx,
        "nt"          : cfg.nt_out,
        "N_samples"   : 1,
        "dx"          : cfg.L / cfg.nx,
        "u_mean"      : u_mean,
        "u_std"       : u_std,
        "u_min"       : u_min,
        "u_max"       : u_max,
        "method"      : "fdm_upwind_central",
        "validation"  : val_results,
    }, pt_path)
    print(f"  → .pt   : {pt_path}")
    print(f"     shape: U = {U_pt.shape}  (N_samples, N_t, N_x)")

    X_grid, T_grid = np.meshgrid(x, t)
    all_rows = np.column_stack([T_grid.ravel(), X_grid.ravel(), U.ravel()])
    csv_path = os.path.join(cfg.data_dir, cfg.csv_filename)
    np.savetxt(csv_path, all_rows,
               delimiter=",", header="t,x,u", comments="", fmt="%.10f")
    print(f"  → .csv  : {csv_path}")
    print(f"     rows : {all_rows.shape[0]:,}  ({cfg.nt_out} × {cfg.nx})")


def main() -> None:
    print("=" * 70)
    print("  BURGERS DATASET — FDM  |  Upwind + Central Diffusion")
    print("=" * 70)

    cfg      = Config()
    x, dx, t = build_grid(cfg)

    dt_adv  = cfg.cfl * dx
    dt_diff = cfg.cfl * dx ** 2 / cfg.nu
    dt_fine = min(dt_adv, dt_diff)

    print(f"\n  Domain  : x ∈ [{cfg.x_start}, {cfg.x_end})  nx={cfg.nx}  dx={dx:.8f}")
    print(f"  Time    : t[0]=0  t[-1]={cfg.T}  nt_out={cfg.nt_out}")
    print(f"  Physics : ν={cfg.nu:.6f}  CFL={cfg.cfl}")
    print(f"  dt_fine : {dt_fine:.2e}\n")

    t0 = time.perf_counter()
    u0 = initial_condition(x)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        U = solve(x, t, u0, cfg)
        for w in caught:
            print(f"\n  ⚠  {w.message}")

    elapsed = time.perf_counter() - t0
    print(f"  Solved in {elapsed:.1f}s  |  u(T) max={np.nanmax(np.abs(U[-1])):.4f}")

    val_results = validate_solution(U, x, t, cfg) if cfg.validate else {}
    plot_solution(U, x, t, cfg)
    save_dataset(U, x, t, val_results, cfg)

    passed = val_results.get("stable", False) and val_results.get("energy_monotone", False)
    print("\n" + "=" * 70)
    print(f"\n  .pt  : {os.path.join(cfg.data_dir, cfg.pt_filename)}")
    print(f"  .csv : {os.path.join(cfg.data_dir, cfg.csv_filename)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
