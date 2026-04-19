import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

# PARAMETERS
NX    = 512          # spatial grid points
X_MIN = -1.0         # domain left boundary
X_MAX =  1.0         # domain right boundary
T_END =  2.0         # total simulation time
NU    =  0.01 / np.pi  # viscosity
CFL   =  0.4         # CFL safety factor

OUTPUT_CSV  = "burgers_fdm_dataset.csv"
OUTPUT_PLOT = "burgers_fdm_solution.png"

# GRID
dx = (X_MAX - X_MIN) / NX
x  = np.linspace(X_MIN, X_MAX, NX, endpoint=False)

dt_adv  = CFL * dx
dt_diff = CFL * dx**2 / NU
dt      = round(min(dt_adv, dt_diff), 8)

t = np.arange(0.0, T_END + dt, dt)
NT = len(t)


# INITIAL CONDITION
def initial_condition(x):
    return -np.sin(np.pi * x)

# FDM SOLVER
def solve(x, t, u0, nu):
    nx = len(x)
    nt = len(t)
    dx = x[1] - x[0]

    U = np.zeros((nt, nx))
    U[0] = u0.copy()
    u    = u0.copy()

    for n in range(1, nt):
        dt_n = t[n] - t[n - 1]

        u_left  = np.roll(u,  1)
        u_right = np.roll(u, -1)

        advection = np.where(
            u >= 0,
            u * (u - u_left)  / dx,
            u * (u_right - u) / dx
        )
        diffusion = nu * (u_right - 2.0 * u + u_left) / dx**2

        u = u - dt_n * advection + dt_n * diffusion
        U[n] = u

        if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 1e4:
            warnings.warn(
                f"INSTABILITY DETECTED at t = {t[n]:.4f}  "
                f"(step {n}/{nt}).  max|u| = {np.max(np.abs(u)):.2e}.  "
                f"Try reducing dt or increasing NX.",
                RuntimeWarning
            )
            U = U[:n]
            t_valid = t[:n]
            return U, t_valid, n

    return U, t, nt - 1

# VALIDATION
def validate(U, t, x, nu):
    print("\n  Validation")
    print("  " + "-" * 40)
    dx = x[1] - x[0]

    max_u = np.max(np.abs(U), axis=1)
    stable = np.all(max_u < 1e4) and np.all(np.isfinite(max_u))
    print(f"  Stability        : {'PASS — max|u| = {:.4f}'.format(max_u.max()) if stable else 'FAIL'}")

    mass       = np.sum(U, axis=1) * dx
    mass_drift = np.max(np.abs(mass - mass[0]))
    mass_ok    = mass_drift < 1e-8
    print(f"  Mass drift       : {mass_drift:.2e}  {'PASS' if mass_ok else 'WARNING — larger than expected'}")

    energy      = 0.5 * np.sum(U**2, axis=1) * dx
    energy_mono = np.all(np.diff(energy) <= 1e-6)
    dE          = 100.0 * (energy[0] - energy[-1]) / energy[0]
    print(f"  Energy dissipated: {dE:.2f}%  {'PASS' if energy_mono else 'WARNING — non-monotone'}")

    print("  " + "-" * 40)
    return {"max_u": float(max_u.max()), "mass_drift": float(mass_drift),
            "energy_dissipated_pct": float(dE), "stable": stable}

# SAVE CSV
def save_csv(t, x, U, path):
    T_grid, X_grid = np.meshgrid(t, x, indexing='ij')
    df = pd.DataFrame({
        "t": T_grid.ravel(),
        "x": X_grid.ravel(),
        "u": U.ravel(),
    })
    df.to_csv(path, index=False, float_format="%.10f")
    return len(df)

# VISUALISATION
def plot_solution(t, x, U, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("1D Burgers' Equation — FDM Solution", fontsize=13, fontweight="bold")

    ax = axes[0]
    stride = max(1, len(t) // 300)
    im = ax.pcolormesh(x, t[::stride], U[::stride], cmap="RdBu_r", shading="auto")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Space-Time  u(x, t)")
    plt.colorbar(im, ax=ax, label="u")

    ax = axes[1]
    snap_times = np.linspace(0, t[-1], 8)
    colors     = plt.get_cmap("plasma")(np.linspace(0, 1, len(snap_times)))
    for ti, col in zip(snap_times, colors):
        idx = int(np.argmin(np.abs(t - ti)))
        ax.plot(x, U[idx], color=col, linewidth=1.5, label=f"t={ti:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("Solution Snapshots")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    dx = x[1] - x[0]
    energy = 0.5 * np.sum(U**2, axis=1) * dx
    ax.plot(t, energy, color="steelblue", linewidth=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("E(t) = ½∫u² dx")
    ax.set_title("Energy Dissipation")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    u0 = initial_condition(x)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        U, t_valid, last_step = solve(x, t, u0, NU)
        for w in caught:
            print(f"\n  ⚠  {w.message}")

    t = t_valid

    stats = validate(U, t, x, NU)

    nrows = save_csv(t, x, U, OUTPUT_CSV)
    plot_solution(t, x, U, OUTPUT_PLOT)
