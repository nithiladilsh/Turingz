"""
================================================================================
BURGERS DATASET VIEWER & INTEGRITY CHECKER
================================================================================
Loads the high-fidelity dataset and produces publication-quality diagnostics:
  1. Space-time heatmap
  2. Temporal evolution line plots
  3. Energy spectrum (validates spectral resolution)
  4. Energy dissipation curve
  5. Summary statistics table
================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
FILE_PATH = os.path.join(os.path.dirname(__file__), "../data/burgers_1d_highfidelity.pt")

print("=" * 60)
print("  BURGERS DATASET VIEWER")
print("=" * 60)

if not os.path.exists(FILE_PATH):
    sys.exit(f"\n[ERROR] File not found: {FILE_PATH}\n"
             f"Run generate_burgers_dataset.py first.\n")

data = torch.load(FILE_PATH, weights_only=False)

u   = data["u"].numpy()            # (nt, nx)  float32 → float64 via numpy
t   = data["t"].numpy()
x   = data["x"].numpy()
nu  = data["nu"]
L   = data["L"]
T   = data["T"]
meta = data.get("solver_meta", {})
val  = data.get("validation",  {})

print(f"\nDataset loaded successfully.")
print(f"  Shape : u {u.shape}  |  x {x.shape}  |  t {t.shape}")
print(f"  ν     : {nu:.6f}")
print(f"  Domain: x ∈ [0, {L:.4f}],  t ∈ [0, {T:.2f}]")
print(f"  Solver: {meta.get('method', 'unknown')}")

# ─────────────────────────────────────────────────────────────
# DATA INTEGRITY CHECK
# ─────────────────────────────────────────────────────────────
print("\n── Data Integrity ─────────────────────────────────────────")
has_nan  = np.any(np.isnan(u))
has_inf  = np.any(np.isinf(u))
print(f"  NaN present : {has_nan}  {'✓' if not has_nan else '✗ CORRUPT!'}")
print(f"  Inf present : {has_inf}  {'✓' if not has_inf else '✗ CORRUPT!'}")
print(f"  u ∈ [{u.min():.4f}, {u.max():.4f}]")
print(f"  mean = {u.mean():.6f}  |  std = {u.std():.6f}")

if val:
    print("\n── Validation Metrics ─────────────────────────────────────")
    if "pde_residual_rms_mean" in val:
        print(f"  PDE residual (mean RMS) : {val['pde_residual_rms_mean']:.3e}")
        print(f"  PDE residual (max  RMS) : {val['pde_residual_rms_max']:.3e}")
    if "energy_monotone" in val:
        print(f"  Energy monotone         : {val['energy_monotone']}")
        print(f"  Energy dissipated       : {val.get('energy_dissipated_pct', 'N/A'):.1f}%")
    if val.get("cole_hopf_l2_mean") is not None:
        print(f"  Cole-Hopf L2 (mean)     : {val['cole_hopf_l2_mean']:.3e}")
        print(f"  Cole-Hopf L2 (max)      : {val['cole_hopf_l2_max']:.3e}")

# ─────────────────────────────────────────────────────────────
# FIGURE LAYOUT: 2×3 grid
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
fig.suptitle(
    f"1D Viscous Burgers' Equation  —  ν = {nu:.5f},  nx = {u.shape[1]},  nt = {u.shape[0]}",
    fontsize=14, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

# ── 1. Space-time heatmap ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])   # spans two columns
im = ax1.pcolormesh(x, t, u, cmap="RdBu_r", shading="auto",
                    vmin=-np.percentile(np.abs(u), 99),
                    vmax=+np.percentile(np.abs(u), 99))
fig.colorbar(im, ax=ax1, label="u(x, t)", pad=0.02)
ax1.set_xlabel("x", fontsize=12)
ax1.set_ylabel("t", fontsize=12)
ax1.set_title("Space-Time Evolution  u(x, t)", fontsize=12)
ax1.set_xlim(0, L)
ax1.set_ylim(0, T)

# ── 2. Temporal line plots ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
n_lines = 6
time_indices = np.round(np.linspace(0, len(t) - 1, n_lines)).astype(int)
colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_lines))
for idx, color in zip(time_indices, colors):
    ax2.plot(x, u[idx], color=color, linewidth=1.4, label=f"t={t[idx]:.2f}")
ax2.set_xlabel("x", fontsize=12)
ax2.set_ylabel("u", fontsize=12)
ax2.set_title("Snapshots at Selected Times", fontsize=12)
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.4)

# ── 3. Energy spectrum at t=0 and t=T ────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
for ti_label, color, ls in [(0, "steelblue", "-"), (-1, "tomato", "--")]:
    u_hat = np.fft.rfft(u[ti_label])
    k_pos = np.arange(len(u_hat))
    E_k   = (np.abs(u_hat) / u.shape[1])**2
    ax3.semilogy(k_pos[1:], E_k[1:], color=color, ls=ls,
                 linewidth=1.3, label=f"t = {t[ti_label]:.2f}")
ax3.set_xlabel("Wavenumber k", fontsize=12)
ax3.set_ylabel("E(k)  [log scale]", fontsize=12)
ax3.set_title("Energy Spectrum", fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, which="both", linestyle="--", alpha=0.4)
ax3.set_xlim(0, u.shape[1] // 2)

# ── 4. Energy dissipation over time ──────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
dx  = L / u.shape[1]
energy = 0.5 * dx * np.sum(u**2, axis=1)
ax4.plot(t, energy, color="seagreen", linewidth=2)
ax4.set_xlabel("t", fontsize=12)
ax4.set_ylabel("E(t) = ½∫u² dx", fontsize=12)
ax4.set_title("Kinetic Energy Dissipation", fontsize=12)
ax4.grid(True, linestyle="--", alpha=0.4)
ax4.set_xlim(0, T)

# ── 5. Pointwise std over time (tracks shock sharpening) ─────
ax5 = fig.add_subplot(gs[1, 2])
u_std_t = u.std(axis=1)
ax5.plot(t, u_std_t, color="darkorchid", linewidth=2)
ax5.set_xlabel("t", fontsize=12)
ax5.set_ylabel("σ(u(·, t))", fontsize=12)
ax5.set_title("Spatial Std Dev Over Time", fontsize=12)
ax5.grid(True, linestyle="--", alpha=0.4)
ax5.set_xlim(0, T)

plt.savefig(os.path.join(os.path.dirname(FILE_PATH), "burgers_diagnostics.png"),
            dpi=150, bbox_inches="tight")
print("\nDiagnostic figure saved: ../data/burgers_diagnostics.png")
plt.show()
print("\nDone.")
