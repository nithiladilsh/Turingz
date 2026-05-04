import torch
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
import os

print("Generating Stable High-Fidelity Burgers Dataset (Spectral + Cole-Hopf)...")

# =========================
# 1. PARAMETERS
# =========================
L = 2.0 * np.pi
nx = 512
x = np.linspace(0, L, nx, endpoint=False)
dx = x[1] - x[0]

T = 2.0
nt = 100
t = np.linspace(0, T, nt)

nu = 0.05 / np.pi
eps = 1e-12  # Small epsilon to avoid division by zero

# =========================
# 2. FFT SETUP (Spectral)
# =========================
k = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
ik = 1j * k

# =========================
# 3. COLE–HOPF INITIAL CONDITION
# =========================
phi0 = np.exp((np.cos(x) - 1) / (2 * nu))

# =========================
# 4. HEAT EQUATION MATRIX
# =========================
diagonals = [
    np.ones(nx - 1),
    -2 * np.ones(nx),
    np.ones(nx - 1),
    np.array([1.0]),      # periodic BC
    np.array([1.0])       # periodic BC
]
offsets = [-1, 0, 1, nx - 1, -(nx - 1)]
A = (nu / dx**2) * diags(diagonals, offsets, format='csc')

# =========================
# 5. KRYLOV SOLVER
# =========================
print("Solving Heat Equation...")
phi = np.zeros((nt, nx), dtype=np.float64)
phi[0] = phi0

for i in range(1, nt):
    dt = t[i] - t[i - 1]
    phi[i] = expm_multiply(A * dt, phi[i - 1])

# =========================
# 6. SPECTRAL INVERSE TRANSFORM (STABLE)
# =========================
print("Applying Stable Spectral Inverse Transformation...")
u = np.zeros((nt, nx), dtype=np.float64)

for i in range(nt):
    phi_hat = np.fft.fft(phi[i])
    dphi_dx = np.fft.ifft(ik * phi_hat).real

    # Stable division
    u_temp = -2.0 * nu * (dphi_dx / (phi[i] + eps))

    # Clip extreme values to prevent explosions
    u[i] = np.clip(u_temp, -10, 10)

# =========================
# 7. NORMALIZATION (ML-READY)
# =========================
u_mean = u.mean()
u_std = u.std()
u_normalized = (u - u_mean) / (u_std + eps)

# =========================
# 8. CONVERT TO TENSORS
# =========================
u_tensor = torch.tensor(u, dtype=torch.float32)
u_norm_tensor = torch.tensor(u_normalized, dtype=torch.float32)
t_tensor = torch.tensor(t, dtype=torch.float32)
x_tensor = torch.tensor(x, dtype=torch.float32)

# =========================
# 9. SAVE DATASET (.PT)
# =========================
os.makedirs('../data', exist_ok=True)
save_path = os.path.join(os.path.dirname(__file__), '../data/burgers_1d_spectral_stable.pt')

torch.save({
    'u': u_tensor,
    'u_normalized': u_norm_tensor,
    't': t_tensor,
    'x': x_tensor,
    'nu': nu,
    'L': L,
    'T': T,
    'nx': nx,
    'nt': nt,
    'dx': dx,
    'mean': u_mean,
    'std': u_std,
    'method': 'cole-hopf_krylov_spectral_stable',
    'description': 'Stable high-fidelity 1D Burgers dataset using Cole-Hopf + Krylov + spectral differentiation'
}, save_path)

print(f"Stable dataset saved: {save_path}")
print(f"Shape: {u_tensor.shape}")

# =========================
# 10. SAVE READABLE CSV
# =========================
print("Saving readable CSV...")
X, T_grid = np.meshgrid(x, t)

data_flat = np.column_stack((
    T_grid.flatten(),
    X.flatten(),
    u.flatten()
))

csv_path = os.path.join(os.path.dirname(__file__), '../data/burgers_spectral_stable_readable.csv')

np.savetxt(
    csv_path,
    data_flat,
    delimiter=",",
    header="t,x,u",
    comments="",
    fmt="%.6f"
)

print(f"CSV also saved to: {csv_path}")