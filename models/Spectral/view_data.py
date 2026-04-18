import torch
import matplotlib.pyplot as plt
import os

print("Loading dataset...")

file_path = os.path.join(os.path.dirname(__file__), '../data/burgers_1d_spectral_stable.pt')
data = torch.load(file_path)

u = data['u'].numpy()
t = data['t'].numpy()
x = data['x'].numpy()

print(f"Data loaded! Shape: {u.shape}")

# --- 1. Heatmap (Improved) ---
plt.figure(figsize=(10, 6))
plt.pcolormesh(x, t, u, cmap='viridis', shading='auto')
plt.colorbar(label='Velocity (u)')
plt.title("Burgers' Equation (Heatmap)")
plt.xlabel("Space (x)")
plt.ylabel("Time (t)")
plt.tight_layout()
plt.show()

# --- 2. Line plots at different times ---
plt.figure(figsize=(10, 6))

time_indices = [0, len(t)//4, len(t)//2, -1]

for idx in time_indices:
    plt.plot(x, u[idx], label=f"t = {t[idx]:.2f}")

plt.title("Wave Evolution Over Time")
plt.xlabel("Space (x)")
plt.ylabel("Velocity (u)")
plt.legend()
plt.grid()
plt.show()