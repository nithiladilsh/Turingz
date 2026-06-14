import json, os
import numpy as np
import torch

DATA = "data/colehopf/burgers_colehopf.pt"
OUT = "results/cross_verification.json"
K, SUBSET_SEED = 64, 0
SLO, SHI, TE = 0.25, 0.75, 1.0

d = torch.load(DATA, weights_only=False, map_location="cpu")
ICs = d["ICs"].numpy()
ur_all = d["u"].numpy()
x, t = d["x"].numpy(), d["t"].numpy()
nu, L = float(d["nu"]), float(d["L"])
N, nt, nx = ur_all.shape
idx = np.sort(np.random.default_rng(SUBSET_SEED).choice(N, K, replace=False))

k = 2 * np.pi * np.arange(nx // 2 + 1) / L
mask = np.arange(nx // 2 + 1) <= nx // 3
dt_target = 1e-4

def rhs(uh):
    u = np.fft.irfft(uh * mask, n=nx)
    return -0.5j * k * np.fft.rfft(u * u)

def step(uh, E, E2, h):
    k1 = rhs(uh)
    k2 = rhs(E2 * uh + 0.5 * h * E2 * k1)
    k3 = rhs(E2 * uh + 0.5 * h * k2)
    k4 = rhs(E * uh + h * E2 * k3)
    uh = E * uh + h / 6 * (E * k1 + 2 * E2 * k2 + 2 * E2 * k3 + k4)
    uh[-1] = 0.0
    return uh

def solve_spec(u0):
    uh = np.fft.rfft(u0); U = np.empty((nt, nx)); U[0] = u0
    for j in range(1, nt):
        h = t[j] - t[j - 1]; m = max(1, round(h / dt_target)); h /= m
        E, E2 = np.exp(-nu * k**2 * h), np.exp(-nu * k**2 * h * 0.5)
        for _ in range(m):
            uh = step(uh, E, E2, h)
        U[j] = np.fft.irfft(uh * mask, n=nx)
    return U

uv = np.stack([solve_spec(ICs[i]) for i in idx])
ur = ur_all[idx]
err = uv - ur
l2 = np.linalg.norm(err, axis=2) / np.linalg.norm(ur, axis=2)
linf = np.abs(err).max(axis=2)
pre, sh = t < SLO, (t >= SLO) & (t <= SHI)
po, ex = (t > SHI) & (t <= TE), t > TE

def win(m):
    return {"pre_shock": float(m[:, pre].max()), "shock_peak": float(m[:, sh].max()),
            "post_shock": float(m[:, po].max()), "in_dist_mean": float(m[:, t <= TE].mean()),
            "extrap_max": float(m[:, ex].max()), "extrap_mean": float(m[:, ex].mean())}

report = {"dataset": DATA, "verifier": "spectral_ifrk4_inline", "n_total": int(N),
          "n_subset": K, "subset_seed": SUBSET_SEED, "subset_idx": idx.tolist(),
          "relative_l2": win(l2), "linf": win(linf),
          "overall_max_l2": float(l2.max()), "overall_mean_l2": float(l2.mean()),
          "overall_max_abs": float(np.abs(err).max()), "per_ic_max_l2": l2.max(1).tolist()}
report["verdict"] = bool(report["overall_max_l2"] < 1e-2)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(report, open(OUT, "w"), indent=2)
print("subset=%d/%d  overall_max_l2=%.2e  mean_l2=%.2e  max_abs=%.2e  verdict=%s" % (
    K, N, report["overall_max_l2"], report["overall_mean_l2"],
    report["overall_max_abs"], report["verdict"]))
