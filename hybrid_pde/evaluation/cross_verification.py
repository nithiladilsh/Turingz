import json
import numpy as np
import torch

TOL = 1e-3
SLO, SHI, TE = 0.25, 0.75, 1.0
REF = "data/colehopf/burgers_colehopf.pt"
VER = "data/spectral/burgers_spectral.pt"

R = torch.load(REF, weights_only=False)
V = torch.load(VER, weights_only=False)
ur, uv, t = R["u"].numpy(), V["u"].numpy(), R["t"].numpy()
assert np.allclose(R["x"].numpy(), V["x"].numpy()) and np.allclose(t, V["t"].numpy())
assert np.allclose(R["ICs"].numpy(), V["ICs"].numpy())

err = uv - ur
l2 = np.linalg.norm(err, axis=2) / np.linalg.norm(ur, axis=2)
linf = np.abs(err).max(axis=2)
fr, fv = np.fft.rfft(ur, axis=2), np.fft.rfft(uv, axis=2)
spec = np.linalg.norm(np.abs(fv) - np.abs(fr), axis=2) / np.linalg.norm(fr, axis=2)
pre, sh = t < SLO, (t >= SLO) & (t <= SHI)
po, ex = (t > SHI) & (t <= TE), t > TE

def wins(m):
    return {"pre_shock": float(m[:, pre].max()), "shock_peak": float(m[:, sh].max()),
            "post_shock": float(m[:, po].max()), "extrap_max": float(m[:, ex].max()),
            "extrap_mean": float(m[:, ex].mean()), "final_time": float(m[:, -1].max())}

rep = {"tolerance": TOL, "windows": {"shock_lo": SLO, "shock_hi": SHI, "train_end": TE},
       "relative_l2": wins(l2), "linf": wins(linf), "spectral_distance": wins(spec),
       "per_sample_max_l2": l2.max(1).tolist(), "per_sample_mean_l2": l2.mean(1).tolist(),
       "overall_max_l2": float(l2.max()), "overall_max_linf": float(linf.max()),
       "ref_source": REF, "ver_source": VER,
       "grid": {g: R[g] for g in ("nx", "nt", "L", "T", "nu", "N_samples")}}
rep["verdict_smooth"] = rep["relative_l2"]["pre_shock"] < TOL
rep["verdict_shock"] = rep["relative_l2"]["shock_peak"] < TOL
rep["verdict_extrap"] = rep["relative_l2"]["extrap_max"] < TOL
rep["verdict_overall"] = rep["overall_max_l2"] < TOL

print("overall_max_l2=%.2e shock=%.2e extrap=%.2e linf=%.2e verdict=%s" % (
    rep["overall_max_l2"], rep["relative_l2"]["shock_peak"],
    rep["relative_l2"]["extrap_max"], rep["overall_max_linf"], rep["verdict_overall"]))
with open("results/cross_verification.json", "w") as f:
    json.dump(rep, f, indent=2)
assert rep["verdict_overall"]
print("CROSS-CHECK PASSED")