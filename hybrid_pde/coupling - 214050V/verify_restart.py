"""
Phase 1 verification (Module 2, Coupling):
  Check A - Solver equivalence: does restart_spectral.solve_full(u0) reproduce the
            TEAM solver's real solve(u0)?  We load the team's actual function code from
            solvers/numerical/spectral.py (numpy-only; we skip its `import torch` and
            module-level dataset build), so this compares against the real solver, not a
            hand-copy.
  Check B - Dataset reproducibility: regenerate the Cole-Hopf initial conditions with the
            fixed seed and confirm the 10 held-out ICs in predictions.npz are exactly the
            test-set ICs (900-909). This proves we can regenerate/extend the held-out set.
"""
import os, numpy as np
from restart_spectral import solve_full, X

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPECTRAL = os.path.join(ROOT, "hybrid_pde", "solvers", "numerical", "spectral.py")
PRED = os.path.join(ROOT, "results", "eval", "predictions.npz")


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-15))


# ---- Check A: equivalence against the team solver's real code ----
src = open(SPECTRAL).read().split("def random_ic")[0]         # constants + rhs + step + solve
src = "\n".join(l for l in src.splitlines() if l.strip() != "import torch")
ns = {}
exec(src, ns)
team_solve = ns["solve"]

rng = np.random.default_rng(0)
def zero_mean_ic(nmodes=4):
    u = sum(rng.standard_normal()*np.sin(2*np.pi*m*X/2.0 + rng.uniform(0, 2*np.pi))
            for m in range(1, nmodes+1))
    return u/(np.abs(u).max()+1e-12)
ics = [np.sin(np.pi*X)] + [zero_mean_ic() for _ in range(4)]

print("Check A - wrapper vs TEAM solve():")
maxd = 0.0
for j, u0 in enumerate(ics):
    d = rel(solve_full(u0.copy()), team_solve(u0.copy()))
    maxd = max(maxd, d)
    print(f"   IC {j}: rel diff = {d:.2e}")
print(f"   -> max = {maxd:.2e}  {'PASS (identical solver)' if maxd < 1e-10 else 'FAIL'}\n")

# ---- Check B: reproduce the held-out ICs from the generator ----
nx = 512; L = 2.0; x = np.linspace(-1, 1, nx, endpoint=False)
def gen_ic(r, n_modes=4):
    u = sum(r.standard_normal()*np.sin(2*np.pi*m*x/L + r.uniform(0, 2*np.pi))
            for m in range(1, n_modes+1))
    return u/(np.abs(u).max()+1e-12)
r = np.random.default_rng(42)                                  # same seed as colehopf.py
ICs = np.stack([np.sin(np.pi*x)] + [gen_ic(r) for _ in range(999)])

d = np.load(PRED)
held_ics = d["u_true_eval"][:, 0]                              # (10, 512) at t=0
dif = rel(ICs[900:910], held_ics)
print("Check B - regenerated ICs[900:909] vs predictions held-out ICs:")
print(f"   rel diff = {dif:.2e}  {'PASS (held-out = test ICs 900-909; regenerable)' if dif < 1e-5 else 'CHECK alignment'}")
