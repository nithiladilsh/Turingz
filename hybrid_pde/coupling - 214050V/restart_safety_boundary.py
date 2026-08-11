import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import viscosity_reanchor_stress as base
from viscosity_reanchor_stress import (TGRID, X, NX, DT_TARGET, EPS,
                                       nearest_index, tail_rel_l2, solve, JPATH, FIGDIR)

T_S = 1.0
BPATH = os.path.join(FIGDIR, "restart_safety_boundary.json")
NUS_ALL = [("1/(100pi)", 100), ("1/(200pi)", 200), ("1/(400pi)", 400),
           ("1/(600pi)", 600), ("1/(800pi)", 800), ("1/(1200pi)", 1200),
           ("1/(1600pi)", 1600)]
TRUTH_CACHE = "/tmp/vrs_truth_{tag}.npz"


def nu_of(d): return 1.0 / (d * np.pi)


def _load():
    if os.path.exists(BPATH):
        with open(BPATH) as f: return json.load(f)
    return {"refinement": {}, "boundary": {}, "config": {
        "t_s": T_S, "dt": DT_TARGET, "grid_nx": NX, "ic": "sin(pi x)",
        "re_cell_def": "max|u(t_s)| * dx / nu",
        "diag_def": "energy fraction in top octave of resolved band (k in [K3/2, K3], K3 = NX/3) of the handoff state"}}


def _save(r):
    os.makedirs(FIGDIR, exist_ok=True)
    with open(BPATH, "w") as f: json.dump(r, f, indent=2)


# ---------- A) grid-refinement control (variable N, careful scheme) ----------
def solve_N(u0, nu, N, dt_target=DT_TARGET):
    """Careful (mask + Nyquist) IF-RK4 at resolution N from t=0 on TGRID."""
    L = 2.0
    K = 2 * np.pi * np.arange(N // 2 + 1) / L
    MASK = np.arange(N // 2 + 1) <= N // 3
    def rhs(uh):
        u = np.fft.irfft(uh * MASK, n=N, axis=-1)
        return -0.5j * K * np.fft.rfft(u * u, axis=-1)
    NT = len(TGRID)
    out = np.empty((NT, N))
    out[0] = u0
    uh = np.fft.rfft(u0)
    for j in range(1, NT):
        h = TGRID[j] - TGRID[j - 1]
        m = max(1, round(h / dt_target)); h = h / m
        E = np.exp(-nu * K ** 2 * h); E2 = np.exp(-nu * K ** 2 * h * 0.5)
        for _ in range(m):
            k1 = rhs(uh)
            k2 = rhs(E2 * uh + 0.5 * h * E2 * k1)
            k3 = rhs(E2 * uh + 0.5 * h * k2)
            k4 = rhs(E * uh + h * E2 * k3)
            uh = E * uh + h / 6 * (E * k1 + 2 * E2 * k2 + 2 * E2 * k3 + k4)
            uh[..., -1] = 0.0
        out[j] = np.fft.irfft(uh * MASK, n=N)
    return out


def refinement(tag, N):
    """Compare careful N-grid solve vs 512-grid solve on the shared points."""
    d = dict(NUS_ALL)[tag]; nu = nu_of(d)
    r = _load()
    t0 = time.time()
    icN = np.sin(np.pi * np.linspace(-1, 1, N, endpoint=False))
    uN = solve_N(icN, nu, N)
    u512 = solve(np.sin(np.pi * X), 0, nu, DT_TARGET, careless=False)
    s = N // NX
    uN_on512 = uN[:, ::s]
    i0 = nearest_index(T_S)
    dev_full = float(np.sqrt(((uN_on512 - u512) ** 2).sum()) / np.sqrt((uN_on512 ** 2).sum()))
    dev_tail = tail_rel_l2(u512[i0:], uN_on512, i0)
    r["refinement"].setdefault(tag, {})[str(N)] = {
        "rel_l2_full": dev_full, "rel_l2_tail": dev_tail}
    _save(r)
    print(f"[refine {tag} N={N}] 512-vs-{N} rel L2: full={dev_full:.3e} tail={dev_tail:.3e} "
          f"({time.time()-t0:.1f}s)", flush=True)


# ---------- B/C) boundary points ----------
def boundary_point(tag):
    d = dict(NUS_ALL)[tag]; nu = nu_of(d)
    r = _load()
    t0 = time.time()
    ic = np.sin(np.pi * X)
    cache = TRUTH_CACHE.format(tag=tag.replace("/", "_"))
    if os.path.exists(cache):
        truth = np.load(cache)["truth"]
    else:
        truth = solve(ic, 0, nu, base.DT_TRUTH, careless=False)
        np.savez_compressed(cache, truth=truth)
    i0 = nearest_index(T_S)
    state = truth[i0]
    dx = X[1] - X[0]
    re_cell = float(np.abs(state).max() * dx / nu)
    # reference-free spectral diagnostic on the handoff state
    uh = np.fft.rfft(state)
    e = np.abs(uh) ** 2
    K3 = NX // 3
    diag = float(e[K3 // 2:K3 + 1].sum() / (e.sum() + EPS))
    row = {"nu": nu, "re_cell": re_cell, "highk_energy_frac": diag}
    for mname, careless in (("verified", False), ("careless", True)):
        traj = solve(state, i0, nu, DT_TARGET, careless=careless)
        tail = tail_rel_l2(traj, truth, i0)
        row[mname + "_tail"] = (None if not np.isfinite(tail) else tail)
        row[mname + "_unstable"] = bool(not np.isfinite(tail))
    r["boundary"][tag] = row
    _save(r)
    print(f"[boundary {tag}] Re_cell={re_cell:.2f} diag={diag:.2e} "
          f"verified={row['verified_tail']:.2e} careless="
          f"{'UNSTABLE' if row['careless_unstable'] else '%.4f' % row['careless_tail']} "
          f"({time.time()-t0:.1f}s)", flush=True)


def crossings():
    r = _load()
    rows = sorted(r["boundary"].values(), key=lambda x: x["re_cell"])
    res = {}
    # threshold
    for thr, name in ((0.01, "1pct"), (0.05, "5pct")):
        rc = None
        for a, b in zip(rows, rows[1:]):
            ta = a["careless_tail"] if a["careless_tail"] is not None else 10.0
            tb = b["careless_tail"] if b["careless_tail"] is not None else 10.0
            if ta < thr <= tb:
                # log-linear interpolation in (Re_cell, tail)
                import math
                f = (math.log(thr) - math.log(ta)) / (math.log(tb) - math.log(ta))
                rc = a["re_cell"] + f * (b["re_cell"] - a["re_cell"])
                break
        res[name] = rc
    r["boundary_crossings_re_cell"] = res
    _save(r)
    print("[crossings] careless tail crosses 1%% at Re_cell ~= %.2f ; 5%% at ~= %.2f"
          % (res["1pct"] or -1, res["5pct"] or -1), flush=True)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = _load()
    rows = sorted(r["boundary"].values(), key=lambda x: x["re_cell"])
    rc = [x["re_cell"] for x in rows]
    vt = [x["verified_tail"] for x in rows]
    ct = [x["careless_tail"] if x["careless_tail"] is not None else np.nan for x in rows]
    un = [x["careless_unstable"] for x in rows]
    dg = [x["highk_energy_frac"] for x in rows]
    x1 = r["boundary_crossings_re_cell"].get("1pct")
    x5 = r["boundary_crossings_re_cell"].get("5pct")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax.plot(rc, vt, "o-", color="tab:blue", label="verified re-anchor (bit-exact restart)")
    ax.plot(rc, ct, "s--", color="tab:red", label="approximate restart (no de-aliasing)")
    for x, y, u in zip(rc, ct, un):
        if u: ax.annotate("UNSTABLE", (x, 0.3), color="tab:red", fontsize=8,
                          ha="center", rotation=90)
    ax.axhline(0.01, color="k", lw=0.7, ls=":", label="1% target")
    if x1: ax.axvspan(x1, max(rc) * 1.05, color="tab:red", alpha=0.07)
    if x1: ax.axvline(x1, color="tab:red", lw=0.8, ls="-.",
                      label=f"safety boundary Re_cell ≈ {x1:.1f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"cell Reynolds number at handoff  $Re_{cell}=\max|u|\,\Delta x/\nu$")
    ax.set_ylabel("tail relative $L_2$ over $[t_s,2]$")
    ax.set_title("Restart-fidelity safety boundary (IC sin($\\pi x$), $t_s$=1.0)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax2.plot(dg, ct, "s", color="tab:red", label="approximate restart tail")
    for x, y, u in zip(dg, ct, un):
        if u: ax2.annotate("UNSTABLE", (x, 0.3), color="tab:red", fontsize=8,
                           ha="center", rotation=90)
    ax2.axhline(0.01, color="k", lw=0.7, ls=":")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("reference-free diagnostic: high-k energy fraction of handoff state")
    ax2.set_ylabel("approximate-restart tail error")
    ax2.set_title("Reference-free predictor of restart safety")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "restart_safety_boundary.png")
    fig.savefig(p, dpi=150); print("[saved]", p, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "refine":   refinement(sys.argv[2], int(sys.argv[3]))
    elif cmd == "point":  boundary_point(sys.argv[2])
    elif cmd == "cross":  crossings()
    elif cmd == "figure": figure()
