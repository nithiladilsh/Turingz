import os
import sys
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from restart_spectral import (TGRID, X, NX, K, MASK, DT_TARGET, NU,
                              nearest_index, solve_from)

FIGDIR = os.path.join(ROOT, "results", "module2", "figures")
EPS = 1e-12
T_S = 1.0
ROUGH_AMP = 0.05           # 5% relative L2 synthetic ML-artefact perturbation
ROUGH_SEED = 214050        # deterministic
DT_TRUTH = DT_TARGET / 4.0     # 2.5e-5
DT_CHECK = DT_TARGET / 2.0     # 5e-5   (convergence self-check)
UNSTABLE_AMP = 10.0

NUS = [("1/(100pi)", 1.0 / (100 * np.pi)),
       ("1/(200pi)", 1.0 / (200 * np.pi)),
       ("1/(400pi)", 1.0 / (400 * np.pi)),
       ("1/(800pi)", 1.0 / (800 * np.pi)),
       ("1/(1600pi)", 1.0 / (1600 * np.pi))]


# local copy of the production stepper
# Identical maths to restart_spectral._step / solve_from (cross-checked below).
def _rhs(uh, careless):
    if careless:
        u = np.fft.irfft(uh, n=NX, axis=-1)               # no de-alias mask
    else:
        u = np.fft.irfft(uh * MASK, n=NX, axis=-1)
    return -0.5j * K * np.fft.rfft(u * u, axis=-1)


def _step(uh, E, E2, h, careless):
    k1 = _rhs(uh, careless)
    k2 = _rhs(E2 * uh + 0.5 * h * E2 * k1, careless)
    k3 = _rhs(E2 * uh + 0.5 * h * k2, careless)
    k4 = _rhs(E * uh + h * E2 * k3, careless)
    uh = E * uh + h / 6 * (E * k1 + 2 * E2 * k2 + 2 * E2 * k3 + k4)
    if not careless:
        uh[..., -1] = 0.0                                  # Nyquist zeroing
    return uh


def solve(u0, i_start, nu, dt_target, careless=False):
    """Restart from u0 at output index i_start, integrate to T.
    Early-exits (fills NaN) once the field is non-finite or blows past
    UNSTABLE_AMP, so an unstable careless run cannot burn CPU for nothing."""
    u0 = np.asarray(u0, dtype=np.float64)
    batched = (u0.ndim == 2)
    U0 = u0 if batched else u0[None, :]
    B = U0.shape[0]
    NT = len(TGRID)
    out = np.full((B, NT - i_start, NX), np.nan)
    out[:, 0] = U0
    uh = np.fft.rfft(U0, axis=-1)
    alive = np.ones(B, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        for j in range(i_start + 1, NT):
            h = TGRID[j] - TGRID[j - 1]
            m = max(1, round(h / dt_target))
            h = h / m
            E = np.exp(-nu * K ** 2 * h)
            E2 = np.exp(-nu * K ** 2 * h * 0.5)
            for _ in range(m):
                uh[alive] = _step(uh[alive], E, E2, h, careless)
            if careless:
                u = np.fft.irfft(uh, n=NX, axis=-1)
            else:
                u = np.fft.irfft(uh * MASK, n=NX, axis=-1)
            bad = ~np.isfinite(u).all(axis=-1) | (np.abs(u).max(axis=-1) > UNSTABLE_AMP)
            newly_dead = alive & bad
            out[alive & ~bad, j - i_start] = u[alive & ~bad]
            out[newly_dead, j - i_start] = u[newly_dead]   # keep the blow-up frame
            alive = alive & ~bad
            if not alive.any():
                break
    return out if batched else out[0]


# metrics (same definitions as handoff_stability_diagnostic)
def d1(u):
    return np.fft.irfft(1j * K * np.fft.rfft(u, axis=-1), n=NX, axis=-1)


def d2(u):
    return np.fft.irfft(-(K ** 2) * np.fft.rfft(u, axis=-1), n=NX, axis=-1)


def pde_residual_rms(u_prev, u_cur, dt, nu):
    ut = (u_cur - u_prev) / dt
    r = ut + u_cur * d1(u_cur) - nu * d2(u_cur)
    return float(np.sqrt((r ** 2).mean()))


def energy(u):
    return 0.5 * (X[1] - X[0]) * (u ** 2).sum(axis=-1)


def rel_l2(a, b):
    return float(np.sqrt(((a - b) ** 2).sum()) / (np.sqrt((b ** 2).sum()) + EPS))


def tail_rel_l2(traj, ref, i0):
    t = TGRID
    with np.errstate(invalid="ignore"):
        c = (np.sqrt(((traj - ref[i0:]) ** 2).sum(-1))
             / (np.sqrt((ref[i0:] ** 2).sum(-1)) + EPS))
    if not np.isfinite(c).all():
        return float("inf")
    return float(np.trapezoid(c, t[i0:]) / (t[-1] - t[i0] + EPS))


def rough_state(u_s, amp, seed):
    """Band-limited high-k perturbation, normalised to `amp` relative L2.
    Band spans NX//8 .. NX//2: partly inside the resolved 2/3 band, partly
    beyond it, mimicking broadband ML-state artefacts."""
    rng = np.random.default_rng(seed)
    coef = np.zeros(NX // 2 + 1, dtype=complex)
    lo, hi = NX // 8, NX // 2
    coef[lo:hi] = rng.standard_normal(hi - lo) + 1j * rng.standard_normal(hi - lo)
    noise = np.fft.irfft(coef, n=NX)
    noise *= amp * np.sqrt((u_s ** 2).sum()) / (np.sqrt((noise ** 2).sum()) + EPS)
    return u_s + noise


JPATH = os.path.join(FIGDIR, "viscosity_reanchor_stress.json")
TRUTH_CACHE = "/tmp/vrs_truth_{label}.npz"


def _load_results():
    if os.path.exists(JPATH):
        with open(JPATH) as f:
            return json.load(f)
    return {"config": {"t_s": T_S, "i0": int(nearest_index(T_S)),
                       "rough_amp": ROUGH_AMP, "rough_seed": ROUGH_SEED,
                       "dt_run": DT_TARGET, "dt_truth": DT_TRUTH,
                       "dt_check": DT_CHECK, "grid_nx": NX, "ic": "sin(pi x)",
                       "unstable_amp": UNSTABLE_AMP},
            "stepper_crosscheck_reldiff": None,
            "cases": []}


def run_case(idx):
    """Run one viscosity case (idx into NUS); appends to the JSON on disk."""
    os.makedirs(FIGDIR, exist_ok=True)
    i0 = nearest_index(T_S)
    ic = np.sin(np.pi * X)                     # canonical shock-forming wave
    results = _load_results()

    if idx == 0 and results["stepper_crosscheck_reldiff"] is None:
        t0 = time.time()
        a = solve(ic, 0, NU, DT_TARGET, careless=False)
        b = solve_from(ic, 0)
        cross = rel_l2(a, b)
        results["stepper_crosscheck_reldiff"] = cross
        print(f"[check] local stepper vs restart_spectral.solve_from: rel diff = {cross:.3e} "
              f"({time.time()-t0:.1f}s)", flush=True)

    for label, nu in [NUS[idx]]:
        t0 = time.time()
        cache = TRUTH_CACHE.format(label=label.replace("/", "_"))
        if os.path.exists(cache):
            truth = np.load(cache)["truth"]
        else:
            truth = solve(ic, 0, nu, DT_TRUTH, careless=False)
            np.savez_compressed(cache, truth=truth)
        check = solve(ic, 0, nu, DT_CHECK, careless=False)
        conv = tail_rel_l2(check[i0:], truth, i0)
        u_s = truth[i0]
        states = {"clean": u_s, "rough": rough_state(u_s, ROUGH_AMP, ROUGH_SEED)}
        case = {"nu_label": label, "nu": float(nu),
                "truth_dt_convergence": conv, "states": {}}
        for sname, state in states.items():
            entry = {}
            for mname, careless in (("verified", False), ("careless", True)):
                traj = solve(state, i0, nu, DT_TARGET, careless=careless)
                finite = np.isfinite(traj).all()
                amp_ok = finite and float(np.abs(traj).max()) <= UNSTABLE_AMP
                dt_out = TGRID[i0 + 1] - TGRID[i0]
                res_b = pde_residual_rms(truth[i0 - 1], state, TGRID[i0] - TGRID[i0 - 1], nu)
                res_a = (pde_residual_rms(traj[0], traj[1], dt_out, nu)
                         if np.isfinite(traj[1]).all() else float("inf"))
                en = energy(traj)
                en_growth = (float(np.nanmax(en) / (energy(state) + EPS))
                             if np.isfinite(en).any() else float("inf"))
                tail = tail_rel_l2(traj, truth, i0)
                entry[mname] = {
                    "tail_rel_l2": tail,
                    "res_before": res_b,
                    "res_after": res_a,
                    "res_spike_ratio": (res_a / (res_b + EPS)) if np.isfinite(res_a) else float("inf"),
                    "energy_growth": en_growth,
                    "unstable": bool(not (finite and amp_ok)),
                    "handoff_jump": rel_l2(traj[0], state),
                    "meets_1pct": bool(np.isfinite(tail) and tail <= 0.01),
                    "meets_5pct": bool(np.isfinite(tail) and tail <= 0.05),
                }
            case["states"][sname] = entry
        results["cases"] = [c for c in results["cases"] if c["nu_label"] != label]
        results["cases"].append(case)
        results["cases"].sort(key=lambda c: -c["nu"])
        c = case["states"]
        print(f"[nu={label}] conv={conv:.2e}  "
              f"clean: verified {c['clean']['verified']['tail_rel_l2']:.4f} "
              f"vs careless {c['clean']['careless']['tail_rel_l2']:.4f}"
              f"{' UNSTABLE' if c['clean']['careless']['unstable'] else ''}  |  "
              f"rough: verified {c['rough']['verified']['tail_rel_l2']:.4f} "
              f"vs careless {c['rough']['careless']['tail_rel_l2']:.4f}"
              f"{' UNSTABLE' if c['rough']['careless']['unstable'] else ''}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    with open(JPATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {JPATH}", flush=True)


def make_figure():
    results = _load_results()
    i0 = nearest_index(T_S)
    ic = np.sin(np.pi * X)
    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nus = [c["nu"] for c in results["cases"]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Low-viscosity spectral re-anchor stress test "
                 "(verified vs careless restart, $t_s$ = 1.0)", fontsize=12)

    def series(state, method, key):
        v = [c["states"][state][method][key] for c in results["cases"]]
        return [np.nan if (x is None or not np.isfinite(x)) else x for x in v]

    styles = {("clean", "verified"): ("tab:blue", "o", "-", "verified · clean"),
              ("clean", "careless"): ("tab:red", "o", "--", "careless · clean"),
              ("rough", "verified"): ("tab:blue", "s", "-", "verified · rough"),
              ("rough", "careless"): ("tab:red", "s", "--", "careless · rough")}

    ax = axes[0, 0]
    for (st, me), (col, mk, ls, lab) in styles.items():
        y = series(st, me, "tail_rel_l2")
        ax.plot(nus, y, marker=mk, ls=ls, color=col, alpha=0.55 if st == "rough" else 1.0, label=lab)
        for xx, yy, c in zip(nus, y, results["cases"]):
            if c["states"][st][me]["unstable"]:
                ax.plot(xx, yy, marker="x", ms=14, color="k")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"viscosity $\nu$ (decreasing $\rightarrow$ sharper shock)")
    ax.set_ylabel("tail relative $L_2$ over $[t_s, 2]$")
    ax.set_title("Accuracy (x = unstable run)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for (st, me), (col, mk, ls, lab) in styles.items():
        ax.plot(nus, series(st, me, "res_spike_ratio"), marker=mk, ls=ls, color=col,
                alpha=0.55 if st == "rough" else 1.0, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"viscosity $\nu$"); ax.set_ylabel("residual after / before")
    ax.set_title("Residual spike at the handoff"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for (st, me), (col, mk, ls, lab) in styles.items():
        ax.plot(nus, series(st, me, "energy_growth"), marker=mk, ls=ls, color=col,
                alpha=0.55 if st == "rough" else 1.0, label=lab)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"viscosity $\nu$"); ax.set_ylabel(r"$\max_t E(t)\,/\,E(t_s)$")
    ax.set_title("Energy growth after handoff (physical: $\\leq 1$)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # error-over-time at the harshest viscosity, rough state
    ax = axes[1, 1]
    label, nu = NUS[-1]
    cache = TRUTH_CACHE.format(label=label.replace("/", "_"))
    truth = (np.load(cache)["truth"] if os.path.exists(cache)
             else solve(ic, 0, nu, DT_TRUTH, careless=False))
    state = rough_state(truth[i0], ROUGH_AMP, ROUGH_SEED)
    for mname, careless, col in (("verified", False, "tab:blue"), ("careless", True, "tab:red")):
        traj = solve(state, i0, nu, DT_TARGET, careless=careless)
        with np.errstate(invalid="ignore"):
            c = (np.sqrt(((traj - truth[i0:]) ** 2).sum(-1))
                 / (np.sqrt((truth[i0:] ** 2).sum(-1)) + EPS))
        ax.plot(TGRID[i0:], c, color=col, label=f"{mname} restart")
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel("relative $L_2$ error")
    ax.set_title(f"Error over time, $\\nu$ = {label}, rough handoff state")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ppath = os.path.join(FIGDIR, "viscosity_reanchor_stress.png")
    fig.savefig(ppath, dpi=150)
    print(f"[saved] {ppath}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--figure":
        make_figure()
    elif len(sys.argv) > 1 and sys.argv[1] == "--case":
        run_case(int(sys.argv[2]))
    else:
        for i in range(len(NUS)):
            run_case(i)
        make_figure()
