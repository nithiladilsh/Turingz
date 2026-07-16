"""
Evidence-hardening extension for the low-viscosity re-anchor stress test (E11).
Author: Dharmapala R.D. (214050V)

Two additions requested at evaluator-review, NO new mechanism, main M2 method
unchanged:

  A) Multi-IC robustness: the stress test repeated over 10 random truncated
     Fourier ICs (modes 1-4, unit amplitude, seed 42 -- same family as the
     dataset generator), at nu = 1/(100pi), 1/(800pi), 1/(1600pi).
     CLEAN handoff states only (the true state at t_s): the clean-state result
     is immune to the "your rough state is synthetic" objection.

  B) Safety-step ablation: which spectral safety step matters?
       verified    : 2/3 de-alias mask + Nyquist zeroing   (production)
       no_dealias  : mask OFF, Nyquist zeroing ON
       no_nyquist  : mask ON,  Nyquist zeroing OFF
       careless    : both OFF

  C) From-t0 scheme-fidelity control: the same 4 variants integrated from
     t = 0 (canonical sin(pi x)).  Pre-registered honest framing: if the
     careless scheme also fails from t=0, the failure is a SCHEME-FIDELITY
     issue, not a restart-index issue -- which is precisely why the Module 2
     restart is verified bit-for-bit against the production solver.

Truth for the multi-IC runs uses dt = 5e-5; dt-convergence vs 2.5e-5 was
established at ~1e-12 in the base experiment (per-nu numbers in the JSON).

Updates results/module2/figures/viscosity_reanchor_stress.json (new keys
"ext_config", "multi_ic", "from_t0") and rebuilds the PNG (2x3 panels).
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import viscosity_reanchor_stress as base
from viscosity_reanchor_stress import (TGRID, X, NX, K, MASK, DT_TARGET, EPS,
                                       nearest_index, tail_rel_l2,
                                       pde_residual_rms, energy, rel_l2, JPATH,
                                       FIGDIR, UNSTABLE_AMP)

T_S = 1.0
N_IC = 10
IC_SEED = 42
DT_TRUTH_EXT = 5e-5
NUS_EXT = [("1/(100pi)", 1.0 / (100 * np.pi)),
           ("1/(800pi)", 1.0 / (800 * np.pi)),
           ("1/(1600pi)", 1.0 / (1600 * np.pi))]
VARIANTS = {"verified":   (True,  True),
            "no_dealias": (False, True),
            "no_nyquist": (True,  False),
            "careless":   (False, False)}
TRUTH_EXT_CACHE = "/tmp/vrs_ext_truth_{i}.npz"


def _rhs_v(uh, mask_on):
    u = np.fft.irfft(uh * MASK, n=NX, axis=-1) if mask_on else np.fft.irfft(uh, n=NX, axis=-1)
    return -0.5j * K * np.fft.rfft(u * u, axis=-1)


def _step_v(uh, E, E2, h, mask_on, nyq_on):
    k1 = _rhs_v(uh, mask_on)
    k2 = _rhs_v(E2 * uh + 0.5 * h * E2 * k1, mask_on)
    k3 = _rhs_v(E2 * uh + 0.5 * h * k2, mask_on)
    k4 = _rhs_v(E * uh + h * E2 * k3, mask_on)
    uh = E * uh + h / 6 * (E * k1 + 2 * E2 * k2 + 2 * E2 * k3 + k4)
    if nyq_on:
        uh[..., -1] = 0.0
    return uh


def solve_v(u0, i_start, nu, dt_target, mask_on, nyq_on):
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
                uh[alive] = _step_v(uh[alive], E, E2, h, mask_on, nyq_on)
            u = np.fft.irfft(uh * MASK, n=NX, axis=-1) if mask_on else np.fft.irfft(uh, n=NX, axis=-1)
            bad = ~np.isfinite(u).all(axis=-1) | (np.abs(u).max(axis=-1) > UNSTABLE_AMP)
            out[:, j - i_start][alive] = u[alive]
            alive = alive & ~bad
            if not alive.any():
                break
    return out if batched else out[0]


def make_ics(n=N_IC, seed=IC_SEED):
    """Random truncated Fourier profiles, modes 1-4, unit amplitude (same
    family as the dataset generator, hybrid_pde/data)."""
    rng = np.random.default_rng(seed)
    ics = []
    for _ in range(n):
        u = np.zeros(NX)
        for k in range(1, 5):
            a, b = rng.standard_normal(2)
            u += a * np.sin(k * np.pi * X) + b * np.cos(k * np.pi * X)
        ics.append(u / np.abs(u).max())
    return np.array(ics)


def _load():
    with open(JPATH) as f:
        return json.load(f)


def _save(r):
    with open(JPATH, "w") as f:
        json.dump(r, f, indent=2)


def ext_truth(i):
    label, nu = NUS_EXT[i]
    cache = TRUTH_EXT_CACHE.format(i=i)
    if os.path.exists(cache):
        print(f"[truth {label}] cached", flush=True); return
    t0 = time.time()
    ics = make_ics()
    truth = solve_v(ics, 0, nu, DT_TRUTH_EXT, True, True)
    np.savez_compressed(cache, truth=truth)
    print(f"[truth {label}] B={len(ics)} dt={DT_TRUTH_EXT} ({time.time()-t0:.1f}s)", flush=True)


def ext_restarts(i):
    label, nu = NUS_EXT[i]
    i0 = nearest_index(T_S)
    truth = np.load(TRUTH_EXT_CACHE.format(i=i))["truth"]
    r = _load()
    r.setdefault("ext_config", {"n_ic": N_IC, "ic_seed": IC_SEED,
                                "dt_truth_ext": DT_TRUTH_EXT,
                                "states": "clean (true state at t_s)",
                                "variants": {k: {"dealias_mask": v[0], "nyquist_zero": v[1]}
                                             for k, v in VARIANTS.items()}})
    mi = r.setdefault("multi_ic", {})
    entry = {}
    t0 = time.time()
    states = truth[:, i0]
    dt_out = TGRID[i0 + 1] - TGRID[i0]
    for vname, (mask_on, nyq_on) in VARIANTS.items():
        traj = solve_v(states, i0, nu, DT_TARGET, mask_on, nyq_on)
        tails, spikes, egrow, unstable = [], [], [], 0
        for b in range(len(states)):
            finite = np.isfinite(traj[b]).all()
            amp_ok = finite and float(np.abs(traj[b]).max()) <= UNSTABLE_AMP
            if not (finite and amp_ok):
                unstable += 1
            tail = tail_rel_l2(traj[b], truth[b], i0)
            tails.append(tail)
            rb = pde_residual_rms(truth[b, i0 - 1], states[b], TGRID[i0] - TGRID[i0 - 1], nu)
            ra = (pde_residual_rms(traj[b, 0], traj[b, 1], dt_out, nu)
                  if np.isfinite(traj[b, 1]).all() else float("inf"))
            spikes.append(ra / (rb + EPS))
            en = energy(traj[b])
            egrow.append(float(np.nanmax(en) / (energy(states[b]) + EPS))
                         if np.isfinite(en).any() else float("inf"))
        ft = [x for x in tails if np.isfinite(x)]
        fs = [x for x in spikes if np.isfinite(x)]
        entry[vname] = {
            "tail_mean": float(np.mean(ft)) if ft else None,
            "tail_std": float(np.std(ft)) if ft else None,
            "tail_max": float(np.max(ft)) if ft else None,
            "spike_mean": float(np.mean(fs)) if fs else None,
            "energy_growth_max": (float(np.max([x for x in egrow if np.isfinite(x)]))
                                  if any(np.isfinite(x) for x in egrow) else None),
            "unstable_count": unstable, "n": len(states),
            "meets_1pct": int(sum(1 for x in tails if np.isfinite(x) and x <= 0.01)),
            "meets_5pct": int(sum(1 for x in tails if np.isfinite(x) and x <= 0.05)),
            "tails": [None if not np.isfinite(x) else x for x in tails],
        }
    mi[label] = entry
    _save(r)
    s = "  ".join(f"{k}: {e['tail_mean']:.4f}±{e['tail_std']:.4f} u={e['unstable_count']}/{e['n']}"
                  if e["tail_mean"] is not None else f"{k}: ALL-UNSTABLE u={e['unstable_count']}/{e['n']}"
                  for k, e in entry.items())
    print(f"[multi-IC {label}] {s} ({time.time()-t0:.1f}s)", flush=True)


def ext_fromt0(i):
    label, nu = NUS_EXT[i]
    ic = np.sin(np.pi * X)
    truth = np.load(TRUTH_EXT_CACHE.format(i=i))["truth"] if False else None
    # canonical-IC truth from the base experiment cache (dt=2.5e-5)
    cache = base.TRUTH_CACHE.format(label=label.replace("/", "_"))
    truth = np.load(cache)["truth"] if os.path.exists(cache) else solve_v(ic, 0, nu, base.DT_TRUTH, True, True)
    r = _load()
    ft = r.setdefault("from_t0", {})
    entry = {}
    t0 = time.time()
    for vname, (mask_on, nyq_on) in VARIANTS.items():
        traj = solve_v(ic, 0, nu, DT_TARGET, mask_on, nyq_on)
        finite = np.isfinite(traj).all()
        amp_ok = finite and float(np.abs(traj).max()) <= UNSTABLE_AMP
        err = tail_rel_l2(traj, truth, 0) if (finite and amp_ok) else float("inf")
        entry[vname] = {"full_traj_err": (err if np.isfinite(err) else None),
                        "unstable": bool(not (finite and amp_ok))}
    ft[label] = entry
    _save(r)
    parts = [k + ": " + ("UNSTABLE" if e["unstable"] else "%.4f" % e["full_traj_err"])
             for k, e in entry.items()]
    print("[from-t0 %s] %s (%.1fs)" % (label, "  ".join(parts), time.time() - t0), flush=True)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = _load()
    i0 = nearest_index(T_S)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle("Low-viscosity spectral re-anchor stress test -- verified vs degraded restarts "
                 "($t_s$ = 1.0; multi-IC n = %d; from-t0 scheme-fidelity control)" % N_IC, fontsize=12)
    vc = {"verified": "tab:blue", "no_dealias": "tab:orange",
          "no_nyquist": "tab:green", "careless": "tab:red"}

    # (a) canonical tail vs nu (base cases)
    ax = axes[0, 0]
    nus = [c["nu"] for c in r["cases"]]
    for st, mk in (("clean", "o"), ("rough", "s")):
        for me, col, ls in (("verified", "tab:blue", "-"), ("careless", "tab:red", "--")):
            y = [c["states"][st][me]["tail_rel_l2"] for c in r["cases"]]
            y = [np.nan if not np.isfinite(v) else v for v in y]
            ax.plot(nus, y, marker=mk, ls=ls, color=col,
                    alpha=0.55 if st == "rough" else 1.0, label=f"{me} · {st}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"$\nu$ (decreasing $\rightarrow$ sharper shock)")
    ax.set_ylabel("tail rel $L_2$ over $[t_s,2]$")
    ax.set_title("(a) Canonical IC: accuracy vs viscosity"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (b) residual spike (base cases)
    ax = axes[0, 1]
    for st, mk in (("clean", "o"), ("rough", "s")):
        for me, col, ls in (("verified", "tab:blue", "-"), ("careless", "tab:red", "--")):
            y = [c["states"][st][me]["res_spike_ratio"] for c in r["cases"]]
            y = [np.nan if not np.isfinite(v) else v for v in y]
            ax.plot(nus, y, marker=mk, ls=ls, color=col,
                    alpha=0.55 if st == "rough" else 1.0, label=f"{me} · {st}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"$\nu$"); ax.set_ylabel("residual after / before")
    ax.set_title("(b) Residual spike at the handoff"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    mi = r.get("multi_ic", {})
    labels = [l for l, _ in NUS_EXT if l in mi]
    xpos = np.arange(len(labels)); w = 0.2

    # (c) multi-IC tails, 4 variants
    ax = axes[0, 2]
    for k, (vname, col) in enumerate(vc.items()):
        means, stds, alluns = [], [], []
        for l in labels:
            e = mi[l][vname]
            if e["tail_mean"] is None:
                means.append(np.nan); stds.append(0); alluns.append(True)
            else:
                means.append(e["tail_mean"]); stds.append(e["tail_std"]); alluns.append(False)
        ax.bar(xpos + (k - 1.5) * w, means, w, yerr=stds, color=col, label=vname, capsize=2)
        for xi, au in zip(xpos, alluns):
            if au:
                ax.text(xi + (k - 1.5) * w, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-4,
                        "UNSTABLE", rotation=90, fontsize=6, color=col, ha="center", va="bottom")
    ax.set_yscale("log"); ax.set_xticks(xpos); ax.set_xticklabels(labels)
    ax.set_ylabel("tail rel $L_2$ (mean ± std over stable ICs)")
    ax.set_title(f"(c) Multi-IC sweep (n = {N_IC}, clean handoff)"); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    # (d) unstable counts
    ax = axes[1, 0]
    for k, (vname, col) in enumerate(vc.items()):
        cnt = [mi[l][vname]["unstable_count"] if l in mi else 0 for l in labels]
        ax.bar(xpos + (k - 1.5) * w, cnt, w, color=col, label=vname)
    ax.set_xticks(xpos); ax.set_xticklabels(labels); ax.set_ylim(0, N_IC)
    ax.set_ylabel(f"unstable runs (of {N_IC})")
    ax.set_title("(d) Instability count per variant"); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    # (e) from-t0 control
    ax = axes[1, 1]
    f0 = r.get("from_t0", {})
    labels0 = [l for l, _ in NUS_EXT if l in f0]
    xp0 = np.arange(len(labels0))
    for k, (vname, col) in enumerate(vc.items()):
        y, uns = [], []
        for l in labels0:
            e = f0[l][vname]
            y.append(np.nan if e["unstable"] else e["full_traj_err"])
            uns.append(e["unstable"])
        ax.bar(xp0 + (k - 1.5) * w, y, w, color=col, label=vname)
        for xi, u in zip(xp0, uns):
            if u:
                ax.text(xi + (k - 1.5) * w, 1e-4, "UNSTABLE", rotation=90,
                        fontsize=6, color=col, ha="center", va="bottom")
    ax.set_yscale("log"); ax.set_xticks(xp0); ax.set_xticklabels(labels0)
    ax.set_ylabel("full-trajectory rel $L_2$ from $t=0$")
    ax.set_title("(e) From-t0 control: failure is scheme fidelity,\nnot the restart index")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")

    # (f) error over time at harshest nu (canonical, rough)
    ax = axes[1, 2]
    label, nu = ("1/(1600pi)", 1.0 / (1600 * np.pi))
    cache = base.TRUTH_CACHE.format(label=label.replace("/", "_"))
    ic = np.sin(np.pi * X)
    truth = np.load(cache)["truth"] if os.path.exists(cache) else solve_v(ic, 0, nu, base.DT_TRUTH, True, True)
    state = base.rough_state(truth[i0], base.ROUGH_AMP, base.ROUGH_SEED)
    for vname, (mask_on, nyq_on) in VARIANTS.items():
        traj = solve_v(state, i0, nu, DT_TARGET, mask_on, nyq_on)
        with np.errstate(invalid="ignore"):
            c = (np.sqrt(((traj - truth[i0:]) ** 2).sum(-1))
                 / (np.sqrt((truth[i0:] ** 2).sum(-1)) + EPS))
        ax.plot(TGRID[i0:], c, color=vc[vname], label=vname)
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel("rel $L_2$ error")
    ax.set_title(f"(f) Error over time, $\\nu$ = {label}, rough handoff")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(FIGDIR, "viscosity_reanchor_stress.png")
    fig.savefig(p, dpi=150)
    print(f"[saved] {p}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "truth":
        ext_truth(int(sys.argv[2]))
    elif cmd == "restarts":
        ext_restarts(int(sys.argv[2]))
    elif cmd == "fromt0":
        ext_fromt0(int(sys.argv[2]))
    elif cmd == "figure":
        figure()
