"""
Handoff continuity & switch-transient stability diagnostic (Module 2).
Author: Dharmapala R.D. (214050V)

PURPOSE
    Answer the examiner question "did you just call a solver?" with evidence.
    The validated coupling is a HARD one-way switch, but it is NOT a discontinuous
    replacement of one trajectory by an unrelated state: the numerical continuation is
    seeded with EXACTLY the FNO-predicted field at the switch index, so the state value
    at the handoff is continuous BY CONSTRUCTION.  This diagnostic quantifies that and
    shows the transition is physically stable, so the dominant hybrid error is the
    quality of the learned STATE at the switch, not a handoff discontinuity.

    It does NOT change the coupling method and does NOT claim any blended coupling.

METRICS (per switch time t_s, averaged over held-out FNO waves)
    1. switch-state jump            ||hybrid[i]-fno[i]|| / ||fno[i]||      (~0 by construction)
    2. one-step temporal transient  ||hybrid[i+1]-hybrid[i]|| vs ||fno[i+1]-fno[i]||
    3. Burgers PDE residual (RMS) immediately before vs after the handoff
    4. residual spike ratio         residual_after / residual_before
    5. field energy before vs after the handoff
    6. hybrid tail error over [t_s, 2]
    7. numerical-work fraction

NEGATIVE CONTROL (clearly labelled)
    A deliberately careless restart that skips the spectral safety steps
    (no 2/3 de-alias mask, no Nyquist zeroing).  Included so the value of the
    correct re-anchor is measurable; most informative on rough / high-frequency
    states (run the OOD variant on a torch machine).

OUTPUTS
    results/module2/figures/handoff_stability_diagnostic.json
    results/module2/figures/handoff_stability_diagnostic.png
"""
import os, sys, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from restart_spectral import (solve_from, nearest_index, TGRID, X, NX, NU, K,
                              MASK, DT_TARGET)

PRED = os.environ.get("MODULE2_PRED", os.path.join(ROOT, "results", "eval", "predictions.npz"))
FIGDIR = os.path.join(ROOT, "results", "module2", "figures")
SWITCHES = [1.0, 1.2, 1.4, 1.6, 1.8]
EPS = 1e-12
N_WAVES = int(os.environ.get("M2_N_WAVES", "0"))     # 0 = all


# ---------- spectral operators (same wavenumbers as the solver) ----------
def d1(u): return np.fft.irfft(1j * K * np.fft.rfft(u, axis=-1), n=NX, axis=-1)
def d2(u): return np.fft.irfft(-(K ** 2) * np.fft.rfft(u, axis=-1), n=NX, axis=-1)

def pde_residual_rms(u_prev, u_cur, dt):
    """RMS of the Burgers residual u_t + u u_x - nu u_xx using two frames."""
    ut = (u_cur - u_prev) / dt
    r = ut + u_cur * d1(u_cur) - NU * d2(u_cur)
    return np.sqrt((r ** 2).mean(axis=-1))

def energy(u): return 0.5 * (X[1] - X[0]) * (u ** 2).sum(axis=-1)

def tail_rel_l2(traj, ref, t, i0):
    c = np.sqrt(((traj[:, i0:] - ref[:, i0:]) ** 2).sum(-1)) / (np.sqrt((ref[:, i0:] ** 2).sum(-1)) + EPS)
    return np.trapezoid(c, t[i0:], axis=-1) / (t[-1] - t[i0] + EPS)


# ---------- negative control: careless restart (no de-alias mask, no Nyquist zero) ----------
def _rhs_careless(uh):
    u = np.fft.irfft(uh, n=NX, axis=-1)                 # NB: no MASK applied
    return -0.5j * K * np.fft.rfft(u * u, axis=-1)

def _step_careless(uh, E, E2, h):
    k1 = _rhs_careless(uh)
    k2 = _rhs_careless(E2 * uh + 0.5 * h * E2 * k1)
    k3 = _rhs_careless(E2 * uh + 0.5 * h * k2)
    k4 = _rhs_careless(E * uh + h * E2 * k3)
    return E * uh + h / 6 * (E * k1 + 2 * E2 * k2 + 2 * E2 * k3 + k4)   # NB: no Nyquist zeroing

def solve_from_careless(u0, i_start):
    U0 = np.atleast_2d(u0).astype(np.float64)
    B = U0.shape[0]
    out = np.empty((B, len(TGRID) - i_start, NX)); out[:, 0] = U0
    uh = np.fft.rfft(U0, axis=-1)
    for j in range(i_start + 1, len(TGRID)):
        h = TGRID[j] - TGRID[j - 1]; m = max(1, round(h / DT_TARGET)); h = h / m
        E = np.exp(-NU * K ** 2 * h); E2 = np.exp(-NU * K ** 2 * h * 0.5)
        for _ in range(m):
            uh = _step_careless(uh, E, E2, h)
        out[:, j - i_start] = np.fft.irfft(uh, n=NX, axis=-1)
    return out


def build_hybrid(fno, i, careless=False):
    tail = (solve_from_careless(fno[:, i], i) if careless else solve_from(fno[:, i], i))
    out = fno.copy(); out[:, i:] = tail
    return out


def main():
    d = np.load(PRED)
    t = d["t"]; true = d["u_true_eval"].astype(float); fno = d["FNO_eval"].astype(float)
    n = fno.shape[0] if N_WAVES == 0 else min(N_WAVES, fno.shape[0])
    true, fno = true[:n], fno[:n]
    nt = len(t); i1 = nearest_index(1.0)
    print(f"handoff stability diagnostic | n={n} waves | switches={SWITCHES}", flush=True)

    rows = []
    for ts in SWITCHES:
        i = nearest_index(ts)
        H = build_hybrid(fno, i)                          # verified re-anchor
        Hc = build_hybrid(fno, i, careless=True)          # negative control
        dt = float(t[i + 1] - t[i])

        jump = np.linalg.norm(H[:, i] - fno[:, i], axis=-1) / (np.linalg.norm(fno[:, i], axis=-1) + EPS)
        trans_h = np.linalg.norm(H[:, i + 1] - H[:, i], axis=-1)
        trans_f = np.linalg.norm(fno[:, i + 1] - fno[:, i], axis=-1)
        res_before = pde_residual_rms(fno[:, i - 1], fno[:, i], dt)         # ML side
        res_after = pde_residual_rms(H[:, i], H[:, i + 1], dt)              # numerical side
        res_after_c = pde_residual_rms(Hc[:, i], Hc[:, i + 1], dt)         # careless side
        e_before = energy(H[:, i]); e_after = energy(H[:, i + 1])
        hyb_tail = tail_rel_l2(H, true, t, i1)

        rows.append(dict(
            t_s=ts, numerical_work=float((nt - i) / nt),
            state_jump=float(jump.mean()),
            transient_hybrid=float(trans_h.mean()), transient_fno=float(trans_f.mean()),
            transient_ratio=float((trans_h / (trans_f + EPS)).mean()),
            residual_before=float(res_before.mean()), residual_after=float(res_after.mean()),
            residual_spike_ratio=float((res_after / (res_before + EPS)).mean()),
            residual_after_careless=float(res_after_c.mean()),
            energy_before=float(e_before.mean()), energy_after=float(e_after.mean()),
            hybrid_tail=float(hyb_tail.mean())))
        r = rows[-1]
        print("  t_s=%.1f  jump=%.2e  trans(h/f)=%.3f  resid(before->after)=%.3f->%.3f  "
              "careless_after=%.3f  hyb_tail=%.3f" %
              (ts, r["state_jump"], r["transient_ratio"], r["residual_before"],
               r["residual_after"], r["residual_after_careless"], r["hybrid_tail"]), flush=True)

    os.makedirs(FIGDIR, exist_ok=True)
    out = dict(note="Module 2 handoff continuity & stability diagnostic. Method unchanged "
                    "(verified hard-switch re-anchor). Negative control = careless restart "
                    "(no 2/3 de-alias mask, no Nyquist zeroing).",
               n_waves=int(n), switches=SWITCHES, rows=rows)
    jp = os.path.join(FIGDIR, "handoff_stability_diagnostic.json")
    with open(jp + ".tmp", "w") as f:
        json.dump(out, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(jp + ".tmp", jp)
    print("wrote", jp, flush=True)

    # ---- table ----
    print("\n%-5s %9s %10s %11s %11s %10s %9s" %
          ("t_s", "st_jump", "trans_r", "res_before", "res_after", "carel_aft", "hyb_tail"))
    print("-" * 72)
    for r in rows:
        print("%-5.1f %9.2e %10.3f %11.3f %11.3f %10.3f %9.3f" %
              (r["t_s"], r["state_jump"], r["transient_ratio"], r["residual_before"],
               r["residual_after"], r["residual_after_careless"], r["hybrid_tail"]))

    # ---- plot: PDE residual around the switch (representative wave, t_s=1.4) ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        i = nearest_index(1.4); w = 8
        H = build_hybrid(fno, i)
        lo, hi = max(1, i - w), min(nt - 1, i + w)
        js = list(range(lo, hi))
        res_fno = [float(pde_residual_rms(fno[0, j - 1], fno[0, j], float(t[j] - t[j - 1]))) for j in js]
        res_hyb = [float(pde_residual_rms(H[0, j - 1], H[0, j], float(t[j] - t[j - 1]))) for j in js]
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        ax.plot([t[j] for j in js], res_fno, "o-", color="#d1495b", lw=2, label="pure FNO (physics residual)")
        ax.plot([t[j] for j in js], res_hyb, "s-", color="#2e7d32", lw=2, label="hybrid re-anchor")
        ax.axvline(t[i], ls="--", color="k", lw=1); ax.text(t[i] + 0.002, ax.get_ylim()[1] * 0.9, "handoff", fontsize=9)
        ax.set(xlabel="time t", ylabel="Burgers residual (RMS)",
               title="Handoff is continuous & stable: residual drops after re-anchor, no spike (t_s=1.4)")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "handoff_stability_diagnostic.png"), dpi=140)
        print("wrote handoff_stability_diagnostic.png", flush=True)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
