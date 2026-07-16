"""
Handoff Viability Gate (HVG) -- Module 2 mechanism prototype.
Author: Dharmapala R.D. (214050V)

WHAT IT IS
    A reference-free, state-quality-aware gate that, at a candidate switch, predicts
    whether re-anchoring from the current ML state will yield a useful hybrid -- WITHOUT
    ground truth and WITHOUT running the expensive numerical continuation.

    It turns this measured Module 2 law into a decision algorithm:
        hybrid accuracy is controlled by the ML state quality at the switch;
        the numerical continuation itself adds almost no error.

CLEAN MODULE BOUNDARY (no overlap)
    M1 trust : "is the ML model becoming unreliable?"      -> trust score / switch flag
    M2 HVG   : "if we switch now, is THIS state salvageable?" -> predicted hybrid error / viability
    M3 control: "can we afford this correction?"           -> cost-aware schedule
    M1 decides failure; HVG estimates handoff usefulness; M3 decides cost. Different questions.

INPUT   u_ML(x, t_s) only (the ML field at the candidate switch)
OUTPUT  {"predicted_hybrid_error": float, "handoff_viable": bool, "viability_score": float}

This is a PROTOTYPE / de-risk: it fits a transparent model on held-out waves with
leave-one-wave-out validation and reports honestly whether reference-free features
predict the real hybrid error. If weak, it stays future work.

Outputs:
    results/module2/figures/handoff_viability_gate.json
    results/module2/figures/handoff_viability_gate.png
"""
import os, sys, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from restart_spectral import solve_from, nearest_index, TGRID, X, NX, NU, K, DX

PRED = os.environ.get("MODULE2_PRED", os.path.join(ROOT, "results", "eval", "predictions.npz"))
FIGDIR = os.path.join(ROOT, "results", "module2", "figures")
SWITCHES = [1.0, 1.2, 1.4, 1.6, 1.8]
EPS = 1e-12
CUTOFF = NX // 3
VIABLE_ERR = 0.10                         # abs-error threshold from the viability rule
FEATURES = ["pde_residual", "highfreq_frac", "spectral_slope", "grad_energy", "total_energy", "t_s"]


def d1(u): return np.fft.irfft(1j * K * np.fft.rfft(u), n=NX)
def d2(u): return np.fft.irfft(-(K ** 2) * np.fft.rfft(u), n=NX)

def state_features(u_prev, u_cur, dt, t_s):
    """Reference-free features from the ML field (and its previous frame for u_t)."""
    res = (u_cur - u_prev) / dt + u_cur * d1(u_cur) - NU * d2(u_cur)
    pde_residual = float(np.sqrt((res ** 2).mean()))
    p = np.abs(np.fft.rfft(u_cur)) ** 2
    highfreq_frac = float(p[CUTOFF + 1:].sum() / (p.sum() + EPS))
    kk = np.arange(1, len(p))
    slope = float(np.polyfit(np.log10(kk), np.log10(p[1:] + EPS), 1)[0])   # spectral rolloff
    grad_energy = float((d1(u_cur) ** 2).mean())
    total_energy = float(0.5 * DX * (u_cur ** 2).sum())
    return [pde_residual, highfreq_frac, slope, grad_energy, total_energy, float(t_s)]


def tail_rel_l2(traj, ref, t, i0):
    c = np.sqrt(((traj[i0:] - ref[i0:]) ** 2).sum(-1)) / (np.sqrt((ref[i0:] ** 2).sum(-1)) + EPS)
    return float(np.trapezoid(c, t[i0:]) / (t[-1] - t[i0] + EPS))


# ---------- transparent numpy models ----------
def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + EPS
    return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd

def linreg_fit(X, y):
    A = np.hstack([np.ones((len(X), 1)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta
def linreg_pred(beta, X):
    return np.hstack([np.ones((len(X), 1)), X]) @ beta

def logreg_fit(X, y, iters=2000, lr=0.3):
    A = np.hstack([np.ones((len(X), 1)), X]); w = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w)); w -= lr * (A.T @ (p - y)) / len(A)
    return w
def logreg_pred(w, X):
    return 1 / (1 + np.exp(-(np.hstack([np.ones((len(X), 1)), X]) @ w)))

def pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + EPS))

def auc(scores, labels):
    labels = np.asarray(labels); pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    d = np.load(PRED)
    t = d["t"]; true = d["u_true_eval"].astype(float); fno = d["FNO_eval"].astype(float)
    n = fno.shape[0]; i1 = nearest_index(1.0)

    # build samples: (wave, t_s) -> reference-free features + true hybrid error label
    feats, labels_err, groups, meta = [], [], [], []
    for ts in SWITCHES:
        i = nearest_index(ts); dt = float(t[i] - t[i - 1])
        H = solve_from(fno[:, i], i)                       # batched hybrid tails (local time)
        tail_t = t[i:]
        for j in range(n):
            feats.append(state_features(fno[j, i - 1], fno[j, i], dt, ts))
            cc = np.sqrt(((H[j] - true[j, i:]) ** 2).sum(-1)) / (np.sqrt((true[j, i:] ** 2).sum(-1)) + EPS)
            labels_err.append(float(np.trapezoid(cc, tail_t) / (tail_t[-1] - tail_t[0] + EPS)))
            groups.append(j); meta.append((j, ts))
    Xf = np.array(feats); yerr = np.array(labels_err); groups = np.array(groups)
    yvia = (yerr < VIABLE_ERR).astype(float)
    print(f"HVG prototype | {len(yerr)} samples ({n} waves x {len(SWITCHES)} switch times) | "
          f"viable(<{VIABLE_ERR:.0%}): {int(yvia.sum())}/{len(yvia)}", flush=True)

    # leave-one-wave-out CV, full features vs t_s-only baseline
    def loo(cols):
        pe = np.zeros(len(yerr)); ps = np.zeros(len(yerr))
        for j in range(n):
            tr, te = groups != j, groups == j
            Xtr, Xte, _, _ = standardize(Xf[tr][:, cols], Xf[te][:, cols])
            pe[te] = linreg_pred(linreg_fit(Xtr, yerr[tr]), Xte)
            ps[te] = logreg_pred(logreg_fit(Xtr, yvia[tr]), Xte)
        return pe, ps
    all_cols = list(range(len(FEATURES))); ts_col = [FEATURES.index("t_s")]
    pe_full, ps_full = loo(all_cols)
    pe_ts, _ = loo(ts_col)

    corr_full = pearson(pe_full, yerr); mae_full = float(np.abs(pe_full - yerr).mean())
    corr_ts = pearson(pe_ts, yerr); mae_ts = float(np.abs(pe_ts - yerr).mean())
    acc = float(((ps_full > 0.5) == (yvia == 1)).mean()); AUC = auc(ps_full, yvia)

    # does state info help BEYOND switch time? partial correlation at fixed t_s
    within = []
    for ts in SWITCHES:
        m = np.array([mt[1] == ts for mt in meta])
        if m.sum() > 2: within.append(pearson(pe_full[m], yerr[m]))
    within_corr = float(np.nanmean(within))

    # interpretable coefficients on standardised full data
    Xs = (Xf - Xf.mean(0)) / (Xf.std(0) + EPS)
    coef = linreg_fit(Xs, yerr)[1:]
    coefs = {FEATURES[k]: float(coef[k]) for k in range(len(FEATURES))}

    out = dict(
        note="Handoff Viability Gate (HVG) prototype. Reference-free state-quality features "
             "predict the hybrid tail error at a candidate switch. Leave-one-wave-out CV. "
             "M2 answers handoff usefulness, distinct from M1 trust (failure) and M3 cost.",
        n_samples=len(yerr), n_waves=int(n), viable_threshold=VIABLE_ERR,
        regression=dict(full_features=dict(pearson_r=corr_full, mae=mae_full),
                        t_s_only_baseline=dict(pearson_r=corr_ts, mae=mae_ts),
                        within_switchtime_r=within_corr),
        classification=dict(accuracy=acc, auc=AUC),
        feature_coefficients=coefs, features=FEATURES)
    os.makedirs(FIGDIR, exist_ok=True)
    jp = os.path.join(FIGDIR, "handoff_viability_gate.json")
    with open(jp + ".tmp", "w") as f:
        json.dump(out, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(jp + ".tmp", jp)

    print("\n--- HVG predictive performance (leave-one-wave-out) ---")
    print("regression full features : r=%.3f  MAE=%.4f" % (corr_full, mae_full))
    print("regression t_s-only      : r=%.3f  MAE=%.4f  (baseline)" % (corr_ts, mae_ts))
    print("state features beyond t_s: within-switch-time r=%.3f" % within_corr)
    print("viability classifier     : acc=%.2f  AUC=%.2f" % (acc, AUC))
    print("feature weights (std)    :", {k: round(v, 3) for k, v in coefs.items()})
    print("wrote", jp, flush=True)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.6, 5.0))
        sc = ax.scatter(yerr, pe_full, c=Xf[:, FEATURES.index("t_s")], cmap="viridis",
                        s=45, edgecolor="k", linewidth=0.4)
        lim = [0, max(yerr.max(), pe_full.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1, label="ideal")
        ax.axhline(VIABLE_ERR, color="grey", ls=":"); ax.axvline(VIABLE_ERR, color="grey", ls=":")
        ax.set(xlabel="actual hybrid tail error", ylabel="HVG predicted error",
               title="Handoff Viability Gate: reference-free prediction vs truth (r=%.2f)" % corr_full)
        fig.colorbar(sc, label="switch time t_s"); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "handoff_viability_gate.png"), dpi=140)
        print("wrote handoff_viability_gate.png", flush=True)
    except Exception as e:
        print("plot skipped:", e)


# ---- the mechanism's inference API (what M3/the runtime would call) ----
def evaluate_handoff_state(u_prev, u_ml, dt, t_s, model):
    x = (np.array(state_features(u_prev, u_ml, dt, t_s)) - model["mu"]) / model["sd"]
    err = float(linreg_pred(model["beta"], x[None, :])[0])
    score = float(logreg_pred(model["w"], x[None, :])[0])
    return {"predicted_hybrid_error": err, "handoff_viable": bool(score > 0.5), "viability_score": score}


if __name__ == "__main__":
    main()
