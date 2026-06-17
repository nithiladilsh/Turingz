import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fno

OUT = fno.OUT
JSON_PATH = os.path.join(OUT, "mode_sweep.json")
PLOT_PATH = os.path.join(OUT, "mode_sweep.png")


def run(modes_list, seeds, epochs, n_train, n_val):
    fno.EPOCHS = epochs
    fno.WIDTH, fno.LAYERS = 32, 4
    dev = fno.dev

    U, ICs, x, t, te, Tmax = fno.load_data()
    tr_mask, ex_mask = t <= te, t > te
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    t_block = t[tr_mask].to(dev)

    rows = []
    for nm in modes_list:
        fno.MODES_T = fno.MODES_X = nm
        iv, ev, n_params = [], [], 0
        t0 = time.perf_counter()
        for sd in seeds:
            model, fl, (mean, std) = fno.train_one(sd, U, ICs, x, t, te, Tmax, train_idx)
            n_params = int(sum(p.numel() for p in model.parameters()))
            m = lambda mask, idx: fno.rel_l2_masked(model, ICs, U, x.to(dev), t.to(dev),
                                                    t_block, mask, idx, Tmax, mean, std)
            iv.append(m(tr_mask, val_idx)); ev.append(m(ex_mask, val_idx))
        iv, ev = np.array(iv), np.array(ev)
        rows.append({"n_modes": nm, "n_parameters": n_params,
                     "wall_s": time.perf_counter() - t0,
                     "val_in_dist_mean": float(iv.mean()), "val_in_dist_std": float(iv.std(ddof=1)),
                     "val_extrap_mean": float(ev.mean()), "val_extrap_std": float(ev.std(ddof=1)),
                     "val_in_dist_per_seed": iv.tolist(), "val_extrap_per_seed": ev.tolist()})
        print("  modes=%2d  params=%8d  val_in=%.4f±%.4f  val_extrap=%.4f±%.4f" %
              (nm, n_params, iv.mean(), iv.std(ddof=1), ev.mean(), ev.std(ddof=1)), flush=True)
    return rows


def plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    mm = [r["n_modes"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].errorbar(mm, [r["val_in_dist_mean"] for r in rows],
                   yerr=[r["val_in_dist_std"] for r in rows], fmt="o-", capsize=3, label="val in-dist")
    ax[0].errorbar(mm, [r["val_extrap_mean"] for r in rows],
                   yerr=[r["val_extrap_std"] for r in rows], fmt="s--", capsize=3, label="val extrap")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("retained Fourier modes"); ax[0].set_ylabel("relative L2")
    ax[0].grid(alpha=.3, which="both"); ax[0].legend()
    ax[1].plot(mm, [r["n_parameters"] / 1e3 for r in rows], "d-")
    ax[1].set_xlabel("retained Fourier modes"); ax[1].set_ylabel("parameters (x1e3)")
    ax[1].grid(alpha=.3)
    plt.tight_layout(); os.makedirs(OUT, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight"); plt.close(fig)


def main(full=False):
    if full:
        modes, seeds, epochs, n_train, n_val = [4, 8, 16, 32], [0, 1, 2, 3, 4], 150, fno.N_TRAIN, fno.N_VAL
    else:
        modes, seeds, epochs, n_train, n_val = [4, 8, 16], [0, 1], 4, 16, 8
    rows = run(modes, seeds, epochs, n_train, n_val)
    means = np.array([r["val_in_dist_mean"] for r in rows])
    best = rows[int(means.argmin())]
    thr = best["val_in_dist_mean"] + best["val_in_dist_std"]
    rec = next(r["n_modes"] for r in sorted(rows, key=lambda r: r["n_modes"])
               if r["val_in_dist_mean"] <= thr)
    out = {"modes": modes, "seeds": seeds, "epochs": epochs, "selection_metric": "val_in_dist",
           "per_mode": rows, "best_n_modes": best["n_modes"], "recommended_n_modes": rec,
           "rule": "smallest m with mean validation in-distribution rel L2 <= best_mean + best_std"}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(JSON_PATH, "w"), indent=2)
    plot(rows)
    print("\n  best n_modes=%d  recommended n_modes=%d  ->  %s" %
          (best["n_modes"], rec, JSON_PATH))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    main(full=ap.parse_args().full)
