import time
import tracemalloc
import statistics as stats


def time_callable(fn, repeats=15, warmup=3, inner=1):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    ms = sorted(s * 1e3 for s in samples)
    n = len(ms)
    q1 = ms[max(0, n // 4)]
    q3 = ms[min(n - 1, (3 * n) // 4)]
    return {
        "median_ms": float(stats.median(ms)),
        "iqr_ms": float(q3 - q1),
        "min_ms": float(ms[0]),
        "mean_ms": float(stats.fmean(ms)),
        "std_ms": float(stats.pstdev(ms)) if n > 1 else 0.0,
        "repeats": n,
    }


def peak_memory(fn):
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 ** 2)


def profile_op(name, fn, repeats=15, warmup=3, inner=1, measure_mem=True):
    rec = {"op": name}
    rec.update(time_callable(fn, repeats=repeats, warmup=warmup, inner=inner))
    if measure_mem:
        try:
            rec["peak_mem_mib"] = float(peak_memory(fn))
        except Exception as e:
            rec["peak_mem_mib"] = None
            rec["mem_error"] = str(e)
    return rec


def profile_numerical_units(G, ic, solvers=("fdm", "spectral"),
                            step_repeats=15, full_repeats=5):
    out = []
    j_mid = G.T_GRID.shape[0] // 2
    t0, t1 = G.T_GRID[j_mid - 1], G.T_GRID[j_mid]
    ch = G.colehopf_solve(ic)
    u_mid = ch[j_mid - 1].copy()
    for name in solvers:
        adv = G.NUMERICAL_SOLVERS[name]["advance"]
        sol = G.NUMERICAL_SOLVERS[name]["solve"]
        out.append(profile_op(f"{name}.step", lambda: adv(u_mid, t0, t1),
                               repeats=step_repeats))
        out.append(profile_op(f"{name}.full_solve", lambda: sol(ic),
                               repeats=full_repeats, warmup=1))
    return out


def profile_ml_step(surrogate, repeats=30, warmup=5):
    fn = surrogate.step_cost_fn()
    if fn is None:
        return None
    return profile_op(f"{surrogate.name}.ml_step", fn, repeats=repeats,
                      warmup=warmup, measure_mem=False)


def profile_corrector(G, ic, solver="spectral", lengths=(1, 2, 4, 8, 16),
                      repeats=7):
    adv = G.NUMERICAL_SOLVERS[solver]["advance"]
    ch = G.colehopf_solve(ic)
    j0 = int(G.TRAIN_MASK.sum())
    out = []
    for Kc in lengths:
        j_end = min(j0 + Kc, G.T_GRID.shape[0] - 1)

        def run(j0=j0, j_end=j_end):
            u = ch[j0].copy()
            for j in range(j0 + 1, j_end + 1):
                u = adv(u, G.T_GRID[j - 1], G.T_GRID[j])
            return u
        rec = profile_op(f"{solver}.corrector_K{Kc}", run, repeats=repeats, warmup=1)
        rec["K_steps"] = int(j_end - j0)
        out.append(rec)
    return out
