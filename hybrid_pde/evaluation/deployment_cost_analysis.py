import os, sys, json, time, gc, threading, platform, subprocess
os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import psutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "colehopf", "burgers_colehopf.pt")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "deployment")
os.makedirs(OUT, exist_ok=True)

torch.set_default_device("cpu")
torch.set_num_threads(1)
REPEAT, WARMUP, TP_REPEAT, SC_REPEAT = 30, 5, 10, 8
NUM_REPEAT, NUM_SC_REPEAT, NUM_WARMUP = 10, 2, 1
EVAL = np.arange(900, 910)
NT_SWEEP = [25, 50, 100, 200, 400]
PINN_WARMUP_ITERS, PINN_MEASURE_ITERS, PINN_ADAM_ITERS = 100, 300, 15000
MODELS_ML = ["PINN", "FNO", "DeepONet"]
MODELS_NUM = ["ColeHopf", "FDM", "Spectral"]
MODELS = MODELS_ML + MODELS_NUM

d = torch.load(DATA, weights_only=False, map_location="cpu")
U, ICs, x, t = d["u"].numpy(), d["ICs"], d["x"], d["t"]
te = float(d["t_train_end"])
nx = U.shape[-1]
xn, tn = x.numpy(), t.numpy()
UE = U[EVAL]

NU = 1.0 / (100 * np.pi)
L = 2.0
dx = L / nx
KX = 2 * np.pi * np.arange(nx // 2 + 1) / L
KMASK = np.arange(nx // 2 + 1) <= nx // 3
DT_SPEC = 1e-4
X_EXT = np.concatenate([xn - L, xn, xn + L])
DIFF = xn[:, None] - X_EXT


def timeit(fn, repeat=REPEAT, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    s = np.empty(repeat)
    for i in range(repeat):
        a = time.perf_counter()
        fn()
        s[i] = time.perf_counter() - a
    return float(s.mean()), float(s.std()), float(np.median(s))


def _measure_rss(model):
    p = psutil.Process()
    box = {"peak": 0, "go": True}

    def poll():
        while box["go"]:
            box["peak"] = max(box["peak"], p.memory_info().rss)
    th = threading.Thread(target=poll, daemon=True)
    gc.collect()
    th.start()
    if model == "FNO":
        fno_infer(load_fno(), EVAL[:1], tn)
    elif model == "DeepONet":
        deeponet_infer(load_deeponet(), EVAL[:1], tn)
    elif model == "PINN":
        pinn_infer(load_pinn(900), EVAL[:1], tn)
    else:
        num_infer(model, EVAL[:1], tn)
    box["go"] = False
    th.join()
    print(f"PEAK_MB {box['peak'] / 1e6:.3f}")


def rss_subprocess(model):
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--rss", model],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("PEAK_MB"):
            return float(line.split()[1])
    return float("nan")


def fit_exp(N, y):
    return float(np.polyfit(np.log(N), np.log(y), 1)[0])


def load_fno():
    from neuralop.models import FNO
    cfg = torch.load(os.path.join(RES, "fno", "fno_config.pt"), map_location="cpu", weights_only=False)
    m = FNO(n_modes=(cfg["n_modes"],), hidden_channels=cfg["hidden_channels"], in_channels=3, out_channels=1)
    m.load_state_dict(torch.load(os.path.join(RES, "fno", "fno.pt"), map_location="cpu", weights_only=False))
    m.eval()
    return m.to("cpu")


def fno_infer(m, idx, t_):
    out = np.zeros((len(idx), len(t_), nx), np.float32)
    tt = torch.tensor(t_, dtype=torch.float32)
    with torch.no_grad():
        for r, s in enumerate(idx):
            inp = torch.stack([ICs[s].unsqueeze(0).repeat(len(t_), 1),
                               tt.view(-1, 1).repeat(1, nx),
                               x.view(1, -1).repeat(len(t_), 1)], dim=1)
            out[r] = m(inp).squeeze(1).numpy()
    return out


def load_deeponet():
    from hybrid_pde.solvers.ml.deepOnet.deeponet import load_model
    torch.set_default_device("cpu")
    s = load_model(os.path.join(RES, "deeponet"), xn)
    s.net = s.net.to("cpu")
    return s


def deeponet_infer(s, idx, t_):
    return s.predict_grid(ICs.numpy()[idx], xn, t_)


def load_pinn(i):
    import deepxde as dde
    torch.set_default_device("cpu")
    net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal")
    net.load_state_dict(torch.load(os.path.join(RES, "pinn", f"pinn_ic{i}.pt"), map_location="cpu"))
    net.eval()
    return net.to("cpu")


def pinn_infer(net, idx, t_):
    X, T = np.meshgrid(xn, t_)
    XT = torch.tensor(np.stack([X.ravel(), T.ravel()], 1), dtype=torch.float32)
    out = np.zeros((len(idx), len(t_), nx), np.float32)
    with torch.no_grad():
        for r in range(len(idx)):
            out[r] = net(XT).numpy().reshape(len(t_), nx)
    return out


def pinn_train_cost(i):
    import deepxde as dde
    torch.set_default_device("cpu")
    NU = 1.0 / (100 * np.pi)
    geomtime = dde.geometry.GeometryXTime(dde.geometry.Interval(-1, 1), dde.geometry.TimeDomain(0, 1.0))

    def pde(z, u):
        u_t = dde.grad.jacobian(u, z, i=0, j=1)
        u_x = dde.grad.jacobian(u, z, i=0, j=0)
        u_xx = dde.grad.hessian(u, z, i=0, j=0)
        return u_t + u * u_x - NU * u_xx
    xe = np.concatenate([xn, [1.0]])
    ye = np.concatenate([ICs.numpy()[i], [ICs.numpy()[i][0]]])
    icf = lambda Z: np.interp(Z[:, 0], xe, ye)[:, None]
    ic = dde.icbc.IC(geomtime, icf, lambda _, on: on)
    bc = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on: on)
    data = dde.data.TimePDE(geomtime, pde, [ic, bc], num_domain=2540, num_boundary=80, num_initial=160)
    net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal").to("cpu")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.train(iterations=PINN_WARMUP_ITERS, display_every=PINN_WARMUP_ITERS)
    a = time.perf_counter()
    model.train(iterations=PINN_MEASURE_ITERS, display_every=PINN_MEASURE_ITERS)
    per_iter = (time.perf_counter() - a) / PINN_MEASURE_ITERS
    adam_s = per_iter * PINN_ADAM_ITERS
    model.compile("L-BFGS")
    b = time.perf_counter()
    model.train()
    lbfgs_s = time.perf_counter() - b
    return adam_s + lbfgs_s, per_iter, adam_s, lbfgs_s


def colehopf_solve(ic, t_):
    cumint = np.concatenate([np.zeros((len(ic), 1)),
                             np.cumsum(0.5 * (ic[:, :-1] + ic[:, 1:]) * dx, axis=1)], axis=1)
    a = -cumint / (2 * NU)
    pe = np.tile(np.exp(a - a.max(axis=1, keepdims=True)), (1, 3))
    out = np.empty((len(ic), len(t_), nx), np.float32)
    out[:, 0] = ic
    for j in range(1, len(t_)):
        K = np.exp(-DIFF ** 2 / (4 * NU * t_[j]))
        out[:, j] = (pe @ (DIFF * K).T) / (pe @ K.T) / t_[j]
    return out


def fdm_step(u, h):
    up, um = np.roll(u, -1), np.roll(u, 1)
    u_x = np.where(u >= 0, (u - um) / dx, (up - u) / dx)
    return u + h * (-u * u_x + NU * (up - 2 * u + um) / dx ** 2)


def fdm_solve(u0, t_):
    dt = 0.4 * min(dx / (np.abs(u0).max() + 1e-9), dx ** 2 / (2 * NU))
    out = np.empty((len(t_), nx), np.float32)
    out[0] = u0
    u, tc = u0.copy(), 0.0
    for kk in range(1, len(t_)):
        while tc < t_[kk] - 1e-12:
            h = min(dt, t_[kk] - tc)
            u = fdm_step(u, h)
            tc += h
        out[kk] = u
    return out


def spec_rhs(uh):
    u = np.fft.irfft(uh * KMASK, n=nx)
    return -0.5j * KX * np.fft.rfft(u * u)


def spec_solve(u0, t_):
    uh = np.fft.rfft(u0)
    out = np.empty((len(t_), nx), np.float32)
    out[0] = u0
    for j in range(1, len(t_)):
        h = t_[j] - t_[j - 1]
        m = max(1, round(h / DT_SPEC))
        h /= m
        E, E2 = np.exp(-NU * KX ** 2 * h), np.exp(-NU * KX ** 2 * h * 0.5)
        for _ in range(m):
            k1 = spec_rhs(uh)
            k2 = spec_rhs(E2 * uh + 0.5 * h * E2 * k1)
            k3 = spec_rhs(E2 * uh + 0.5 * h * k2)
            k4 = spec_rhs(E * uh + h * E2 * k3)
            uh = E * uh + h / 6 * (E * k1 + 2 * E2 * k2 + 2 * E2 * k3 + k4)
            uh[-1] = 0.0
        out[j] = np.fft.irfft(uh * KMASK, n=nx)
    return out


def num_infer(name, idx, t_):
    ic = ICs.numpy()[idx]
    if name == "ColeHopf":
        return colehopf_solve(ic, t_)
    f = fdm_solve if name == "FDM" else spec_solve
    return np.stack([f(ic[r], t_) for r in range(len(idx))])


def err_split(pred):
    c = (np.linalg.norm(pred - UE, axis=2) / (np.linalg.norm(UE, axis=2) + 1e-12)).mean(0)
    return float(c[tn <= te].mean()) * 100, float(c[tn > te].mean()) * 100


rel = json.load(open(os.path.join(RES, "eval", "reliability_summary.json")))["reliability_unseen"]
disk = {"PINN": os.path.getsize(os.path.join(RES, "pinn", "pinn_ic900.pt")),
        "FNO": os.path.getsize(os.path.join(RES, "fno", "fno.pt")),
        "DeepONet": os.path.getsize(os.path.join(RES, "deeponet", "model.pt"))}

if len(sys.argv) > 2 and sys.argv[1] == "--rss":
    _measure_rss(sys.argv[2])
    sys.exit(0)

stats = {}

print("measuring FNO (inference, throughput, memory, scaling) ...", flush=True)
fno = load_fno()
stats["FNO"] = {"params": int(sum(p.numel() for p in fno.parameters()))}
lat, sd, med = timeit(lambda: fno_infer(fno, EVAL[:1], tn))
stats["FNO"]["infer_ms"], stats["FNO"]["infer_ms_std"], stats["FNO"]["infer_ms_median"] = lat * 1e3, sd * 1e3, med * 1e3
tp = timeit(lambda: fno_infer(fno, EVAL, tn), repeat=TP_REPEAT)[0]
stats["FNO"]["throughput_ic_s"] = len(EVAL) / tp
stats["FNO"]["rss_mb"] = rss_subprocess("FNO")
sc = [timeit(lambda nt_=k: fno_infer(fno, EVAL[:1], np.linspace(0, 2, nt_)), repeat=SC_REPEAT)[0] for k in NT_SWEEP]
stats["FNO"]["scaling"] = {"N": [k * nx for k in NT_SWEEP], "t": sc, "exp": fit_exp(np.array(NT_SWEEP) * nx, np.array(sc))}
stats["FNO"]["deploy_s"] = lat

print("measuring DeepONet ...", flush=True)
don = load_deeponet()
stats["DeepONet"] = {"params": don.num_parameters()}
lat, sd, med = timeit(lambda: deeponet_infer(don, EVAL[:1], tn))
stats["DeepONet"]["infer_ms"], stats["DeepONet"]["infer_ms_std"], stats["DeepONet"]["infer_ms_median"] = lat * 1e3, sd * 1e3, med * 1e3
tp = timeit(lambda: deeponet_infer(don, EVAL, tn), repeat=TP_REPEAT)[0]
stats["DeepONet"]["throughput_ic_s"] = len(EVAL) / tp
stats["DeepONet"]["rss_mb"] = rss_subprocess("DeepONet")
sc = [timeit(lambda nt_=k: deeponet_infer(don, EVAL[:1], np.linspace(0, 2, nt_)), repeat=SC_REPEAT)[0] for k in NT_SWEEP]
stats["DeepONet"]["scaling"] = {"N": [k * nx for k in NT_SWEEP], "t": sc, "exp": fit_exp(np.array(NT_SWEEP) * nx, np.array(sc))}
stats["DeepONet"]["deploy_s"] = lat

print("measuring PINN inference + scaling ...", flush=True)
pnet = load_pinn(900)
stats["PINN"] = {"params": int(sum(p.numel() for p in pnet.parameters()))}
lat, sd, med = timeit(lambda: pinn_infer(pnet, EVAL[:1], tn))
stats["PINN"]["infer_ms"], stats["PINN"]["infer_ms_std"], stats["PINN"]["infer_ms_median"] = lat * 1e3, sd * 1e3, med * 1e3
stats["PINN"]["rss_mb"] = rss_subprocess("PINN")
sc = [timeit(lambda nt_=k: pinn_infer(pnet, EVAL[:1], np.linspace(0, 2, nt_)), repeat=SC_REPEAT)[0] for k in NT_SWEEP]
stats["PINN"]["scaling"] = {"N": [k * nx for k in NT_SWEEP], "t": sc, "exp": fit_exp(np.array(NT_SWEEP) * nx, np.array(sc))}
cache = os.path.join(OUT, "pinn_train_cost.json")
if os.path.exists(cache) and not os.environ.get("RETRAIN_PINN"):
    print("using cached PINN training cost (delete pinn_train_cost.json to re-measure)", flush=True)
    c = json.load(open(cache))
    train_s, per_iter, adam_s, lbfgs_s = c["train_s"], c["train_per_iter_s"], c["adam_s"], c["lbfgs_s"]
else:
    print("measuring PINN training cost (one-time, ~13 min) ...", flush=True)
    train_s, per_iter, adam_s, lbfgs_s = pinn_train_cost(900)
    json.dump({"train_s": train_s, "train_per_iter_s": per_iter, "adam_s": adam_s, "lbfgs_s": lbfgs_s},
              open(cache, "w"), indent=2)
stats["PINN"]["train_s"], stats["PINN"]["train_per_iter_s"] = train_s, per_iter
stats["PINN"]["adam_s"], stats["PINN"]["lbfgs_s"] = adam_s, lbfgs_s
stats["PINN"]["deploy_s"] = train_s + lat
stats["PINN"]["throughput_ic_s"] = 1.0 / stats["PINN"]["deploy_s"]

for name in MODELS_NUM:
    print(f"measuring {name} ...", flush=True)
    lat, sd, med = timeit(lambda nm=name: num_infer(nm, EVAL[:1], tn), repeat=NUM_REPEAT, warmup=NUM_WARMUP)
    sc = [timeit(lambda nm=name, k_=k: num_infer(nm, EVAL[:1], np.linspace(0, 2, k_)), repeat=NUM_SC_REPEAT, warmup=NUM_WARMUP)[0] for k in NT_SWEEP]
    ein, eex = err_split(num_infer(name, EVAL, tn))
    stats[name] = {"params": 0, "disk_mb": 0.0, "params_mb": 0.0,
                   "infer_ms": lat * 1e3, "infer_ms_std": sd * 1e3, "infer_ms_median": med * 1e3,
                   "throughput_ic_s": 1.0 / lat, "rss_mb": rss_subprocess(name),
                   "scaling": {"N": [k * nx for k in NT_SWEEP], "t": sc, "exp": fit_exp(np.array(NT_SWEEP) * nx, np.array(sc))},
                   "deploy_s": lat, "err_in": ein, "err_extrap": eex}

for m in MODELS_ML:
    stats[m]["disk_mb"] = disk[m] / 1e6
    stats[m]["params_mb"] = stats[m]["params"] * 4 / 1e6
    stats[m]["err_in"] = rel[m]["in_window"] * 100
    stats[m]["err_extrap"] = rel[m]["extrapolation"] * 100

C = {"PINN": "#d62728", "FNO": "#1f77b4", "DeepONet": "#2ca02c",
     "ColeHopf": "#9467bd", "FDM": "#8c564b", "Spectral": "#e377c2"}

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
xb = np.arange(len(MODELS))
ax[0].bar(xb, [stats[m]["infer_ms"] for m in MODELS], yerr=[stats[m]["infer_ms_std"] for m in MODELS],
          color=[C[m] for m in MODELS], capsize=4)
ax[0].set_yscale("log"); ax[0].set_xticks(xb); ax[0].set_xticklabels(MODELS)
ax[0].set_ylabel("per-IC inference latency (ms, log)"); ax[0].set_title("Runtime — full grid forward pass")
ax[0].grid(True, axis="y", alpha=0.3)
ax[1].bar(xb, [stats[m]["throughput_ic_s"] for m in MODELS], color=[C[m] for m in MODELS])
ax[1].set_yscale("log"); ax[1].set_xticks(xb); ax[1].set_xticklabels(MODELS)
ax[1].set_ylabel("deployment throughput (new ICs / s, log)"); ax[1].set_title("Throughput — new ICs served per second")
ax[1].grid(True, axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "1_runtime_throughput.png"), dpi=150); plt.close()

LBL = {"ColeHopf": (8, 9), "Spectral": (-58, -14), "FNO": (10, -4)}
plt.figure(figsize=(8, 5.5))
for m in MODELS:
    plt.scatter(stats[m]["deploy_s"], stats[m]["err_extrap"], s=160, color=C[m], zorder=3,
                edgecolor="k", label=f"{m} (extrap)")
    plt.scatter(stats[m]["deploy_s"], stats[m]["err_in"], s=110, color=C[m], zorder=3,
                marker="^", edgecolor="k", alpha=0.7)
    plt.annotate(m, (stats[m]["deploy_s"], stats[m]["err_extrap"]), textcoords="offset points",
                 xytext=LBL.get(m, (8, 6)), fontsize=10)
plt.xscale("log"); plt.xlabel("deployment cost per new IC (s, log)")
plt.ylabel("relative error vs Cole-Hopf (%)")
plt.title("Cost–accuracy trade-off  (circle = extrapolation t>1, triangle = in-window t<=1)")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "2_cost_accuracy.png"), dpi=150); plt.close()

plt.figure(figsize=(8, 5.5))
for m in MODELS:
    s = stats[m]["scaling"]
    plt.plot(s["N"], np.array(s["t"]) * 1e3, "o-", color=C[m], lw=2, label=f"{m}  (slope {s['exp']:.2f})")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("grid points evaluated  N = nt x nx  (log)"); plt.ylabel("latency (ms, log)")
plt.title("Computational complexity — latency vs problem size")
plt.grid(True, which="both", alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "3_scalability.png"), dpi=150); plt.close()

plt.figure(figsize=(8, 5.5))
N = np.arange(0, 201)
for m in MODELS:
    plt.plot(N, N * stats[m]["deploy_s"], color=C[m], lw=2, label=f"{m}  ({stats[m]['deploy_s']:.3g} s/IC)")
plt.yscale("log"); plt.xlabel("number of new ICs served")
plt.ylabel("cumulative serving wall-time (s, log)")
plt.title("Amortization — serving cost vs number of new ICs")
plt.grid(True, which="both", alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "4_amortization.png"), dpi=150); plt.close()

stats["_env"] = {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
                 "logical_cores": psutil.cpu_count(), "torch_threads": torch.get_num_threads(),
                 "torch": torch.__version__, "repeats": REPEAT}
json.dump(stats, open(os.path.join(OUT, "cost_summary.json"), "w"), indent=2)

print(f"\nDEPLOYMENT & COMPUTATIONAL COST  (CPU, 1 thread, {REPEAT} timed reps)")
print(f"{'model':10s}{'params':>10}{'disk MB':>9}{'infer ms':>11}{'med ms':>10}{'thrpt IC/s':>12}{'RSS MB':>9}{'scale exp':>11}{'deploy s':>11}")
for m in MODELS:
    s = stats[m]
    print(f"{m:10s}{s['params']:>10}{s['disk_mb']:>9.2f}{s['infer_ms']:>10.2f}{s['infer_ms_median']:>10.2f}"
          f"{s['throughput_ic_s']:>12.2f}{s['rss_mb']:>9.1f}{s['scaling']['exp']:>11.2f}{s['deploy_s']:>11.4g}")
print(f"\nSaved 4 plots and cost_summary.json to {OUT}")
