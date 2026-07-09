import os
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(_ROOT, "data", "colehopf", "burgers_colehopf.pt")
N_TRAIN, N_VAL = 800, 100


def load():
    d = torch.load(DATA, weights_only=False, map_location="cpu")
    U = d["u"].numpy()
    ICs = d["ICs"].numpy() if "ICs" in d else U[:, 0, :].copy()
    x = d["x"].numpy()
    t = d["t"].numpy()
    te = float(d["t_train_end"])
    Tmax = float(d["T"]) if "T" in d else float(t.max())
    return U, ICs, x, t, te, Tmax


def split(N):
    return (np.arange(N_TRAIN),
            np.arange(N_TRAIN, N_TRAIN + N_VAL),
            np.arange(N_TRAIN + N_VAL, N))
