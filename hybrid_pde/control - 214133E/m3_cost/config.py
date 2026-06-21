import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(MODULE_DIR, "..", ".."))
RESULTS = os.path.join(ROOT, "results")
M3_RESULTS = os.path.join(MODULE_DIR, "results")
os.makedirs(M3_RESULTS, exist_ok=True)

NU = 1.0 / (100 * 3.141592653589793)
L, NX = 2.0, 512
T, NT = 2.0, 200
N_SAMPLES = 1000
T_TRAIN_END = 1.0
T_START = 0.01
SEED = 42

N_TRAIN, N_VAL = 800, 100

DEEPONET_FIELD = os.path.join(RESULTS, "deeponet", "extrapolation_fields.npz")
META_NPZ = os.path.join(RESULTS, "deeponet", "meta.npz")
