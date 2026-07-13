from pathlib import Path
import numpy as np

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
M3_RESULTS_DIR = RESULTS_DIR / "deployment"
COLEHOPF_PT = DATA_DIR / "colehopf" / "burgers_colehopf.pt"

NU = 1.0 / (100.0 * np.pi)
DOMAIN_LENGTH = 2.0

TEST_IC_INDICES = list(range(900, 910))
SEED = 0
DEFAULT_ACCURACY_TARGETS = [0.30, 0.20, 0.10, 0.05, 0.02, 0.01]
