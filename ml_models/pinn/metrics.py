import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.metrics import ( 
    SHOCK_T_LO,
    SHOCK_T_HI,
    compute_metrics,
    aggregate_over_samples,
)
