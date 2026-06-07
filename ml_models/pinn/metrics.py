"""
PINN metrics shim — re-exports the shared, model-agnostic implementation.

The accuracy metrics now live in common/metrics.py (single source of truth) so
PINN, FNO and DeepONet are all graded by the same code. This module keeps the
old import path (`from .metrics import compute_metrics`) working unchanged.
"""

import os
import sys

# Ensure the project root is importable so `common` resolves.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.metrics import (  # noqa: F401,E402
    SHOCK_T_LO,
    SHOCK_T_HI,
    compute_metrics,
    aggregate_over_samples,
)
