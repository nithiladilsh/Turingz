"""
================================================================================
OUT-OF-DISTRIBUTION TEST SPEC  —  single source of truth   (Module 2)
Team : Turingz   File : common/ood_spec.py

The robustness study (Module 2) evaluates every trained solver (FNO, PINN,
DeepONet) on the SAME three deliberately out-of-FAMILY initial conditions, so
the cross-model comparison is fair. This file is the one place those three ICs
and their physics are defined; all three solver runners import from here, the
same way every model imports the train/test partition from canonical_split.py.

How this differs from the canonical-split "OOD" ICs (samples 6, 7)
-----------------------------------------------------------------
canonical_split.TEST_IDX = [6, 7] are held-out draws from the SAME IC family the
operators trained on (random Fourier, same nu) — they measure "generalises to
unseen IC samples". The three cases HERE are out-of-family by construction —
different spatial frequency, different functional shape, different viscosity —
and measure "generalises to unseen KINDS of input". Keep the two notions
distinct in the report.

The three shifts (report sections 4.3.2 / 5.6 / 6.6.2)
-----------------------------------------------------
    OOD-1  higher-frequency sinusoid  sin(3 pi x)        spatial-frequency shift
    OOD-2  localised Gaussian bump    exp(-x^2/2 sigma^2) functional-shape shift
    OOD-3  canonical sin(pi x), nu'   nu' = 1/(50 pi)     physical-parameter shift

Grid / physics are inherited from the Cole-Hopf reference generator
(numerical_solvers/colehopf/colehopf.py) so OOD errors are directly comparable
to the in-distribution numbers: x in [-1, 1) with nx=512 (periodic,
endpoint=False), t = [0, 0.01, ..., 2.0] with nt=200, t_train_end=1.0.
Only OOD-3 changes nu, which is why only OOD-3 needs its reference regenerated
with the shifted viscosity (the other two reuse nu_train).
================================================================================
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

# ── pull the validated Cole-Hopf primitives + the canonical grid/physics ─────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
_COLEHOPF_DIR = os.path.join(_PROJECT_ROOT, "numerical_solvers", "colehopf")
if _COLEHOPF_DIR not in sys.path:
    sys.path.insert(0, _COLEHOPF_DIR)

from colehopf import Config, build_grid, solve_burgers  # noqa: E402


# ── canonical physics constants (read from the Cole-Hopf Config) ─────────────
_CFG = Config()
NU_TRAIN: float = _CFG.nu                  # 1/(100 pi), the viscosity used in training
NU_OOD3: float = 1.0 / (50.0 * np.pi)      # shifted viscosity for OOD-3 (sharper shocks)
L: float = _CFG.L
T_TRAIN_END: float = _CFG.t_train_end


def native_grid() -> tuple[np.ndarray, np.ndarray]:
    """The exact (x, t) grid the Cole-Hopf reference + FNO native grid use.

    x : (512,)  periodic, endpoint=False, in [-1, 1)
    t : (200,)  non-uniform: [0.0, 0.01, ..., 2.0]
    Both PINN/DeepONet and the reference must use THIS grid so every field is
    compared pointwise on the same nodes.
    """
    x, _dx, t = build_grid(_CFG)
    return x, t


# =============================================================================
#  OOD case definition
# =============================================================================
@dataclass(frozen=True)
class OODCase:
    """One out-of-family test case.

    name        : human-readable label for tables/plots.
    short       : filesystem-safe id used in CSV/JSON keys.
    description : one-line rationale (what shift it probes).
    nu          : viscosity for THIS case's Cole-Hopf reference.
    ic_fn       : u(x, 0) as a function of the spatial grid.
    """
    name: str
    short: str
    description: str
    nu: float
    ic_fn: Callable[[np.ndarray], np.ndarray]

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.ic_fn(np.asarray(x, dtype=np.float64)), dtype=np.float64)


# ── the three locked OOD initial conditions ──────────────────────────────────
def _ic_high_freq_sin(x: np.ndarray) -> np.ndarray:
    # sin(3 pi x): three full waves across [-1, 1). Probes spatial frequencies
    # above the sin(pi x)-family the FNO trained on, right at its mode-truncation
    # limit. Periodic-clean (matches at the x=-1/+1 seam).
    return np.sin(3.0 * np.pi * x)


def _ic_gaussian_bump(x: np.ndarray, sigma: float = 0.2) -> np.ndarray:
    # exp(-x^2 / 2 sigma^2): a sharp, non-sinusoidal, non-zero-mean shape.
    # sigma=0.2 keeps the tails ~3.7e-6 at the seam, so it is effectively
    # periodic on this grid (no artificial jump for Cole-Hopf or the FNO).
    return np.exp(-(x ** 2) / (2.0 * sigma ** 2))


def _ic_sin_pi(x: np.ndarray) -> np.ndarray:
    # canonical training IC; only the viscosity (nu) changes for OOD-3.
    return np.sin(np.pi * x)


OOD_CASES: List[OODCase] = [
    OODCase(
        name="OOD-1 higher-frequency sinusoid",
        short="ood1_highfreq",
        description="sin(3 pi x), spatial frequencies outside the training family",
        nu=NU_TRAIN,
        ic_fn=_ic_high_freq_sin,
    ),
    OODCase(
        name="OOD-2 localised Gaussian bump",
        short="ood2_gaussian",
        description="exp(-x^2 / 2*0.2^2), non-sinusoidal localised shape",
        nu=NU_TRAIN,
        ic_fn=_ic_gaussian_bump,
    ),
    OODCase(
        name="OOD-3 shifted-viscosity sin(pi x)",
        short="ood3_shifted_nu",
        description="canonical IC with nu' = 1/(50 pi), changes the PDE itself",
        nu=NU_OOD3,
        ic_fn=_ic_sin_pi,
    ),
]


def ood_cases() -> List[OODCase]:
    """The three locked OOD cases, shared by FNO / PINN / DeepONet runners."""
    return list(OOD_CASES)


# =============================================================================
#  Cole-Hopf reference for an OOD case  (shared, pure-numpy)
# =============================================================================
def cole_hopf_reference(case: OODCase, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Exact Cole-Hopf reference u_ref(t, x) for one OOD case, shape (nt, nx).

    Uses the same validated solver as the training dataset. For OOD-1/OOD-2 the
    viscosity equals nu_train; for OOD-3 it is the shifted nu carried on the
    case. Row 0 is the IC; rows 1.. are the analytic heat-kernel solution.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    u_ic = case.initial_condition(x)
    return solve_burgers(x, t, u_ic, case.nu, L)


def summary() -> dict:
    return {
        "nu_train": NU_TRAIN,
        "nu_ood3": NU_OOD3,
        "t_train_end": T_TRAIN_END,
        "cases": [
            {"short": c.short, "name": c.name, "nu": c.nu, "desc": c.description}
            for c in OOD_CASES
        ],
    }


if __name__ == "__main__":
    import json

    print("OOD spec (single source of truth, Module 2)")
    print(json.dumps(summary(), indent=2))

    x, t = native_grid()
    print(f"\ngrid: x in [{x[0]:.3f}, {x[-1]:.3f}] nx={len(x)}  "
          f"t in [{t[0]:.2f}, {t[-1]:.2f}] nt={len(t)}")
    for c in OOD_CASES:
        u = cole_hopf_reference(c, x, t)
        print(f"  {c.short:16s} nu={c.nu:.5e}  u(T) max={np.max(np.abs(u[-1])):.4f}  "
              f"finite={np.isfinite(u).all()}")
