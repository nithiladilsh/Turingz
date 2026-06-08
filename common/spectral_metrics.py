"""
================================================================================
SPECTRAL (FOURIER-AMPLITUDE) DISTANCE  —  single source of truth   (Module 2)
Team : Turingz   File : common/spectral_metrics.py

The robustness study reports two numbers per (model, OOD input): the pointwise
relative-L2 error (common/metrics.py) and the SPECTRAL distance defined here.
This module is the one implementation of the spectral distance, used identically
for FNO, PINN and DeepONet so the comparison is apples-to-apples.

What the spectral distance is
-----------------------------
For a snapshot u(., t), an FFT gives an amplitude spectrum |U_k(t)| — how much
sine-wave-of-frequency-k is in the signal. The spectral distance between the
predicted and reference snapshots at time t is a weighted L1 distance between
their amplitude spectra:

    D(t) = sum_k  w_k * | |U_pred_k(t)| - |U_ref_k(t)| |

with selectable mode weights w_k. This is the practical "weighted
Fourier-amplitude" proxy for the Wasserstein-Fourier distance of Fesser et al.;
it captures the same intuition (compare magnitude spectra, weight modes) without
an optimal-transport solve per time step.

Why it is an early-warning signal
----------------------------------
Relative L2 is a single scalar — "wrong here". D(t) breaks the error down by
frequency, so it reveals WHAT KIND of error is happening (high-frequency junk vs
wrong-amplitude bulk). In an extrapolating operator the spectrum typically starts
to drift before the pointwise error blows up, so D(t) tends to cross a "departed
from baseline" threshold EARLIER than relative L2. That lead time is the
early-warning claim; first_crossing_time / early_warning_lead quantify it.

A note on the weighting (read before defending a choice)
--------------------------------------------------------
    "low_freq"  w_k = 1/(1+k)  emphasises bulk (low-mode) content — the
                               Wasserstein-Fourier flavour, energy lives here.
    "uniform"   w_k = 1        every mode counts equally.
    "high_freq" w_k = k        emphasises high-mode drift.
There is a real tension: the early-warning rationale is that the model first
leaks energy into HIGH modes, but "low_freq" weighting de-emphasises exactly
those modes. So the weighting that gives the best (earliest) warning is an
empirical question, not a given. Run all three (the runner does) and report which
weighting yields the largest early_warning_lead rather than asserting one a priori.
================================================================================
"""

from __future__ import annotations

from typing import Dict

import numpy as np


# =============================================================================
#  Amplitude spectra
# =============================================================================
def amplitude_spectrum(field: np.ndarray) -> np.ndarray:
    """One-sided amplitude spectrum |rfft(u)| along the spatial axis.

    field : (nt, nx) or (nx,). Returns (nt, n_modes) or (n_modes,) with
    n_modes = nx // 2 + 1. x is assumed periodic and uniform (the project grid),
    which is exactly the setting rfft expects, so no windowing is applied.
    """
    field = np.asarray(field, dtype=np.float64)
    return np.abs(np.fft.rfft(field, axis=-1))


def mode_weights(n_modes: int, weight_mode: str = "low_freq") -> np.ndarray:
    """Per-mode weights w_k, shape (n_modes,)."""
    k = np.arange(n_modes, dtype=np.float64)
    if weight_mode == "low_freq":
        return 1.0 / (1.0 + k)
    if weight_mode == "uniform":
        return np.ones_like(k)
    if weight_mode == "high_freq":
        return k
    raise ValueError(f"unknown weight_mode: {weight_mode!r} "
                     "(expected 'low_freq', 'uniform' or 'high_freq')")


# =============================================================================
#  Spectral distance D(t)
# =============================================================================
def spectral_distance_per_t(pred: np.ndarray, ref: np.ndarray,
                            weight_mode: str = "low_freq") -> np.ndarray:
    """Weighted Fourier-amplitude distance at each time step, shape (nt,).

    pred, ref : (nt, nx) on the same periodic grid. Row index is time.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert pred.shape == ref.shape, (pred.shape, ref.shape)

    sp_p = amplitude_spectrum(pred)          # (nt, n_modes)
    sp_r = amplitude_spectrum(ref)
    w = mode_weights(sp_p.shape[1], weight_mode)   # (n_modes,)
    return np.sum(w[None, :] * np.abs(sp_p - sp_r), axis=1)


# =============================================================================
#  Early-warning crossing / lead time
# =============================================================================
def first_crossing_time(series: np.ndarray, t: np.ndarray,
                        baseline_window: tuple[float, float] = (0.0, 0.25),
                        factor: float = 2.0) -> float:
    """First time `series` exceeds factor * its baseline-window mean.

    The baseline is taken over an early, in-distribution-time window where the
    model is trusted (default the pre-shock window [0, 0.25], well inside the
    training window). Returns nan if the series never crosses, or if the baseline
    window is empty.
    """
    series = np.asarray(series, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    in_base = (t >= baseline_window[0]) & (t <= baseline_window[1])
    if not in_base.any():
        return float("nan")

    baseline = float(np.mean(series[in_base]))
    threshold = factor * baseline
    crossed = np.where(series > threshold)[0]
    if crossed.size == 0:
        return float("nan")
    return float(t[crossed[0]])


def early_warning_lead(rel_l2: np.ndarray, spec_dist: np.ndarray, t: np.ndarray,
                       baseline_window: tuple[float, float] = (0.0, 0.25),
                       factor: float = 2.0) -> Dict[str, float]:
    """Lead time of the spectral early-warning over the pointwise error.

    Both curves are thresholded at `factor` x their own baseline mean. The lead
    is (time rel-L2 crosses) - (time spectral distance crosses): positive means
    the spectral signal fired earlier, which is the desired result.
    """
    cross_l2 = first_crossing_time(rel_l2, t, baseline_window, factor)
    cross_sd = first_crossing_time(spec_dist, t, baseline_window, factor)
    lead = (cross_l2 - cross_sd
            if not (np.isnan(cross_l2) or np.isnan(cross_sd)) else float("nan"))
    return {
        "first_cross_rel_l2": cross_l2,
        "first_cross_spec_dist": cross_sd,
        "early_warning_lead": lead,
        "baseline_window": list(baseline_window),
        "factor": factor,
    }
