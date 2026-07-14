"""
Module 2 (Coupling) - Phase 5 automated tests.

Guards the handoff pipeline so a later refactor cannot silently change the
numbers. Runs offline from results/eval/predictions.npz (no torch needed).

Run from the repo root:  pytest tests/test_coupling.py -q
"""
import os
import sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "hybrid_pde", "coupling - 214050V"))

from restart_spectral import solve_from, nearest_index, lowpass, TGRID, NU, NX  # noqa: E402

PRED = os.path.join(ROOT, "results", "eval", "predictions.npz")


def _rl2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-15))


@pytest.fixture(scope="module")
def data():
    d = np.load(PRED)
    return {
        "t": d["t"],
        "true": d["u_true_eval"].astype(np.float64),
        "FNO": d["FNO_eval"].astype(np.float64),
        "DeepONet": d["DeepONet_eval"].astype(np.float64),
        "PINN": d["PINN"].astype(np.float64),
    }


# 1. Restart consistency: from a TRUE state, reproduce the reference tail.
def test_restart_consistency(data):
    i = nearest_index(1.4)
    tail = solve_from(data["true"][:, i], i)
    ref = data["true"][:, i:]
    max_err = max(_rl2(tail[j], ref[j]) for j in range(tail.shape[0]))
    assert max_err < 5e-3, f"restart tail error too large: {max_err:.2e}"


# 2. Zero handoff mismatch: first numerical state == handed-over state.
def test_zero_handoff_mismatch(data):
    i = nearest_index(1.2)
    handed = data["FNO"][:, i]
    tail = solve_from(handed, i)
    assert np.allclose(tail[:, 0], handed, atol=1e-10)


# 3. Time-grid alignment and shape: hybrid matches the reference exactly.
def test_time_grid_and_shape(data):
    i = nearest_index(1.0)
    tail = solve_from(data["FNO"][:, i], i)
    hybrid = np.concatenate([data["FNO"][:, :i], tail], axis=1)
    assert hybrid.shape == data["true"].shape
    assert np.allclose(data["t"], TGRID)
    # no duplicated / missing switch point
    assert tail.shape[1] == data["true"].shape[1] - i


# 4. No ground-truth leakage: the real hybrid must NOT equal the true-state
#    upper bound, and the handed state must differ from truth.
def test_no_ground_truth_leakage(data):
    i = nearest_index(1.4)
    hybrid = solve_from(data["FNO"][:, i], i)
    upper_bound = solve_from(data["true"][:, i], i)
    assert not np.allclose(hybrid, upper_bound, atol=1e-6)
    assert _rl2(data["FNO"][0, i], data["true"][0, i]) > 1e-3


# 5. Determinism: same restart twice -> identical arrays.
def test_determinism(data):
    i = nearest_index(1.4)
    a = solve_from(data["FNO"][:, i], i)
    b = solve_from(data["FNO"][:, i], i)
    assert np.allclose(a, b)


# 6. Model-agnostic interface: same call for FNO, DeepONet, PINN.
@pytest.mark.parametrize("model", ["FNO", "DeepONet", "PINN"])
def test_model_agnostic(data, model):
    i = nearest_index(1.0)
    tail = solve_from(data[model][:, i], i)
    assert tail.shape[0] == data[model].shape[0]
    assert tail.shape[-1] == NX
    assert np.isfinite(tail).all()


# 7. Filtering: raw (frac=1) is identity; a real filter removes the high band.
def test_filtering(data):
    u = data["FNO"][:, nearest_index(1.0)]
    assert np.allclose(lowpass(u, 1.0), u)            # raw default = identity
    filt = lowpass(u, 0.5)
    cut = int(0.5 * (NX // 3))
    uh = np.fft.rfft(filt, axis=-1)
    assert np.allclose(uh[..., cut + 1:], 0.0)        # band above cutoff removed


# 8. Contract: viscosity, shape, finiteness.
def test_contract(data):
    assert np.isclose(NU, 1.0 / (100.0 * np.pi))
    st = data["FNO"][0, nearest_index(1.0)]
    assert st.shape == (NX,)
    assert np.isfinite(st).all()
