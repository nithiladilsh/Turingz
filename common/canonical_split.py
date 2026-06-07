"""
================================================================================
CANONICAL DATASET SPLIT  —  single source of truth
Team : Turingz   File : common/canonical_split.py

Every solver (PINN, FNO, DeepONet) and every evaluation module (reliability,
robustness, cost) imports the train/test split from THIS file. Before this
existed each model defined its own split, which made the cross-model
comparison invalid. Now there is exactly one partition.

Team decision (8 Cole-Hopf initial conditions):
    * Operators (FNO, DeepONet) TRAIN on 6 ICs and are TESTED on 2 held-out
      ("unseen") ICs, to measure generalization to new initial conditions.
    * Sample 0 (the smooth focal case, u0 = sin(pi x)) stays in TRAIN so the
      operators always see a smooth IC.
    * The PINN is single-instance: it cannot generalize to an unseen IC, so it
      trains ONE model PER IC across all 8. The split does not change how the
      PINN trains; it only labels each result as in-distribution or unseen so
      the three models line up in the comparison tables.

The temporal split is the same for all: train on t <= T_TRAIN_END, evaluate
(extrapolate) on t > T_TRAIN_END.
================================================================================
"""

import os

# ── Reference dataset (relative to the project root) ─────────────────────────
DATASET_REL_PATH = os.path.join("data", "colehopf", "burgers_1d_cole_hopf.pt")

# ── Fixed experiment constants ───────────────────────────────────────────────
N_SAMPLES   = 8
SEED        = 42      # one seed for every model (was 0 for FNO, 42 for others)
T_TRAIN_END = 1.0     # temporal train / extrapolation boundary

# ── Initial-condition partition (operator train vs held-out unseen test) ─────
TRAIN_IDX = [0, 1, 2, 3, 4, 5]   # operators learn from these ICs
TEST_IDX  = [6, 7]               # held-out, never seen by operators in training
EVAL_IDX  = list(range(N_SAMPLES))  # every IC is evaluated

# ── Integrity guards: the split must be a clean partition ────────────────────
assert sorted(TRAIN_IDX + TEST_IDX) == EVAL_IDX, "split must cover all samples once"
assert set(TRAIN_IDX).isdisjoint(TEST_IDX),      "train and test must not overlap"
assert 0 in TRAIN_IDX,                            "focal sample 0 must stay in train"


def regime_of(sample: int) -> str:
    """'in_dist' if the operators trained on this IC, else 'ood' (unseen)."""
    return "in_dist" if sample in TRAIN_IDX else "ood"


def operator_train_idx() -> list:
    """IC indices the operator models (FNO, DeepONet) train on."""
    return list(TRAIN_IDX)


def operator_test_idx() -> list:
    """Held-out IC indices used to test operator generalization."""
    return list(TEST_IDX)


def pinn_training_plan() -> list:
    """The PINN trains one model per IC. Returns (sample, regime) for all ICs,
    so a runner can train the full PINN set and tag each with its regime."""
    return [(i, regime_of(i)) for i in EVAL_IDX]


def resolve_dataset_path(project_root: str) -> str:
    """Absolute path to the canonical reference dataset."""
    return os.path.join(project_root, DATASET_REL_PATH)


def summary() -> dict:
    return {
        "n_samples":   N_SAMPLES,
        "train_idx":   list(TRAIN_IDX),
        "test_idx":    list(TEST_IDX),
        "eval_idx":    list(EVAL_IDX),
        "t_train_end": T_TRAIN_END,
        "seed":        SEED,
        "policy":      "operators train on 6 ICs, held-out test on 2; "
                       "PINN one model per IC; same temporal split for all",
    }


if __name__ == "__main__":
    import json
    print("Canonical split (single source of truth)")
    print(json.dumps(summary(), indent=2))
    print("\nPINN training plan (one model per IC):")
    for s, r in pinn_training_plan():
        print(f"  sample {s}: {r}")
