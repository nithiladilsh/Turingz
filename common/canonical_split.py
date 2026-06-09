import os

DATASET_REL_PATH = os.path.join("data", "colehopf", "burgers_1d_cole_hopf.pt")

N_SAMPLES   = 8
SEED        = 42
T_TRAIN_END = 1.0

TRAIN_IDX = [0, 1, 2, 3, 4, 5]
TEST_IDX  = [6, 7]
EVAL_IDX  = list(range(N_SAMPLES))

assert sorted(TRAIN_IDX + TEST_IDX) == EVAL_IDX, "split must cover all samples once"
assert set(TRAIN_IDX).isdisjoint(TEST_IDX),      "train and test must not overlap"
assert 0 in TRAIN_IDX,                            "focal sample 0 must stay in train"


def regime_of(sample: int) -> str:
    return "in_dist" if sample in TRAIN_IDX else "ood"


def operator_train_idx() -> list:
    return list(TRAIN_IDX)


def operator_test_idx() -> list:
    return list(TEST_IDX)


def pinn_training_plan() -> list:
    return [(i, regime_of(i)) for i in EVAL_IDX]


def resolve_dataset_path(project_root: str) -> str:
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
    print("Canonical split")
    print(json.dumps(summary(), indent=2))
    print("\nPINN training plan:")
    for s, r in pinn_training_plan():
        print(f"  sample {s}: {r}")
