from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class PINNConfig:
    # Problem instance
    sample: int = 0                      # which Cole-Hopf IC to solve (0 = sin(pi x))
    t_train_end: float = 1.0             # time domain cap for training (extrapolation test beyond)

    # Architecture
    hidden: List[int] = field(default_factory=lambda: [64, 64, 64, 64, 64])
    activation: str = "tanh"
    hard_periodic: bool = True           # enforce periodicity via sin/cos feature transform
    n_harmonics: int = 4                 # harmonics in periodic feature map (matches dataset n_modes)

    # Collocation
    num_domain: int = 16000
    num_boundary: int = 400              # used only when hard_periodic is False
    num_initial: int = 400

    # Optional supervised anchoring (data-informed PINN ablation)
    use_data_anchors: bool = False
    n_anchors: int = 2000

    # Loss weights (order: pde, ic, [periodic_u, periodic_ux], [anchors])
    w_pde: float = 1.0
    w_ic: float = 10.0
    w_bc: float = 1.0
    w_data: float = 1.0

    # Optimization
    lr: float = 1.0e-3
    adam_iters: int = 15000
    lbfgs: bool = True
    display_every: int = 1000

    # Reproducibility / precision
    seed: int = 42
    float64: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PINNConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)
