from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class PINNConfig:
    sample: int = 0                      
    t_train_end: float = 1.0           

    hidden: List[int] = field(default_factory=lambda: [64, 64, 64, 64, 64])
    activation: str = "tanh"
    hard_periodic: bool = True           
    n_harmonics: int = 4                

    num_domain: int = 16000
    num_boundary: int = 400            
    num_initial: int = 400

    use_data_anchors: bool = False
    n_anchors: int = 2000

    w_pde: float = 1.0
    w_ic: float = 10.0
    w_bc: float = 1.0
    w_data: float = 1.0

    lr: float = 1.0e-3
    adam_iters: int = 15000
    lbfgs: bool = True
    display_every: int = 1000

    seed: int = 42
    float64: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PINNConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)
