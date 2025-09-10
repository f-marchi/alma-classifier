from dataclasses import dataclass

@dataclass
class Config:
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float = 0.2
    num_features: int = None
    num_classes: int = None