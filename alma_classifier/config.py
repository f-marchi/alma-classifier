"""
Configuration classes for ALMA Subtype v2 models.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    """Configuration for TabularTransformer model."""
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    num_features: Optional[int] = None
    num_classes: Optional[int] = None
