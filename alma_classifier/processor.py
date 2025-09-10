"""
Data preprocessing utilities for ALMA classifier.
"""
from dataclasses import dataclass
from typing import Optional
from sklearn.preprocessing import PowerTransformer, StandardScaler


@dataclass
class DataProcessor:
    """Simple data processor wrapper for storing preprocessing objects."""
    power_transformer: Optional[PowerTransformer] = None
    scaler: Optional[StandardScaler] = None
