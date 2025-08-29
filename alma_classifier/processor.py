"""
Data preprocessing utilities for ALMA Subtype v2.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataProcessor:
    """Simple data processor wrapper for storing preprocessing objects."""
    power_transformer: Optional[object] = None
    scaler: Optional[object] = None
