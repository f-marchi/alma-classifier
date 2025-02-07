"""ALMA Classifier package for epigenomic classification."""
from .predictor import ALMAPredictor
from .utils import export_results

__version__ = "0.1.2"

__all__ = ["ALMAPredictor", "export_results"]
