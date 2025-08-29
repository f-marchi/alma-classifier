"""ALMA Classifier package for epigenomic classification."""

from .predictor import ALMAPredictor
from .preprocessing import process_methylation_data, apply_pacmap
from .bed_processing import process_bed_to_methylation, is_bed_file
from .utils import export_results
from .models import load_models, validate_models, validate_models_v2

# Optional v2 imports - only available if PyTorch is installed and working
__v2_available__ = False
try:
    # Test if torch can be imported without immediate CUDA issues
    import torch
    # Only import our v2 modules if torch works
    from .alma_v2_core import ALMAv2
    from .transformer_models import ShukuchiAutoencoder, TabularTransformer
    from .ensemble import EnsemblePredictor
    from .config import TransformerConfig
    __v2_available__ = True
except (ImportError, OSError):
    # PyTorch not available or has CUDA issues
    __v2_available__ = False

__all__ = [
    "ALMAPredictor", 
    "process_methylation_data", 
    "apply_pacmap",
    "process_bed_to_methylation",
    "is_bed_file",
    "export_results",
    "load_models",
    "validate_models",
    "validate_models_v2"
]

# Add v2 components to __all__ if available
if __v2_available__:
    __all__.extend([
        "ALMAv2",
        "ShukuchiAutoencoder", 
        "TabularTransformer",
        "EnsemblePredictor",
        "TransformerConfig"
    ])