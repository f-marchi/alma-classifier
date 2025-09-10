"""
Top‑level public API.

>>> from alma_classifier import ALMA, aml_signature
"""
from .core import ALMA            # noqa: F401
from .signature import hazard as aml_signature  # noqa: F401
from .download import get_models_dir, is_models_downloaded  # noqa: F401
from .bed_utils import process_bed_to_methylation, is_bed_file  # noqa: F401

__all__ = ["ALMA", "aml_signature", "get_models_dir", "is_models_downloaded", 
           "process_bed_to_methylation", "is_bed_file"]
