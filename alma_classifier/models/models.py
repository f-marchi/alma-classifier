"""Model loading and management utilities."""
import joblib
import pacmap
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any

def get_model_path() -> Path:
    """Get path to model files."""
    return Path(__file__).parent

def load_models() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load pre-trained PaCMAP and LightGBM models.

    Returns:
        Tuple containing PaCMAP and LightGBM model dictionaries
    """
    warnings.filterwarnings('ignore')
    model_path = get_model_path()

    # Validate model files exist
    if not validate_models():
        raise RuntimeError(
            "Required model files are missing. Please run:\n"
            "python -m alma_classifier.download_models"
        )

    # Load PaCMAP models
    pacmap_models = {
        '2d': pacmap.load(str(model_path / 'pacmap_2d_model_alma')),
        '5d': pacmap.load(str(model_path / 'pacmap_5d_model_alma'))
    }

    # Load LightGBM models
    lgbm_models = {
        'subtype': joblib.load(str(model_path / 'lgbm_dx_model.pkl')),
        'risk': joblib.load(str(model_path / 'lgbm_px_model.pkl'))
    }

    return pacmap_models, lgbm_models

def validate_models() -> bool:
    """
    Validate that all required model files exist.

    Returns:
        bool: True if all models exist, False otherwise
    """
    model_path = get_model_path()
    required_files = [
        'pacmap_2d_model_alma.ann',
        'pacmap_2d_model_alma.pkl',
        'pacmap_5d_model_alma.ann',
        'pacmap_5d_model_alma.pkl',
        'lgbm_dx_model.pkl',
        'lgbm_px_model.pkl',
        'imputer_model.joblib'
    ]

    for file in required_files:
        if not (model_path / file).exists():
            return False
    return True