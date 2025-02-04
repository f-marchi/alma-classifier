"""Model loading and management utilities."""
import joblib
import pacmap
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any

def get_model_path() -> Path:
    """Get path to model files."""
    return Path(__file__).parent / "models"

def load_models() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load pre-trained PaCMAP and LightGBM models.
    
    Returns:
        Tuple containing PaCMAP and LightGBM model dictionaries
    """
    warnings.filterwarnings('ignore')
    model_path = get_model_path()

    # Load PaCMAP model
    pacmap_model = pacmap.load(str(model_path / 'pacmap_5d_model_alma'))

    # Load LightGBM models
    lgbm_models = {
        'subtype': joblib.load(str(model_path / 'lgbm_dx_model.pkl')),
        'risk': joblib.load(str(model_path / 'lgbm_px_model.pkl'))
    }

    return pacmap_model, lgbm_models

def validate_models() -> Tuple[bool, str]:
    """
    Validate that all required model files exist.
    
    Returns:
        Tuple[bool, str]: (True if all models exist, error message if any)
    """
    model_path = get_model_path()
    required_files = [
        'pacmap_5d_model_alma.pkl',
        'pacmap_5d_model_alma.ann',
        'lgbm_dx_model.pkl',
        'lgbm_px_model.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
            
    if missing_files:
        msg = (
            f"Missing model files: {', '.join(missing_files)}.\n"
            "Please run 'python -m alma_classifier.download_models' "
            "to download required models."
        )
        return False, msg
    return True, ""
