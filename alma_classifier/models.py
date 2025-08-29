"""Model loading and management utilities."""
import joblib
import pacmap
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any

def get_model_path() -> Path:
    """Get path to model files."""
    return Path(__file__).parent / "models"

def get_model_path_v2() -> Path:
    """Get path to v2 model files."""
    base_models_dir = Path(__file__).parent / "models" / "alma_v2"
    
    # Check for different possible extraction paths
    models_subdir = base_models_dir / "models"
    data_dir = base_models_dir / "data"
    
    if models_subdir.exists():
        return models_subdir
    elif data_dir.exists():
        return data_dir
    
    return base_models_dir

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

def validate_models_v2() -> Tuple[bool, str]:
    """
    Validate that all required v2 model files exist.
    
    Returns:
        Tuple[bool, str]: (True if all models exist, error message if any)
    """
    model_path = get_model_path_v2()
    
    # Check for required model directories
    required_dirs = [
        "alma_autoencoders/alma_shukuchi",
        "alma_transformers/alma_shingan/fold_models"
    ]
    
    missing_dirs = []
    for req_dir in required_dirs:
        if not (model_path / req_dir).exists():
            missing_dirs.append(req_dir)
    
    # Check for some key files
    key_files = [
        "alma_autoencoders/alma_shukuchi/shukuchi_model_featselect.pth",
        "alma_transformers/alma_shingan/model_config.json"
    ]
    
    missing_files = []
    for key_file in key_files:
        if not (model_path / key_file).exists():
            missing_files.append(key_file)
    
    if missing_dirs or missing_files:
        missing_items = missing_dirs + missing_files
        msg = (
            f"Missing ALMA v2 model files/directories: {', '.join(missing_items)}.\n"
            "Please run 'alma-classifier --download-models-v2' "
            "to download required v2 models."
        )
        return False, msg
    return True, ""
