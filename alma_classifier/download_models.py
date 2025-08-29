"""Script to download pre-trained model files."""
import os
import sys
import urllib.request
import tarfile
from pathlib import Path
from typing import Dict, List

MODEL_URLS = {
    'imputer_model.joblib': 'https://github.com/f-marchi/ALMA-classifier/releases/download/0.1.0/imputer_model.joblib',
    'lgbm_dx_model.pkl': 'https://github.com/f-marchi/ALMA-classifier/releases/download/0.1.0/lgbm_dx_model.pkl',
    'lgbm_px_model.pkl': 'https://github.com/f-marchi/ALMA-classifier/releases/download/0.1.0/lgbm_px_model.pkl',
    'pacmap_5d_model_alma.ann': 'https://github.com/f-marchi/ALMA-classifier/releases/download/0.1.0/pacmap_5d_model_alma.ann',
    'pacmap_5d_model_alma.pkl': 'https://github.com/f-marchi/ALMA-classifier/releases/download/0.1.0/pacmap_5d_model_alma.pkl'
}

# ALMA v2 models configuration
DEFAULT_RELEASE_URL_V2 = "https://github.com/pedro-orsini/alma-transformer/releases/download/0.2.0/alma-models.tar.gz"

def get_model_dir() -> Path:
    """Get the models directory path."""
    return Path(__file__).parent / "models"

def get_models_dir_v2() -> Path:
    """Get the directory where v2 models should be stored."""
    base_models_dir = Path(__file__).parent / "models" / "alma_v2"
    base_models_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for different possible extraction paths
    models_subdir = base_models_dir / "models"
    data_dir = base_models_dir / "data"
    
    if models_subdir.exists():
        return models_subdir
    elif data_dir.exists():
        return data_dir
    
    return base_models_dir

def get_local_archive_path_v2() -> Path:
    """Get path to local v2 archive file."""
    return Path(__file__).parent / "models" / "alma_v2" / "alma-models.tar.gz"

def is_models_downloaded_v2() -> bool:
    """Check if v2 models are already downloaded."""
    models_dir = get_models_dir_v2()
    
    # Check for required model directories
    required_dirs = [
        "alma_autoencoders/alma_shukuchi",
        "alma_transformers/alma_shingan/fold_models"
    ]
    
    for req_dir in required_dirs:
        if not (models_dir / req_dir).exists():
            return False
    
    # Check for some key files
    key_files = [
        "alma_autoencoders/alma_shukuchi/shukuchi_model_featselect.pth",
        "alma_transformers/alma_shingan/model_config.json"
    ]
    
    for key_file in key_files:
        if not (models_dir / key_file).exists():
            return False
    
    return True

def download_models() -> None:
    """Download all required model files."""
    model_dir = get_model_dir()
    model_dir.mkdir(exist_ok=True)
    
    print("Downloading model files...")
    for filename, url in MODEL_URLS.items():
        target_path = model_dir / filename
        if not target_path.exists():
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, target_path)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
                sys.exit(1)
        else:
            print(f"File {filename} already exists, skipping...")
    
    print("\nAll model files downloaded successfully!")
    print(f"Model files are located in: {model_dir}")

def download_models_v2() -> None:
    """Download ALMA v2 transformer models."""
    model_dir = get_models_dir_v2()
    archive_path = get_local_archive_path_v2()
    
    # Create directory if it doesn't exist
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    
    if is_models_downloaded_v2():
        print("ALMA v2 models already downloaded and extracted.")
        return
    
    print("Downloading ALMA v2 model archive...")
    try:
        urllib.request.urlretrieve(DEFAULT_RELEASE_URL_V2, archive_path)
        print(f"Successfully downloaded archive to {archive_path}")
    except Exception as e:
        print(f"Error downloading archive: {str(e)}")
        sys.exit(1)
    
    # Extract the archive
    print("Extracting models...")
    try:
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=model_dir)
        print(f"Successfully extracted models to {model_dir}")
        
        # Remove the archive file after successful extraction
        archive_path.unlink()
        print("Cleaned up archive file.")
        
    except Exception as e:
        print(f"Error extracting archive: {str(e)}")
        sys.exit(1)
    
    print("\nALMA v2 models downloaded and installed successfully!")
    print(f"Model files are located in: {model_dir}")

def main():
    """Main entry point for model download."""
    try:
        download_models()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def main_v2():
    """Main entry point for v2 model download."""
    try:
        download_models_v2()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
