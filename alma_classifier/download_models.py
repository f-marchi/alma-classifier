"""Script to download pre-trained model files."""
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

MODEL_URLS = {
    'imputer_model.joblib': 'https://github.com/f-marchi/ALMA-classifier/releases/download/0.1.0/imputer_model.joblib',
    'lgbm_dx_model.pkl': 'https://github.com/f-marchi/ALMA-classifier/releases/download/0.1.0/lgbm_dx_model.pkl',
    'lgbm_px_model.pkl': 'https://github.com/f-marchi/ALMA-classifier/releases/download/0.1.0/lgbm_px_model.pkl',
    'pacmap_5d_model_alma.ann': 'https://github.com/f-marchi/ALMA-classifier/releases/download/0.1.0/pacmap_5d_model_alma.ann',
    'pacmap_5d_model_alma.pkl': 'https://github.com/f-marchi/ALMA-classifier/releases/download/0.1.0/pacmap_5d_model_alma.pkl'
}

def get_model_dir() -> Path:
    """Get the models directory path."""
    return Path(__file__).parent / "models"

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

def main():
    """Main entry point for model download."""
    try:
        download_models()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
