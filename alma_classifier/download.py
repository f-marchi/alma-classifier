"""
Model download utilities for ALMA classifier.
"""
import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default GitHub release URL
DEFAULT_RELEASE_URL = "https://github.com/f-marchi/ALMA-classifier/releases/download/0.2.0a0/alma-models.tar.gz"

def get_models_dir() -> Path:
    """Return the directory where models are stored (read/write safe).

    Resolution order:
    1) Environment variable ALMA_MODELS_DIR (respected in Docker image)
    2) If models bundled in the image under package "../models" exist, use them (read-only OK)
    3) User data dir (XDG_DATA_HOME or ~/.local/share)/alma-classifier/models
    """
    # 1) Env override (recommended in Docker)
    env_dir = os.getenv("ALMA_MODELS_DIR")
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        data_dir = p / "data"
        return data_dir if data_dir.exists() else p

    # 2) Use packaged models directory if present (do NOT try to create under site-packages)
    pkg_models = Path(__file__).parent.parent / "models"
    if pkg_models.exists():
        data_dir = pkg_models / "data"
        return data_dir if data_dir.exists() else pkg_models

    # 3) User data directory (writable)
    xdg_data_home = os.getenv("XDG_DATA_HOME")
    if xdg_data_home:
        user_base = Path(xdg_data_home)
    else:
        user_base = Path.home() / ".local" / "share"
    base_models_dir = user_base / "alma-classifier" / "models"
    base_models_dir.mkdir(parents=True, exist_ok=True)
    data_dir = base_models_dir / "data"
    return data_dir if data_dir.exists() else base_models_dir

def get_local_archive_path() -> Path:
    """Get path to local archive file."""
    return Path(__file__).parent / "alma-models.tar.gz"

def is_models_downloaded() -> bool:
    """Check if minimal required model assets are present.

    Required files (diagnostic only):
    1. kept_cpg_columns.pkl
    2. minmax_scaler.joblib
    3. shukuchi_model_featselect.pth
    4. ensemble_predictor.pkl (per transformer task)
    5. label_encoder.pkl      (per transformer task)
    """
    models_dir = get_models_dir()

    required_files = [
        # Autoencoder
        "alma_autoencoders/alma_shukuchi/kept_cpg_columns.pkl",
        "alma_autoencoders/alma_shukuchi/minmax_scaler.joblib",
        "alma_autoencoders/alma_shukuchi/shukuchi_model_featselect.pth",
    # Diagnostic transformer
    "alma_transformers/alma_shingan/ensemble_predictor.pkl",
    "alma_transformers/alma_shingan/label_encoder.pkl",
    ]

    for rel in required_files:
        if not (models_dir / rel).exists():
            return False
    return True

def download_file(url: str, destination: Path, desc: str = "Downloading") -> None:
    """Download a file with progress bar."""
    logger.info(f"Downloading {desc} from {url}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def extract_models(archive_path: Path, destination: Path) -> None:
    """Extract the models archive."""   
    with tarfile.open(archive_path, 'r:gz') as tar:
        # Extract all files
        tar.extractall(destination)

def download_models() -> bool:
    """Download ALMA classifier models."""
    if is_models_downloaded():
        logger.info("Models already downloaded.")
        return True
    
    # Get the models directory
    base_models_dir = get_models_dir()
    
    local_archive = get_local_archive_path()
    
    try:
        # Check if local archive exists
        if local_archive.exists():
            logger.info(f"Found local archive at {local_archive}")
            extract_models(local_archive, base_models_dir)
            
            # Verify extraction was successful
            if is_models_downloaded():
                logger.info(f"Models successfully extracted to {base_models_dir}")
                return True
            else:
                logger.warning("Local archive extraction failed, downloading from remote")
        
        # Create temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = temp_path / "alma-models.tar.gz"
            
            # Download the archive
            download_file(DEFAULT_RELEASE_URL, archive_path, "ALMA models")
            
            # Extract models
            extract_models(archive_path, base_models_dir)

        logger.info(f"Models successfully downloaded to {base_models_dir}.")
        logger.info("You may now run a demo: `alma-classifier --demo` or use your own data: `alma-classifier -i path/to/your_data.pkl`.")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download models: {e}.")
        return False

def get_demo_data_path() -> Path | None:
    """Get path to demo data if available."""
    models_dir = get_models_dir()
    demo_file = models_dir / "example_dataset.csv.gz"
    
    if demo_file.exists():
        return demo_file
    return None


def main() -> None:  # pragma: no cover - thin wrapper
    """Module entrypoint for `python -m alma_classifier.download`.

    Exits with status 1 if model download fails so Docker build can fail fast.
    """
    if not download_models():
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
