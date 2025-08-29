"""
Full pipeline: autoencoder → latent → diagnostic transformer for ALMA Subtype v2
"""
import json
import pickle
import joblib
import logging
import sys
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


class ALMAv2:
    """Load models once, call .predict() on arbitrary sample batches."""
    
    def __init__(self, base: Union[str, Path, None] = None):
        if base:
            base = Path(base)
        else:
            # Check if models are downloaded, if not suggest download
            if not self._is_models_downloaded():
                raise FileNotFoundError(
                    "ALMA v2 models not found. Please run 'alma-classifier --download-models-v2' first."
                )
            base = self._get_models_dir()
        
        self.base = base
        
        # Try to import torch, but make it optional for now
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            self.device = None
            logging.warning("PyTorch not available. ALMA Subtype v2 will not be functional.")

        self.auto = self.scaler = self.cpg = None
        self.diag = self.dlabels = None

    def _get_models_dir(self) -> Path:
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

    def _is_models_downloaded(self) -> bool:
        """Check if v2 models are already downloaded."""
        models_dir = self._get_models_dir()
        
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

    def load_auto(self):
        """Load autoencoder model."""
        if self.device is None:
            raise ImportError("PyTorch is required for ALMA Subtype v2 model")
            
        import torch
        from .transformer_models import ShukuchiAutoencoder
        
        p = self.base / "alma_autoencoders" / "alma_shukuchi"
        self.cpg = pd.read_pickle(p / "kept_cpg_columns.pkl")
        self.scaler = joblib.load(p / "minmax_scaler.joblib")

        meta = torch.load(p / "shukuchi_metadata_featselect.pth", weights_only=False)
        
        # Try loading the model - it might be a complete model object or state dict
        model_data = torch.load(p / "shukuchi_model_featselect.pth", map_location=self.device, weights_only=False)
        
        if isinstance(model_data, ShukuchiAutoencoder):
            # Complete model was saved
            self.auto = model_data.to(self.device).eval()
        else:
            # State dict was saved
            self.auto = (
                ShukuchiAutoencoder(meta["input_size"], meta["latent_size"])
                .to(self.device)
                .eval()
            )
            self.auto.load_state_dict(model_data)

    def _convert_old_state_dict(self, old_state_dict):
        """Convert old state dict keys to new format for diagnostic model."""
        # For TabularTransformer (diagnostic model)
        # Map from old keys to new keys
        key_mapping = {
            'pos_embed': 'pos',
            'input_norm': 'in_norm',
            'input_dropout': 'in_drop',
            'feature_embed': 'embed',
            'feature_transform': 'to_tok',
            'transformer': 'tx',
            'global_pool': 'pool',
            'classifier': 'cls'
        }
        
        new_state_dict = {}
        for old_key, value in old_state_dict.items():
            new_key = old_key
            # Sort by key length (longest first) to avoid partial matches
            for old_prefix, new_prefix in sorted(key_mapping.items(), key=lambda x: len(x[0]), reverse=True):
                if old_key.startswith(old_prefix + '.') or old_key == old_prefix:
                    # Only replace at the beginning and ensure it's a complete component
                    if old_key == old_prefix:
                        new_key = new_prefix
                    elif old_key.startswith(old_prefix + '.'):
                        new_key = new_prefix + old_key[len(old_prefix):]
                    break
            new_state_dict[new_key] = value
        
        return new_state_dict

    def _load_tx(self, name):
        """Load transformer model."""
        if self.device is None:
            raise ImportError("PyTorch is required for ALMA Subtype v2 model")
            
        import torch
        from .transformer_models import TabularTransformer
        from .config import TransformerConfig
        from .ensemble import EnsemblePredictor
        
        d = self.base / "alma_transformers" / name
        config_data = json.load(open(d / "model_config.json"))
        cfg = TransformerConfig(**{k: v for k, v in config_data.items() if k in ['d_model', 'n_heads', 'n_layers', 'dropout']})
        cfg.num_features = config_data['num_features']
        cfg.num_classes = config_data.get('num_classes')
        labels = pickle.load(open(d / "label_encoder.pkl", "rb"))

        folds = []
        for fd in sorted((d / "fold_models").glob("fold_*")):
            proc = pickle.load(open(fd / "processor.pkl", "rb"))
            
            # Load model - extract state dict from saved model
            model_data = torch.load(fd / "model_state_dict.pth", map_location=self.device, weights_only=False)
            
            if isinstance(model_data, TabularTransformer):
                # Complete model was saved - extract its state dict
                state_dict = model_data.state_dict()
            else:
                # State dict was saved directly
                state_dict = model_data
            
            # Convert old state dict keys to new format if needed
            state_dict = self._convert_old_state_dict(state_dict)
            
            # Create diagnostic model
            model = (
                TabularTransformer(cfg.num_features, cfg.num_classes, cfg)
                .to(self.device)
                .eval()
            )
            
            model.load_state_dict(state_dict)
            
            folds.append({"model": model, "processor": proc})
        return EnsemblePredictor(folds, self.device), labels

    def load_diag(self):
        """Load diagnostic model."""
        self.diag, self.dlabels = self._load_tx("alma_shingan")

    def _prep(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare data for prediction."""
        missing = set(self.cpg) - set(df.columns)
        if missing:
            df.loc[:, list(missing)] = 0.5
        return self.scaler.transform(
            df[self.cpg].fillna(0.5).values.astype(np.float32)
        )

    def _latent(self, X_scaled: np.ndarray) -> np.ndarray:
        """Extract latent features using autoencoder."""
        if self.device is None:
            raise ImportError("PyTorch is required for ALMA Subtype v2 model")
            
        import torch
        
        with torch.no_grad():
            _, z = self.auto(torch.from_numpy(X_scaled).to(self.device))
        return z.cpu().numpy()

    def predict(self, input_file: Union[Path, str], output_file: Optional[Union[Path, str]] = None):
        """Generate predictions for input data."""
        df = pd.read_pickle(input_file)
        latent = self._latent(self._prep(df))

        res = pd.DataFrame({"sample_id": df.index}, index=df.index)

        # Diagnostic
        y, c, u, _ = self.diag.predict_with_conf(latent)
        res["ALMA Subtype v2"] = self.dlabels.inverse_transform(y)
        res["Diagnostic Confidence v2"] = c

        out = Path(output_file) if output_file else Path(input_file).with_name("alma_v2_predictions.csv")
        res.to_csv(out, index=False)
        return out
