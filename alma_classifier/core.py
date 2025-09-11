"""ALMA end‑to‑end inference pipeline.
"""
import pickle, joblib, logging, sys
from pathlib import Path

import numpy as np, pandas as pd, torch

from .config import Config  # kept for backwards unpickling safety
from .models import ShukuchiAutoencoder, TabularTransformer  # noqa: F401 (for unpickling)
from .processor import DataProcessor  # noqa: F401 (for unpickling legacy objects)
from .ensemble import EnsemblePredictor
from . import signature
from .download import get_models_dir, is_models_downloaded

# This makes classes available in the main namespace for unpickling
sys.modules['__main__'].shukuchi = ShukuchiAutoencoder
sys.modules['__main__'].TabularTransformer = TabularTransformer
sys.modules['__main__'].Config = Config
sys.modules['__main__'].DataProcessor = DataProcessor
sys.modules['__main__'].EnsemblePredictor = EnsemblePredictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

class ALMA:
    """Load models once, call .predict() on arbitrary sample batches."""
    def __init__(self, base: str | Path = None):
        if base:
            base = Path(base)
        else:
            # Check if models are downloaded, if not suggest download
            if not is_models_downloaded():
                raise FileNotFoundError(
                    "Models not found. Please run 'alma-classifier --download-models' first."
                )
            base = get_models_dir()
        
        self.base = base
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.auto = self.scaler = self.cpg = None
        self.diag = self.dlabels = None

    # ---------- loaders ----------
    def _ensure_features_axis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure data is shaped as samples x features.

        Requirement: number of features must be greater than number of samples.
        If not, assume the input is transposed and fix by transposing.
        """
        n_samples, n_features = df.shape[0], df.shape[1]
        if n_features > n_samples:
            return df
        logging.info(
            "Input appears transposed (features=%d, samples=%d). Auto-transposing.",
            n_features, n_samples,
        )
        return df.T

    def load_auto(self):
        """Load autoencoder + scaler."""
        p = self.base / "alma_autoencoders" / "alma_shukuchi"
        self.cpg = pd.read_pickle(p / "kept_cpg_columns.pkl")
        self.scaler = joblib.load(p / "minmax_scaler.joblib")

        model_path = p / "shukuchi_model_featselect.pth"
        model_data = torch.load(model_path, map_location=self.dev, weights_only=False)

        if isinstance(model_data, ShukuchiAutoencoder):  # full model saved
            self.auto = model_data.to(self.dev).eval()
            return

        # Infer latent size from encoder final linear weight shape
        # Expected keys like 'encoder.6.weight' (nn.Sequential indexing)
        latent_size = None
        for k, v in model_data.items():
            if k.endswith("encoder.6.weight") and hasattr(v, "shape"):
                latent_size = v.shape[0]
                break
        if latent_size is None:
            # Fallback: try alternative key pattern (older saved models may name differently)
            for k, v in model_data.items():
                if "encoder" in k and k.endswith("weight") and hasattr(v, "shape"):
                    # Take first matching linear with out_features < in_features (heuristic)
                    if v.shape[0] < v.shape[1]:
                        latent_size = v.shape[0]
                        break
        if latent_size is None:
            raise RuntimeError("Could not infer latent size from autoencoder state_dict; please re-save model object.")

        input_size = len(self.cpg)
        self.auto = ShukuchiAutoencoder(input_size, latent_size).to(self.dev).eval()
        self.auto.load_state_dict(model_data)

    def _load_tx(self, name):
        """Load transformer ensemble using only ensemble_predictor + label_encoder."""
        d = self.base / "alma_transformers" / name
        labels = pickle.load(open(d / "label_encoder.pkl", "rb"))
        ep_path = d / "ensemble_predictor.pkl"

        # Models were saved on a CUDA box; enforce CPU mapping when CUDA is unavailable.
        # Try torch.load first (covers torch.save artifacts); fall back to raw pickle.
        obj = None
        # 1) Try torch.load with explicit CPU mapping
        try:
            obj = torch.load(ep_path, map_location=lambda storage, loc: storage.cpu(), weights_only=False)
        except Exception as e1:
            # 2) Fallback: monkeypatch torch.serialization.default_restore_location to coerce cuda->cpu
            # Access torch.serialization via attribute to satisfy static analysis
            _ts = torch.serialization
            orig_restore = _ts.default_restore_location
            def _forced_cpu_restore(storage, location):
                if isinstance(location, str) and location.startswith("cuda"):
                    location = "cpu"
                return orig_restore(storage, location)
            _ts.default_restore_location = _forced_cpu_restore
            try:
                with open(ep_path, "rb") as fh:
                    obj = pickle.load(fh)
            except Exception as e2:
                # 3) Re-raise with combined context
                raise RuntimeError(f"Failed to load ensemble predictor on CPU. torch.load error: {e1}; pickle fallback error: {e2}") from e2
            finally:
                _ts.default_restore_location = orig_restore

        if isinstance(obj, EnsemblePredictor):
            # Ensure models are on correct device
            for f in getattr(obj, 'folds', []):
                m = f.get("model")
                if m is not None:
                    self._patch_legacy_model_attrs(m)
                    m.to(self.dev).eval()
            obj.device = self.dev
            return obj, labels

        # Legacy structure (list/dict) -> reconstruct EnsemblePredictor
        if isinstance(obj, dict) and 'folds' in obj:
            folds = obj['folds']
        elif isinstance(obj, list):
            folds = obj
        else:
            raise TypeError(f"Unexpected object in {ep_path}: {type(obj)}")
        for f in folds:
            m = f.get("model")
            if m is not None:
                self._patch_legacy_model_attrs(m)
                m.to(self.dev).eval()
        return EnsemblePredictor(folds, self.dev), labels

    # --------- legacy attribute compatibility ---------
    def _patch_legacy_model_attrs(self, model):
        """Alias old/new attribute names so forward() works across saved variants.

        Diagnostic TabularTransformer forward expects: in_norm, in_drop, embed, pos, to_tok, tx, pool, cls
        """
        long_to_short = {
            'input_norm': 'in_norm',
            'input_dropout': 'in_drop',
            'feature_embed': 'embed',
            'pos_embed': 'pos',
            'feature_transform': 'to_tok',
            'transformer': 'tx',
            'global_pool': 'pool',
            'classifier': 'cls',
        }

        # If model is TabularTransformer but lacks short names, add them
        needed_short = any(hasattr(model, k) for k in long_to_short.keys()) and not hasattr(model, 'in_drop')
        if needed_short:
            for long, short in long_to_short.items():
                if hasattr(model, long) and not hasattr(model, short):
                    setattr(model, short, getattr(model, long))

    def load_diag(self): self.diag, self.dlabels = self._load_tx("alma_shingan")

    # ---------- inference ----------
    def _prep(self, df: pd.DataFrame) -> np.ndarray:
        missing = set(self.cpg) - set(df.columns)
        if missing:
            df.loc[:, list(missing)] = 0.5
        return self.scaler.transform(
            df[self.cpg].fillna(0.5).values.astype(np.float32)
        )

    def _latent(self, X_scaled: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            _, z = self.auto(torch.from_numpy(X_scaled).to(self.dev))
        return z.cpu().numpy()

    def _load_csv_file(self, csv_path: Path) -> pd.DataFrame:
        """Load CSV file (compressed or uncompressed) with methylation beta values."""
        import gzip
        
        if csv_path.name.endswith('.csv.gz'):
            # Handle compressed CSV
            with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f, index_col=0)
        else:
            # Handle uncompressed CSV
            df = pd.read_csv(csv_path, index_col=0)
        
        df = self._ensure_features_axis(df)
        print(f"Loaded CSV with {len(df)} samples and {len(df.columns)} features")
        return df

    def predict(self, input_file: Path | str, output_file: Path | str | None = None, all_probs: bool = False):
        from .bed_utils import is_bed_file, process_bed_to_methylation
        
        input_path = Path(input_file)
        
        # Check if input is a BED file and process accordingly
        if is_bed_file(input_path):
            print(f"Detected BED file format: {input_path.name}")
            df = process_bed_to_methylation(input_path)
            df = self._ensure_features_axis(df)
        elif input_path.name.endswith('.csv.gz') or input_path.suffix == '.csv':
            # Handle CSV files (compressed or uncompressed)
            print(f"Detected CSV file format: {input_path.name}")
            df = self._load_csv_file(input_path)
        else:
            # Assume it's a pickle file
            df = pd.read_pickle(input_file)
            df = self._ensure_features_axis(df)
        
        latent = self._latent(self._prep(df))

        res = pd.DataFrame({"sample_id": df.index}, index=df.index)

        # Diagnostic
        y, c, u, p = self.diag.predict_with_conf(latent)
        res["ALMA Subtype v2"] = self.dlabels.inverse_transform(y)
        res["Diagnostic Confidence"] = c

        # Optionally include per-class probabilities
        if all_probs:
            try:
                # Map probability columns to human-readable labels in encoder order
                # Ensure we inverse transform an ordered range to align indices
                idxs = np.arange(p.shape[1])
                labels = self.dlabels.inverse_transform(idxs)
                for i, lab in enumerate(labels):
                    # Keep label as-is but prepend a clear prefix
                    col_name = f"Prob {lab}"
                    # Avoid collision just in case
                    if col_name in res.columns:
                        col_name = f"Prob_{i}_{lab}"
                    res[col_name] = p[:, i]
            except Exception as e:
                logging.warning("Failed to append class probabilities: %s", e)

        # 38‑CpG AML signature
        try:
            res = res.join(signature.hazard(df))
        except Exception as e:
            logging.warning("38CpG signature skipped: %s", e)

        out = Path(output_file) if output_file else Path(input_file).with_name("alma_predictions.csv")
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            res.to_csv(out, index=False)
        except PermissionError as e:
            raise PermissionError(f"Cannot write output CSV to '{out}'. Please choose a writable location (e.g., use --output ~/ALMA-results/yourfile.csv). Original error: {e}")
        return out
