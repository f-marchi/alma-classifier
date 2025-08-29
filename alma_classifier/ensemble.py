"""
Ensemble predictor for ALMA Subtype v2.
"""
import numpy as np


class EnsemblePredictor:
    """Averaging‑probability ensemble over k folds."""
    def __init__(self, folds, device):
        self.folds = folds
        self.device = device

    def _proba(self, X: np.ndarray) -> np.ndarray:
        """Compute average probabilities across all folds."""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("PyTorch is required for ALMA Subtype v2 model")
            
        preds = []
        for fm in self.folds:
            proc = fm["processor"]
            Xp = proc.power_transformer.transform(X) if proc.power_transformer else X
            Xp = proc.scaler.transform(Xp)
            with torch.no_grad():  # Ensure no gradients are computed
                logits = fm["model"](torch.from_numpy(Xp).float().to(self.device))
                preds.append(F.softmax(logits, 1).cpu().numpy())
        return np.mean(preds, axis=0)

    def predict_with_conf(self, X: np.ndarray):
        """Generate predictions with confidence scores."""
        p = self._proba(X)
        y = p.argmax(1)
        conf = p.max(1)
        unc = (-np.sum(p * np.log(p + 1e-8), 1) / np.log(p.shape[1]))
        return y, conf, unc, p
