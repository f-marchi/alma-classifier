import numpy as np
import torch
import torch.nn.functional as F


class EnsemblePredictor:
    """Averaging‑probability ensemble over k folds."""
    def __init__(self, folds, device: torch.device):
        # Accept either list of folds or an object with fold_models
        if hasattr(folds, 'fold_models') and not isinstance(folds, list):
            folds = getattr(folds, 'fold_models')
        self.folds = folds
        self.device = device
        # Provide legacy attribute names for code expecting them
        self.fold_models = self.folds
        self.num_models = len(self.folds)

    def __setstate__(self, state):  # for pickle backward compatibility
        self.__dict__.update(state)
        # Normalize attribute names
        if not hasattr(self, 'folds') and hasattr(self, 'fold_models'):
            self.folds = self.fold_models
        if not hasattr(self, 'fold_models') and hasattr(self, 'folds'):
            self.fold_models = self.folds
        if not hasattr(self, 'num_models'):
            self.num_models = len(self.folds)

    def _proba(self, X: np.ndarray) -> np.ndarray:
        preds = []
        # Support legacy attribute 'fold_models'
        fold_list = self.folds if hasattr(self, 'folds') else getattr(self, 'fold_models')
        for fm in fold_list:
            proc = fm["processor"]
            Xp = proc.power_transformer.transform(X) if proc.power_transformer else X
            Xp = proc.scaler.transform(Xp)
            with torch.no_grad():  # Ensure no gradients are computed
                logits = fm["model"](torch.from_numpy(Xp).float().to(self.device))
                preds.append(F.softmax(logits, 1).cpu().numpy())
        return np.mean(preds, axis=0)

    def predict_with_conf(self, X: np.ndarray):
        p = self._proba(X)
        y = p.argmax(1)
        conf = p.max(1)
        unc = (-np.sum(p * np.log(p + 1e-8), 1) / np.log(p.shape[1]))
        return y, conf, unc, p
