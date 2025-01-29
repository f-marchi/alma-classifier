
"""Main predictor class for ALMA classifier."""
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any
from pathlib import Path

from .models import load_models
from .preprocessing import process_methylation_data, apply_pacmap

class ALMAPredictor:
    """
    ALMA (Acute Leukemia Methylome Atlas) predictor class.
    
    Provides methods for:
    - Epigenetic subtype classification
    - AML risk stratification
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize ALMA predictor.
        
        Args:
            confidence_threshold: Minimum probability threshold for predictions
        """
        self.pacmap_model, self.lgbm_models = load_models()
        self.confidence_threshold = confidence_threshold
        
    def predict(
        self,
        data: Union[pd.DataFrame, str, Path],
        sample_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate predictions for new samples.
        
        Args:
            data: Methylation beta values as DataFrame or file path
            sample_type: Optional sample type info (unused)
            
        Returns:
            DataFrame with predictions and confidence scores
        """
        # Process input data
        methyl_data = process_methylation_data(data)
        
        # Apply PaCMAP dimension reduction
        features = apply_pacmap(methyl_data, self.pacmap_model)
        
        # Generate predictions
        return self._predict_subtype(features)
    
    def _predict_subtype(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate epigenetic subtype predictions."""
        # Get model predictions
        preds = self.lgbm_models['subtype'].predict(features)
        probs = self.lgbm_models['subtype'].predict_proba(features)
        
        # Create results DataFrame
        results = pd.DataFrame(index=features.index)
        results['AL Epigenomic Subtype'] = pd.Series(preds, index=features.index)
        
        # Add probability columns
        prob_cols = [f'P({c})' for c in self.lgbm_models['subtype'].classes_]
        results[prob_cols] = pd.DataFrame(probs, index=features.index)
        
        # Add confidence indicator
        max_prob = results[prob_cols].max(axis=1)
        results['AL Epigenomic Subtype'][max_prob < self.confidence_threshold] = np.nan
        results[f'Subtype >{self.confidence_threshold*100}% Confidence'] = max_prob >= self.confidence_threshold
        
        # Check for AML or MDS and run risk prediction if found
        if any('AML' in pred or 'MDS' in pred for pred in preds):
            risk_results = self._predict_risk(features)
            results = pd.concat([results, risk_results], axis=1)
        else:
            results['AML Epigenomic Risk'] = "AML or MDS not detected"
            results['P(Remission) at 5y'] = np.nan
            results['P(Death) at 5y'] = np.nan
            results[f'Risk >{self.confidence_threshold*100}% Confidence'] = np.nan
        
        return results
    
    def _predict_risk(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate AML risk predictions."""
        # Get model predictions
        preds = self.lgbm_models['risk'].predict(features)
        probs = self.lgbm_models['risk'].predict_proba(features)
        
        # Create results DataFrame
        results = pd.DataFrame(index=features.index)
        results['AML Epigenomic Risk'] = pd.Series(preds, index=features.index)
        
        # Map predictions to risk levels
        results['AML Epigenomic Risk'] = results['AML Epigenomic Risk'].map(
            {'Alive': 'Low', 'Dead': 'High'})
        
        # Add probability columns 
        results['P(Remission) at 5y'] = probs[:,0]
        results['P(Death) at 5y'] = probs[:,1]
        
        # Add confidence indicator
        max_prob = np.max(probs, axis=1)
        results['AML Epigenomic Risk'][max_prob < self.confidence_threshold] = np.nan
        results[f'Risk >{self.confidence_threshold*100}% Confidence'] = max_prob >= self.confidence_threshold
        
        return results
