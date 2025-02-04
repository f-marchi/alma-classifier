"""Main predictor class for ALMA classifier."""
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path
from .models import load_models
from .preprocessing import process_methylation_data, apply_pacmap

class ALMAPredictor:
    """
    ALMA (Acute Leukemia Methylome Atlas) predictor class.
    
    Provides methods for:
    - Epigenetic subtype classification
    - AML risk stratification for AML/MDS samples only
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
        include_38cpg: bool = True,
        show_progress: bool = True) -> pd.DataFrame:
        """
        Generate predictions for new samples.
        
        Args:
            data: Methylation beta values as DataFrame or file path
            include_38cpg: Whether to include 38CpG AML signature predictions
            
        Returns:
            DataFrame with predictions and confidence scores
        """
        # Process input data
        methyl_data = process_methylation_data(data)
        
        # Apply PaCMAP dimension reduction
        features = apply_pacmap(methyl_data, self.pacmap_model)
        
        # Generate subtype predictions first
        subtype_results = self._predict_subtype(features)
        
        # Initialize empty signature results
        signature_results = pd.DataFrame(index=methyl_data.index)
        signature_columns = ['38CpG-HazardScore', '38CpG-AMLsignature']
        for col in signature_columns:
            signature_results[col] = np.nan
            
        # Generate 38CpG signature predictions if requested, only for AML/MDS samples
        if include_38cpg:
            from .aml_signature import generate_coxph_score
            is_aml_mds = subtype_results['ALMA Subtype'].str.startswith(('AML', 'MDS'), na=False)
            
            if is_aml_mds.any():
                aml_mds_signatures = generate_coxph_score(methyl_data[is_aml_mds])
                signature_results.loc[is_aml_mds] = aml_mds_signatures
        
        # Generate subtype predictions first
        subtype_results = self._predict_subtype(features)
        
        # Initialize empty risk results with same index as features
        risk_results = pd.DataFrame(index=features.index)
        
        # Add risk prediction columns with NaN values
        risk_columns = ['AML Epigenomic Risk', 'P(Death) at 5y']
        for col in risk_columns:
            risk_results[col] = np.nan
        
        # Generate risk predictions only for AML/MDS samples
        is_aml_mds = subtype_results['ALMA Subtype'].str.startswith(('AML', 'MDS'), na=False)
        
        if is_aml_mds.any():
            aml_features = features[is_aml_mds]
            risk_predictions = self._predict_risk(aml_features)
            # Update only the rows that have AML/MDS predictions
            risk_results.loc[is_aml_mds] = risk_predictions
            
        # Combine all results
        return pd.concat([subtype_results, risk_results, signature_results], axis=1)
    
    def _predict_subtype(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate epigenetic subtype predictions."""
        # Get model predictions
        preds = self.lgbm_models['subtype'].predict(features)
        probs = self.lgbm_models['subtype'].predict_proba(features)
        
        # Create results DataFrame
        results = pd.DataFrame(index=features.index)
        results['ALMA Subtype'] = pd.Series(preds, index=features.index)
        
        # Add probability only for predicted class
        predicted_probs = np.array([probs[i, self.lgbm_models['subtype'].classes_ == pred][0] 
                                  for i, pred in enumerate(preds)])
        results['P(Predicted Subtype)'] = predicted_probs
        
        # Replace low confidence predictions with "Not confident"
        results.loc[predicted_probs < self.confidence_threshold, 'ALMA Subtype'] = "Not confident"
        
        # Add second most probable subtype for predictions between 0.5 and 0.8
        second_best_mask = (predicted_probs >= 0.5) & (predicted_probs < 0.8)
        results['Other potential subtype'] = np.nan
        results['P(other potential subtype)'] = np.nan
        
        for i in range(len(features)):
            if second_best_mask[i]:
                sorted_indices = np.argsort(probs[i])[::-1]
                second_best_class = self.lgbm_models['subtype'].classes_[sorted_indices[1]]
                results.loc[features.index[i], 'Other potential subtype'] = second_best_class
                results.loc[features.index[i], 'P(other potential subtype)'] = probs[i][sorted_indices[1]]
        
        return results
    
    def _predict_risk(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate AML risk predictions for AML/MDS samples."""
        # Get model predictions
        preds = self.lgbm_models['risk'].predict(features)
        probs = self.lgbm_models['risk'].predict_proba(features)
        
        # Create results DataFrame
        results = pd.DataFrame(index=features.index)
        results['AML Epigenomic Risk'] = pd.Series(preds, index=features.index)
        
        # Map predictions to risk levels
        results['AML Epigenomic Risk'] = results['AML Epigenomic Risk'].map(
            {'Alive': 'Low', 'Dead': 'High'})
        
        # Add only P(Death) probability
        results['P(Death) at 5y'] = probs[:,1]
        
        # Replace low confidence predictions with "Not confident"
        max_prob = np.max(probs, axis=1)
        results.loc[max_prob < self.confidence_threshold, 'AML Epigenomic Risk'] = "Not confident"
        
        return results