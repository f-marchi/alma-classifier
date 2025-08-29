"""Main predictor class for ALMA classifier."""
import numpy as np
import pandas as pd
from typing import Union, Optional
from pathlib import Path
from .models import load_models, validate_models_v2
from .preprocessing import process_methylation_data, apply_pacmap

class ALMAPredictor:
    """
    ALMA (Acute Leukemia Methylome Atlas) predictor class.
    
    Provides methods for:
    - Epigenetic subtype classification
    - AML risk stratification for AML/MDS samples only
    - ALMA Subtype v2 predictions (transformer-based)
    """
    
    def __init__(self, confidence_threshold: float = 0.5, include_v2: bool = False):
        """
        Initialize ALMA predictor.
        
        Args:
            confidence_threshold: Minimum probability threshold for predictions
            include_v2: Whether to include ALMA Subtype v2 predictions
        """
        self.pacmap_model, self.lgbm_models = load_models()
        self.confidence_threshold = confidence_threshold
        self.include_v2 = include_v2
        self.alma_v2 = None
        
        # Initialize v2 model if requested
        if self.include_v2:
            self._init_v2_model()
    
    def _init_v2_model(self):
        """Initialize ALMA v2 model."""
        try:
            # Check if v2 models are available
            models_valid, error_msg = validate_models_v2()
            if not models_valid:
                print(f"Warning: {error_msg}")
                print("ALMA Subtype v2 will be disabled.")
                self.include_v2 = False
                return
                
            from .alma_v2_core import ALMAv2
            self.alma_v2 = ALMAv2()
            
            # Load the models
            try:
                self.alma_v2.load_auto()
                self.alma_v2.load_diag()
                print("ALMA Subtype v2 models loaded successfully.")
            except ImportError as e:
                print(f"Warning: {str(e)}")
                print("ALMA Subtype v2 will be disabled.")
                self.include_v2 = False
                self.alma_v2 = None
            except Exception as e:
                print(f"Warning: Error loading ALMA v2 models: {str(e)}")
                print("ALMA Subtype v2 will be disabled.")
                self.include_v2 = False
                self.alma_v2 = None
                
        except ImportError:
            print("Warning: ALMA Subtype v2 dependencies not available.")
            print("ALMA Subtype v2 will be disabled.")
            self.include_v2 = False
        
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
        
        # Generate v2 predictions if enabled
        v2_results = pd.DataFrame(index=methyl_data.index)
        if self.include_v2 and self.alma_v2 is not None:
            try:
                # Save methylation data temporarily for v2 prediction
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                    methyl_data.to_pickle(tmp_file.name)
                    
                    # Get v2 predictions
                    v2_output = self.alma_v2.predict(tmp_file.name)
                    v2_df = pd.read_csv(v2_output)
                    
                    # Align with original index
                    v2_results['ALMA Subtype v2'] = pd.Series(
                        v2_df['ALMA Subtype v2'].values, 
                        index=methyl_data.index
                    )
                    v2_results['Diagnostic Confidence v2'] = pd.Series(
                        v2_df['Diagnostic Confidence v2'].values, 
                        index=methyl_data.index
                    )
                    
                    # Clean up temp files
                    import os
                    os.unlink(tmp_file.name)
                    if v2_output.exists():
                        v2_output.unlink()
                        
            except Exception as e:
                print(f"Warning: Error generating ALMA v2 predictions: {str(e)}")
                v2_results['ALMA Subtype v2'] = np.nan
                v2_results['Diagnostic Confidence v2'] = np.nan
        else:
            v2_results['ALMA Subtype v2'] = np.nan
            v2_results['Diagnostic Confidence v2'] = np.nan
            
        # Combine all results
        return pd.concat([subtype_results, risk_results, signature_results, v2_results], axis=1)
    
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