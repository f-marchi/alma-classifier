"""Utility functions for ALMA classifier."""
import pandas as pd
from typing import Optional

def check_lymphoblastic_samples(
    risk_predictions: pd.DataFrame,
    sample_type: pd.Series
) -> pd.DataFrame:
    """
    Remove ALL samples from risk predictions.
    
    Args:
        risk_predictions: DataFrame with risk predictions
        sample_type: Series with sample type information
        
    Returns:
        Filtered risk predictions
    """
    all_indicators = [
        'NOPHO ALL92-2000',
        'French GRAALL 2003–2005', 
        'TARGET ALL'
    ]
    
    if any(ind in sample_type for ind in all_indicators):
        return pd.DataFrame(index=risk_predictions.index)
    return risk_predictions

def export_results(
    predictions: pd.DataFrame,
    output_path: str,
    format: str = 'excel'
) -> None:
    """
    Export prediction results to file.
    
    Args:
        predictions: DataFrame with predictions
        output_path: Path to save results
        format: Output format ('excel' or 'csv')
    """
    if format == 'excel':
        predictions.to_excel(output_path)
    elif format == 'csv':
        predictions.to_csv(output_path)
    else:
        raise ValueError("Unsupported format. Use 'excel' or 'csv'")
