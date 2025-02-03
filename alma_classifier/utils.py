"""Utility functions for ALMA classifier."""
import pandas as pd

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
    # Round float columns to 3 decimal places
    float_cols = predictions.select_dtypes(include=['float64']).columns
    predictions[float_cols] = predictions[float_cols].round(3)
    
    if format == 'excel':
        predictions.to_excel(output_path)
    elif format == 'csv':
        predictions.to_csv(output_path)
    else:
        raise ValueError("Unsupported format. Use 'excel' or 'csv'")
