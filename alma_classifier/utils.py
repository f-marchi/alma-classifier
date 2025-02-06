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
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            predictions.to_excel(writer, sheet_name='Predictions', index=True)
            
            # Create guide sheet
            guide = pd.DataFrame({
                'Column': [
                    'ALMA Subtype',
                    'P(Predicted Subtype)',
                    'Other potential subtype',
                    'P(other potential subtype)',
                    'AML Epigenomic Risk',
                    'P(Death) at 5y',
                    '38CpG-HazardScore',
                    '38CpG-AMLsignature'
                ],
                'Description': [
                    'Predicted WHO 2022 subtype or "Not confident" if below confidence threshold',
                    'Probability score for the predicted subtype (0-1)',
                    'Second most likely subtype (only shown for predictions with 0.5-0.8 confidence)',
                    'Probability score for the second most likely subtype',
                    'AML Epigenomic Risk classification (High/Low)',
                    'Probability of death within 5 years based on AML Epigenomic Risk',
                    'Continuous risk score based on 38 CpG signature (AML/MDS only)',
                    'Binary risk classification based on 38 CpG signature (High/Low)'
                ]
            })
            guide.to_excel(writer, sheet_name='Guide', index=False)
    elif format == 'csv':
        predictions.to_csv(output_path)
    else:
        raise ValueError("Unsupported format. Use 'excel' or 'csv'")
