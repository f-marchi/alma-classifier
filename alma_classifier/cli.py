"""Command line interface for ALMA classifier."""
import argparse
import sys
from pathlib import Path
from .predictor import ALMAPredictor
from .utils import export_results

def main():
    """Execute ALMA classifier from command line."""
    parser = argparse.ArgumentParser(
        description="ALMA Classifier - Epigenomic classification for methylation data"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to input methylation data file (.pkl, .csv, or .xlsx)"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path for output predictions (.xlsx or .csv)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions (default: 0.5)"
    )
    parser.add_argument(
        "--sample-type",
        type=str,
        help="Optional sample type information"
    )
    
    args = parser.parse_args()
    
    try:
        # Check if pacmap is installed
        try:
            import pacmap
        except ImportError:
            print("Error: pacmap package is required but not installed.")
            print("Please install it using: pip install pacmap==0.7.0")
            sys.exit(1)
            
        # Initialize predictor
        predictor = ALMAPredictor(confidence_threshold=args.confidence)
        
        # Generate predictions
        results = predictor.predict(
            data=args.input,
            sample_type=args.sample_type
        )
        
        # Export results
        output_format = 'excel' if args.output.endswith('.xlsx') else 'csv'
        export_results(results, args.output, format=output_format)
        
        print(f"Successfully generated predictions: {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
