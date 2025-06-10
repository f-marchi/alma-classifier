"""Command line interface for ALMA classifier."""
import argparse
import sys
import warnings
from pathlib import Path
from .predictor import ALMAPredictor
from .utils import export_results
from .models import validate_models

def main():
    """Execute ALMA classifier from command line."""
    parser = argparse.ArgumentParser(
        description="ALMA Classifier - Epigenomic classification for methylation data"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run classifier on example dataset"
    )
    parser.add_argument(
        "--input",
        required=False,
        type=str,
        help="Path to input methylation data file (.pkl, .csv, .bed, or .bed.gz)"
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

    args = parser.parse_args()
    
    try:
        if not args.demo and not args.input:
            parser.error("--input is required when not using --demo")
            
        if args.demo:
            from pathlib import Path
            args.input = str(Path(__file__).parent / "data" / "example_dataset.pkl")
            if not args.output:
                args.output = "example_predictions.xlsx"
            print(f"Running demo with example dataset: {args.input}")
            print(f"Results will be saved to: {args.output}")

        # Check if pacmap is installed
        try:
            import pacmap
        except ImportError as e:
            missing_pkg = str(e).split("'")[1]
            print(f"Error: {missing_pkg} package is required but not installed.")
            print(f"Please install it using: pip install {missing_pkg}")
            sys.exit(1)
            
        # Validate models
        models_valid, error_msg = validate_models()
        if not models_valid:
            print("Error: Missing model files")
            print(error_msg)
            sys.exit(1)
            
        # Initialize predictor
        predictor = ALMAPredictor(confidence_threshold=args.confidence)
        
        print("Starting prediction process...")
        
        # Configure warnings to be more visible
        warnings.simplefilter("always", UserWarning)
        
        # Generate predictions
        results = predictor.predict(
            data=args.input,
        )
        
        # Export results
        output_format = 'excel' if args.output.endswith('.xlsx') else 'csv'
        export_results(results, args.output, format=output_format)
        
        print(f"Successfully generated predictions: {args.output}")
        
    except ValueError as e:
        # Handle CpG coverage errors and other validation errors
        print(f"Data validation error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()