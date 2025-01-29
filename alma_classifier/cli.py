"""Command-line interface for ALMA classifier."""
import click
from pathlib import Path
import sys

from .predictor import ALMAPredictor
from .utils import export_results

try:
    import pacmap
except ImportError:
    click.echo("Warning: pacmap package not found. Please install it with: pip install pacmap", err=True)
    sys.exit(1)

@click.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Path to input methylation data file (.pkl, .csv, .xlsx)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Path for output results file (.xlsx)')
@click.option('--sample-type', '-s', type=str,
              help='Optional sample type information')
@click.option('--confidence', '-c', type=float, default=0.5,
              help='Confidence threshold for predictions (default: 0.5)')
def main(input, output, sample_type, confidence):
    """ALMA Classifier - Epigenomic classification for methylation data."""
    try:
        # Initialize predictor
        predictor = ALMAPredictor(confidence_threshold=confidence)
        
        # Generate predictions
        click.echo(f"Processing input file: {input}")
        results = predictor.predict(
            data=input,
            sample_type=sample_type
        )
        
        # Export results
        click.echo(f"Saving results to: {output}")
        export_results(results, output, format='excel')
        
        click.echo("Classification completed successfully!")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
