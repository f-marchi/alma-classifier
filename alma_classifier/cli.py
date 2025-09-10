"""
Command-line interface for ALMA classifiers.
"""
import argparse
import sys
from pathlib import Path

from .core import ALMA
from .utils import set_deterministic
from .download import download_models, get_demo_data_path

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="alma-classifier",
        description="🩸🧬 ALMA Classifier – Epigenomic diagnosis of acute leukemia (research use only) 🧬🩸"
    )
    ap.add_argument("-i", "--input_data", help="Input file: .pkl with β‑values, .csv/.csv.gz with β‑values, or .bed/.bed.gz nanopore file")
    ap.add_argument("-o", "--output", help=".csv output (default: alongside input data)")
    ap.add_argument("--download-models", action="store_true", help="Download model weights from GitHub release")
    ap.add_argument("--demo", action="store_true", help="Run demo with example dataset")
    ap.add_argument("--all_probs", action="store_true", help="Include all subtype/class probabilities as separate columns in the output")
    # If the user invoked the command with no arguments at all, show help (equivalent to -h)
    if len(sys.argv) == 1:
        ap.print_help()
        return

    args = ap.parse_args()

    # Handle model download
    if args.download_models:
        success = download_models()
        if not success:
            print("Failed to download models. Please check your internet connection and try again.")
            return
        print("Models downloaded successfully!")
        return

    # Handle demo mode
    if args.demo:
        demo_data = get_demo_data_path()
        if not demo_data:
            print("Demo data not found. Please run 'alma-classifier --download-models' first.")
            return
        
        print(f"Running demo with example dataset: {demo_data}")
        # Convert demo data to pkl format if needed (assuming it's CSV)
        if demo_data.suffix == '.gz' and demo_data.stem.endswith('.csv'):
            import pandas as pd
            import gzip
            with gzip.open(demo_data, 'rt') as f:
                demo_df = pd.read_csv(f, index_col=0)
            temp_pkl = demo_data.parent / "demo_temp.pkl"
            demo_df.to_pickle(temp_pkl)
            input_data = temp_pkl
        elif demo_data.suffix == '.csv':
            import pandas as pd
            demo_df = pd.read_csv(demo_data, index_col=0)
            temp_pkl = demo_data.parent / "demo_temp.pkl"
            demo_df.to_pickle(temp_pkl)
            input_data = temp_pkl
        else:
            input_data = demo_data
        
        # Set output to a demo results file in results/ folder
        results_dir = Path.cwd() / "results"
        results_dir.mkdir(exist_ok=True)
        output_file = results_dir / "demo_predictions.csv"
    else:
        # Regular mode requires input data
        if not args.input_data:
            ap.error("--input_data is required (unless using --demo or --download-models)")
        input_data = args.input_data
        
        # If no output specified, create results/ folder and use default name
        if args.output is None:
            results_dir = Path.cwd() / "results"
            results_dir.mkdir(exist_ok=True)
            input_path = Path(input_data)
            # Handle different file extensions appropriately
            if input_path.name.endswith('.bed.gz'):
                stem = input_path.name.replace('.bed.gz', '')
            elif input_path.name.endswith('.csv.gz'):
                stem = input_path.name.replace('.csv.gz', '')
            elif input_path.suffix in ['.bed', '.csv']:
                stem = input_path.stem
            else:
                stem = input_path.stem
            output_file = results_dir / f"{stem}_predictions.csv"
        else:
            output_file = args.output

    set_deterministic()
    alma = ALMA()
    alma.load_auto(); alma.load_diag()
    out = alma.predict(input_data, output_file, all_probs=args.all_probs)
    print(f"Predictions saved to: {out}")

    # Clean up temp file if created for demo
    if args.demo and 'temp_pkl' in locals():
        temp_pkl.unlink(missing_ok=True)


"""Executable entry‑point registered in pyproject as the console‑script."""
def main_cli():
    """Entry point for the alma-classifier command."""
    main()

if __name__ == "__main__":
    main_cli()
