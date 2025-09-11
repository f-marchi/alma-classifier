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
    def _ensure_parent_writable(path: Path) -> bool:
        """Ensure parent dir exists and is writable. Returns True if OK, else False."""
        parent = path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            test = parent / ".__alma_write_test__"
            with open(test, "w") as _:
                pass
            test.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def _pick_output_path(preferred_dir: Path, filename: str) -> Path:
        """Pick a writable output path. If preferred_dir isn't writable, fallback to ~/ALMA-results."""
        preferred = preferred_dir / filename
        if _ensure_parent_writable(preferred):
            return preferred
        # Fallback in home
        home_dir = Path.home() / "ALMA-results"
        home_dir.mkdir(parents=True, exist_ok=True)
        alt = home_dir / filename
        # Last resort: ensure we can write there; if not, use CWD
        if _ensure_parent_writable(alt):
            print(f"Results directory not writable: {preferred_dir}. Saving to: {alt.parent}")
            return alt
        cwd_alt = Path.cwd() / filename
        if _ensure_parent_writable(cwd_alt):
            print(f"Results directory not writable: {preferred_dir}. Saving to current directory: {cwd_alt.parent}")
            return cwd_alt
        # Give up: return preferred (will error in predict), but with a clearer message later
        return preferred

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

        # Pick output: honor --output if provided, else default to CWD
        if args.output:
            output_file = Path(args.output)
            if not _ensure_parent_writable(output_file):
                ap.error(f"Cannot write to the specified output path: {output_file}. Choose a writable location.")
        else:
            output_file = _pick_output_path(Path.cwd(), "demo_predictions.csv")
    else:
        # Regular mode requires input data
        if not args.input_data:
            ap.error("--input_data is required (unless using --demo or --download-models)")
        input_data = args.input_data
        
        # If no output specified, default to current working directory
        if args.output is None:
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
            output_file = _pick_output_path(Path.cwd(), f"{stem}_predictions.csv")
        else:
            output_file = Path(args.output)
            if not _ensure_parent_writable(output_file):
                ap.error(f"Cannot write to the specified output path: {output_file}. Choose a writable location.")

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
