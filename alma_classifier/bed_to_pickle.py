import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Column names remain the same as before
BED_COLUMNS = [
    "chrom", "start_position", "end_position", "modified_base_code", "score",
    "strand", "start_position2", "end_position2", "color", "Nvalid_cov",
    "fraction_modified", "Nmod", "Ncanonical", "Nother_mod", "Ndelete",
    "Nfail", "Ndiff", "Nnocall"
]

def read_pacmap_reference(file_path):
    ref_df = pd.read_csv(
        file_path, 
        sep='\t', 
        usecols=['chrm', 'start', 'name'],
        names=['chrm', 'start', 'end', 'name', 'score', 'strand'], 
        dtype={'chrm': str, 'start': int, 'name': str}
    )
    
    # Create coordinate column
    ref_df['coordinate'] = ref_df['chrm'] + ':' + ref_df['start'].astype(str)
    
    # Handle duplicate coordinates by creating a mapping dictionary
    coord_to_names = ref_df.groupby('coordinate')['name'].agg(list).to_dict()
    
    return ref_df[['name']].set_index('name'), coord_to_names

def read_sample_data(file_path):
    return pd.read_csv(
        file_path, 
        sep='\t', 
        usecols=['chrom', 'start_position', 'modified_base_code', 'fraction_modified'],
        names=BED_COLUMNS, 
        dtype={'chrom': str, 'start_position': int, 'modified_base_code': str, 'fraction_modified': float}
    )

def process_sample(ref_df, coord_to_names, sample_df, sample_name):
    # Create coordinate column for sample
    sample_df['coordinate'] = sample_df['chrom'] + ':' + sample_df['start_position'].astype(str)
    
    # Create a dictionary to store beta values for all CpG names
    beta_values = {}
    
    # Process each coordinate in the sample
    for coord, frac_mod in zip(sample_df['coordinate'], sample_df['fraction_modified']):
        if coord in coord_to_names:
            # Get all CpG names for this coordinate
            cpg_names = coord_to_names[coord]
            beta = round(frac_mod / 100, 3)
            # Assign the same beta value to all CpGs at this coordinate
            for name in cpg_names:
                beta_values[name] = beta
    
    # Create a series with all reference CpGs
    result = pd.Series(beta_values, name=sample_name)
    result = result.reindex(ref_df.index)
    
    return result

def process_directory(directory_path, ref_df, coord_to_names):
    results = []
    bed_files = sorted(directory_path.glob('*.bed'))
    
    print(f"Found {len(bed_files)} BED files to process")
    for bed_file in bed_files:
        sample_name = bed_file.stem
        print(f"Processing {sample_name}...")
        sample_df = read_sample_data(bed_file)
        
        # Process sample
        result = process_sample(ref_df, coord_to_names, sample_df, sample_name)
        results.append(result)
    
    # Concatenate all results
    return pd.concat(results, axis=1).T.sort_index(axis=1)

def main():
    parser = argparse.ArgumentParser(description='Convert BED files to pickle format')
    parser.add_argument('--bed-dir', type=str, default='bed',
                        help='Directory containing BED files (default: bed)')
    parser.add_argument('--pacmap-reference', type=str, required=True,
                        help='Path to pacmap reference BED file')
    parser.add_argument('--output-dir', type=str, default='bed',
                        help='Output directory for pickle file (default: bed)')
    
    args = parser.parse_args()
    
    bed_dir = Path(args.bed_dir)
    pacmap_reference_path = Path(args.pacmap_reference)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Reading pacmap reference...")
    ref_df, coord_to_names = read_pacmap_reference(pacmap_reference_path)
    
    print("Processing BED files...")
    result = process_directory(bed_dir, ref_df, coord_to_names)
    
    dataset_title = f'{result.shape[0]}samples_{result.shape[1]}cpgs_nanopore_bvalues.pkl'
    output_path = output_dir / dataset_title
    
    print(f"Saving results to {output_path}...")
    result.to_pickle(output_path)
    print("Done!")
    
    return result

if __name__ == '__main__':
    main()

# Example usage (Python3.8)
# python bed_to_pickle.py --pacmap-reference ref/pacmap_reference.bed