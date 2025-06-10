"""Data preprocessing utilities."""
import numpy as np
import pandas as pd
import gzip
import joblib
import warnings
from typing import Tuple, Union, Optional, Any
from pathlib import Path
from .bed_processing import process_bed_to_methylation, is_bed_file

def read_feature_reference(ref_path: Union[str, Path]) -> pd.Series:
    """Read feature reference file and return CpG names."""
    ref_df = pd.read_csv(ref_path, sep='\t', header=None, usecols=[3], names=['cpg_name'])
    return ref_df['cpg_name']

def load_model_and_impute(df: pd.DataFrame, model_path: Union[str, Path]) -> pd.DataFrame:
    """Load imputer model and impute missing features."""
    loaded_imputer = joblib.load(model_path)
    imputer_features = loaded_imputer.feature_names_in_
    df_aligned = df.reindex(columns=imputer_features)
    imputed_data = loaded_imputer.transform(df_aligned)
    imputed_df = pd.DataFrame(imputed_data, columns=imputer_features, index=df_aligned.index)
    return imputed_df.round(3).astype('float32')

def process_methylation_data(
    data: Union[pd.DataFrame, str, Path],
    columns: Optional[list] = None
) -> pd.DataFrame:
    """Process methylation beta values data."""
    # Load data if path provided
    if isinstance(data, (str, Path)):
        data_path = Path(data)
        
        # Check if it's a BED file
        if is_bed_file(data_path):
            df = process_bed_to_methylation(data_path)
        elif str(data_path).endswith('.pkl'):
            df = pd.read_pickle(data_path)
        elif str(data_path).endswith('.csv'):
            df = pd.read_csv(data_path, index_col=0)
        else:
            raise ValueError("Unsupported file format. Use .pkl, .csv, .bed, or .bed.gz")
    else:
        df = data.copy()

    # Validate data format
    if df.empty:
        raise ValueError("Empty dataset provided")
    
    # Get reference features
    ref_path = Path(__file__).parent / "data" / "pacmap_reference.bed"
    reference_features = read_feature_reference(ref_path)
    
    # Filter to only keep features present in reference
    common_features = df.columns.intersection(reference_features)
    if len(common_features) == 0:
        raise ValueError("No matching features found between input data and reference")
    
    df = df[common_features]
    
    # Load imputer model and handle missing values
    model_path = Path(__file__).parent / "models" / "imputer_model.joblib"
    df = load_model_and_impute(df, model_path)
    
    # Validate beta values
    if df.min().min() < -2 or df.max().max() > 2:
        raise ValueError("Methylation values exceed expected range [0, 1]. Are you sure these are beta values?")
        
    return df

def apply_pacmap(
    data: pd.DataFrame,
    pacmap_model: Any
) -> pd.DataFrame:
    """Apply PaCMAP dimension reduction."""
    embedding = pacmap_model.transform(data.to_numpy(dtype='float16'))
    cols = [f'PaCMAP {i+1} of 5' for i in range(5)]
    df = pd.DataFrame(embedding, columns=cols, index=data.index)

    return df